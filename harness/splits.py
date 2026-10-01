"""search/dev/held-out data splits, per docs/plan.md section 4's anti-cheating
rules: "Three splits, never mixed... Store held-out hashes in the database
and refuse any run that reads them outside a final-report flag."

data/splits.json (not the database) is the source of truth. A split's
identity comes from hashing its actual content, never from a CLI path
string or filename -- otherwise `--eval-split dev --kld-text held_out.txt`
would silently mislabel held-out data as dev with no warning. db.py mirrors
this manifest into a `split_files` table on every connect(), so
`db.rebuild()` (which deletes the .duckdb file) can't lose the lock.

Week 4 ("red-team your own verifier") is where this gets attacked on
purpose. This module only needs to build the mechanism, not prove it's
unbreakable.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import re
from pathlib import Path

MANIFEST_PATH = Path("data/splits.json")
SPLITS = ("search", "dev", "held_out")


class HeldOutViolation(PermissionError):
    pass


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def paragraphs(text: str, min_words: int = 5) -> list[str]:
    """Blank-line-separated, whitespace-normalized chunks -- the unit both
    paragraph hashing (held-out leak detection) and the long-context
    haystack builder work in.

    Drops chunks under min_words: Wikipedia's plain-text extracts are full of
    structural noise shorter than that -- section headers ("== References
    =="), bare table-cell fragments left over from infobox markup ("|", "2",
    "1/2") -- that recur verbatim across any two unrelated articles. Verified
    live: comparing dev.txt against held_out.txt without this filter flagged
    157 "shared paragraphs" that were entirely this kind of boilerplate (the
    longest was 4 words), not real content overlap, which would have made
    the held-out leak check useless noise. 5 words comfortably exceeds every
    observed boilerplate fragment while still catching real sentences."""
    parts = re.split(r"\n\s*\n", text)
    return [" ".join(p.split()) for p in parts if len(p.split()) >= min_words]


def paragraph_hashes(text: str) -> list[str]:
    return [_sha256(p.encode())[:16] for p in paragraphs(text)]


def _empty_manifest() -> dict:
    return {"version": 1, "text": {}, "tasks": {}}


def load_manifest(path: Path = MANIFEST_PATH) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found -- build data/search.txt and data/held_out.txt "
                                 f"(see scripts/fetch_wiki.py) and lock_held_out() first")
    return json.loads(path.read_text())


def classify_text(path: Path, manifest: dict) -> str:
    """Which split a file belongs to, by content hash -- never by path or
    filename, so a copy or rename can't masquerade as a different split.

    Hashes the decoded text re-encoded as UTF-8 (not raw bytes): on Windows,
    text-mode writes translate "\\n" to "\\r\\n", so read_bytes() and
    read_text().encode() disagree on line endings for the exact same file.
    Every hash in this module must go through this same path-independent
    encoding or locked/computed hashes silently stop matching."""
    sha = _sha256(path.read_text(encoding="utf-8").encode("utf-8"))
    for name in SPLITS:
        entry = manifest["text"].get(name)
        if entry and entry["sha256"] == sha:
            return name
    return "adhoc"


def load_eval_text(path: Path, manifest: dict, allow_held_out: bool) -> tuple[str, str, str]:
    """CHOKEPOINT: every quality measure that reads free text (KL, the
    long-context haystack) calls this instead of Path.read_text directly.

    Returns (text, split, sha256_short). Raises HeldOutViolation if:
      - the file IS the held-out split and allow_held_out is false
      - the held-out file's content no longer matches its locked hash
        (edited after locking -- always refused, flag or not)
      - the file isn't held-out itself but shares a paragraph with it,
        and allow_held_out is false (catches "copy half of held_out.txt
        into my own file" leaks, not just using the file directly)
    """
    text = path.read_text(encoding="utf-8")
    sha_full = _sha256(text.encode())
    split = classify_text(path, manifest)

    # classify_text identifies by content hash, which means drifted held-out
    # content (edited after locking) no longer hashes to anything and comes
    # back "adhoc" -- silently skipping the held-out check entirely. Also
    # check by path, so editing the held-out file in place is still caught.
    held_out_entry = manifest["text"].get("held_out")
    is_held_out_path = held_out_entry and path.resolve() == Path(held_out_entry["path"]).resolve()

    if split == "held_out" or is_held_out_path:
        locked = manifest["text"]["held_out"]["sha256"]
        if sha_full != locked:
            raise HeldOutViolation(f"{path} has changed since it was locked -- held-out set must not be edited")
        if not allow_held_out:
            raise HeldOutViolation(f"{path} is the held-out split; pass --allow-held-out for final reporting only")
        return text, "held_out", sha_full[:16]

    if not allow_held_out and "held_out" in manifest["text"]:
        held_paras = set(manifest["text"]["held_out"]["paragraph_sha256"])
        overlap = held_paras & set(paragraph_hashes(text))
        if overlap:
            raise HeldOutViolation(
                f"{path} shares {len(overlap)} paragraph(s) with the held-out split -- "
                "refusing without --allow-held-out")
    return text, split, sha_full[:16]


def task_doc_indices(task: str, split: str, manifest: dict, allow_held_out: bool,
                      limit: int | None = None) -> list[int]:
    """CHOKEPOINT for lm-eval task items: deterministic partition of a task's
    docs into search/dev/held_out by hashing (seed, task, doc_index) -- no
    overlap by construction, and reproducible without storing the partition.
    Raises HeldOutViolation for split='held_out' without allow_held_out."""
    if split == "held_out" and not allow_held_out:
        raise HeldOutViolation(f"task {task!r} split 'held_out' requires --allow-held-out")
    cfg = manifest.get("tasks", {}).get(task)
    if cfg is None:
        raise KeyError(f"no split config for task {task!r} in {MANIFEST_PATH} -- "
                        f"add one, or use split='unsplit' to run the whole task")
    n, seed, fractions = cfg["n_docs"], cfg["seed"], cfg["fractions"]
    order = sorted(range(n), key=lambda i: hashlib.sha256(f"{seed}:{task}:{i}".encode()).hexdigest())
    start = 0
    parts = {}
    for name in SPLITS:
        end = start + round(fractions.get(name, 0.0) * n)
        parts[name] = order[start:end]
        start = end
    indices = parts.get(split, [])
    return indices[:limit] if limit else indices


def record_text_split(name: str, path: Path, titles: list[str] | None = None,
                       revids: list[str] | None = None, manifest_path: Path = MANIFEST_PATH) -> dict:
    """Record the search or dev split's hash/provenance. Not for held_out --
    use lock_held_out, which also stores paragraph hashes and refuses to
    silently relock different content."""
    assert name in ("search", "dev"), "use lock_held_out() for the held_out split"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else _empty_manifest()
    text = path.read_text(encoding="utf-8")
    manifest["text"][name] = {"path": str(path), "sha256": _sha256(text.encode()),
                               "titles": titles or [], "revids": revids or []}
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return manifest["text"][name]


def lock_held_out(path: Path, titles: list[str] | None = None, revids: list[str] | None = None,
                   manifest_path: Path = MANIFEST_PATH) -> dict:
    """Locks the held-out split: records its whole-file hash and per-paragraph
    hashes. Refuses to silently relock a held_out entry to different content
    (delete the manifest entry yourself first if you really mean to replace it
    -- that's a deliberate speed bump, not a usability bug)."""
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else _empty_manifest()
    text = path.read_text(encoding="utf-8")
    sha = _sha256(text.encode())
    existing = manifest["text"].get("held_out")
    if existing and existing["sha256"] != sha:
        raise ValueError(f"held_out is already locked to different content than {path} -- "
                          f"delete manifest['text']['held_out'] in {manifest_path} first if this is deliberate")
    manifest["text"]["held_out"] = {
        "path": str(path), "sha256": sha, "titles": titles or [], "revids": revids or [],
        "paragraph_sha256": paragraph_hashes(text),
        "locked_at": dt.datetime.now(dt.timezone.utc).isoformat(),
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return manifest["text"]["held_out"]


def set_task_split(task: str, n_docs: int, fractions: dict[str, float] | None = None, seed: int = 0,
                    manifest_path: Path = MANIFEST_PATH) -> dict:
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else _empty_manifest()
    manifest["tasks"][task] = {"n_docs": n_docs, "seed": seed,
                                "fractions": fractions or {"search": 0.6, "dev": 0.2, "held_out": 0.2}}
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return manifest["tasks"][task]


def ngram_overlap(a: str, b: str, n: int = 13) -> float:
    """Fraction of a's word n-grams that also appear in b -- a cheap, real
    contamination check between splits (Wikipedia articles quote each other
    more than you'd expect)."""
    def ngrams(s: str) -> set[tuple[str, ...]]:
        words = s.split()
        return {tuple(words[i:i + n]) for i in range(len(words) - n + 1)}
    ga, gb = ngrams(a), ngrams(b)
    return len(ga & gb) / len(ga) if ga else 0.0
