"""Fetch Wikipedia article text for the search/dev/held-out splits.

data/dev.txt already exists (Week 1's 10-article corpus) and is left alone --
this script only builds data/search.txt and data/held_out.txt, from article
titles disjoint from dev's 10 and from each other, and records each split's
hash (plus titles/revision ids, for provenance) in data/splits.json.

Usage:
  python -m harness.fetch_wiki --split search
  python -m harness.fetch_wiki --split held_out
  python -m harness.fetch_wiki              # both, plus records dev.txt's hash
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

from harness import splits

USER_AGENT = "aioptimization-research/1.0 (local research harness; contact: odograsmussen@gmail.com)"
API = "https://en.wikipedia.org/w/api.php"

# Disjoint from dev.txt's 10 titles (Quantum_computing, Byzantine_Empire,
# Climate_change, History_of_the_Internet, World_War_II, Great_Barrier_Reef,
# Python_(programming_language), Industrial_Revolution, Photosynthesis,
# Renaissance) and from each other.
TITLES = {
    "search": [
        "Artificial_intelligence", "Ancient_Egypt", "DNA", "Volcano", "Renewable_energy",
        "Opera", "Chess", "Coral_reef", "Printing_press", "Antarctica",
        "Electric_vehicle", "Mount_Everest", "William_Shakespeare", "Vaccine", "Solar_System",
    ],
    "held_out": [
        "Blockchain", "Great_Wall_of_China", "Black_hole", "Agriculture", "Jazz",
        "Nuclear_power", "Mars", "Telescope",
    ],
}


def _fetch_one(title: str) -> tuple[str, str]:
    params = urllib.parse.urlencode({
        "action": "query", "prop": "extracts|revisions", "rvprop": "ids", "explaintext": 1,
        "format": "json", "titles": title, "redirects": 1,
    })
    req = urllib.request.Request(f"{API}?{params}", headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read())
    pages = data["query"]["pages"]
    page = next(iter(pages.values()))
    extract = page.get("extract", "")
    revid = str(page.get("revisions", [{}])[0].get("revid", ""))
    return extract, revid


def fetch_split(name: str, out_dir: Path = Path("data")) -> Path:
    assert name in TITLES, f"no title list for {name!r}"
    titles = TITLES[name]
    out_path = out_dir / f"{name}.txt"
    revids = []
    with open(out_path, "w", encoding="utf-8") as f:
        for title in titles:
            print(f"fetching {title}")
            extract, revid = _fetch_one(title)
            f.write(extract + "\n\n")
            revids.append(revid)
            time.sleep(1.5)  # Wikipedia API rate limit etiquette

    if name == "held_out":
        splits.lock_held_out(out_path, titles=titles, revids=revids)
    else:
        splits.record_text_split(name, out_path, titles=titles, revids=revids)
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["search", "held_out", "both"], default="both")
    args = ap.parse_args()
    names = ["search", "held_out"] if args.split == "both" else [args.split]
    for name in names:
        path = fetch_split(name)
        print(f"wrote {path}, recorded in {splits.MANIFEST_PATH}")

    # record dev.txt's own hash too, so it's classified correctly by
    # splits.classify_text (it was written manually in Week 1, before this
    # script or the manifest existed)
    dev_path = Path("data/dev.txt")
    if dev_path.exists():
        splits.record_text_split("dev", dev_path, titles=[
            "Quantum_computing", "Byzantine_Empire", "Climate_change", "History_of_the_Internet",
            "World_War_II", "Great_Barrier_Reef", "Python_(programming_language)",
            "Industrial_Revolution", "Photosynthesis", "Renaissance"])
        print(f"recorded existing {dev_path} in {splits.MANIFEST_PATH}")

    # lm-eval task item splits, so GSM8K/HumanEval respect search/dev/held_out too
    splits.set_task_split("gsm8k", n_docs=1319)
    splits.set_task_split("humaneval", n_docs=164)
    print(f"configured task splits for gsm8k, humaneval in {splits.MANIFEST_PATH}")


if __name__ == "__main__":
    main()
