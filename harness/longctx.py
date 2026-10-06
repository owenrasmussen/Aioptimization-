"""Synthetic needle-in-a-haystack and multi-hop retrieval tests.

Haystack text is sized to an exact token count via the server's own
/tokenize endpoint, tokenizing each paragraph once up front rather than a
word-count estimate (which can run 10-20% over on dense Wikipedia text --
at a 32768-token target that's enough to overflow the context window and
fail the request for a reason that has nothing to do with model quality).

Answers use a fixed format (KX-\\d{7}) so scoring is an exact match on the
first extracted code, not a substring check -- a plain "is the code
somewhere in the response" check gives false positives when distractor
codes are present, or when one code is a substring of another.
"""
from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass, field
from pathlib import Path

from harness.splits import paragraphs as split_into_paragraphs

CODE_RE = re.compile(r"KX-\d{7}(?!\d)")  # (?!\d): don't match a 7-digit prefix of a longer number

# Deliberately synthetic so they don't collide with real content in a
# Wikipedia-sourced haystack.
_NAME_POOL = [
    "Zephyrion", "Quintara", "Blackwood-7", "Marrowfen", "Calyxtide",
    "Driftglass", "Umbervale", "Pellucid-9", "Starling Keep", "Thornwick",
    "Velmora", "Cindergate", "Oakhollow-3", "Fenrath", "Brightmoor",
]


def tokenize_paragraphs(paragraphs: list[str], tokenize) -> list[int]:
    """tokenize: str -> list[int] (e.g. LlamaServer.tokenize). Returns the
    token count of each paragraph, computed once and reused for every trial
    built from this haystack."""
    return [len(tokenize(p)) for p in paragraphs]


def fill_haystack(paragraphs: list[str], para_tokens: list[int], budget: int, offset: int) -> list[str]:
    """Greedily select whole paragraphs starting at `offset` (wrapping around
    the pool), stopping before the budget would be exceeded. Never splits a
    paragraph, so token counts stay exact without truncating mid-sentence."""
    n = len(paragraphs)
    if n == 0:
        return []
    selected, total, i, seen = [], 0, offset % n, 0
    while seen < n:
        t = para_tokens[i]
        if total + t > budget:
            break
        selected.append(paragraphs[i])
        total += t
        i = (i + 1) % n
        seen += 1
    return selected


def insert_at_depth(paras: list[str], sentence: str, depth: float) -> list[str]:
    idx = max(0, min(len(paras), round(depth * len(paras))))
    return paras[:idx] + [sentence] + paras[idx:]


def _code(rng: random.Random) -> str:
    return f"KX-{rng.randint(1_000_000, 9_999_999)}"


@dataclass(frozen=True)
class Trial:
    kind: str  # "needle" | "multihop"
    target_len: int
    seed: int
    messages: list[dict] = field(repr=False)
    expected: str
    max_tokens: int = 32


def build_needle(paragraphs: list[str], para_tokens: list[int], target_tokens: int, depth: float,
                  seed: int, max_tokens: int = 32, reserve: int = 256) -> Trial:
    """`reserve` must cover everything the token budget check below doesn't
    see: the inserted needle sentence (not subtracted from `budget`), the
    question text, and the chat template's own special tokens -- none of
    which fill_haystack's accounting knows about. Verified live: at
    target_tokens=32768 (this model's native max), a 128-token reserve
    still let one multihop trial's real prompt land at 32805 tokens and
    get rejected by the server as over context -- 256 leaves comfortable
    margin for needle's smaller overhead too."""
    rng = random.Random(seed)
    topic = rng.choice(_NAME_POOL)
    code = _code(rng)
    budget = max(target_tokens - reserve, 0)
    haystack = fill_haystack(paragraphs, para_tokens, budget, rng.randrange(max(len(paragraphs), 1)))
    haystack = insert_at_depth(haystack, f"The secret code for {topic} is {code}.", depth)
    question = (f"Question: what is the secret code for {topic}? "
                f"Answer with only the code, in the form KX-XXXXXXX.")
    prompt = "\n\n".join(haystack) + "\n\n" + question
    return Trial("needle", target_tokens, seed, [{"role": "user", "content": prompt}], code, max_tokens)


def build_multihop(paragraphs: list[str], para_tokens: list[int], target_tokens: int,
                    depths: tuple[float, float], seed: int, max_tokens: int = 32,
                    n_distractors: int = 2, reserve: int = 256) -> Trial:
    """Two linked facts (project -> person -> code) at different depths, plus
    distractor chains for other projects. Without distractors, a model could
    answer by finding the only code in the haystack without actually doing
    the hop -- this makes the hop necessary."""
    rng = random.Random(seed)
    names = rng.sample(_NAME_POOL, min(2 + 2 * n_distractors, len(_NAME_POOL)))
    project, person = names[0], names[1]
    code = _code(rng)
    sentences = [(f"Project {project} is led by {person}.", depths[0]),
                 (f"{person}'s access code is {code}.", depths[1])]
    for i in range(n_distractors):
        if 2 + 2 * i + 1 >= len(names):
            break
        dp, dperson = names[2 + 2 * i], names[2 + 2 * i + 1]
        sentences.append((f"Project {dp} is led by {dperson}.", rng.random()))
        sentences.append((f"{dperson}'s access code is {_code(rng)}.", rng.random()))

    budget = max(target_tokens - reserve, 0)
    haystack = fill_haystack(paragraphs, para_tokens, budget, rng.randrange(max(len(paragraphs), 1)))
    for sentence, d in sorted(sentences, key=lambda x: -x[1]):  # highest depth first keeps indices valid
        haystack = insert_at_depth(haystack, sentence, d)

    question = (f"Question: what is the access code for the person leading Project {project}? "
                f"First find who leads Project {project}, then find that person's access code. "
                f"Answer with only the code, in the form KX-XXXXXXX.")
    prompt = "\n\n".join(haystack) + "\n\n" + question
    return Trial("multihop", target_tokens, seed, [{"role": "user", "content": prompt}], code, max_tokens)


def score(response: str, expected: str) -> bool:
    m = CODE_RE.search(response)
    return bool(m) and m.group(0) == expected


def make_trials(paragraphs: list[str], para_tokens: list[int], lengths: list[int],
                 needle_depths: tuple[float, ...] = (0.1, 0.5, 0.9),
                 multihop_depth_pairs: tuple[tuple[float, float], ...] = ((0.2, 0.7), (0.7, 0.2)),
                 seeds: tuple[int, ...] = (0, 1), max_tokens: int = 32) -> list[Trial]:
    trials = []
    for length in lengths:
        for depth in needle_depths:
            for seed in seeds:
                trials.append(build_needle(paragraphs, para_tokens, length, depth, seed, max_tokens))
        for depths in multihop_depth_pairs:
            for seed in seeds:
                trials.append(build_multihop(paragraphs, para_tokens, length, depths, seed, max_tokens))
    return trials


def run_trials(server, trials: list[Trial], out_jsonl: Path) -> dict[str, dict]:
    """Runs each trial against `server` (needs .chat(messages, max_tokens, seed)).
    Writes one JSONL line per trial (for later paired analysis, not just the
    aggregate) and returns {f"{kind}_{length}": {score, n, n_correct, ...}}."""
    by_key: dict[tuple[str, int], list[tuple[bool, int | None]]] = {}
    lines = []
    for t in trials:
        resp = server.chat(t.messages, max_tokens=t.max_tokens, seed=t.seed)
        content = resp["choices"][0]["message"]["content"]
        correct = score(content, t.expected)
        prompt_tokens = (resp.get("usage") or {}).get("prompt_tokens")
        by_key.setdefault((t.kind, t.target_len), []).append((correct, prompt_tokens))
        lines.append(json.dumps({"kind": t.kind, "target_len": t.target_len, "seed": t.seed,
                                  "expected": t.expected, "response": content, "correct": correct,
                                  "prompt_tokens": prompt_tokens}))
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    out_jsonl.write_text("\n".join(lines))

    summary = {}
    for (kind, length), items in by_key.items():
        n = len(items)
        n_correct = sum(1 for c, _ in items if c)
        pts = [p for _, p in items if p is not None]
        summary[f"{kind}_{length}"] = {
            "score": n_correct / n, "n": n, "n_correct": n_correct,
            "prompt_tokens_mean": sum(pts) / len(pts) if pts else None,
            "prompt_tokens_min": min(pts) if pts else None,
            "prompt_tokens_max": max(pts) if pts else None,
        }
    return summary


def paragraphs_from_text(text: str) -> list[str]:
    return split_into_paragraphs(text)
