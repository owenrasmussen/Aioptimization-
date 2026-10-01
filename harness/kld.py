"""Week 1: KL divergence of quantized models against a reference (e.g. Q8_0).

Step 1 saves the reference logits once; step 2 compares each candidate to them.
The actual subprocess/log handling lives in harness/llama.py, shared with the
Week 2 orchestrator (run.py) so both paths behave identically.

Example:
  python -m harness.kld --ref models/qwen-4b-Q8_0.gguf --text data/dev.txt \\
      models/qwen-4b-Q3_K_M.gguf models/qwen-4b-Q4_K_M.gguf models/qwen-4b-Q6_K.gguf
"""
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

from harness import llama

DEFAULT_BIN = llama.DEFAULT_BIN

# Lines llama-perplexity prints in its KL summary, e.g. "Mean    KLD:   0.012345 ±   0.000123"
PATTERNS = {
    "ppl": r"Mean PPL\(Q\)\s*:\s*([\d.]+)",
    "kld_mean": r"Mean\s+KLD:\s*([\d.]+)",
    "kld_p99": r"99\.0%\s+KLD:\s*([\d.]+)",
    "kld_max": r"Maximum KLD:\s*([\d.]+)",
    "top1_agree_pct": r"Same top p:\s*([\d.]+)",
}


def parse(log: str) -> dict:
    out = {}
    for k, pat in PATTERNS.items():
        m = re.search(pat, log)
        out[k] = float(m.group(1)) if m else None
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True, help="reference model, e.g. Q8_0 (say so in writeups)")
    ap.add_argument("--text", required=True, help="evaluation text file")
    ap.add_argument("--ctx", type=int, default=512)
    ap.add_argument("--bin-dir", default=os.environ.get("LLAMA_BIN", DEFAULT_BIN))
    ap.add_argument("--out", default="results/kld")
    ap.add_argument("candidates", nargs="+")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    base = out / f"{Path(args.ref).stem}_{Path(args.text).stem}_c{args.ctx}.kld"

    if not base.exists():
        print(f"saving reference logits -> {base}")
    llama.reference_logits(Path(args.ref), Path(args.text), args.ctx, base, bin_dir=args.bin_dir)

    results = []
    for cand in args.candidates:
        log_path = out / f"{Path(cand).stem}_c{args.ctx}.log"
        r = {"model": cand, "ref": args.ref, "ctx": args.ctx, "text": args.text,
             "size_bytes": Path(cand).stat().st_size,
             **llama.kl_divergence(Path(cand), base, Path(args.text), args.ctx, [], log_path,
                                    bin_dir=args.bin_dir)}
        results.append(r)
        print(json.dumps(r))

    (out / f"summary_c{args.ctx}.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
