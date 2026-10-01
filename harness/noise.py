"""Week 1: measure run-to-run noise of llama-bench for one config.

Runs llama-bench N separate times (one repeat each), waiting for the GPU to cool
between runs, and reports median, spread and a bootstrap CI for prompt and
generation speed. The CV you get here is the smallest difference you can
believe later. The actual subprocess call lives in harness/llama.py, shared
with the Week 2 orchestrator.

Example:
  python -m harness.noise --model models/qwen-4b-Q4_K_M.gguf --repeats 10 --max-temp 45
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path

from harness import gpu, llama, stats

DEFAULT_BIN = llama.DEFAULT_BIN


def run_once(bin_dir: str, model: str, pp: int, tg: int, extra: list[str],
             log_path: Path | None = None) -> list[dict]:
    import tempfile

    log_path = log_path or Path(tempfile.gettempdir()) / "harness_noise_run_once.log"
    return llama.bench(Path(model), pp, tg, 1, list(extra), log_path, bin_dir=bin_dir)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--pp", type=int, default=512, help="prompt tokens")
    ap.add_argument("--tg", type=int, default=128, help="generated tokens")
    ap.add_argument("--max-temp", type=int, default=45, help="wait until GPU is at or below this (C)")
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--bin-dir", default=os.environ.get("LLAMA_BIN", DEFAULT_BIN))
    ap.add_argument("--out", default="results/noise")
    ap.add_argument("extra", nargs="*", help="extra llama-bench flags after --, e.g. -- -fa 1")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    for i in range(args.warmup):
        run_once(args.bin_dir, args.model, args.pp, args.tg, args.extra, out / f"warmup{i}.log")

    pp_ts, tg_ts, runs = [], [], []
    for i in range(args.repeats):
        start_temp = gpu.wait_for_cool(args.max_temp)
        recs = run_once(args.bin_dir, args.model, args.pp, args.tg, args.extra, out / f"repeat{i}.log")
        for r in recs:
            (pp_ts if r["n_prompt"] > 0 else tg_ts).append(r["avg_ts"])
        runs.append({"repeat": i, "start_temp_c": start_temp, "records": recs})
        print(f"repeat {i}: start {start_temp}C  " +
              "  ".join(f"{'pp' if r['n_prompt'] else 'tg'} {r['avg_ts']:.1f} t/s" for r in recs))

    report = {
        "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
        "model": args.model,
        "pp": args.pp, "tg": args.tg, "extra_flags": args.extra,
        "environment": gpu.environment(),
        "llamacpp_commit": gpu.llamacpp_commit(str(Path(args.bin_dir).parents[1])),
        "prompt_tps": {**stats.summarize(pp_ts), "median_ci95": stats.bootstrap_ci(pp_ts)},
        "gen_tps": {**stats.summarize(tg_ts), "median_ci95": stats.bootstrap_ci(tg_ts)},
        "runs": runs,
    }
    path = out / f"{Path(args.model).stem}_{dt.datetime.now():%Y%m%d_%H%M%S}.json"
    path.write_text(json.dumps(report, indent=2))

    for k in ("prompt_tps", "gen_tps"):
        s = report[k]
        print(f"{k}: median {s['median']:.1f}  CV {s['cv_pct']:.2f}%  "
              f"95% CI [{s['median_ci95'][0]:.1f}, {s['median_ci95'][1]:.1f}]")
    print(f"saved {path}")


if __name__ == "__main__":
    main()
