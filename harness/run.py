"""The orchestrator: a candidate spec goes in, a result record comes out.

python -m harness.run specs/example.json --repeats 5 --pp 512 --tg 128 \\
    --max-temp 45 --kld-ref models/qwen2.5-3b-instruct-Q8_0.gguf --kld-text data/dev.txt \\
    --lm-eval-tasks gsm8k,humaneval --run-longctx

For each spec: build (or reuse, from the artifact cache keyed by artifact_hash)
the quantized model, measure pp and tg speed as separate llama-bench
invocations (so each phase's sampled energy belongs to that phase, not a
load+pp+tg blend), measure KL divergence against a fixed reference model with
the spec's own KV-cache flags applied, optionally run a task suite
(GSM8K/HumanEval, via harness.lmeval) and the long-context suite
(needle/multi-hop, via harness.longctx) against a real llama-server, and write
both a raw JSON result record (results/runs/<id>/record.json, the source of
truth) and rows into the DuckDB store (results/runs.duckdb, a rebuildable
summary of those records).

Every quality measure that reads free text (KL, the long-context haystack) or
task docs (lm-eval) goes through harness.splits' chokepoints first, which
classify the input by content hash against data/splits.json and refuse the
held-out split without --allow-held-out.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import time
from pathlib import Path
from uuid import uuid4

from harness import db, gpu, llama, lmeval, longctx, memory, splits
from harness.server import LlamaServer, ServerFailed
from harness.spec import CandidateSpec


def _hash_dict(d: dict) -> str:
    return hashlib.sha256(json.dumps(d, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:16]


def ensure_artifact(spec: CandidateSpec, cache_dir: Path, log_dir: Path, bin_dir: str) -> dict:
    """Build (or reuse) the weights file for spec.artifact_hash. This *is* the
    artifact cache: a file at the expected path means it's already built."""
    if spec.quant in ("F16", "BF16"):
        path = Path(spec.base_model)
        return {"path": str(path), "size_bytes": path.stat().st_size,
                "build_seconds": None, "build_cmd": None, "built_at": None}

    cache_dir.mkdir(parents=True, exist_ok=True)
    dst = cache_dir / f"{spec.artifact_hash}.gguf"
    if dst.exists():
        return {"path": str(dst), "size_bytes": dst.stat().st_size,
                "build_seconds": None, "build_cmd": None, "built_at": None}

    if spec.tensor_overrides or spec.imatrix:
        raise NotImplementedError("per-tensor overrides / imatrix are not wired up yet (Level 2, later)")

    log_path = log_dir / f"quantize_{spec.artifact_hash}.log"
    build_seconds = llama.quantize(Path(spec.base_model), dst, spec.quant, log_path, bin_dir=bin_dir)
    return {"path": str(dst), "size_bytes": dst.stat().st_size, "build_seconds": build_seconds,
            "build_cmd": f"llama-quantize {spec.base_model} {dst} {spec.quant}",
            "built_at": dt.datetime.now(dt.timezone.utc).isoformat()}


def _predicted_total_mb(spec: CandidateSpec, artifact_path: Path, overhead_gb: float) -> float | None:
    try:
        shape = memory.shape_from_gguf(str(artifact_path))
    except Exception:
        return None
    weight_bytes = artifact_path.stat().st_size
    total = memory.total_bytes(weight_bytes, shape, spec.ctx, spec.kv_type_k, spec.kv_type_v,
                                int(overhead_gb * 1e9))
    return total / 2**20


def _new_run_entry(workload_id: str, workload: dict, repeat: int, run_dir: Path, name: str,
                    max_temp: int) -> tuple[dict, Path]:
    start_temp = gpu.wait_for_cool(max_temp)
    env_now = gpu.environment()
    log_path = run_dir / f"{name}.log"
    entry = {
        "run_id": uuid4().hex[:16], "workload_id": workload_id, "workload": workload,
        "repeat": repeat, "seed": None, "start_temp_c": start_temp,
        "start_sm_clock_mhz": env_now["sm_clock_mhz"],
        "started_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "wall_seconds": None, "status": "ok", "command": None, "log_path": str(log_path),
        "speed": None, "resources": None, "quality": None, "failure": None,
    }
    return entry, log_path


def measure_bench(spec: CandidateSpec, artifact_path: Path, kind: str, pp: int, tg: int, repeat: int,
                   extra_flags: list[str], run_dir: Path, max_temp: int, bin_dir: str,
                   overhead_gb: float) -> dict:
    workload = {"kind": kind, "input_len": pp, "output_len": tg, "concurrency": 1}
    workload_id = _hash_dict(workload)
    entry, log_path = _new_run_entry(workload_id, workload, repeat, run_dir, f"{kind}_{repeat}", max_temp)
    t0 = time.monotonic()
    try:
        with gpu.Sampler() as sampler:
            recs = llama.bench(artifact_path, pp, tg, 1, extra_flags, log_path, depth=spec.ctx,
                                bin_dir=bin_dir)
        entry["wall_seconds"] = time.monotonic() - t0
        r = recs[0]
        entry["speed"] = {"pp_tps": r["avg_ts"] if pp else None, "tg_tps": r["avg_ts"] if tg else None,
                           "ttft_ms": None, "p50_ms": None, "p95_ms": None, "raw": r}
        res = sampler.summary(n_tokens=tg or None)
        res["predicted_total_mb"] = _predicted_total_mb(spec, artifact_path, overhead_gb)
        res["nvml_log"] = None
        entry["resources"] = res
    except llama.RunFailed as e:
        entry["status"] = "failed"
        entry["failure"] = {"type": "bench", "message": str(e), "log_path": str(e.log_path)}
    return entry


def measure_kld(spec: CandidateSpec, artifact_path: Path, ref: Path, text: Path, text_hash: str,
                 split: str, ctx: int, chunks: int | None, cache_dir: Path, run_dir: Path,
                 max_temp: int, bin_dir: str) -> dict:
    workload = {"kind": "kld", "ref": ref.name, "prompt_set_hash": text_hash, "split": split,
                "input_len": ctx, "output_len": 0, "concurrency": 1, "chunks": chunks}
    workload_id = _hash_dict(workload)
    entry, log_path = _new_run_entry(workload_id, workload, 0, run_dir, "kld", max_temp)
    t0 = time.monotonic()
    # Keyed on ref+text+ctx+chunks: a reference-logits file is only valid for the
    # exact (model, text, ctx, chunks) it was built from. Missing the text/chunks
    # from this key was a real bug -- pointing --kld-text at a different file with
    # the same --kld-ref would have silently reused the wrong reference logits.
    ref_base = cache_dir / f"ref_{ref.stem}_{text_hash}_c{ctx}_n{chunks or 0}.kld"
    try:
        llama.reference_logits(ref, text, ctx, ref_base, chunks=chunks, bin_dir=bin_dir)
        with gpu.Sampler() as sampler:
            kld = llama.kl_divergence(artifact_path, ref_base, text, ctx, llama.spec_flags(spec), log_path,
                                       chunks=chunks, bin_dir=bin_dir)
        entry["wall_seconds"] = time.monotonic() - t0
        entry["quality"] = {"eval_name": "kld", "split": split, "score": kld.get("ppl"),
                             "kl_mean": kld.get("kld_mean"), "kl_p99": kld.get("kld_p99"),
                             "kl_max": kld.get("kld_max"), "top1_agree_pct": kld.get("top1_agree_pct")}
        entry["resources"] = sampler.summary()
    except llama.RunFailed as e:
        entry["status"] = "failed"
        entry["failure"] = {"type": "kld", "message": str(e), "log_path": str(e.log_path)}
    return entry


def _load_eval_text(path: Path, allow_held_out: bool) -> tuple[str, str, str]:
    """Wraps splits.load_eval_text, falling back to treating the file as
    unclassified ("adhoc") if data/splits.json doesn't exist yet (e.g. a
    fresh clone before scripts/fetch_wiki.py has run) -- there's nothing to
    protect against yet in that case, so this degrades safely rather than
    hard-failing Week 1/2-style usage."""
    try:
        manifest = splits.load_manifest()
    except FileNotFoundError:
        text = path.read_text(encoding="utf-8")
        return text, "adhoc", hashlib.sha256(text.encode()).hexdigest()[:16]
    return splits.load_eval_text(path, manifest, allow_held_out)


def measure_lm_eval(spec: CandidateSpec, artifact_path: Path, tasks: list[str], eval_split: str,
                     allow_held_out: bool, limit: int | None, run_dir: Path, max_temp: int,
                     bin_dir: str) -> list[dict]:
    """One short-context server (GSM8K/HumanEval need only a few thousand
    tokens of context, nowhere near the long-context suite's needs), one
    quality run entry per task."""
    server_ctx = 4096
    try:
        manifest = splits.load_manifest()
    except FileNotFoundError:
        manifest = None

    entries = []
    try:
        with LlamaServer(artifact_path, spec, server_ctx, run_dir / "lmeval_server.log",
                          bin_dir=bin_dir) as server:
            for task in tasks:
                split_used = eval_split
                indices = None
                if manifest is not None:
                    try:
                        indices = splits.task_doc_indices(task, eval_split, manifest, allow_held_out, limit)
                    except KeyError:
                        split_used = "unsplit"
                if manifest is None or split_used == "unsplit":
                    split_used = "unsplit"
                    indices = list(range(limit)) if limit else None

                workload = {"kind": "lmeval", "task": task, "split": split_used,
                            "concurrency": 1, "server_n_ctx": server_ctx,
                            "n_items": len(indices) if indices is not None else None}
                entry, task_log = _new_run_entry(_hash_dict(workload), workload, 0, run_dir,
                                                  f"lmeval_{task}", max_temp)
                t0 = time.monotonic()
                try:
                    with gpu.Sampler() as sampler:
                        if task == "humaneval":
                            q = lmeval.evaluate_humaneval(server, indices, run_dir / "quality")
                        else:
                            q = lmeval.evaluate_gsm8k(server.base_url, spec.spec_hash, indices,
                                                       run_dir / "quality")
                    entry["wall_seconds"] = time.monotonic() - t0
                    entry["quality"] = {**q, "split": split_used}
                    entry["resources"] = sampler.summary()
                except Exception as e:
                    entry["status"] = "failed"
                    entry["failure"] = {"type": "lmeval", "message": str(e), "log_path": str(task_log)}
                entries.append(entry)
    except ServerFailed as e:
        for task in tasks:
            workload = {"kind": "lmeval", "task": task, "split": eval_split}
            entry, _ = _new_run_entry(_hash_dict(workload), workload, 0, run_dir,
                                       f"lmeval_{task}_serverfail", max_temp)
            entry["status"] = "failed"
            entry["failure"] = {"type": "server", "message": str(e), "log_path": str(e.log_path)}
            entries.append(entry)
    return entries


def measure_longctx(spec: CandidateSpec, artifact_path: Path, lengths: list[int], haystack_text: str,
                     text_split: str, seeds: tuple[int, ...], run_dir: Path, max_temp: int,
                     bin_dir: str) -> list[dict]:
    paras = longctx.paragraphs_from_text(haystack_text)
    server_ctx = max(lengths) + 512  # headroom for the question and the generation

    try:
        with LlamaServer(artifact_path, spec, server_ctx, run_dir / "longctx_server.log",
                          bin_dir=bin_dir) as server:
            para_tokens = longctx.tokenize_paragraphs(paras, server.tokenize)
            trials = longctx.make_trials(paras, para_tokens, lengths, seeds=seeds)
            out_jsonl = run_dir / "quality" / "longctx_trials.jsonl"
            summary = longctx.run_trials(server, trials, out_jsonl)
    except ServerFailed as e:
        entries = []
        for length in lengths:
            for kind in ("needle", "multihop"):
                eval_name = f"{kind}_{length}"
                workload = {"kind": "longctx", "eval_name": eval_name, "split": text_split}
                entry, _ = _new_run_entry(_hash_dict(workload), workload, 0, run_dir,
                                           f"longctx_{eval_name}_serverfail", max_temp)
                entry["status"] = "failed"
                entry["failure"] = {"type": "server", "message": str(e), "log_path": str(e.log_path)}
                entries.append(entry)
        return entries

    # Recorded after the suite runs, not before each trial -- unlike bench/KL,
    # this is a quality-only measurement and isn't thermal-sensitive, so one
    # wait_for_cool before the server starts (not threaded through every trial)
    # is an acceptable simplification.
    entries = []
    for eval_name, s in summary.items():
        length = int(eval_name.rsplit("_", 1)[1])
        workload = {"kind": "longctx", "eval_name": eval_name, "split": text_split,
                    "input_len": length, "output_len": 32, "concurrency": 1,
                    "server_n_ctx": server_ctx, "seeds": list(seeds)}
        entry, _ = _new_run_entry(_hash_dict(workload), workload, 0, run_dir, f"longctx_{eval_name}", max_temp)
        entry["quality"] = {"eval_name": eval_name, "split": text_split, "score": s["score"],
                             "n_items": s["n"],
                             "details": {"n_correct": s["n_correct"],
                                         "prompt_tokens_mean": s["prompt_tokens_mean"],
                                         "prompt_tokens_min": s["prompt_tokens_min"],
                                         "prompt_tokens_max": s["prompt_tokens_max"],
                                         "trials_path": str(out_jsonl)}}
        entries.append(entry)
    return entries


def run_spec(spec_path: Path, args: argparse.Namespace) -> dict:
    spec = CandidateSpec.from_json(spec_path)
    spec_hash, artifact_hash = spec.spec_hash, spec.artifact_hash

    run_dir = Path(args.out) / f"{spec_hash}_{dt.datetime.now():%Y%m%d_%H%M%S}"
    run_dir.mkdir(parents=True, exist_ok=True)

    artifact = ensure_artifact(spec, Path(args.cache_dir), run_dir, args.bin_dir)
    artifact_path = Path(artifact["path"])

    env = gpu.environment()
    commit = gpu.llamacpp_commit(str(Path(args.bin_dir).resolve().parents[1]))
    stable = gpu.stable_env(env, commit)
    env_id = _hash_dict(stable)

    extra_flags = llama.spec_flags(spec) + list(spec.engine_flags)

    runs = []
    for kind, pp, tg in (("bench_pp", args.pp, 0), ("bench_tg", 0, args.tg)):
        for rep in range(args.repeats):
            runs.append(measure_bench(spec, artifact_path, kind, pp, tg, rep, extra_flags, run_dir,
                                       args.max_temp, args.bin_dir, args.overhead_gb))

    # llama-bench's own JSON already reports model size/param count -- no need
    # to parse the GGUF separately for bits-per-weight.
    n_params = bpw = None
    for r in runs:
        raw = (r.get("speed") or {}).get("raw")
        if raw and raw.get("model_n_params"):
            n_params = raw["model_n_params"]
            bpw = raw["model_size"] * 8 / raw["model_n_params"]
            break

    if not args.no_kld:
        kld_path = Path(args.kld_text)
        # The chokepoint: classifies kld_path by its actual content hash (never
        # by the --kld-text string itself) and raises if it's the held-out
        # split, or leaks held-out paragraphs, without --allow-held-out.
        _, kld_split, kld_text_hash = _load_eval_text(kld_path, args.allow_held_out)
        runs.append(measure_kld(spec, artifact_path, Path(args.kld_ref), kld_path, kld_text_hash,
                                 kld_split, args.kld_ctx, args.kld_chunks, Path(args.cache_dir), run_dir,
                                 args.max_temp, args.bin_dir))

    if args.lm_eval_tasks:
        runs += measure_lm_eval(spec, artifact_path, args.lm_eval_tasks, args.eval_split,
                                 args.allow_held_out, args.lm_eval_limit, run_dir, args.max_temp,
                                 args.bin_dir)

    if args.run_longctx:
        haystack_path = Path(args.longctx_text or args.kld_text)
        haystack_text, longctx_split, _ = _load_eval_text(haystack_path, args.allow_held_out)
        runs += measure_longctx(spec, artifact_path, args.longctx_lengths, haystack_text, longctx_split,
                                 tuple(range(args.longctx_seeds)), run_dir, args.max_temp, args.bin_dir)

    record = {
        "record_version": 2, "spec_hash": spec_hash, "artifact_hash": artifact_hash, "env_id": env_id,
        "spec": spec.to_dict(),
        "artifact": {**artifact, "n_params": n_params, "bpw": bpw},
        "environment": stable,
        "runs": runs,
    }
    (run_dir / "record.json").write_text(json.dumps(record, indent=2, default=str))

    con = db.connect(args.db)
    db.insert_record(con, record)
    con.close()
    print(f"spec {spec_hash} ({spec.quant}, kv={spec.kv_type_k}/{spec.kv_type_v}, ctx={spec.ctx}): "
          f"{sum(1 for r in runs if r['status'] == 'ok')}/{len(runs)} runs ok -> {run_dir}")
    return record


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("specs", nargs="+", help="one or more candidate spec JSON files")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--pp", type=int, default=512)
    ap.add_argument("--tg", type=int, default=128)
    ap.add_argument("--max-temp", type=int, default=70)
    ap.add_argument("--overhead-gb", type=float, default=0.8, help="see docs/week1-results.md for calibration")
    ap.add_argument("--bin-dir", default=llama.DEFAULT_BIN)
    ap.add_argument("--cache-dir", default="models/cache")
    ap.add_argument("--out", default="results/runs")
    ap.add_argument("--db", default=db.DB_PATH)
    ap.add_argument("--no-kld", action="store_true")
    ap.add_argument("--kld-ref", help="reference model for KL divergence (required unless --no-kld)")
    ap.add_argument("--kld-text", help="eval text file (required unless --no-kld)")
    ap.add_argument("--kld-ctx", type=int, default=512)
    ap.add_argument("--kld-chunks", type=int, default=50)
    ap.add_argument("--eval-split", choices=list(splits.SPLITS), default="dev",
                     help="which split's docs to use for lm-eval tasks (default: dev)")
    ap.add_argument("--allow-held-out", action="store_true",
                     help="required to read the held-out split or task docs -- final reporting only")
    ap.add_argument("--lm-eval-tasks", default="",
                     help="comma-separated lm-eval tasks to run, e.g. gsm8k,humaneval (default: none)")
    ap.add_argument("--lm-eval-limit", type=int, default=None,
                     help="cap docs per task (fast iteration); omit for the full configured split")
    ap.add_argument("--run-longctx", action="store_true", help="run the needle/multi-hop long-context suite")
    ap.add_argument("--longctx-lengths", type=int, nargs="+", default=[8192, 16384, 32768])
    ap.add_argument("--longctx-seeds", type=int, default=2, help="number of seeds per (kind, length)")
    ap.add_argument("--longctx-text", help="haystack text file (default: --kld-text)")
    args = ap.parse_args()
    args.lm_eval_tasks = [t.strip() for t in args.lm_eval_tasks.split(",") if t.strip()]

    if not args.no_kld and not (args.kld_ref and args.kld_text):
        ap.error("--kld-ref and --kld-text are required unless --no-kld")
    if args.run_longctx and not (args.longctx_text or args.kld_text):
        ap.error("--run-longctx needs --longctx-text or --kld-text to source the haystack")
    if args.eval_split == "held_out" and not args.allow_held_out:
        ap.error("--eval-split held_out requires --allow-held-out")

    for spec_path in args.specs:
        run_spec(Path(spec_path), args)


if __name__ == "__main__":
    main()
