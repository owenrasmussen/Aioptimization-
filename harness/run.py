"""Week 2 orchestrator: a candidate spec goes in, a result record comes out.

python -m harness.run specs/example.json --repeats 5 --pp 512 --tg 128 \\
    --max-temp 45 --kld-ref models/qwen2.5-3b-instruct-Q8_0.gguf --kld-text data/dev.txt

For each spec: build (or reuse, from the artifact cache keyed by artifact_hash)
the quantized model, measure pp and tg speed as separate llama-bench
invocations (so each phase's sampled energy belongs to that phase, not a
load+pp+tg blend), measure KL divergence against a fixed reference model with
the spec's own KV-cache flags applied, and write both a raw JSON result
record (results/runs/<id>/record.json, the source of truth) and rows into the
DuckDB store (results/runs.duckdb, a rebuildable summary of those records).
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import time
from pathlib import Path
from uuid import uuid4

from harness import db, gpu, llama, memory
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


def measure_kld(spec: CandidateSpec, artifact_path: Path, ref: Path, text: Path, ctx: int,
                 chunks: int | None, cache_dir: Path, run_dir: Path, max_temp: int,
                 bin_dir: str) -> dict:
    text_hash = hashlib.sha256(text.read_bytes()).hexdigest()[:16]
    workload = {"kind": "kld", "ref": ref.name, "prompt_set_hash": text_hash,
                "input_len": ctx, "output_len": 0, "concurrency": 1, "chunks": chunks}
    workload_id = _hash_dict(workload)
    entry, log_path = _new_run_entry(workload_id, workload, 0, run_dir, "kld", max_temp)
    t0 = time.monotonic()
    ref_base = cache_dir / f"ref_{ref.stem}_c{ctx}.kld"
    try:
        llama.reference_logits(ref, text, ctx, ref_base, chunks=chunks, bin_dir=bin_dir)
        with gpu.Sampler() as sampler:
            kld = llama.kl_divergence(artifact_path, ref_base, text, ctx, llama.kv_flags(spec), log_path,
                                       chunks=chunks, bin_dir=bin_dir)
        entry["wall_seconds"] = time.monotonic() - t0
        entry["quality"] = {"eval_name": "kld", "split": "dev", "score": kld.get("ppl"),
                             "kl_mean": kld.get("kld_mean"), "kl_p99": kld.get("kld_p99"),
                             "kl_max": kld.get("kld_max"), "top1_agree_pct": kld.get("top1_agree_pct")}
        entry["resources"] = sampler.summary()
    except llama.RunFailed as e:
        entry["status"] = "failed"
        entry["failure"] = {"type": "kld", "message": str(e), "log_path": str(e.log_path)}
    return entry


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

    extra_flags = llama.kv_flags(spec) + list(spec.engine_flags)
    if spec.batch_size:
        extra_flags += ["-b", str(spec.batch_size)]
    if spec.ubatch_size:
        extra_flags += ["-ub", str(spec.ubatch_size)]

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
        runs.append(measure_kld(spec, artifact_path, Path(args.kld_ref), Path(args.kld_text),
                                 args.kld_ctx, args.kld_chunks, Path(args.cache_dir), run_dir,
                                 args.max_temp, args.bin_dir))

    record = {
        "record_version": 1, "spec_hash": spec_hash, "artifact_hash": artifact_hash, "env_id": env_id,
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
    args = ap.parse_args()

    if not args.no_kld and not (args.kld_ref and args.kld_text):
        ap.error("--kld-ref and --kld-text are required unless --no-kld")

    for spec_path in args.specs:
        run_spec(Path(spec_path), args)


if __name__ == "__main__":
    main()
