"""The orchestrator: candidate specs go in, interleaved measurements and a
result record come out.

python -m harness.run specs/a.json specs/b.json --repeats 10 --pp 512 --tg 128 \\
    --max-temp 45 --kld-ref models/qwen2.5-3b-instruct-Q8_0.gguf --kld-text data/dev.txt \\
    --lm-eval-tasks gsm8k,humaneval --run-longctx

Passing N specs interleaves their bench repeats (round-robin, rotating order
each round) rather than measuring spec A fully, then spec B fully -- the
latter is a real methodological mistake (docs/plan.md section 4: interleave
so thermal drift affects every arm equally), and Week 2/3 made it. There is
deliberately only one execution path here, so it can't be made again by
accident; `harness/compare.py` reads the results back out of the DB rather
than offering a second way to run things.

For each spec ("arm"): build (or reuse, from the artifact cache keyed by
artifact_hash, verified by actual GGUF quant type -- not just a file existing
at the expected path) the quantized model, measure pp/tg speed interleaved
across all arms (each bench record checked against what was actually asked
for, and against a physics "speed of light" bound), then per arm: KL
divergence, optionally a task suite (GSM8K/HumanEval) and the long-context
suite, against a real llama-server. Every quality measure that reads free
text or task docs goes through harness.splits' chokepoints first. Writes a
raw JSON result record (results/runs/<id>/record.json, the source of truth)
twice -- once after the bench phase, again after quality -- so a crash
during one arm's quality suite in a larger sweep doesn't lose the bench data
already collected for every arm; and rows into DuckDB (results/runs.duckdb,
a rebuildable summary of those records).
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from uuid import uuid4

from harness import db, gpu, llama, lmeval, longctx, memory, physics, splits
from harness.server import LlamaServer, ServerFailed
from harness.spec import CandidateSpec

_QUANT_FILE_TYPE = {  # general.file_type values, checked against llama.cpp's llama_ftype enum
    "F32": 0, "F16": 1, "Q4_0": 2, "Q4_1": 3, "Q8_0": 7, "Q5_0": 8, "Q5_1": 9,
    "Q2_K": 10, "Q3_K_S": 11, "Q3_K_M": 12, "Q3_K_L": 13, "Q4_K_S": 14, "Q4_K_M": 15,
    "Q5_K_S": 16, "Q5_K_M": 17, "Q6_K": 18, "BF16": 32,
}


def _hash_dict(d: dict) -> str:
    return hashlib.sha256(json.dumps(d, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:16]


def _actual_file_type(gguf_path: Path) -> int | None:
    from gguf import GGUFReader

    r = GGUFReader(gguf_path)
    f = r.fields.get("general.file_type")
    return None if f is None else int(f.parts[f.data[0]][0])


def _gguf_params_and_bpw(gguf_path: Path) -> tuple[int, float]:
    """n_params and bits-per-weight computed directly from the GGUF's own
    tensor metadata, not from llama-bench's JSON -- a gate-only (--no-bench)
    stage never runs llama-bench at all, and artifacts.n_params/bpw are
    written with INSERT OR IGNORE, so relying on bench output would leave
    them permanently NULL for any artifact whose first measurement skipped
    bench."""
    from gguf import GGUFReader

    r = GGUFReader(gguf_path)
    n_params = sum(t.n_elements for t in r.tensors)
    total_bytes = sum(t.n_bytes for t in r.tensors)
    return n_params, total_bytes * 8 / n_params


def ensure_artifact(spec: CandidateSpec, cache_dir: Path, log_dir: Path, bin_dir: str) -> dict:
    """Build (or reuse) the weights file for spec.artifact_hash. This *is* the
    artifact cache: a file at the expected path means it's already built --
    but a cache hit also checks the file's actual GGUF quant type against
    spec.quant (Week 4 red-team finding: nothing previously stopped a wrong
    file, e.g. a Q3_K_M copied over a Q4_K_M cache path, from being silently
    measured and reported under the wrong label). Raises on a quant missing
    from _QUANT_FILE_TYPE rather than silently skipping the check -- a
    missing map entry should fail loud, not quietly turn the check off for
    exactly the quants nobody's verified it against yet."""
    if spec.quant in ("F16", "BF16"):
        path = Path(spec.base_model)
        n_params, bpw = _gguf_params_and_bpw(path)
        return {"path": str(path), "size_bytes": path.stat().st_size, "n_params": n_params, "bpw": bpw,
                "build_seconds": None, "build_cmd": None, "built_at": None}

    cache_dir.mkdir(parents=True, exist_ok=True)
    dst = cache_dir / f"{spec.artifact_hash}.gguf"
    if dst.exists():
        if spec.quant not in _QUANT_FILE_TYPE:
            raise ValueError(f"quant {spec.quant!r} has no entry in _QUANT_FILE_TYPE -- add one "
                              f"before using it, so cache-poisoning detection isn't silently skipped")
        actual = _actual_file_type(dst)
        if actual is not None and actual != _QUANT_FILE_TYPE[spec.quant]:
            raise ValueError(f"cached artifact {dst} has file_type={actual}, expected "
                              f"{_QUANT_FILE_TYPE[spec.quant]} for quant={spec.quant!r} -- "
                              f"the cache is poisoned or stale, delete it")
        n_params, bpw = _gguf_params_and_bpw(dst)
        return {"path": str(dst), "size_bytes": dst.stat().st_size, "n_params": n_params, "bpw": bpw,
                "build_seconds": None, "build_cmd": None, "built_at": None}

    if spec.tensor_overrides or spec.imatrix:
        raise NotImplementedError("per-tensor overrides / imatrix are not wired up yet (Level 2, later)")

    log_path = log_dir / f"quantize_{spec.artifact_hash}.log"
    build_seconds = llama.quantize(Path(spec.base_model), dst, spec.quant, log_path, bin_dir=bin_dir)
    n_params, bpw = _gguf_params_and_bpw(dst)
    return {"path": str(dst), "size_bytes": dst.stat().st_size, "n_params": n_params, "bpw": bpw,
            "build_seconds": build_seconds,
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
        "schedule": None, "checks": [],
    }
    return entry, log_path


def _check_bench_record(recs: list[dict], spec: CandidateSpec, artifact_path: Path, pp: int, tg: int,
                         depth: int) -> list[str]:
    """Returns a list of problems (empty = clean). llama-bench can return
    more than one record, or one that doesn't match what was actually
    asked for, if flags smuggle extra -p/-n/-d values or the KV/batch
    settings drifted -- this is the check that the measurement actually
    measured the spec, not just that llama-bench exited 0."""
    problems = []
    if len(recs) != 1:
        return [f"expected 1 bench record, got {len(recs)}"]
    r = recs[0]
    if r.get("n_prompt") != pp:
        problems.append(f"n_prompt={r.get('n_prompt')} != requested {pp}")
    if r.get("n_gen") != tg:
        problems.append(f"n_gen={r.get('n_gen')} != requested {tg}")
    if r.get("n_depth") != depth:
        problems.append(f"n_depth={r.get('n_depth')} != requested {depth}")
    if r.get("type_k") != spec.kv_type_k:
        problems.append(f"type_k={r.get('type_k')!r} != spec {spec.kv_type_k!r}")
    if r.get("type_v") != spec.kv_type_v:
        problems.append(f"type_v={r.get('type_v')!r} != spec {spec.kv_type_v!r}")
    want_fa = 1 if spec.flash_attn else 0
    if r.get("flash_attn") not in (want_fa, bool(want_fa)):
        problems.append(f"flash_attn={r.get('flash_attn')!r} != spec {want_fa}")
    if r.get("n_gpu_layers") != 99:
        problems.append(f"n_gpu_layers={r.get('n_gpu_layers')} != 99")
    if spec.batch_size and r.get("n_batch") != spec.batch_size:
        problems.append(f"n_batch={r.get('n_batch')} != spec {spec.batch_size}")
    if spec.ubatch_size and r.get("n_ubatch") != spec.ubatch_size:
        problems.append(f"n_ubatch={r.get('n_ubatch')} != spec {spec.ubatch_size}")
    got_model = Path(r.get("model_filename", "")).resolve()
    if got_model != artifact_path.resolve():
        problems.append(f"model_filename={got_model} != artifact {artifact_path.resolve()}")
    return problems


def measure_bench(spec: CandidateSpec, artifact_path: Path, kind: str, pp: int, tg: int, repeat: int,
                   extra_flags: list[str], run_dir: Path, max_temp: int, bin_dir: str,
                   overhead_gb: float, bandwidth: dict, schedule: dict | None = None) -> dict:
    depth = spec.ctx
    workload = {"kind": kind, "input_len": pp, "output_len": tg, "depth": depth, "concurrency": 1}
    workload_id = _hash_dict(workload)
    entry, log_path = _new_run_entry(workload_id, workload, repeat, run_dir, f"{kind}_{repeat}", max_temp)
    entry["schedule"] = schedule
    t0 = time.monotonic()
    try:
        with gpu.Sampler() as sampler:
            recs = llama.bench(artifact_path, pp, tg, 1, extra_flags, log_path, depth=depth,
                                bin_dir=bin_dir)
        entry["wall_seconds"] = time.monotonic() - t0

        problems = _check_bench_record(recs, spec, artifact_path, pp, tg, depth)
        if problems:
            entry["status"] = "invalid"
            entry["failure"] = {"type": "bench_mismatch", "message": "; ".join(problems),
                                 "log_path": str(log_path)}
            return entry

        r = recs[0]
        tps = r["avg_ts"]
        entry["speed"] = {"pp_tps": tps if pp else None, "tg_tps": tps if tg else None,
                           "ttft_ms": None, "p50_ms": None, "p95_ms": None, "raw": r}
        res = sampler.summary(n_tokens=tg or None)
        res["predicted_total_mb"] = _predicted_total_mb(spec, artifact_path, overhead_gb)
        res["nvml_log"] = None
        entry["resources"] = res

        # The physics "speed of light" check: a result that exceeds the
        # hardware's theoretical ceiling is a bug or a cheat, not good noise.
        if tg and bandwidth.get("gbps"):
            bpt = physics.decode_bytes_per_token(artifact_path, depth, spec.kv_type_k, spec.kv_type_v)
            if bpt:
                check = physics.check_decode(tps, bpt["total"], bandwidth["gbps"])
                entry["checks"].append(check)
                if not check["passed"]:
                    entry["status"] = "invalid"
                    entry["failure"] = {"type": "physics", "message": f"tg_tps {tps} exceeds "
                                         f"bandwidth bound {check['bound']:.1f}", "log_path": str(log_path)}
        if pp and bandwidth.get("peak_tops") and r.get("model_n_params"):
            check = physics.check_prefill(tps, r["model_n_params"], bandwidth["peak_tops"])
            entry["checks"].append(check)
            if not check["passed"]:
                entry["status"] = "invalid"
                entry["failure"] = {"type": "physics", "message": f"pp_tps {tps} exceeds "
                                     f"compute bound {check['bound']:.1f}", "log_path": str(log_path)}
    except llama.RunFailed as e:
        entry["status"] = "failed"
        entry["failure"] = {"type": "bench", "message": str(e), "log_path": str(e.log_path)}
    return entry


def measure_kld(spec: CandidateSpec, artifact_path: Path, ref: Path, text: Path, text_hash: str,
                 split: str, ctx: int, chunks: int | None, cache_dir: Path, run_dir: Path,
                 max_temp: int, bin_dir: str) -> dict:
    # eval_name/log name include ctx and chunks: a spec measured at two KL
    # fidelities (e.g. a cheap gate pass and a later full pass) would
    # otherwise collide under the single name "kld" -- the second overwrites
    # the first both in the quality table (same (run_id, eval_name) isn't
    # the issue; it's that compare's "latest by eval_name" lookup can't tell
    # them apart) and on disk (both write to the same kld.log).
    eval_name = f"kld_c{ctx}_n{chunks or 0}"
    workload = {"kind": "kld", "ref": ref.name, "prompt_set_hash": text_hash, "split": split,
                "input_len": ctx, "output_len": 0, "concurrency": 1, "chunks": chunks}
    workload_id = _hash_dict(workload)
    entry, log_path = _new_run_entry(workload_id, workload, 0, run_dir, eval_name, max_temp)
    t0 = time.monotonic()
    ref_base = cache_dir / f"ref_{ref.stem}_{text_hash}_c{ctx}_n{chunks or 0}.kld"
    try:
        llama.reference_logits(ref, text, ctx, ref_base, chunks=chunks, bin_dir=bin_dir)
        with gpu.Sampler() as sampler:
            kld = llama.kl_divergence(artifact_path, ref_base, text, ctx, llama.spec_flags(spec), log_path,
                                       chunks=chunks, bin_dir=bin_dir)
        entry["wall_seconds"] = time.monotonic() - t0
        entry["quality"] = {"eval_name": eval_name, "split": split, "score": kld.get("ppl"),
                             "kl_mean": kld.get("kld_mean"), "kl_p99": kld.get("kld_p99"),
                             "kl_max": kld.get("kld_max"), "top1_agree_pct": kld.get("top1_agree_pct")}
        entry["resources"] = sampler.summary()
    except llama.RunFailed as e:
        entry["status"] = "failed"
        entry["failure"] = {"type": "kld", "message": str(e), "log_path": str(e.log_path)}
    return entry


def _load_eval_text(path: Path, allow_held_out: bool) -> tuple[str, str, str]:
    """Wraps splits.load_eval_text, falling back to treating the file as
    unclassified ("adhoc") if data/splits.json doesn't exist yet."""
    try:
        manifest = splits.load_manifest()
    except FileNotFoundError:
        text = path.read_text(encoding="utf-8")
        return text, "adhoc", hashlib.sha256(text.encode()).hexdigest()[:16]
    return splits.load_eval_text(path, manifest, allow_held_out)


def measure_lm_eval(spec: CandidateSpec, artifact_path: Path, tasks: list[str], eval_split: str,
                     allow_held_out: bool, limit: int | None, run_dir: Path, max_temp: int,
                     bin_dir: str) -> list[dict]:
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
    server_ctx = max(lengths) + 512

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


@dataclass
class Arm:
    idx: int
    spec: CandidateSpec
    spec_path: Path
    run_dir: Path
    artifact: dict
    artifact_path: Path
    extra_flags: list[str]
    bench_runs: list[dict] = field(default_factory=list)
    quality_runs: list[dict] = field(default_factory=list)


def prepare_arm(idx: int, spec_path: Path, session_id: str, args: argparse.Namespace) -> Arm:
    """Loads the spec and builds/verifies its artifact -- done for every arm
    before any GPU time is spent on benchmarking, so a quantize failure on
    arm 5 of 6 is caught before arms 0-4 waste a bench run that'll be
    reported alongside a half-prepared sweep."""
    spec = CandidateSpec.from_json(spec_path)
    run_dir = Path(args.out) / f"{spec.spec_hash}_{session_id}_a{idx}"
    run_dir.mkdir(parents=True, exist_ok=True)
    artifact = ensure_artifact(spec, Path(args.cache_dir), run_dir, args.bin_dir)
    extra_flags = llama.spec_flags(spec) + list(spec.engine_flags)
    return Arm(idx, spec, spec_path, run_dir, artifact, Path(artifact["path"]), extra_flags)


def bench_interleaved(arms: list[Arm], session_id: str, args: argparse.Namespace) -> None:
    """Rotates arm order each round (a Latin square, not fixed A,B,A,B --
    removes position bias too) and runs kind-major within a round (all
    arms' pp, then all arms' tg), so paired measurements across arms land
    as close together in time as possible: this is what interleaving is
    for -- sharing thermal drift and any other slow confound across every
    arm equally, per docs/plan.md section 4."""
    n = len(arms)
    bandwidth = physics.peak_bandwidth_gbps(override=args.bandwidth_gbps)
    bandwidth["peak_tops"] = args.peak_tops or physics.KNOWN_PEAK_DENSE_TOPS.get(
        gpu.environment()["gpu"])

    for rnd in range(args.repeats):
        order = arms[rnd % n:] + arms[:rnd % n]
        for kind, pp, tg in (("bench_pp", args.pp, 0), ("bench_tg", 0, args.tg)):
            for pos, arm in enumerate(order):
                schedule = {"session_id": session_id, "arm": arm.idx, "round": rnd, "order_pos": pos}
                entry = measure_bench(arm.spec, arm.artifact_path, kind, pp, tg, rnd, arm.extra_flags,
                                       arm.run_dir, args.max_temp, args.bin_dir, args.overhead_gb,
                                       bandwidth, schedule=schedule)
                arm.bench_runs.append(entry)


def run_quality(arm: Arm, args: argparse.Namespace) -> None:
    if args.kld:
        kld_path = Path(args.kld_text)
        _, kld_split, kld_text_hash = _load_eval_text(kld_path, args.allow_held_out)
        # --kld is repeatable (CTX:CHUNKS), so one spec can be KL-measured at
        # multiple fidelities in one run -- e.g. a cheap gate-stage check and
        # a full survivor check both land in the same record. measure_kld's
        # eval_name already bakes in ctx/chunks, so these don't collide.
        for ctx, chunks in args.kld:
            arm.quality_runs.append(measure_kld(arm.spec, arm.artifact_path, Path(args.kld_ref), kld_path,
                                                 kld_text_hash, kld_split, ctx, chunks,
                                                 Path(args.cache_dir), arm.run_dir, args.max_temp,
                                                 args.bin_dir))
    if args.lm_eval_tasks:
        arm.quality_runs += measure_lm_eval(arm.spec, arm.artifact_path, args.lm_eval_tasks,
                                             args.eval_split, args.allow_held_out, args.lm_eval_limit,
                                             arm.run_dir, args.max_temp, args.bin_dir)
    if args.run_longctx:
        haystack_path = Path(args.longctx_text or args.kld_text)
        haystack_text, longctx_split, _ = _load_eval_text(haystack_path, args.allow_held_out)
        arm.quality_runs += measure_longctx(arm.spec, arm.artifact_path, args.longctx_lengths,
                                             haystack_text, longctx_split, tuple(range(args.longctx_seeds)),
                                             arm.run_dir, args.max_temp, args.bin_dir)


def write_record(arm: Arm, env: dict, env_id: str, args: argparse.Namespace) -> dict:
    # n_params/bpw now come straight from ensure_artifact (computed from the
    # GGUF's own tensor metadata), not scavenged from a bench record's JSON --
    # a --no-bench gate stage never runs llama-bench, so that fallback would
    # have left them permanently NULL (artifacts is INSERT OR IGNORE, no
    # second chance to fill them in later).
    runs = arm.bench_runs + arm.quality_runs
    record = {
        "record_version": 4, "spec_hash": arm.spec.spec_hash, "artifact_hash": arm.spec.artifact_hash,
        "env_id": env_id, "spec": arm.spec.to_dict(), "artifact": arm.artifact,
        "environment": env, "runs": runs,
    }
    if getattr(args, "sweep_id", None):
        record["sweep"] = {"sweep_id": args.sweep_id, "stage": args.stage}
    (arm.run_dir / "record.json").write_text(json.dumps(record, indent=2, default=str))
    con = db.connect(args.db)
    db.insert_record(con, record)
    con.close()
    return record


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("specs", nargs="+", help="one or more candidate spec JSON files (interleaved if >1)")
    ap.add_argument("--repeats", type=int, default=10,
                     help="min 5 -- fewer rounds can't reach significance in the bootstrap/sign test")
    ap.add_argument("--pp", type=int, default=512)
    ap.add_argument("--tg", type=int, default=128)
    ap.add_argument("--max-temp", type=int, default=70)
    ap.add_argument("--overhead-gb", type=float, default=0.8, help="see docs/week1-results.md for calibration")
    ap.add_argument("--bandwidth-gbps", type=float, default=None, help="override the physics bandwidth bound")
    ap.add_argument("--peak-tops", type=float, default=None, help="override the physics compute bound")
    ap.add_argument("--bin-dir", default=llama.DEFAULT_BIN)
    ap.add_argument("--cache-dir", default="models/cache")
    ap.add_argument("--out", default="results/runs")
    ap.add_argument("--db", default=db.DB_PATH)
    ap.add_argument("--no-bench", action="store_true",
                     help="skip the interleaved bench phase (gate stages: quality only)")
    ap.add_argument("--no-kld", action="store_true")
    ap.add_argument("--kld-ref", help="reference model for KL divergence (required unless --no-kld)")
    ap.add_argument("--kld-text", help="eval text file (required unless --no-kld)")
    ap.add_argument("--kld", action="append", metavar="CTX:CHUNKS", default=None,
                     help="repeatable -- one KL measurement per entry, e.g. --kld 512:50 --kld 16384:2 "
                          "(default: 512:50). Each gets its own eval_name, so a spec can be KL-measured "
                          "at multiple fidelities in one run without the results colliding.")
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
    ap.add_argument("--sweep-id", help="tag written into the record/DB for later grid analysis (optional)")
    ap.add_argument("--stage", default=None, help="e.g. 'gate' or 'full' -- only meaningful with --sweep-id")
    args = ap.parse_args()
    args.lm_eval_tasks = [t.strip() for t in args.lm_eval_tasks.split(",") if t.strip()]
    if args.no_kld:
        args.kld = []  # --no-kld always wins, even if --kld was also (contradictorily) passed
    else:
        entries = args.kld if args.kld is not None else ["512:50"]
        args.kld = []
        for entry in entries:
            ctx_s, _, chunks_s = entry.partition(":")
            args.kld.append((int(ctx_s), int(chunks_s) if chunks_s else None))

    if args.kld and not (args.kld_ref and args.kld_text):
        ap.error("--kld-ref and --kld-text are required unless --no-kld")
    if args.run_longctx and not (args.longctx_text or args.kld_text):
        ap.error("--run-longctx needs --longctx-text or --kld-text to source the haystack")
    if args.eval_split == "held_out" and not args.allow_held_out:
        ap.error("--eval-split held_out requires --allow-held-out")
    if not args.no_bench and args.repeats < 5:
        ap.error("--repeats must be at least 5 (docs/plan.md section 4: 'run at least 5 repeats')")
    if args.sweep_id and not args.stage:
        ap.error("--sweep-id requires --stage")

    session_id = uuid4().hex[:8]
    env = gpu.environment()
    commit = gpu.llamacpp_commit(str(Path(args.bin_dir).resolve().parents[1]))
    stable = gpu.stable_env(env, commit)
    env_id = _hash_dict(stable)

    arms = [prepare_arm(i, Path(p), session_id, args) for i, p in enumerate(args.specs)]
    if not args.no_bench:
        bench_interleaved(arms, session_id, args)
        for arm in arms:
            write_record(arm, stable, env_id, args)  # bench-only checkpoint
    for arm in arms:
        run_quality(arm, args)
        write_record(arm, stable, env_id, args)  # full record
        ok = sum(1 for r in arm.bench_runs + arm.quality_runs if r["status"] == "ok")
        total = len(arm.bench_runs) + len(arm.quality_runs)
        print(f"arm {arm.idx} {arm.spec.spec_hash} ({arm.spec.quant}, kv={arm.spec.kv_type_k}/"
              f"{arm.spec.kv_type_v}, fa={arm.spec.flash_attn}, ctx={arm.spec.ctx}): "
              f"{ok}/{total} runs ok -> {arm.run_dir}")

    if len(arms) > 1:
        print(f"session {session_id}: python -m harness.compare --session {session_id}")


if __name__ == "__main__":
    main()
