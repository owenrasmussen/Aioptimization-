"""Grid enumeration and survivor selection for a sweep -- planning and
analysis only. This module never executes anything on the GPU beyond
building artifacts (which is itself a planning step, not a measurement);
every actual benchmark/quality run still goes through `python -m
harness.run`, keeping Week 4's one-execution-path discipline intact.

python -m harness.grid enumerate --sweep w5
python -m harness.grid promote --sweep w5 --baseline specs/qwen3b-q4km-kvf16-fa.json \\
    --gate-eval kld_c512_n20 --split search
"""
from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path

from harness import db, llama, memory
from harness.spec import CandidateSpec

WEEK5_QUANTS = ("Q3_K_M", "Q4_K_M", "Q5_K_M", "Q6_K", "Q8_0")
WEEK5_KV = (("f16", "f16"), ("q8_0", "q8_0"), ("q4_0", "q4_0"))


def fa_all_quants_enabled(bin_dir: str) -> bool:
    """Checks third_party/llama.cpp/build/CMakeCache.txt for
    GGML_CUDA_FA_ALL_QUANTS. Verified live this build has it OFF, meaning
    CUDA flash-attention kernels only exist for matched K/V type pairs
    (f16/f16, q8_0/q8_0, q4_0/q4_0) -- a mixed pair (q8_0 K / q4_0 V) was
    smoke-tested and did NOT crash, which is the riskier outcome: it means
    there's a fallback path whose performance isn't verified against a
    FA_ALL_QUANTS=ON build. Returns False (conservative) if the cache file
    can't be found at all, same as if the flag were off."""
    cache_path = Path(bin_dir).resolve().parents[1] / "CMakeCache.txt"
    if not cache_path.exists():
        return False
    m = re.search(r"GGML_CUDA_FA_ALL_QUANTS:BOOL=(\w+)", cache_path.read_text(errors="ignore"))
    return bool(m) and m.group(1).upper() == "ON"


@dataclass(frozen=True)
class Candidate:
    spec: CandidateSpec
    artifact_path: Path
    weight_bytes: int
    predicted_total_mb: float
    fits: bool


def enumerate_grid(base_model: Path, quants: tuple[str, ...], kv_pairs: tuple[tuple[str, str], ...],
                    ctx_lengths: list[int], budget_mb: float, overhead_gb: float, cache_dir: Path,
                    bin_dir: str, log_dir: Path) -> list[Candidate]:
    """Builds every candidate's real artifact (via the same ensure_artifact
    run.py uses -- building here *is* planning, not measurement) and checks
    it against the budget using the real file size, not a pre-build bpw
    estimate (checked live against this project's own files: generic bpw
    tables are off by several percent in a model-specific, vocab-size-driven
    way that could silently let an over-budget config through)."""
    from harness import run as run_mod  # local import: avoids a module-load cycle

    if not fa_all_quants_enabled(bin_dir):
        mixed = [(k, v) for k, v in kv_pairs if k != v]
        kv_pairs = tuple((k, v) for k, v in kv_pairs if k == v)
        if mixed:
            print(f"GGML_CUDA_FA_ALL_QUANTS is off in this build -- skipping {len(mixed)} "
                  f"mixed K/V pair(s): {mixed}")

    candidates = []
    for quant in quants:
        for kv_k, kv_v in kv_pairs:
            for ctx in ctx_lengths:
                spec = CandidateSpec(base_model=str(base_model), quant=quant, kv_type_k=kv_k,
                                      kv_type_v=kv_v, flash_attn=True, ctx=ctx)
                artifact = run_mod.ensure_artifact(spec, cache_dir, log_dir, bin_dir)
                artifact_path = Path(artifact["path"])
                shape = memory.shape_from_gguf(str(artifact_path))
                predicted_mb = memory.total_bytes(artifact["size_bytes"], shape, ctx, kv_k, kv_v,
                                                   int(overhead_gb * 1e9)) / 2**20
                fits = predicted_mb <= budget_mb
                candidates.append(Candidate(spec, artifact_path, artifact["size_bytes"], predicted_mb, fits))
    return candidates


def write_specs(candidates: list[Candidate], out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for c in candidates:
        name = f"{c.spec.quant}_k{c.spec.kv_type_k}-v{c.spec.kv_type_v}_c{c.spec.ctx}.json"
        path = out_dir / name
        c.spec.to_json(path)
        paths.append(path)
    return paths


def nondominated_fronts(points: list[tuple[float, ...]]) -> list[int]:
    """Front index per point (0 = best/non-dominated), minimizing every
    coordinate. This is the actual multi-objective sort "multi-fidelity
    pruning" needs: promoting by front, not by one metric alone, is what
    keeps the cheap/low-quality corner of the Pareto front (e.g. the
    smallest weights + smallest KV combo) from being excluded just because
    it scores worst on quality -- it's still non-dominated if nothing beats
    it on BOTH axes."""
    n = len(points)

    def dominates(p: tuple[float, ...], q: tuple[float, ...]) -> bool:
        return all(a <= b for a, b in zip(p, q)) and any(a < b for a, b in zip(p, q))

    fronts = [-1] * n
    remaining = set(range(n))
    front_idx = 0
    while remaining:
        current = [i for i in remaining if not any(j != i and dominates(points[j], points[i])
                                                     for j in remaining)]
        for i in current:
            fronts[i] = front_idx
            remaining.discard(i)
        front_idx += 1
    return fronts


def promote(con, sweep_id: str, gate_eval_name: str, split: str, baseline_spec_hash: str | None = None,
            fronts: int = 2, min_n: int = 4, max_n: int = 8, overhead_gb: float = 0.8,
            out_dir: Path | None = None) -> dict:
    """Reads gate-stage KL and predicted VRAM for every spec_hash in the
    sweep, promotes the union of the first `fronts` non-dominated fronts of
    (kl_mean, predicted_total_mb), both minimized, clipped to [min_n, max_n]
    survivors ordered by front then KL. predicted_total_mb is recomputed
    fresh from specs+artifacts (not read from resources.predicted_total_mb,
    which only exists for runs that went through a bench phase -- a
    --no-bench gate stage never populates it)."""
    rows = con.execute("""
        SELECT r.spec_hash, q.kl_mean, s.ctx, s.kv_type_k, s.kv_type_v, a.path, a.size_bytes
        FROM quality q
        JOIN runs r ON q.run_id = r.run_id
        JOIN specs s ON r.spec_hash = s.spec_hash
        JOIN artifacts a ON s.artifact_hash = a.artifact_hash
        WHERE q.eval_name = ? AND q.split = ? AND r.status = 'ok' AND q.kl_mean IS NOT NULL
        ORDER BY r.started_at
    """, [gate_eval_name, split]).fetchall()

    by_spec: dict[str, tuple[float, float]] = {}
    for spec_hash, kl_mean, ctx, kv_k, kv_v, path, size_bytes in rows:
        if spec_hash in by_spec:
            continue
        shape = memory.shape_from_gguf(path)
        predicted_mb = memory.total_bytes(size_bytes, shape, ctx, kv_k, kv_v, int(overhead_gb * 1e9)) / 2**20
        by_spec[spec_hash] = (kl_mean, predicted_mb)

    spec_hashes = list(by_spec.keys())
    points = [by_spec[h] for h in spec_hashes]
    front_of = nondominated_fronts(points)
    order = sorted(range(len(spec_hashes)), key=lambda i: (front_of[i], points[i][0]))

    survivors: list[str] = []
    for i in order:
        if len(survivors) >= max_n:
            break
        if front_of[i] < fronts or len(survivors) < min_n:
            survivors.append(spec_hashes[i])
    if baseline_spec_hash:
        if baseline_spec_hash in survivors:
            survivors.remove(baseline_spec_hash)
        survivors.insert(0, baseline_spec_hash)

    result = {
        "sweep_id": sweep_id, "gate_eval_name": gate_eval_name, "split": split, "fronts_promoted": fronts,
        "all_points": {h: {"kl_mean": by_spec[h][0], "predicted_total_mb": by_spec[h][1],
                            "front": front_of[i]} for i, h in enumerate(spec_hashes)},
        "survivors": survivors,
    }

    out_dir = out_dir or Path(f"results/sweeps/{sweep_id}")
    out_dir.mkdir(parents=True, exist_ok=True)
    survivors_dir = out_dir / "survivors"
    survivors_dir.mkdir(exist_ok=True)
    survivor_paths = []
    for h in survivors:
        spec_json = con.execute("SELECT spec_json FROM specs WHERE spec_hash=?", [h]).fetchone()[0]
        path = survivors_dir / f"{h}.json"
        path.write_text(json.dumps(json.loads(spec_json), indent=2, sort_keys=True))
        survivor_paths.append(str(path))
    result["survivor_paths"] = survivor_paths
    (out_dir / "promote.json").write_text(json.dumps(result, indent=2, default=str))
    (out_dir / "survivors.txt").write_text("\n".join(survivor_paths), newline="\n")
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    enum_ap = sub.add_parser("enumerate")
    enum_ap.add_argument("--sweep", required=True)
    enum_ap.add_argument("--base-model", default="models/qwen2.5-3b-instruct-f16.gguf")
    enum_ap.add_argument("--quants", nargs="+", default=list(WEEK5_QUANTS))
    enum_ap.add_argument("--ctx", type=int, nargs="+", default=[32768])
    enum_ap.add_argument("--budget-mb", type=float, default=11 * 1024)
    enum_ap.add_argument("--overhead-gb", type=float, default=0.8)
    enum_ap.add_argument("--cache-dir", default="models/cache")
    enum_ap.add_argument("--bin-dir", default=llama.DEFAULT_BIN)
    enum_ap.add_argument("--out", default=None, help="default: specs/<sweep>/")

    promote_ap = sub.add_parser("promote")
    promote_ap.add_argument("--sweep", required=True)
    promote_ap.add_argument("--gate-eval", required=True, help="e.g. kld_c512_n20")
    promote_ap.add_argument("--split", default="search")
    promote_ap.add_argument("--baseline", help="spec JSON whose hash is always promoted, first")
    promote_ap.add_argument("--fronts", type=int, default=2)
    promote_ap.add_argument("--min-n", type=int, default=4)
    promote_ap.add_argument("--max-n", type=int, default=8)
    promote_ap.add_argument("--overhead-gb", type=float, default=0.8)
    promote_ap.add_argument("--db", default=db.DB_PATH)

    args = ap.parse_args()
    if args.cmd == "enumerate":
        out_dir = Path(args.out) if args.out else Path("specs") / args.sweep
        candidates = enumerate_grid(Path(args.base_model), tuple(args.quants), WEEK5_KV, args.ctx,
                                     args.budget_mb, args.overhead_gb, Path(args.cache_dir),
                                     args.bin_dir, Path(f"results/sweeps/{args.sweep}/build_logs"))
        fit_count = sum(1 for c in candidates if c.fits)
        paths = write_specs(candidates, out_dir)
        print(f"{len(candidates)} enumerated, {fit_count} fit the {args.budget_mb:.0f}MB budget "
              f"-> {out_dir} ({len(paths)} spec files)")
    elif args.cmd == "promote":
        con = db.connect(args.db)
        baseline_hash = CandidateSpec.from_json(args.baseline).spec_hash if args.baseline else None
        result = promote(con, args.sweep, args.gate_eval, args.split, baseline_hash, args.fronts,
                          args.min_n, args.max_n, args.overhead_gb)
        con.close()
        print(f"{len(result['all_points'])} gate-tier configs, promoted {len(result['survivors'])} "
              f"-> results/sweeps/{args.sweep}/survivors.txt")


if __name__ == "__main__":
    main()
