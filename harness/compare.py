"""Analysis over the DB -- not a second way to run specs.

python -m harness.compare --session <id> [--baseline-arm 0]
python -m harness.compare --latest

Deliberately read-only: harness/run.py is the only execution path, and
keeping exactly one path is what stops the sequential-vs-interleaved mistake
(Weeks 2-3's comparisons were run fully sequentially, not interleaved) from
quietly recurring. This module only reads back what run.py already wrote.

Compares every arm against one baseline, not all pairs -- with N arms that's
N-1 comparisons rather than a false-positive-prone O(N^2). Pairs bench
rounds (dropping any round where either side is missing or invalid) and
uses stats.paired_rel_diff_ci, which resamples round indices together so
the pairing interleaving paid for isn't thrown away. Reports a quality
delta alongside every speed delta -- docs/plan.md's "a fast config that
produces garbage is not fast" rule, enforced here: an arm with no quality
measurement prints QUALITY UNVERIFIED instead of a clean-looking speed win.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict

from harness import db, stats


def latest_session(con) -> str | None:
    row = con.execute("SELECT session_id FROM schedule ORDER BY rowid DESC LIMIT 1").fetchone()
    return row[0] if row else None


def session_rows(con, session_id: str) -> list[tuple]:
    return con.execute("""
        SELECT sc.arm, sc.round, w.kind, sp.pp_tps, sp.tg_tps, r.status, r.spec_hash
        FROM schedule sc
        JOIN runs r ON sc.run_id = r.run_id
        JOIN workloads w ON r.workload_id = w.workload_id
        LEFT JOIN speed sp ON sp.run_id = r.run_id
        WHERE sc.session_id = ?
        ORDER BY sc.arm, sc.round
    """, [session_id]).fetchall()


def _latest_quality(con, spec_hash: str) -> dict[tuple[str, str], dict]:
    """Keyed on (eval_name, split), not eval_name alone: a cheap gate-split
    KL and a full dev-split KL can legitimately share an eval_name (same
    ctx/chunks, different split) without one overwriting the other here --
    eval_name alone already distinguishes most KL fidelities (measure_kld
    bakes ctx/chunks into the name), but split is the second axis that can
    still collide without this."""
    rows = con.execute("""
        SELECT q.eval_name, q.split, q.score, q.n_items, q.kl_mean, r.started_at
        FROM quality q JOIN runs r ON q.run_id = r.run_id
        WHERE r.spec_hash = ?
        ORDER BY r.started_at DESC
    """, [spec_hash]).fetchall()
    out = {}
    for eval_name, split, score, n_items, kl_mean, _ in rows:
        key = (eval_name, split)
        if key not in out:  # first (most recent) row wins
            out[key] = {"score": score, "n_items": n_items, "kl_mean": kl_mean}
    return out


def _spec_dict(con, spec_hash: str) -> dict:
    row = con.execute("SELECT spec_json FROM specs WHERE spec_hash=?", [spec_hash]).fetchone()
    return json.loads(row[0]) if row else {}


def _paired_series(rows: list[tuple], arm: int, kind: str) -> dict[int, tuple[float | None, str]]:
    out = {}
    for a, rnd, k, pp_tps, tg_tps, status, _ in rows:
        if a != arm or k != kind:
            continue
        tps = pp_tps if kind == "bench_pp" else tg_tps
        out[rnd] = (tps, status)
    return out


def _pair_and_drop(baseline_series: dict, arm_series: dict) -> tuple[list[float], list[float], int]:
    rounds = sorted(set(baseline_series) & set(arm_series))
    a_vals, b_vals, dropped = [], [], 0
    for rnd in rounds:
        ta, sa = baseline_series[rnd]
        tb, sb = arm_series[rnd]
        if sa == "ok" and sb == "ok" and ta is not None and tb is not None:
            a_vals.append(ta)
            b_vals.append(tb)
        else:
            dropped += 1
    return a_vals, b_vals, dropped


def _verdict(lo: float, hi: float) -> str:
    if lo > 0:
        return "faster"
    if hi < 0:
        return "slower"
    return "no detectable difference"


def compare_session(con, session_id: str, baseline_arm: int = 0, alpha: float = 0.05) -> dict:
    rows = session_rows(con, session_id)
    if not rows:
        raise ValueError(f"no schedule rows found for session {session_id!r}")
    arms = sorted({r[0] for r in rows})
    spec_hash_by_arm = {a: next(r[6] for r in rows if r[0] == a) for a in arms}
    baseline_spec = _spec_dict(con, spec_hash_by_arm[baseline_arm])

    result = {"session_id": session_id, "baseline_arm": baseline_arm, "n_comparisons": len(arms) - 1,
              "arms": {}}
    for arm in arms:
        entry = {"spec_hash": spec_hash_by_arm[arm], "speed": {}, "quality": None, "warnings": []}

        if arm != baseline_arm:
            arm_spec = _spec_dict(con, spec_hash_by_arm[arm])
            diffs = [k for k in set(baseline_spec) | set(arm_spec) if baseline_spec.get(k) != arm_spec.get(k)]
            if len(diffs) > 1:
                entry["warnings"].append(f"baseline and arm {arm} differ in {len(diffs)} fields "
                                          f"({sorted(diffs)}), not just the one being compared")

        for kind in ("bench_pp", "bench_tg"):
            base_series = _paired_series(rows, baseline_arm, kind)
            arm_series = _paired_series(rows, arm, kind)
            a_vals, b_vals, dropped = _pair_and_drop(base_series, arm_series)
            if len(a_vals) < 5:
                entry["speed"][kind] = {"verdict": "insufficient", "n_paired": len(a_vals), "n_dropped": dropped}
                continue
            lo, hi = stats.paired_rel_diff_ci(a_vals, b_vals, alpha=alpha)
            entry["speed"][kind] = {
                "verdict": _verdict(lo, hi) if arm != baseline_arm else "baseline",
                "rel_diff_ci": [lo, hi], "n_paired": len(a_vals), "n_dropped": dropped,
                "baseline_summary": stats.summarize(a_vals), "arm_summary": stats.summarize(b_vals),
            }

        base_quality = _latest_quality(con, spec_hash_by_arm[baseline_arm])
        arm_quality = _latest_quality(con, spec_hash_by_arm[arm])
        if not arm_quality:
            entry["quality"] = "QUALITY UNVERIFIED"
        else:
            q = {}
            for (eval_name, split), aq in arm_quality.items():
                key = f"{eval_name}@{split}"
                bq = base_quality.get((eval_name, split))
                if bq and aq.get("n_items") and bq.get("n_items") and aq.get("score") is not None \
                        and bq.get("score") is not None:
                    k_a = round(bq["score"] * bq["n_items"])
                    k_b = round(aq["score"] * aq["n_items"])
                    lo, hi = stats.bootstrap_prop_diff_ci(k_a, bq["n_items"], k_b, aq["n_items"], alpha=alpha)
                    q[key] = {"baseline_score": bq["score"], "arm_score": aq["score"],
                              "prop_diff_ci": [lo, hi]}
                elif aq.get("kl_mean") is not None:
                    q[key] = {"baseline_kl_mean": bq.get("kl_mean") if bq else None,
                              "arm_kl_mean": aq["kl_mean"]}
                else:
                    q[key] = aq
            entry["quality"] = q
        result["arms"][arm] = entry
    return result


def print_report(result: dict) -> None:
    print(f"session {result['session_id']}, baseline arm {result['baseline_arm']}, "
          f"{result['n_comparisons']} comparison(s)")
    for arm, entry in result["arms"].items():
        print(f"\n--- arm {arm} ({entry['spec_hash']}) ---")
        for w in entry["warnings"]:
            print(f"  WARNING: {w}")
        for kind, s in entry["speed"].items():
            if s["verdict"] == "insufficient":
                print(f"  {kind}: insufficient paired rounds ({s['n_paired']}, {s['n_dropped']} dropped)")
            elif s["verdict"] == "baseline":
                print(f"  {kind}: baseline, median {s['baseline_summary']['median']:.1f}, "
                      f"CV {s['baseline_summary']['cv_pct']:.2f}%")
            else:
                lo, hi = s["rel_diff_ci"]
                print(f"  {kind}: {s['verdict']} ({lo:+.1%}, {hi:+.1%}), n={s['n_paired']}, "
                      f"{s['n_dropped']} dropped")
        if entry["quality"] == "QUALITY UNVERIFIED":
            print("  quality: QUALITY UNVERIFIED")
        elif entry["quality"]:
            for name, q in entry["quality"].items():
                print(f"  quality[{name}]: {q}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--session")
    ap.add_argument("--latest", action="store_true")
    ap.add_argument("--baseline-arm", type=int, default=0)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--db", default=db.DB_PATH)
    ap.add_argument("--out", help="write the full result as JSON to this path")
    args = ap.parse_args()

    con = db.connect(args.db)
    session_id = args.session
    if args.latest or not session_id:
        session_id = latest_session(con)
        if not session_id:
            raise SystemExit("no sessions found in the DB")
    result = compare_session(con, session_id, args.baseline_arm, args.alpha)
    con.close()
    print_report(result)
    if args.out:
        from pathlib import Path
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(result, indent=2, default=str))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
