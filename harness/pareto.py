"""Reporting for a grid sweep: a table, two plots, and the short-KL-vs-
long-context-quality agreement analysis. DB-only -- reads back what
harness.run/harness.grid already wrote, executes nothing.

python -m harness.pareto --sweep w5 --stage full --out results/pareto/w5/
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from harness import db, memory, stats

matplotlib = __import__("matplotlib")
matplotlib.use("Agg")  # no display needed; this always runs headless
import matplotlib.pyplot as plt  # noqa: E402


def sweep_table(con, sweep_id: str, stage: str) -> list[dict]:
    """One row per spec_hash in this sweep+stage: quant/KV/bpw/predicted and
    measured VRAM, speed medians, and every quality metric found for that
    spec_hash (KL at whatever fidelities were measured, task scores,
    long-context scores) -- whatever's in the DB, not assumed to all exist."""
    spec_hashes = [r[0] for r in con.execute(
        "SELECT DISTINCT spec_hash FROM sweep_members WHERE sweep_id=? AND stage=?",
        [sweep_id, stage]).fetchall()]

    rows = []
    for spec_hash in spec_hashes:
        s = con.execute("SELECT quant, kv_type_k, kv_type_v, ctx, artifact_hash FROM specs "
                         "WHERE spec_hash=?", [spec_hash]).fetchone()
        if not s:
            continue
        quant, kv_k, kv_v, ctx, artifact_hash = s
        a = con.execute("SELECT path, size_bytes, bpw FROM artifacts WHERE artifact_hash=?",
                         [artifact_hash]).fetchone()
        path, size_bytes, bpw = a if a else (None, None, None)

        predicted_mb = None
        if path and Path(path).exists():
            try:
                shape = memory.shape_from_gguf(path)
                predicted_mb = memory.total_bytes(size_bytes, shape, ctx, kv_k, kv_v,
                                                   int(0.8 * 1e9)) / 2**20
            except Exception:
                pass

        measured_vram = con.execute("""
            SELECT max(res.peak_vram_delta_mb) FROM resources res
            JOIN runs r ON res.run_id = r.run_id
            JOIN sweep_members sm ON sm.run_id = r.run_id
            WHERE sm.sweep_id=? AND sm.stage=? AND r.spec_hash=? AND r.status='ok'
        """, [sweep_id, stage, spec_hash]).fetchone()[0]

        speed = {}
        for kind, col in (("pp", "pp_tps"), ("tg", "tg_tps")):
            vals = [v[0] for v in con.execute(f"""
                SELECT sp.{col} FROM speed sp
                JOIN runs r ON sp.run_id = r.run_id
                JOIN sweep_members sm ON sm.run_id = r.run_id
                WHERE sm.sweep_id=? AND sm.stage=? AND r.spec_hash=? AND r.status='ok' AND sp.{col} IS NOT NULL
            """, [sweep_id, stage, spec_hash]).fetchall()]
            speed[kind] = stats.summarize(vals)["median"] if vals else None

        quality = {}
        for eval_name, split, score, kl_mean, n_items in con.execute("""
            SELECT q.eval_name, q.split, q.score, q.kl_mean, q.n_items
            FROM quality q JOIN runs r ON q.run_id = r.run_id
            JOIN sweep_members sm ON sm.run_id = r.run_id
            WHERE sm.sweep_id=? AND sm.stage=? AND r.spec_hash=?
        """, [sweep_id, stage, spec_hash]).fetchall():
            quality[eval_name] = {"split": split, "score": score, "kl_mean": kl_mean, "n_items": n_items}

        rows.append({"spec_hash": spec_hash, "quant": quant, "kv_type_k": kv_k, "kv_type_v": kv_v,
                     "ctx": ctx, "bpw": bpw, "predicted_total_mb": predicted_mb,
                     "measured_vram_delta_mb": measured_vram, "pp_tps": speed.get("pp"),
                     "tg_tps": speed.get("tg"), "quality": quality})
    return rows


def _kl_for(row: dict, prefer_ctx: int | None = None) -> tuple[str, float] | None:
    """Picks one KL eval_name/value from a row's quality dict, preferring
    one matching prefer_ctx (encoded in the eval_name as kld_c<ctx>_n<chunks>)."""
    kl_entries = [(name, q["kl_mean"]) for name, q in row["quality"].items()
                  if name.startswith("kld_") and q["kl_mean"] is not None]
    if not kl_entries:
        return None
    if prefer_ctx is not None:
        for name, kl in kl_entries:
            if f"_c{prefer_ctx}_" in name:
                return name, kl
    return kl_entries[0]


def front(rows: list[dict], x_key: str, y_key: str, x_max: bool = True, y_min: bool = True) -> list[dict]:
    pts = [(r[x_key], r[y_key]) for r in rows if r.get(x_key) is not None and r.get(y_key) is not None]
    sign_x, sign_y = (-1 if x_max else 1), (1 if y_min else -1)
    from harness.grid import nondominated_fronts
    scored = [(sign_x * x, sign_y * y) for x, y in pts]
    fronts = nondominated_fronts(scored)
    return [r for r, f in zip([r for r in rows if r.get(x_key) is not None and r.get(y_key) is not None],
                               fronts) if f == 0]


def plot_tradeoff(rows: list[dict], path: Path, long_ctx: int = 16384) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    quant_colors = {}
    cmap = plt.get_cmap("tab10")
    kv_markers = {"f16": "o", "q8_0": "s", "q4_0": "^"}

    plotted = []
    for row in rows:
        kl = _kl_for(row, prefer_ctx=long_ctx)
        if row.get("tg_tps") is None or kl is None or kl[1] <= 0:
            continue
        quant = row["quant"]
        if quant not in quant_colors:
            quant_colors[quant] = cmap(len(quant_colors) % 10)
        marker = kv_markers.get(row["kv_type_k"], "x")
        ax.scatter(row["tg_tps"], kl[1], color=quant_colors[quant], marker=marker, s=80,
                   label=f"{quant}/{row['kv_type_k']}")
        plotted.append({**row, "_kl": kl[1]})

    fr = front(plotted, "tg_tps", "_kl", x_max=True, y_min=True)
    fr_sorted = sorted(fr, key=lambda r: r["tg_tps"])
    if fr_sorted:
        ax.plot([r["tg_tps"] for r in fr_sorted], [r["_kl"] for r in fr_sorted],
                "k--", alpha=0.4, label="non-dominated front")

    ax.set_yscale("log")
    ax.set_xlabel("generation speed (tok/s)")
    ax.set_ylabel(f"KL mean vs. f16 (ctx {long_ctx}, log scale)")
    ax.set_title("Speed vs. long-context quality")
    handles, labels = ax.get_legend_handles_labels()
    uniq = dict(zip(labels, handles))
    ax.legend(uniq.values(), uniq.keys(), fontsize=8, loc="best")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_heatmap(rows: list[dict], path: Path, long_ctx: int = 16384, short_ctx: int = 512) -> None:
    quants = sorted({r["quant"] for r in rows})
    kvs = sorted({r["kv_type_k"] for r in rows})
    grid_vals = [[None] * len(kvs) for _ in quants]
    annot = [[""] * len(kvs) for _ in quants]
    for row in rows:
        i, j = quants.index(row["quant"]), kvs.index(row["kv_type_k"])
        long_kl = _kl_for(row, prefer_ctx=long_ctx)
        short_kl = _kl_for(row, prefer_ctx=short_ctx)
        if long_kl:
            grid_vals[i][j] = long_kl[1]
        annot[i][j] = f"long={long_kl[1]:.3f}\nshort={short_kl[1]:.4f}" if long_kl and short_kl else ""

    fig, ax = plt.subplots(figsize=(6, 5))
    import math
    display_vals = [[math.log10(v) if v and v > 0 else None for v in row] for row in grid_vals]
    flat = [v for row in display_vals for v in row if v is not None]
    im = ax.imshow(display_vals, cmap="viridis", aspect="auto",
                   vmin=min(flat) if flat else 0, vmax=max(flat) if flat else 1)
    ax.set_xticks(range(len(kvs)))
    ax.set_xticklabels(kvs)
    ax.set_yticks(range(len(quants)))
    ax.set_yticklabels(quants)
    ax.set_xlabel("KV type")
    ax.set_ylabel("weight quant")
    ax.set_title(f"long-context KL (log10) by weights x KV\n(annotated: long={{long}}, short={{short}} KL)")
    for i in range(len(quants)):
        for j in range(len(kvs)):
            if annot[i][j]:
                ax.text(j, i, annot[i][j], ha="center", va="center", color="white", fontsize=7)
    fig.colorbar(im, ax=ax, label="log10(KL mean)")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def agreement(gate_rows: list[dict], short_ctx: int = 512, long_ctx: int = 16384,
              fronts: int = 2, min_n: int = 4, max_n: int = 8) -> dict:
    """Does short-context KL predict long-context quality? Computed over
    EVERY gate-tier config, not just survivors -- gating on the metric
    being validated would bias the correlation. Also reports gate recall:
    the fraction of configs a long-KL-based gate would promote that the
    actual short-KL-based gate also promoted -- the number that decides
    whether the gate is trustworthy enough to reuse on a bigger model."""
    from harness.grid import nondominated_fronts

    pairs = [(r["spec_hash"], _kl_for(r, short_ctx), _kl_for(r, long_ctx)) for r in gate_rows]
    pairs = [(h, s[1], l[1]) for h, s, l in pairs if s and l]
    if len(pairs) < 3:
        return {"error": f"only {len(pairs)} configs have both short and long KL -- too few to correlate"}

    short_vals = [p[1] for p in pairs]
    long_vals = [p[2] for p in pairs]
    corr = stats.spearman_ci(short_vals, long_vals)

    predicted_mb = {r["spec_hash"]: r["predicted_total_mb"] for r in gate_rows}
    hashes = [p[0] for p in pairs]

    def promoted_by(values):
        pts = [(v, predicted_mb.get(h) or 0) for h, v in zip(hashes, values)]
        fr = nondominated_fronts(pts)
        order = sorted(range(len(hashes)), key=lambda i: (fr[i], pts[i][0]))
        promoted = set()
        for i in order:
            if len(promoted) >= max_n:
                break
            if fr[i] < fronts or len(promoted) < min_n:
                promoted.add(hashes[i])
        return promoted

    by_short = promoted_by(short_vals)
    by_long = promoted_by(long_vals)
    recall = len(by_short & by_long) / len(by_long) if by_long else None

    return {"spearman": corr, "n_configs": len(pairs), "promoted_by_short": sorted(by_short),
            "promoted_by_long": sorted(by_long), "gate_recall": recall}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", required=True)
    ap.add_argument("--stage", default="full")
    ap.add_argument("--gate-stage", default="gate", help="stage to use for the agreement analysis")
    ap.add_argument("--long-ctx", type=int, default=16384)
    ap.add_argument("--short-ctx", type=int, default=512)
    ap.add_argument("--db", default=db.DB_PATH)
    ap.add_argument("--out", default=None, help="default: results/pareto/<sweep>/")
    args = ap.parse_args()

    con = db.connect(args.db)
    rows = sweep_table(con, args.sweep, args.stage)
    gate_rows = sweep_table(con, args.sweep, args.gate_stage)
    con.close()

    out_dir = Path(args.out) if args.out else Path("results/pareto") / args.sweep
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "table.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["spec_hash", "quant", "kv_type_k", "kv_type_v", "ctx", "bpw", "predicted_total_mb",
                    "measured_vram_delta_mb", "pp_tps", "tg_tps"])
        for r in rows:
            w.writerow([r["spec_hash"], r["quant"], r["kv_type_k"], r["kv_type_v"], r["ctx"], r["bpw"],
                        r["predicted_total_mb"], r["measured_vram_delta_mb"], r["pp_tps"], r["tg_tps"]])

    plot_tradeoff(rows, out_dir / "tradeoff.png", long_ctx=args.long_ctx)
    plot_heatmap(gate_rows, out_dir / "heatmap.png", long_ctx=args.long_ctx, short_ctx=args.short_ctx)
    agree = agreement(gate_rows, short_ctx=args.short_ctx, long_ctx=args.long_ctx)
    (out_dir / "summary.json").write_text(json.dumps(agree, indent=2, default=str))

    print(f"wrote {out_dir}/table.csv, tradeoff.png, heatmap.png, summary.json")
    if "spearman" in agree:
        print(f"short-vs-long KL: rho={agree['spearman']['rho']:.2f} "
              f"CI={agree['spearman']['ci']}, gate_recall={agree['gate_recall']}")
    else:
        print(agree.get("error"))


if __name__ == "__main__":
    main()
