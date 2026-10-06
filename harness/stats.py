"""Small statistics helpers: summaries and bootstrap confidence intervals."""
from __future__ import annotations

import random
import statistics


def summarize(xs: list[float]) -> dict:
    med = statistics.median(xs)
    sd = statistics.stdev(xs) if len(xs) > 1 else 0.0
    return {
        "n": len(xs),
        "median": med,
        "mean": statistics.fmean(xs),
        "stdev": sd,
        "cv_pct": 100 * sd / statistics.fmean(xs) if xs and statistics.fmean(xs) else 0.0,
        "min": min(xs),
        "max": max(xs),
    }


def bootstrap_ci(xs: list[float], stat=statistics.median, n_boot: int = 10_000,
                 alpha: float = 0.05, seed: int = 0) -> tuple[float, float]:
    """Percentile bootstrap CI for a statistic of one sample."""
    rng = random.Random(seed)
    boots = sorted(stat(rng.choices(xs, k=len(xs))) for _ in range(n_boot))
    return boots[int(n_boot * alpha / 2)], boots[int(n_boot * (1 - alpha / 2)) - 1]


def bootstrap_diff_ci(a: list[float], b: list[float], stat=statistics.median,
                      n_boot: int = 10_000, alpha: float = 0.05, seed: int = 0) -> tuple[float, float]:
    """CI for stat(b) - stat(a). A difference is real only if the interval excludes 0."""
    rng = random.Random(seed)
    diffs = sorted(
        stat(rng.choices(b, k=len(b))) - stat(rng.choices(a, k=len(a))) for _ in range(n_boot)
    )
    return diffs[int(n_boot * alpha / 2)], diffs[int(n_boot * (1 - alpha / 2)) - 1]


def paired_rel_diff_ci(a: list[float], b: list[float], stat=statistics.median,
                       n_boot: int = 10_000, alpha: float = 0.05, seed: int = 0) -> tuple[float, float]:
    """CI for stat(b_i/a_i - 1) over PAIRED rounds (a[i] and b[i] are the same
    round, measured close together in time by interleaving) -- resamples
    round INDICES, not a and b independently, so the pairing interleaving
    pays for isn't thrown away the way the unpaired bootstrap_diff_ci would.
    Relative (not absolute), so it stays meaningful comparing models of very
    different raw speed. len(a) must equal len(b)."""
    if len(a) != len(b):
        raise ValueError(f"paired series must be the same length, got {len(a)} and {len(b)}")
    n = len(a)
    rng = random.Random(seed)
    rel = sorted(
        stat([b[i] / a[i] - 1 for i in idx])
        for idx in (rng.choices(range(n), k=n) for _ in range(n_boot))
    )
    return rel[int(n_boot * alpha / 2)], rel[int(n_boot * (1 - alpha / 2)) - 1]


def bootstrap_prop_diff_ci(k_a: int, n_a: int, k_b: int, n_b: int, n_boot: int = 10_000,
                           alpha: float = 0.05, seed: int = 0) -> tuple[float, float]:
    """CI for p_b - p_a, two independent proportions, rebuilt from counts
    alone (k successes out of n items). Works directly on data already in
    the DB (quality.score * quality.n_items) -- no per-item records needed,
    so it applies retroactively to any task-suite score already measured."""
    rng = random.Random(seed)
    xs_a = [1] * k_a + [0] * (n_a - k_a)
    xs_b = [1] * k_b + [0] * (n_b - k_b)
    diffs = sorted(
        statistics.fmean(rng.choices(xs_b, k=n_b)) - statistics.fmean(rng.choices(xs_a, k=n_a))
        for _ in range(n_boot)
    )
    return diffs[int(n_boot * alpha / 2)], diffs[int(n_boot * (1 - alpha / 2)) - 1]


def _ranks(xs: list[float]) -> list[float]:
    """1-indexed ranks, ties given the average rank of the tied block --
    the standard definition Spearman's rho is built on."""
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        avg_rank = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank
        i = j + 1
    return ranks


def _pearson(xs: list[float], ys: list[float]) -> float:
    mx, my = statistics.fmean(xs), statistics.fmean(ys)
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    return cov / (vx * vy) ** 0.5 if vx and vy else 0.0


def spearman(x: list[float], y: list[float]) -> float:
    """Spearman rank correlation (Pearson correlation of the ranks, average
    ranks for ties). Pure Python -- no scipy, no numpy."""
    if len(x) != len(y):
        raise ValueError(f"x and y must be the same length, got {len(x)} and {len(y)}")
    return _pearson(_ranks(x), _ranks(y))


def spearman_ci(x: list[float], y: list[float], n_boot: int = 10_000, alpha: float = 0.05,
                seed: int = 0) -> dict:
    """{"rho": ..., "ci": (lo, hi), "n": ...} -- the actual answer to "does
    metric x predict the ranking of metric y", with an honest interval
    rather than a single number eyeballed off a plot. Resamples CONFIGS
    (paired x[i]/y[i]), not x and y independently -- the two metrics must
    stay matched to the same underlying config in every bootstrap draw."""
    n = len(x)
    rho = spearman(x, y)
    rng = random.Random(seed)
    boots = sorted(
        spearman([x[i] for i in idx], [y[i] for i in idx])
        for idx in (rng.choices(range(n), k=n) for _ in range(n_boot))
    )
    lo, hi = boots[int(n_boot * alpha / 2)], boots[int(n_boot * (1 - alpha / 2)) - 1]
    return {"rho": rho, "ci": (lo, hi), "n": n}
