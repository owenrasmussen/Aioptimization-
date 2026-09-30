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
