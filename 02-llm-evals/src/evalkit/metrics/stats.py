"""Statistics for small evaluation sets.

With 40 questions, one question moves a rate by 2.5 points, so a bare "0.85 vs 0.82" means little.
Everything here answers: how sure are we? (confidence intervals) and is version B really better than A
on the same questions? (paired tests).
"""

from __future__ import annotations

import math
import random
from statistics import mean
from typing import Callable, Sequence, TypeVar

T = TypeVar("T")


def wilson_ci(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% confidence interval for a proportion; behaves well near 0 and 1, unlike the normal approximation."""
    if n == 0:
        return (0.0, 1.0)
    p = successes / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (max(0.0, centre - half), min(1.0, centre + half))


def bootstrap_ci(items: Sequence[T], statistic: Callable[[Sequence[T]], float] = mean,
                 n_resamples: int = 2000, alpha: float = 0.05, seed: int = 0) -> tuple[float, float]:
    """Percentile bootstrap: resample the items with replacement, recompute the statistic, take quantiles.

    Works for any statistic, including ones over pairs (e.g. kappa of [(judge, human), ...]).
    """
    if not items:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    n = len(items)
    values = sorted(statistic([items[rng.randrange(n)] for _ in range(n)]) for _ in range(n_resamples))
    return (_quantile(values, alpha / 2), _quantile(values, 1 - alpha / 2))


def paired_bootstrap(a: Sequence[float], b: Sequence[float], n_resamples: int = 2000,
                     seed: int = 0) -> dict[str, float]:
    """Compare two systems scored on the SAME items: mean(b) - mean(a), its 95% CI and a two-sided p-value.

    Pairing removes item difficulty from the comparison, so it detects real differences with fewer items.
    """
    if len(a) != len(b) or not a:
        raise ValueError("need two non-empty score lists of equal length (same items, same order)")
    diffs = [y - x for x, y in zip(a, b)]
    delta = mean(diffs)
    rng = random.Random(seed)
    n = len(diffs)
    boots = sorted(mean(diffs[rng.randrange(n)] for _ in range(n)) for _ in range(n_resamples))
    # p-value: how often the resampled delta lands on the other side of zero (two-sided)
    if delta == 0:
        p = 1.0
    else:
        other_side = sum((d <= 0) if delta > 0 else (d >= 0) for d in boots) / n_resamples
        p = min(1.0, 2 * other_side)
    return {"delta": delta, "ci_low": _quantile(boots, 0.025), "ci_high": _quantile(boots, 0.975), "p": p}


def mcnemar_exact(b: int, c: int) -> float:
    """Exact McNemar test for paired pass/fail outcomes.

    b = items only system A got right, c = items only system B got right. Items both got right or both
    got wrong say nothing about which is better. Under "no difference", b ~ Binomial(b + c, 0.5).
    Returns the two-sided p-value.
    """
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def pearson(x: Sequence[float], y: Sequence[float]) -> float:
    if len(x) != len(y) or len(x) < 2:
        raise ValueError("need two equal-length sequences with at least 2 values")
    mx, my = mean(x), mean(y)
    sxy = sum((a - mx) * (b - my) for a, b in zip(x, y))
    sxx = sum((a - mx) ** 2 for a in x)
    syy = sum((b - my) ** 2 for b in y)
    return sxy / math.sqrt(sxx * syy) if sxx and syy else 0.0


def ranks(values: Sequence[float]) -> list[float]:
    """Average ranks (ties share the mean of their positions), 1-based."""
    order = sorted(range(len(values)), key=values.__getitem__)
    out = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        for k in range(i, j + 1):
            out[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return out


def spearman(x: Sequence[float], y: Sequence[float]) -> float:
    """Rank correlation: does a higher score from one rater mean a higher score from the other?"""
    return pearson(ranks(x), ranks(y))


def percentile(values: Sequence[float], q: float) -> float:
    """q in [0, 100], linear interpolation (same as numpy's default)."""
    if not values:
        return float("nan")
    return _quantile(sorted(values), q / 100)


def _quantile(sorted_values: Sequence[float], q: float) -> float:
    pos = (len(sorted_values) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * (pos - lo)
