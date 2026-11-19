"""Paired comparisons for eval v2. No model downloads.

The procedures are the ones named in configs/eval_v2.yaml:
a percentile paired bootstrap of the mean per-query nDCG difference, and a
two-sided sign test that drops ties.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np


def paired_mean_difference(system: Sequence[float], baseline: Sequence[float]) -> np.ndarray:
    """Per-query score(system) - score(baseline)."""
    a = np.asarray(list(system), dtype=float)
    b = np.asarray(list(baseline), dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"score vectors differ in length: {a.shape} vs {b.shape}")
    if a.ndim != 1 or a.size == 0:
        raise ValueError("need a non-empty 1-d pair of scores")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("scores must be finite")
    return a - b


def paired_bootstrap_ci(
    system: Sequence[float],
    baseline: Sequence[float],
    *,
    n_resamples: int = 10_000,
    seed: int = 20260928,
    level: float = 0.95,
) -> dict[str, float | int]:
    """Percentile CI for the mean paired difference.

    Each replicate draws the queries with replacement and averages
    ``system - baseline`` on that draw. The interval is the empirical
    ``(1-level)/2`` and ``1-(1-level)/2`` quantiles (NumPy linear).
    """
    if n_resamples < 1:
        raise ValueError("n_resamples must be >= 1")
    if not 0.0 < level < 1.0:
        raise ValueError("level must be between 0 and 1")
    diffs = paired_mean_difference(system, baseline)
    n = int(diffs.size)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, n, size=(n_resamples, n))
    means = diffs[draws].mean(axis=1)
    alpha = (1.0 - level) / 2.0
    low, high = np.quantile(means, [alpha, 1.0 - alpha], method="linear")
    return {
        "n": n,
        "n_resamples": int(n_resamples),
        "seed": int(seed),
        "level": float(level),
        "mean_difference": float(diffs.mean()),
        "ci_low": float(low),
        "ci_high": float(high),
    }


def _binom_pmf_half(n: int) -> list[int]:
    """Unnormalized Bin(n, 1/2) masses: C(n, k). The total is 2**n."""
    return [math.comb(n, k) for k in range(n + 1)]


def two_sided_sign_test(differences: Sequence[float]) -> dict[str, float | int | None]:
    """Exact two-sided sign test on paired differences.

    Ties (difference == 0) are dropped. The p-value is the sum of binomial
    probabilities at p=1/2 that are no larger than the probability of the
    observed number of positive signs (the small-p method), which is what
    ``scipy.stats.binomtest(..., alternative='two-sided')`` reports.
    """
    diffs = np.asarray(list(differences), dtype=float)
    if diffs.ndim != 1:
        raise ValueError("differences must be 1-d")
    if not np.isfinite(diffs).all():
        raise ValueError("differences must be finite")
    n_positive = int(np.sum(diffs > 0))
    n_negative = int(np.sum(diffs < 0))
    n_ties = int(diffs.size - n_positive - n_negative)
    n_nonzero = n_positive + n_negative
    if n_nonzero == 0:
        return {
            "n": int(diffs.size),
            "n_nonzero": 0,
            "n_positive": 0,
            "n_negative": 0,
            "n_ties": n_ties,
            "p_value": None,
        }
    masses = _binom_pmf_half(n_nonzero)
    observed = masses[n_positive]
    tail = sum(mass for mass in masses if mass <= observed)
    p_value = tail / float(2**n_nonzero)
    return {
        "n": int(diffs.size),
        "n_nonzero": n_nonzero,
        "n_positive": n_positive,
        "n_negative": n_negative,
        "n_ties": n_ties,
        "p_value": float(p_value),
    }


def quadratic_weighted_kappa(
    labels_a: Sequence[int],
    labels_b: Sequence[int],
    *,
    n_classes: int = 4,
) -> dict[str, float | int | None]:
    """Quadratic-weighted Cohen's kappa.

    Weight for classes i and j is ``(i - j)^2 / (n_classes - 1)^2``.
    Raw agreement is the exact-match rate. Kappa is None when the expected
    weighted disagreement is 0.
    """
    a = [int(x) for x in labels_a]
    b = [int(x) for x in labels_b]
    if len(a) != len(b):
        raise ValueError("label sequences differ in length")
    if n_classes < 2:
        raise ValueError("n_classes must be >= 2")
    if any(x < 0 or x >= n_classes for x in a + b):
        raise ValueError(f"labels must be in 0..{n_classes - 1}")
    n = len(a)
    if n == 0:
        return {"n": 0, "n_agree": 0, "agreement": None, "kappa": None}
    observed = [[0 for _ in range(n_classes)] for _ in range(n_classes)]
    for left, right in zip(a, b):
        observed[left][right] += 1
    hist_a = [sum(row) for row in observed]
    hist_b = [sum(observed[i][j] for i in range(n_classes)) for j in range(n_classes)]
    denom = float((n_classes - 1) ** 2)
    weighted_obs = 0.0
    weighted_exp = 0.0
    for i in range(n_classes):
        for j in range(n_classes):
            weight = ((i - j) ** 2) / denom
            weighted_obs += weight * observed[i][j]
            weighted_exp += weight * (hist_a[i] * hist_b[j] / n)
    agree = sum(observed[i][i] for i in range(n_classes))
    kappa = None if weighted_exp == 0.0 else 1.0 - (weighted_obs / weighted_exp)
    return {"n": n, "n_agree": agree, "agreement": agree / n, "kappa": kappa}


def cohens_kappa(labels_a: Sequence[int], labels_b: Sequence[int]) -> dict[str, float | int | None]:
    """Cohen's kappa for two binary label sequences, plus raw agreement."""
    a = [int(x) for x in labels_a]
    b = [int(x) for x in labels_b]
    if len(a) != len(b):
        raise ValueError("label sequences differ in length")
    if any(x not in (0, 1) for x in a + b):
        raise ValueError("labels must be 0 or 1")
    n = len(a)
    if n == 0:
        return {"n": 0, "agreement": None, "kappa": None}
    agree = sum(x == y for x, y in zip(a, b))
    p_o = agree / n
    p_a1 = sum(a) / n
    p_b1 = sum(b) / n
    p_e = p_a1 * p_b1 + (1.0 - p_a1) * (1.0 - p_b1)
    kappa = None if p_e == 1.0 else (p_o - p_e) / (1.0 - p_e)
    return {"n": n, "n_agree": agree, "agreement": p_o, "kappa": kappa}
