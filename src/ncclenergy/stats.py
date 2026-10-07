"""Descriptive bootstrap statistics over trial-block values.

The three completed blocks are resampled at the block level. Their sequential
execution does not guarantee independent errors, so intervals are descriptive.
"""

from __future__ import annotations

import numpy as np


def boot_means(x: np.ndarray, n_boot: int, rng: np.random.Generator) -> np.ndarray:
    """Bootstrap distribution of the mean of x (shape (n_boot,))."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.full(n_boot, np.nan)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    return x[idx].mean(axis=1)


def ci(dist: np.ndarray, level: float = 0.95) -> tuple[float, float]:
    d = dist[np.isfinite(dist)]
    if len(d) == 0:
        return float("nan"), float("nan")
    a = (1 - level) / 2
    return float(np.quantile(d, a)), float(np.quantile(d, 1 - a))


def mean_ci(x, n_boot: int, rng: np.random.Generator) -> tuple[float, float, float]:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return float("nan"), float("nan"), float("nan")
    lo, hi = ci(boot_means(x, n_boot, rng))
    return float(x.mean()), lo, hi


def prob_best(samples: dict[str, np.ndarray], n_boot: int, rng: np.random.Generator) -> dict[str, float]:
    """P(config has the lowest mean) under independent bootstrap of each config's trials."""
    keys = list(samples)
    mat = np.vstack([boot_means(samples[k], n_boot, rng) for k in keys])
    mat = np.where(np.isfinite(mat), mat, np.inf)
    winners = np.argmin(mat, axis=0)
    counts = np.bincount(winners, minlength=len(keys))
    return {k: float(c) / n_boot for k, c in zip(keys, counts)}


def ratio_ci(a, b, n_boot: int, rng: np.random.Generator) -> tuple[float, float, float]:
    """mean(a)/mean(b) - 1 with a bootstrap interval (independent resampling)."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    point = float(np.nanmean(a) / np.nanmean(b) - 1)
    dist = boot_means(a, n_boot, rng) / boot_means(b, n_boot, rng) - 1
    lo, hi = ci(dist)
    return point, lo, hi


def pareto_front(points: list[tuple[str, float, float]]) -> set[str]:
    """Labels of points not dominated in (x, y), both minimized."""
    front = set()
    for k, x, y in points:
        if not (np.isfinite(x) and np.isfinite(y)):
            continue
        dominated = any((x2 <= x and y2 <= y) and (x2 < x or y2 < y)
                        for k2, x2, y2 in points if k2 != k and np.isfinite(x2) and np.isfinite(y2))
        if not dominated:
            front.add(k)
    return front
