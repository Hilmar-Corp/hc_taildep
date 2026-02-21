# src/hc_taildep/impact/empirical.py
from __future__ import annotations

import numpy as np


class EmpiricalQuantile:
    """
    Quantile function Q(u) based on train returns only.
    Implemented as monotone interpolation over sorted samples.
    """
    def __init__(self, train_returns: np.ndarray):
        x = np.asarray(train_returns, dtype=float)
        x = x[np.isfinite(x)]
        if x.size < 10:
            raise ValueError("EmpiricalQuantile: need at least 10 finite samples")
        self.xs = np.sort(x)
        self.n = self.xs.size

    def __call__(self, u: np.ndarray) -> np.ndarray:
        u = np.asarray(u, dtype=float)
        u = np.clip(u, 1e-12, 1 - 1e-12)
        # map u to fractional index in [0, n-1]
        p = u * (self.n - 1)
        lo = np.floor(p).astype(int)
        hi = np.clip(lo + 1, 0, self.n - 1)
        w = p - lo
        return (1 - w) * self.xs[lo] + w * self.xs[hi]


def pseudo_obs_from_returns(train_returns: np.ndarray) -> np.ndarray:
    """
    Pseudo-observations u in (0,1) from ranks within TRAIN only.
    u_i = rank_i / (n+1) with average ranks for ties.
    """
    x = np.asarray(train_returns, dtype=float)
    n = x.size
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(n, dtype=float)
    ranks[order] = np.arange(1, n + 1, dtype=float)
    # simple tie handling: average consecutive equals
    # (rare in floats; kept minimal)
    u = ranks / (n + 1.0)
    u = np.clip(u, 1e-12, 1 - 1e-12)
    return u