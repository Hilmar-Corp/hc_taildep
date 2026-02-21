from __future__ import annotations

import numpy as np


def logpdf(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """
    Independence copula: c(u,v)=1 -> log c = 0.
    """
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)
    if u.shape != v.shape:
        raise ValueError("u and v must have same shape")
    return np.zeros_like(u, dtype=float)