from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class BootstrapResult:
    delta_lambda_hat: float
    ci95: tuple[float, float]
    pvalue_two_sided: float
    n_calm: int
    n_stress: int
    B: int
    block_len: int
    seed: int


def _block_bootstrap_indices(n: int, block_len: int, rng: np.random.Generator) -> np.ndarray:
    """
    Sample indices of length n by concatenating random contiguous blocks of length block_len.
    """
    if n <= 0:
        return np.array([], dtype=int)
    L = int(block_len)
    if L <= 1:
        return rng.integers(0, n, size=n, dtype=int)

    if L > n:
        # fallback iid bootstrap if block too large
        return rng.integers(0, n, size=n, dtype=int)

    out = np.empty(n, dtype=int)
    k = 0
    while k < n:
        start = int(rng.integers(0, n - L + 1))
        block = np.arange(start, start + L, dtype=int)
        take = min(L, n - k)
        out[k : k + take] = block[:take]
        k += take
    return out


def bootstrap_delta_lambda(
    u: np.ndarray,
    v: np.ndarray,
    mask_calm: np.ndarray,
    mask_stress: np.ndarray,
    *,
    fit_lambda_fn: Callable[[np.ndarray, np.ndarray], float],
    B: int = 1000,
    block_len: int = 10,
    seed: int = 123,
) -> tuple[BootstrapResult, np.ndarray]:
    """
    Block bootstrap within each regime (calm and stress) separately.

    fit_lambda_fn(u_sub, v_sub) -> lambda_hat (float)
    Returns (result, delta_samples)
    """
    rng = np.random.default_rng(int(seed))
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)
    mask_calm = np.asarray(mask_calm, dtype=bool)
    mask_stress = np.asarray(mask_stress, dtype=bool)

    u_c = u[mask_calm]
    v_c = v[mask_calm]
    u_s = u[mask_stress]
    v_s = v[mask_stress]
    n_c = int(u_c.size)
    n_s = int(u_s.size)

    if n_c < 2 or n_s < 2 or B <= 0:
        res = BootstrapResult(
            delta_lambda_hat=float("nan"),
            ci95=(float("nan"), float("nan")),
            pvalue_two_sided=float("nan"),
            n_calm=n_c,
            n_stress=n_s,
            B=int(B),
            block_len=int(block_len),
            seed=int(seed),
        )
        return res, np.array([], dtype=float)

    lam_c_hat = float(fit_lambda_fn(u_c, v_c))
    lam_s_hat = float(fit_lambda_fn(u_s, v_s))
    delta_hat = lam_s_hat - lam_c_hat

    deltas = np.empty(int(B), dtype=float)
    for b in range(int(B)):
        idx_c = _block_bootstrap_indices(n_c, block_len, rng)
        idx_s = _block_bootstrap_indices(n_s, block_len, rng)
        lam_c = float(fit_lambda_fn(u_c[idx_c], v_c[idx_c]))
        lam_s = float(fit_lambda_fn(u_s[idx_s], v_s[idx_s]))
        deltas[b] = lam_s - lam_c

    lo = float(np.quantile(deltas, 0.025))
    hi = float(np.quantile(deltas, 0.975))
    p_ge0 = float(np.mean(deltas >= 0.0))
    p_le0 = float(np.mean(deltas <= 0.0))
    p_two = float(2.0 * min(p_ge0, p_le0))
    p_two = float(np.clip(p_two, 0.0, 1.0))

    res = BootstrapResult(
        delta_lambda_hat=float(delta_hat),
        ci95=(lo, hi),
        pvalue_two_sided=p_two,
        n_calm=n_c,
        n_stress=n_s,
        B=int(B),
        block_len=int(block_len),
        seed=int(seed),
    )
    return res, deltas