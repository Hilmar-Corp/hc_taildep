from __future__ import annotations

import numpy as np
from .utils import logsumexp


def forward_filter_log(
    logf: np.ndarray,  # (T,K) log emission density
    A: np.ndarray,     # (K,K)
    pi0: np.ndarray,   # (K,)
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Forward recursion in log-space.
    Returns:
      pi_pred[t] = p(z_t | y_{<=t-1})  (pre-update)
      pi_filt[t] = p(z_t | y_{<=t})    (post-update)
      logp[t]    = log p(y_t | y_{<=t-1})
    """
    logf = np.asarray(logf, dtype=float)
    A = np.asarray(A, dtype=float)
    pi0 = np.asarray(pi0, dtype=float)

    T, K = logf.shape
    if A.shape != (K, K):
        raise ValueError("A shape mismatch")
    if pi0.shape != (K,):
        raise ValueError("pi0 shape mismatch")

    A = np.clip(A, 1e-15, 1.0)
    A = A / A.sum(axis=1, keepdims=True)
    logA = np.log(A)

    pi_pred = np.full((T, K), np.nan, dtype=float)
    pi_filt = np.full((T, K), np.nan, dtype=float)
    logp = np.full(T, np.nan, dtype=float)

    # init lp = log(pi0) normalized
    lp0 = np.log(np.clip(pi0, 1e-15, 1.0))
    z0 = float(logsumexp(lp0))
    if not np.isfinite(z0):
        lp = np.full(K, -np.log(K), dtype=float)
    else:
        lp = lp0 - z0

    for t in range(T):
        if t == 0:
            lp_pred = lp
        else:
            lp_pred = logsumexp(lp.reshape(K, 1) + logA, axis=0)

        zpred = float(logsumexp(lp_pred))
        if not np.isfinite(zpred):
            lp_pred_n = np.full(K, -np.log(K), dtype=float)
            pi_pred[t] = np.ones(K, dtype=float) / K
        else:
            lp_pred_n = lp_pred - zpred
            pi_pred[t] = np.exp(lp_pred_n)

        # update
        la = lp_pred_n + logf[t]
        z = float(logsumexp(la))
        logp[t] = z

        if not np.isfinite(z):
            # keep prior (deterministic)
            lp = lp_pred_n
            pi_filt[t] = pi_pred[t]
        else:
            lp = la - z
            pi_filt[t] = np.exp(lp)

    return pi_pred, pi_filt, logp