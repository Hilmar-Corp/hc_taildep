from __future__ import annotations

import numpy as np
from .utils import logsumexp


def forward_backward_log(
    logf: np.ndarray,  # (T,K)
    A: np.ndarray,     # (K,K)
    pi0: np.ndarray,   # (K,)
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Forward-backward on TRAIN only (robust).
    Guarantees: no NaNs in gamma/xi even if some emissions are impossible.
    """
    logf = np.asarray(logf, dtype=float)
    A = np.asarray(A, dtype=float)
    pi0 = np.asarray(pi0, dtype=float)

    T, K = logf.shape

    A = np.clip(A, 1e-15, 1.0)
    A = A / A.sum(axis=1, keepdims=True)
    logA = np.log(A)

    logpi0 = np.log(np.clip(pi0, 1e-15, 1.0))
    z0 = float(logsumexp(logpi0))
    if not np.isfinite(z0):
        logpi0 = np.full(K, -np.log(K), dtype=float)
    else:
        logpi0 = logpi0 - z0

    log_alpha = np.full((T, K), -np.log(K), dtype=float)
    logp = np.full(T, -np.inf, dtype=float)

    # forward
    for t in range(T):
        if t == 0:
            la = logpi0 + logf[t]
        else:
            la = logsumexp(log_alpha[t - 1].reshape(K, 1) + logA, axis=0) + logf[t]

        z = float(logsumexp(la))
        logp[t] = z

        if not np.isfinite(z):
            # If all emissions impossible: keep alpha uniform, likelihood -inf.
            log_alpha[t] = np.full(K, -np.log(K), dtype=float)
        else:
            log_alpha[t] = la - z

    ll = float(np.sum(logp[np.isfinite(logp)])) if np.isfinite(logp).any() else float("-inf")

    # backward (normalized; safe when logp[t+1] = -inf)
    log_beta = np.zeros((T, K), dtype=float)
    log_beta[T - 1] = 0.0

    for t in range(T - 2, -1, -1):
        x = logA + (logf[t + 1] + log_beta[t + 1]).reshape(1, K)
        b = logsumexp(x, axis=1)

        if np.isfinite(logp[t + 1]):
            b = b - logp[t + 1]
        # else: skip normalization term (would be -(-inf) => +inf), keep b as-is

        # If b becomes non-finite, reset to 0 (neutral) deterministically.
        b = np.where(np.isfinite(b), b, 0.0)
        log_beta[t] = b

    # gamma
    lg = log_alpha + log_beta
    zrow = logsumexp(lg, axis=1).reshape(T, 1)

    # if row is non-finite (should not happen), make uniform
    bad = ~np.isfinite(zrow).reshape(T)
    if bad.any():
        lg[bad] = np.full((bad.sum(), K), -np.log(K), dtype=float)
        zrow = logsumexp(lg, axis=1).reshape(T, 1)

    lg = lg - zrow
    gamma = np.exp(lg)
    # final clamp
    gamma = np.where(np.isfinite(gamma), gamma, 1.0 / K)

    # xi
    xi = np.zeros((max(T - 1, 0), K, K), dtype=float)
    for t in range(T - 1):
        log_xi = (
            log_alpha[t].reshape(K, 1)
            + logA
            + logf[t + 1].reshape(1, K)
            + log_beta[t + 1].reshape(1, K)
        )
        z = float(logsumexp(log_xi))
        if not np.isfinite(z):
            xi[t] = np.ones((K, K), dtype=float) / (K * K)
        else:
            log_xi = log_xi - z
            xit = np.exp(log_xi)
            xit = np.where(np.isfinite(xit), xit, 0.0)
            s = float(xit.sum())
            if s <= 0:
                xi[t] = np.ones((K, K), dtype=float) / (K * K)
            else:
                xi[t] = xit / s

    return gamma, xi, ll