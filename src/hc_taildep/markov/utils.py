from __future__ import annotations

import numpy as np


def logsumexp(a: np.ndarray, axis=None) -> np.ndarray:
    """Stable log-sum-exp.

    Deterministic handling:
      - NaNs are treated as missing (mapped to -inf).
      - If any +inf is present along `axis`, result is +inf.
      - If all entries are -inf along `axis` (and no +inf), result is -inf.

    This avoids invalid operations like (+/-)inf subtraction that can create NaNs.
    """
    a = np.asarray(a, dtype=float)

    if axis is None:
        if np.isposinf(a).any():
            return np.array(np.inf).reshape(())
        a_f = np.where(np.isnan(a), -np.inf, a)
        m = np.max(a_f)
        if np.isneginf(m):
            return np.array(-np.inf).reshape(())
        out = m + np.log(np.sum(np.exp(a_f - m)))
        return np.array(out).reshape(())

    a_f = np.where(np.isnan(a), -np.inf, a)

    posinf = np.isposinf(a_f)
    has_posinf = np.any(posinf, axis=axis, keepdims=True)

    # Replace +inf with finite for intermediate ops; final overwritten to +inf.
    a_work = np.where(posinf, 0.0, a_f)

    m = np.max(a_work, axis=axis, keepdims=True)
    all_neginf = np.isneginf(m) & (~has_posinf)

    # Avoid (-inf) - (-inf)
    m_work = np.where(all_neginf, 0.0, m)

    # Safe diff: also avoid any NaN intermediate.
    diff = a_work - m_work
    diff = np.where(all_neginf, 0.0, diff)
    diff = np.where(has_posinf, 0.0, diff)

    exp_term = np.exp(diff)
    exp_term = np.where(all_neginf, 0.0, exp_term)
    exp_term = np.where(has_posinf, 0.0, exp_term)

    sumexp = np.sum(exp_term, axis=axis, keepdims=True)

    logsum = np.full_like(sumexp, -np.inf, dtype=float)
    np.log(sumexp, out=logsum, where=(sumexp > 0.0))

    out = m_work + logsum
    out = np.where(has_posinf, np.inf, out)
    out = np.where(all_neginf, -np.inf, out)

    return np.squeeze(out, axis=axis)


def stationary_dist(A: np.ndarray) -> np.ndarray:
    """
    Stationary distribution pi s.t. pi = pi A.
    """
    A = np.asarray(A, dtype=float)
    K = A.shape[0]
    w, v = np.linalg.eig(A.T)
    i = int(np.argmin(np.abs(w - 1.0)))
    pi = np.real(v[:, i])
    pi = np.maximum(pi, 0.0)
    if pi.sum() == 0:
        pi = np.ones(K) / K
    else:
        pi = pi / pi.sum()
    return pi


def implied_durations(A: np.ndarray) -> np.ndarray:
    """
    Expected duration in each state: 1/(1 - A_kk)
    """
    A = np.asarray(A, dtype=float)
    diag = np.clip(np.diag(A), 1e-9, 1 - 1e-9)
    return 1.0 / (1.0 - diag)


def order_states_by_key(params: dict, key: str = "rho", increasing: bool = True) -> tuple[dict, np.ndarray]:
    """
    Canonical ordering to prevent label switching.
    Returns (new_params, perm) where perm maps old->new indices.
    Supported keys:
      - "rho" for both gauss and t
      - "lambda" for t: monotone proxy based on rho and nu (more rho, lower nu => more tail)
    """
    K = int(params["K"])
    if K != 2:
        raise ValueError("Core ordering only implemented for K=2 in J6 core.")
    if key == "rho":
        rhos = np.array([params["theta"][k]["rho"] for k in range(K)], dtype=float)
        perm = np.argsort(rhos) if increasing else np.argsort(-rhos)
    elif key == "lambda":
        rhos = np.array([params["theta"][k]["rho"] for k in range(K)], dtype=float)
        nus = np.array([params["theta"][k].get("nu", 30.0) for k in range(K)], dtype=float)
        score = rhos - 0.10 * np.log(np.clip(nus - 2.0, 1e-9, None))
        perm = np.argsort(score) if increasing else np.argsort(-score)
    else:
        raise ValueError(f"Unknown ordering key={key}")

    perm = perm.astype(int)
    if np.array_equal(perm, np.array([0, 1])):
        return params, perm

    A = np.asarray(params["A"], dtype=float)
    A2 = A[perm][:, perm]
    pi0 = np.asarray(params["pi0"], dtype=float)[perm]

    theta2 = []
    for newk in range(K):
        oldk = int(perm[newk])
        theta2.append(dict(params["theta"][oldk]))

    out = dict(params)
    out["A"] = A2
    out["pi0"] = pi0
    out["theta"] = theta2
    return out, perm


def clamp_rho(rho: float, rho_clamp: float) -> float:
    r = float(rho)
    r = np.clip(r, -1.0 + rho_clamp, 1.0 - rho_clamp)
    return float(r)


def normalize_rows(A: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    A = np.asarray(A, dtype=float)
    A = np.maximum(A, eps)
    A = A / A.sum(axis=1, keepdims=True)
    return A