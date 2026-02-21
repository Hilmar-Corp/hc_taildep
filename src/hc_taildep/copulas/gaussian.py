from __future__ import annotations

import numpy as np
from scipy.stats import norm


def _clamp_rho(rho: float, clamp: float) -> float:
    r = float(rho)
    c = float(clamp)
    if not (0.0 < c < 1.0):
        raise ValueError("clamp must be in (0,1)")
    r = max(min(r, 1.0 - c), -1.0 + c)
    return r


def fit(u_train: np.ndarray, v_train: np.ndarray, *, rho_clamp: float = 1e-6) -> float:
    """
    Gaussian copula MLE is equivalent to corr of z=Phi^{-1}(u), w=Phi^{-1}(v).
    Deterministic + stable with clamping.
    """
    u = np.asarray(u_train, dtype=float)
    v = np.asarray(v_train, dtype=float)

    # Filter non-finite values (robust for toy tests)
    m = np.isfinite(u) & np.isfinite(v)
    u = u[m]
    v = v[m]
    if u.size < 2:
        return 0.0

    z = norm.ppf(u)
    w = norm.ppf(v)
    m2 = np.isfinite(z) & np.isfinite(w)
    z = z[m2]
    w = w[m2]
    if z.size < 2:
        return 0.0

    rho = float(np.corrcoef(z, w)[0, 1])
    if not np.isfinite(rho):
        rho = 0.0
    return _clamp_rho(rho, rho_clamp)


def logpdf(u: np.ndarray, v: np.ndarray, rho: float, *, rho_clamp: float = 1e-6) -> np.ndarray:
    """
    log c_G(u,v;rho) = -0.5*log(1-rho^2) + (rho*z*w - 0.5*rho^2*(z^2+w^2)) / (1-rho^2)
    where z=Phi^{-1}(u), w=Phi^{-1}(v).
    """
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)
    if u.shape != v.shape:
        raise ValueError("u and v must have same shape")

    rho = _clamp_rho(rho, rho_clamp)
    z = norm.ppf(u)
    w = norm.ppf(v)

    one_m_r2 = 1.0 - rho * rho
    # stable guard
    one_m_r2 = max(one_m_r2, 1e-15)

    term0 = -0.5 * np.log(one_m_r2)
    term1 = (rho * z * w - 0.5 * (rho * rho) * (z * z + w * w)) / one_m_r2
    out = term0 + term1

    # sanity: must be finite if inputs are finite + u in (0,1)
    out = np.where(np.isfinite(out), out, -np.inf)
    return out