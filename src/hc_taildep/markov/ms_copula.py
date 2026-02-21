from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.stats import norm

from hc_taildep.copulas import gaussian as gcop
from hc_taildep.copulas import student_t as tcop

from .filtering import forward_filter_log
from .forward_backward import forward_backward_log
from .utils import (
    clamp_rho,
    normalize_rows,
    stationary_dist,
    order_states_by_key,
    implied_durations,
)

# Deterministic floor to prevent all -inf emissions and NaN cascades.
LOGF_FLOOR = -1e12
U_EPS = 1e-12


@dataclass
class MSCopulaFit:
    K: int
    family: str  # "gauss" or "t"
    A: np.ndarray  # (K,K)
    pi0: np.ndarray  # (K,)
    theta: list[dict[str, float]]  # per state: {"rho":..., "nu":... (optional)}
    ll_train: float
    n_eff: np.ndarray
    fit_status: str
    ordering_key: str


def _clip_u(u: np.ndarray, eps: float = U_EPS) -> np.ndarray:
    u = np.asarray(u, dtype=float)
    return np.clip(u, eps, 1.0 - eps)


def _normal_scores(u: np.ndarray) -> np.ndarray:
    return norm.ppf(_clip_u(u))


def _weighted_corr(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
    w = np.asarray(w, dtype=float)
    w = np.maximum(w, 0.0)
    s = w.sum()
    if s <= 0:
        return 0.0
    w = w / s
    mx = float(np.sum(w * x))
    my = float(np.sum(w * y))
    xc = x - mx
    yc = y - my
    vx = float(np.sum(w * xc * xc))
    vy = float(np.sum(w * yc * yc))
    if vx <= 0 or vy <= 0:
        return 0.0
    cov = float(np.sum(w * xc * yc))
    return float(cov / np.sqrt(vx * vy))



def _sanitize_logf(lf: np.ndarray) -> np.ndarray:
    lf = np.asarray(lf, dtype=float)
    bad = ~np.isfinite(lf)
    if bad.any():
        lf = lf.copy()
        lf[bad] = LOGF_FLOOR
    return lf


def _resample_indices_from_weights(
    rng: np.random.Generator,
    w: np.ndarray,
    *,
    n: int,
) -> np.ndarray:
    """Deterministic (seeded) resampling with replacement from weights.

    Used to approximate a weighted M-step when the downstream fitter does not
    accept weights.
    """
    w = np.asarray(w, dtype=float)
    w = np.where(np.isfinite(w) & (w > 0.0), w, 0.0)
    s = float(w.sum())
    if s <= 0.0:
        # uniform fallback (still deterministic given rng)
        w = np.ones_like(w, dtype=float) / float(w.size)
    else:
        w = w / s
    idx = rng.choice(w.size, size=int(n), replace=True, p=w)
    return idx.astype(int)


def _logf_gauss(u: np.ndarray, v: np.ndarray, rhos: np.ndarray, rho_clamp: float) -> np.ndarray:
    T = u.size
    K = rhos.size
    out = np.empty((T, K), dtype=float)
    uu = _clip_u(u)
    vv = _clip_u(v)
    for k in range(K):
        out[:, k] = gcop.logpdf(uu, vv, float(rhos[k]), rho_clamp=rho_clamp)
    return _sanitize_logf(out)


def _logf_t(u: np.ndarray, v: np.ndarray, thetas: list[dict[str, float]]) -> np.ndarray:
    T = u.size
    K = len(thetas)
    out = np.empty((T, K), dtype=float)
    uu = _clip_u(u)
    vv = _clip_u(v)
    for k in range(K):
        rho = float(thetas[k]["rho"])
        nu = float(thetas[k]["nu"])
        nu = float(np.clip(nu, 3.0, 60.0))
        out[:, k] = tcop.logpdf(uu, vv, rho, nu)
    return _sanitize_logf(out)


def _init_from_threshold(
    u: np.ndarray,
    v: np.ndarray,
    x: np.ndarray,
    *,
    calm_q: float,
    stress_q: float,
    family: str,
    rho_clamp: float,
    nu_grid: list[float],
    nu_bounds: tuple[float, float],
) -> list[dict[str, float]]:
    xfin = x[np.isfinite(x)]
    if xfin.size == 0:
        # fallback: no x; just global fit
        if family == "gauss":
            rho = float(gcop.fit(_clip_u(u), _clip_u(v), rho_clamp=rho_clamp))
            return [{"rho": rho}, {"rho": rho}]
        p = tcop.fit(_clip_u(u), _clip_u(v), nu_grid=nu_grid, nu_bounds=nu_bounds, rho_clamp=rho_clamp)
        return [{"rho": float(p.rho), "nu": float(p.nu)}, {"rho": float(p.rho), "nu": float(p.nu)}]

    thr_c = float(np.quantile(xfin, calm_q))
    thr_s = float(np.quantile(xfin, stress_q))
    m_c = np.isfinite(x) & (x <= thr_c)
    m_s = np.isfinite(x) & (x >= thr_s)

    if m_c.sum() < 50 or m_s.sum() < 50:
        m_c = np.ones_like(x, dtype=bool)
        m_s = np.ones_like(x, dtype=bool)

    uu = _clip_u(u)
    vv = _clip_u(v)

    if family == "gauss":
        rho_c = float(gcop.fit(uu[m_c], vv[m_c], rho_clamp=rho_clamp))
        rho_s = float(gcop.fit(uu[m_s], vv[m_s], rho_clamp=rho_clamp))
        return [{"rho": rho_c}, {"rho": rho_s}]
    if family == "t":
        p_c = tcop.fit(uu[m_c], vv[m_c], nu_grid=nu_grid, nu_bounds=nu_bounds, rho_clamp=rho_clamp)
        p_s = tcop.fit(uu[m_s], vv[m_s], nu_grid=nu_grid, nu_bounds=nu_bounds, rho_clamp=rho_clamp)
        return [{"rho": float(p_c.rho), "nu": float(p_c.nu)}, {"rho": float(p_s.rho), "nu": float(p_s.nu)}]
    raise ValueError("family must be gauss or t")


def fit_ms_copula_train(
    u: np.ndarray,
    v: np.ndarray,
    x: np.ndarray,
    *,
    family: str,
    K: int,
    init_A: np.ndarray,
    rho_clamp: float,
    nu_grid: list[float],
    nu_bounds: tuple[float, float],
    calm_q: float,
    stress_q: float,
    max_iter: int = 50,
    tol: float = 1e-6,
    min_state_eff_n: int = 150,
    ordering_key: str = "rho",
    seed: int = 123,
) -> MSCopulaFit:
    rng = np.random.default_rng(seed)
    if K != 2:
        raise ValueError("J6 core uses K=2")

    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)
    x = np.asarray(x, dtype=float)

    m = np.isfinite(u) & np.isfinite(v)
    u = u[m]
    v = v[m]
    x = x[m]
    T = u.size

    A = normalize_rows(np.asarray(init_A, dtype=float))
    pi0 = stationary_dist(A)

    # Family-specific safeguards
    if family == "t":
        # t-copula MS is more fragile; allow smaller effective state sizes
        # and tighten nu-range to avoid pathological heavy-tail fits.
        min_eff = int(max(50, (2 * int(min_state_eff_n)) // 3))  # e.g. 150 -> 100

        # Clamp nu away from 2 (variance boundary) for numeric stability.
        nu_lo = float(max(3.0, float(nu_bounds[0])))
        nu_hi = float(min(60.0, float(nu_bounds[1])))
        if nu_hi <= nu_lo:
            nu_hi = nu_lo + 1.0
        nu_bounds_fam = (nu_lo, nu_hi)

        # Restrict grid to bounds; if empty, use a deterministic fallback grid.
        nu_grid_fam = [float(x) for x in nu_grid if (float(x) >= nu_lo and float(x) <= nu_hi)]
        if not nu_grid_fam:
            nu_grid_fam = [nu_lo, 4.0, 5.0, 7.0, 10.0, 15.0, 20.0, 30.0, nu_hi]
    else:
        min_eff = int(min_state_eff_n)
        nu_bounds_fam = nu_bounds
        nu_grid_fam = nu_grid

    if T < 300:
        theta0 = [{"rho": 0.0}, {"rho": 0.0}] if family == "gauss" else [{"rho": 0.0, "nu": 10.0}, {"rho": 0.0, "nu": 10.0}]
        return MSCopulaFit(
            K=K, family=family, A=A, pi0=np.ones(K)/K, theta=theta0,
            ll_train=float("nan"), n_eff=np.zeros(K),
            fit_status="too_short_train", ordering_key=ordering_key
        )

    theta = _init_from_threshold(
        u, v, x,
        calm_q=calm_q, stress_q=stress_q,
        family=family, rho_clamp=rho_clamp,
        nu_grid=nu_grid_fam, nu_bounds=nu_bounds_fam,
    )

    ll_prev = None
    fit_status = "ok"

    for it in range(max_iter):
        if family == "gauss":
            rhos = np.array([theta[0]["rho"], theta[1]["rho"]], dtype=float)
            logf = _logf_gauss(u, v, rhos, rho_clamp=rho_clamp)
        else:
            logf = _logf_t(u, v, theta)

        gamma, xi, ll = forward_backward_log(logf, A, pi0)

        if not np.isfinite(ll):
            fit_status = "nonfinite_ll"
            break

        n_eff = gamma.sum(axis=0)
        if np.any(n_eff < min_eff):
            fit_status = "degenerate_state"
            # deterministic softening to avoid collapse
            gamma = 0.95 * gamma + 0.05 * (np.ones_like(gamma) / K)
            n_eff = gamma.sum(axis=0)

        # transitions
        denom = np.clip(gamma[:-1].sum(axis=0).reshape(K, 1), 1e-12, None)
        A_new = xi.sum(axis=0) / denom
        A = normalize_rows(A_new)
        pi0 = stationary_dist(A)

        # theta update
        if family == "gauss":
            z1 = _normal_scores(u)
            z2 = _normal_scores(v)
            for k in range(K):
                rho_k = _weighted_corr(z1, z2, gamma[:, k])
                theta[k]["rho"] = clamp_rho(rho_k, rho_clamp=rho_clamp)
        else:
            # Soft-resampling M-step (deterministic via seed):
            # approximate weighted fitting by sampling indices according to gamma[:,k].
            uu = _clip_u(u)
            vv = _clip_u(v)
            for k in range(K):
                # target sample size ~ effective count, but enforce minimum
                n_k = int(max(min_eff, int(np.round(float(n_eff[k])))))
                idx = _resample_indices_from_weights(rng, gamma[:, k], n=n_k)
                if idx.size < min_eff:
                    continue
                p = tcop.fit(
                    uu[idx],
                    vv[idx],
                    nu_grid=nu_grid_fam,
                    nu_bounds=nu_bounds_fam,
                    rho_clamp=rho_clamp,
                )
                theta[k]["rho"] = float(p.rho)
                theta[k]["nu"] = float(np.clip(float(p.nu), nu_bounds_fam[0], nu_bounds_fam[1]))

        # canonical ordering
        params = {"K": K, "A": A, "pi0": pi0, "theta": theta}
        params, _ = order_states_by_key(params, key=ordering_key, increasing=True)
        A = params["A"]
        pi0 = params["pi0"]
        theta = params["theta"]

        # convergence
        if ll_prev is not None:
            rel = abs(ll - ll_prev) / max(1.0, abs(ll_prev))
            if rel < tol:
                ll_prev = ll
                break
        ll_prev = ll

    # final diagnostics pass
    if family == "gauss":
        rhos = np.array([theta[0]["rho"], theta[1]["rho"]], dtype=float)
        logf = _logf_gauss(u, v, rhos, rho_clamp=rho_clamp)
    else:
        logf = _logf_t(u, v, theta)

    gamma, _, ll = forward_backward_log(logf, A, pi0)
    n_eff = gamma.sum(axis=0)

    if not np.isfinite(ll):
        fit_status = "nonfinite_ll"

    if np.any(n_eff < min_eff):
        # enforce status even if loop ended "ok"
        fit_status = "degenerate_state"

    return MSCopulaFit(
        K=K, family=family, A=A, pi0=pi0, theta=theta,
        ll_train=float(ll), n_eff=n_eff, fit_status=fit_status,
        ordering_key=ordering_key
    )


def score_ms_oos_forward(
    u: np.ndarray,
    v: np.ndarray,
    *,
    family: str,
    A: np.ndarray,
    pi0: np.ndarray,
    theta: list[dict[str, float]],
    rho_clamp: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    OOS scoring, forward-only:
      returns (pi_pred, pi_filt, logp) where logp[t] = log p(y_t|y_{<=t-1}).
    """
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)
    m = np.isfinite(u) & np.isfinite(v)

    K = len(theta)
    logf = np.full((u.size, K), LOGF_FLOOR, dtype=float)

    if m.any():
        if family == "gauss":
            rhos = np.array([theta[k]["rho"] for k in range(K)], dtype=float)
            logf[m] = _logf_gauss(u[m], v[m], rhos, rho_clamp=rho_clamp)
        else:
            logf[m] = _logf_t(u[m], v[m], theta)

    pi_pred, pi_filt, logp = forward_filter_log(logf, A, pi0)

    # For invalid u/v, keep NaN outputs (audit-friendly)
    logp[~m] = np.nan
    pi_pred[~m] = np.nan
    pi_filt[~m] = np.nan
    return pi_pred, pi_filt, logp


def ms_diagnostics(A: np.ndarray) -> dict[str, Any]:
    A = np.asarray(A, dtype=float)
    return {
        "A": A.tolist(),
        "stationary": stationary_dist(A).tolist(),
        "durations": implied_durations(A).tolist(),
    }