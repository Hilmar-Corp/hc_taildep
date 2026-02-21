from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from hc_taildep.copulas import gaussian as gcop
from hc_taildep.copulas import student_t as tcop


def robust_mad(x: np.ndarray, eps: float = 1e-12) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return 1.0
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    return float(max(mad, eps))


def logistic_weights(z: np.ndarray, a: float, b: float) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    x = a + b * z
    # stable sigmoid
    out = np.empty_like(x, dtype=float)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[~pos])
    out[~pos] = ex / (1.0 + ex)
    return np.clip(out, 1e-9, 1.0 - 1e-9)


def fisher_mix_rho(rho_calm: float, rho_stress: float, w: np.ndarray, rho_clamp: float) -> np.ndarray:
    """
    Mix rho in Fisher space: eta=atanh(rho), eta_t=(1-w)*eta_c + w*eta_s, rho=tanh(eta_t).
    """
    rc = float(np.clip(rho_calm, -1.0 + rho_clamp, 1.0 - rho_clamp))
    rs = float(np.clip(rho_stress, -1.0 + rho_clamp, 1.0 - rho_clamp))
    eta_c = np.arctanh(rc)
    eta_s = np.arctanh(rs)
    w = np.asarray(w, dtype=float)
    eta_t = (1.0 - w) * eta_c + w * eta_s
    rho_t = np.tanh(eta_t)
    return np.clip(rho_t, -1.0 + rho_clamp, 1.0 - rho_clamp)


def mix_nu(nu_calm: float, nu_stress: float, w: np.ndarray, nu_bounds: tuple[float, float]) -> np.ndarray:
    """
    Mix nu in kappa=log(nu-2) space: nu = 2+exp(kappa).
    """
    nu_min, nu_max = float(nu_bounds[0]), float(nu_bounds[1])
    nc = float(np.clip(nu_calm, nu_min, nu_max))
    ns = float(np.clip(nu_stress, nu_min, nu_max))
    kc = np.log(max(nc - 2.0, 1e-12))
    ks = np.log(max(ns - 2.0, 1e-12))
    w = np.asarray(w, dtype=float)
    kt = (1.0 - w) * kc + w * ks
    nu_t = 2.0 + np.exp(kt)
    return np.clip(nu_t, nu_min, nu_max)


@dataclass(frozen=True)
class ThetaT:
    rho: float
    nu: float


@dataclass(frozen=True)
class ThetaG:
    rho: float


def score_logit_t(
    u: np.ndarray,
    v: np.ndarray,
    z: np.ndarray,
    *,
    theta_calm: ThetaT,
    theta_stress: ThetaT,
    a: float,
    b: float,
    rho_clamp: float,
    nu_bounds: tuple[float, float],
) -> np.ndarray:
    w = logistic_weights(z, a, b)
    rho_t = fisher_mix_rho(theta_calm.rho, theta_stress.rho, w, rho_clamp=rho_clamp)
    nu_t = mix_nu(theta_calm.nu, theta_stress.nu, w, nu_bounds=nu_bounds)
    # vectorized logpdf by looping (tcop.logpdf expects arrays; but params vary by t)
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)
    out = np.empty_like(u, dtype=float)
    for i in range(u.size):
        out[i] = float(tcop.logpdf(np.array([u[i]]), np.array([v[i]]), float(rho_t[i]), float(nu_t[i]))[0])
    return out


def score_logit_gauss(
    u: np.ndarray,
    v: np.ndarray,
    z: np.ndarray,
    *,
    rho_calm: float,
    rho_stress: float,
    a: float,
    b: float,
    rho_clamp: float,
) -> np.ndarray:
    w = logistic_weights(z, a, b)
    rho_t = fisher_mix_rho(rho_calm, rho_stress, w, rho_clamp=rho_clamp)
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)
    out = np.empty_like(u, dtype=float)
    for i in range(u.size):
        out[i] = float(gcop.logpdf(np.array([u[i]]), np.array([v[i]]), float(rho_t[i]), rho_clamp=rho_clamp)[0])
    return out


def grid_fit_ab_for_tcopula(
    u_val: np.ndarray,
    v_val: np.ndarray,
    z_val: np.ndarray,
    *,
    theta_calm: ThetaT,
    theta_stress: ThetaT,
    a_grid: Iterable[float],
    b_grid: Iterable[float],
    rho_clamp: float,
    nu_bounds: tuple[float, float],
) -> tuple[float, float, float]:
    """
    Deterministic grid over (a,b), pick best sum logscore on validation set.
    Returns (a_best, b_best, best_sum).
    """
    best = (-np.inf, 0.0, 0.0)  # (sum, a, b)
    for a in a_grid:
        for b in b_grid:
            lp = score_logit_t(
                u_val,
                v_val,
                z_val,
                theta_calm=theta_calm,
                theta_stress=theta_stress,
                a=float(a),
                b=float(b),
                rho_clamp=rho_clamp,
                nu_bounds=nu_bounds,
            )
            s = float(np.sum(lp[np.isfinite(lp)]))
            if s > best[0]:
                best = (s, float(a), float(b))
    return best[1], best[2], best[0]
