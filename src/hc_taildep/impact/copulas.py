# src/hc_taildep/impact/copulas.py
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, Tuple

import numpy as np
from scipy.special import gammaln
from scipy.stats import kendalltau, norm, t as student_t


# =========================
# Helpers
# =========================

def _clip_u(u: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    u = np.asarray(u, dtype=float)
    return np.clip(u, eps, 1.0 - eps)


def kendall_tau(u: np.ndarray, v: np.ndarray) -> float:
    """Kendall tau on pseudo-observations (robust nan handling)."""
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)
    tau = kendalltau(u, v, nan_policy="omit").correlation
    if tau is None or not np.isfinite(tau):
        return 0.0
    return float(tau)


def rho_from_tau(tau: float) -> float:
    """Elliptical copulas: rho = sin(pi/2 * tau)."""
    return float(np.sin(np.pi * 0.5 * np.clip(tau, -0.999, 0.999)))


# =========================
# Parameter containers
# =========================

@dataclass(frozen=True)
class GaussParams:
    rho: float


@dataclass(frozen=True)
class TParams:
    rho: float
    nu: float


@dataclass(frozen=True)
class ClaytonParams:
    theta: float  # > 0


@dataclass(frozen=True)
class GumbelParams:
    theta: float  # >= 1


# =========================
# Fitters (train-only)
# =========================

def fit_gauss_from_u(u: np.ndarray, v: np.ndarray) -> GaussParams:
    tau = kendall_tau(u, v)
    rho = rho_from_tau(tau)
    rho = float(np.clip(rho, -0.999, 0.999))
    return GaussParams(rho=rho)


def _t_copula_logpdf(u: np.ndarray, v: np.ndarray, rho: float, nu: float) -> np.ndarray:
    """log c(u,v) for t-copula in 2D."""
    u = _clip_u(u)
    v = _clip_u(v)

    x = student_t.ppf(u, df=nu)
    y = student_t.ppf(v, df=nu)

    r = float(np.clip(rho, -0.999, 0.999))
    det = 1.0 - r * r
    det = max(det, 1e-12)

    quad = (x * x - 2.0 * r * x * y + y * y) / det

    # log multivariate t density (2D)
    log_f2 = (
        gammaln((nu + 2.0) / 2.0)
        - gammaln(nu / 2.0)
        - np.log(nu * np.pi)
        - 0.5 * np.log(det)
        - ((nu + 2.0) / 2.0) * np.log1p(quad / nu)
    )

    # subtract marginals to get copula density
    log_fx = student_t.logpdf(x, df=nu)
    log_fy = student_t.logpdf(y, df=nu)
    return log_f2 - (log_fx + log_fy)


def fit_t_from_u_grid(u: np.ndarray, v: np.ndarray, nu_grid=(4, 6, 8, 12, 20, 30)) -> TParams:
    """
    Lightweight t-copula fit:
    - rho from Kendall tau (elliptical link)
    - nu picked by grid maximizing copula log-likelihood
    """
    tau = kendall_tau(u, v)
    rho0 = float(np.clip(rho_from_tau(tau), -0.999, 0.999))

    best_ll = -np.inf
    best_nu = float(nu_grid[0])
    for nu in nu_grid:
        ll = float(np.sum(_t_copula_logpdf(u, v, rho0, float(nu))))
        if ll > best_ll:
            best_ll = ll
            best_nu = float(nu)

    return TParams(rho=rho0, nu=best_nu)


def fit_clayton_from_u(u: np.ndarray, v: np.ndarray) -> ClaytonParams | None:
    """
    Clayton via tau inversion: theta = 2*tau/(1-tau), valid for tau in (0,1).
    Returns None if tau <= 0 (unsupported without rotations).
    """
    tau = kendall_tau(u, v)
    if not (tau > 1e-6):
        return None
    theta = 2.0 * tau / (1.0 - tau)
    if not np.isfinite(theta) or theta <= 0.0:
        return None
    return ClaytonParams(theta=float(theta))


def fit_gumbel_from_u(u: np.ndarray, v: np.ndarray) -> GumbelParams | None:
    """
    Gumbel via tau inversion: theta = 1/(1-tau), valid for tau in [0,1).
    Returns None if tau <= 0 (unsupported without rotations).
    """
    tau = kendall_tau(u, v)
    if not (tau > 1e-6):
        return None
    theta = 1.0 / (1.0 - tau)
    theta = max(theta, 1.0)
    if not np.isfinite(theta) or theta < 1.0:
        return None
    return GumbelParams(theta=float(theta))


# =========================
# Samplers
# =========================

def sample_indep(n: int, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    u = rng.random(n)
    v = rng.random(n)
    return _clip_u(u), _clip_u(v)


def sample_gauss(params: GaussParams, n: int, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    r = float(np.clip(params.rho, -0.999, 0.999))
    z1 = rng.standard_normal(n)
    z2 = r * z1 + np.sqrt(max(1.0 - r * r, 0.0)) * rng.standard_normal(n)
    u1 = norm.cdf(z1)
    u2 = norm.cdf(z2)
    return _clip_u(u1), _clip_u(u2)


def sample_t(params: TParams, n: int, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    r = float(np.clip(params.rho, -0.999, 0.999))
    nu = float(params.nu)

    x1 = rng.standard_normal(n)
    x2 = r * x1 + np.sqrt(max(1.0 - r * r, 0.0)) * rng.standard_normal(n)
    s = rng.chisquare(df=nu, size=n)
    scale = np.sqrt(s / nu)

    t1 = x1 / scale
    t2 = x2 / scale
    u1 = student_t.cdf(t1, df=nu)
    u2 = student_t.cdf(t2, df=nu)
    return _clip_u(u1), _clip_u(u2)


def sample_clayton(params: ClaytonParams, n: int, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    """
    Clayton (lower-tail dependent) via frailty:
      V ~ Gamma(1/theta, 1), Ei ~ Exp(1),
      Ui = (1 + Ei / V)^(-1/theta)
    """
    theta = float(params.theta)
    v = rng.gamma(shape=1.0 / theta, scale=1.0, size=n)
    e1 = rng.exponential(scale=1.0, size=n)
    e2 = rng.exponential(scale=1.0, size=n)
    u1 = (1.0 + e1 / v) ** (-1.0 / theta)
    u2 = (1.0 + e2 / v) ** (-1.0 / theta)
    return _clip_u(u1), _clip_u(u2)


# =========================
# Positive stable for Gumbel frailty
# =========================

def _positive_stable_raw(alpha: float, n: int, rng: np.random.Generator) -> np.ndarray:
    """
    Kanter-type raw positive stable draw (alpha in (0,1)).
    This produces the correct shape but may have a multiplicative scale constant.
    We normalize that constant deterministically via LT calibration.
    """
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must be in (0,1)")

    U = rng.uniform(0.0, np.pi, size=n)
    W = rng.exponential(scale=1.0, size=n)

    Sa = np.sin(alpha * U)
    Su = np.sin(U)
    S1a = np.sin((1.0 - alpha) * U)

    # Raw Kanter form (shape right; scale constant fixed by calibration below)
    part1 = Sa / (Su ** (1.0 / alpha))
    part2 = (S1a / W) ** ((1.0 - alpha) / alpha)
    S = part1 * part2

    # numeric guard
    S = np.clip(S, 0.0, np.inf)
    return S


@lru_cache(maxsize=64)
def _stable_scale_for_unit_laplace(alpha: float) -> float:
    """
    Determine scale s such that S = S_raw / s approximately satisfies:
      E[exp(-t S)] = exp(-t^alpha)  (unit-scale LT)

    We estimate c from LT at t=1:
      E[exp(-S_raw)] = exp(-c)  => c = -log(E[exp(-S_raw)])
    If S_raw has LT exp(-c t^alpha), scaling by s = c^(1/alpha) normalizes to c=1.
    """
    # deterministic seed tied to alpha -> stable & auditable
    seed = int((abs(float(alpha)) * 1_000_000) % (2**31 - 1))
    rng = np.random.default_rng(seed)

    m = 20_000  # good tradeoff: stable enough, cheap (cached)
    Sraw = _positive_stable_raw(alpha, m, rng)

    lt1 = float(np.mean(np.exp(-Sraw)))
    lt1 = max(lt1, 1e-300)
    c = -float(np.log(lt1))

    if not np.isfinite(c) or c <= 0.0:
        return 1.0

    s = float(c ** (1.0 / alpha))
    if not np.isfinite(s) or s <= 0.0:
        s = 1.0
    return s


def _positive_stable(alpha: float, n: int, rng: np.random.Generator) -> np.ndarray:
    """
    Positive alpha-stable S with Laplace transform exp(-t^alpha), 0<alpha<1.
    Implemented as:
      S = S_raw / s(alpha)
    where s(alpha) is deterministically calibrated and cached.
    """
    Sraw = _positive_stable_raw(float(alpha), n, rng)
    s = _stable_scale_for_unit_laplace(float(alpha))
    return Sraw / s


# =========================
# Gumbel (upper) and survival Gumbel (lower)
# =========================

def sample_gumbel(params: GumbelParams, n: int, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    """
    Standard Gumbel copula (UPPER-tail dependence).
    Parameter: theta >= 1.0 ; alpha = 1/theta in (0,1].
    Frailty representation:
      V ~ positive-stable(alpha) with LT exp(-t^alpha)
      Ei ~ Exp(1)
      Ui = exp(-(Ei / V)^alpha)
    """
    theta = float(params.theta)
    if theta <= 1.0:
        return sample_indep(n, rng)

    alpha = 1.0 / theta
    V = _positive_stable(alpha, n, rng)
    E1 = rng.exponential(scale=1.0, size=n)
    E2 = rng.exponential(scale=1.0, size=n)

    u1 = np.exp(-((E1 / V) ** alpha))
    u2 = np.exp(-((E2 / V) ** alpha))
    return _clip_u(u1), _clip_u(u2)


def sample_gumbel_survival(params: GumbelParams, n: int, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    """
    Survival (rotated) Gumbel copula (LOWER-tail dependence).
    Constructed as rotation of the standard (upper-tail) Gumbel:
      (U,V)_surv = (1-U, 1-V) where (U,V) ~ Gumbel(theta)
    """
    u, v = sample_gumbel(params, n, rng)
    return _clip_u(1.0 - u), _clip_u(1.0 - v)


# =========================
# Tail dependence diagnostics (Monte Carlo)
# =========================

def tail_dependence_mc(u: np.ndarray, v: np.ndarray, q: float = 0.05) -> Dict[str, float]:
    """
    Finite-q estimator (diagnostic):
      lambda_L(q) ≈ P(U<q, V<q) / q
      lambda_U(q) ≈ P(U>1-q, V>1-q) / q
    Note: at finite q this is biased; use small q (e.g. 0.01) and treat as diagnostic.
    """
    u = _clip_u(u)
    v = _clip_u(v)
    q = float(q)
    q = min(max(q, 1e-6), 0.25)

    lam_l = float(((u < q) & (v < q)).mean() / q)
    lam_u = float(((u > 1.0 - q) & (v > 1.0 - q)).mean() / q)
    return {"lambda_L": lam_l, "lambda_U": lam_u}