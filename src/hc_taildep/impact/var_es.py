# src/hc_taildep/impact/var_es.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional
from functools import lru_cache

import numpy as np
from scipy.stats import norm, t as student_t


CopulaFamily = Literal["indep", "gauss", "t", "clayton", "gumbel"]



def _clip_u(u: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return np.clip(u, eps, 1.0 - eps)


def _positive_stable_raw(alpha: float, n: int, rng: np.random.Generator) -> np.ndarray:
    """Kanter-type raw positive alpha-stable draw (alpha in (0,1)).

    This produces the correct *shape* but may include a multiplicative scale constant.
    We normalize that constant deterministically via Laplace-transform calibration.

    References (standard): Marshall–Olkin / Kanter construction for positive stable.
    """
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must be in (0,1)")

    U = rng.uniform(0.0, np.pi, size=n)
    W = rng.exponential(scale=1.0, size=n)

    Sa = np.sin(alpha * U)
    Su = np.sin(U)
    S1a = np.sin((1.0 - alpha) * U)

    # Raw Kanter form (scale fixed by calibration below)
    part1 = Sa / (Su ** (1.0 / alpha))
    part2 = (S1a / W) ** ((1.0 - alpha) / alpha)
    S = part1 * part2

    # numerical guard
    return np.clip(S, 0.0, np.inf)


@lru_cache(maxsize=64)
def _stable_scale_for_unit_laplace(alpha: float) -> float:
    """Deterministically estimate scale s so that S = S_raw / s has LT exp(-t^alpha).

    If S_raw has Laplace transform approximately exp(-c t^alpha), then scaling by
    s = c^(1/alpha) makes it exp(-(t)^alpha).

    We estimate c from t=1: E[exp(-S_raw)] = exp(-c) => c = -log(mean(exp(-S_raw))).
    """
    seed = int((abs(float(alpha)) * 1_000_000) % (2**31 - 1))
    rng = np.random.default_rng(seed)

    m = 20_000  # cached; keep modest for speed
    Sraw = _positive_stable_raw(float(alpha), m, rng)

    lt1 = float(np.mean(np.exp(-Sraw)))
    lt1 = max(lt1, 1e-300)
    c = -float(np.log(lt1))

    if not np.isfinite(c) or c <= 0.0:
        return 1.0

    s = float(c ** (1.0 / float(alpha)))
    if not np.isfinite(s) or s <= 0.0:
        s = 1.0
    return s


def _positive_stable(alpha: float, n: int, rng: np.random.Generator) -> np.ndarray:
    """Positive alpha-stable S with Laplace transform exp(-t^alpha), 0<alpha<1."""
    Sraw = _positive_stable_raw(float(alpha), int(n), rng)
    s = _stable_scale_for_unit_laplace(float(alpha))
    return Sraw / s

def _sample_clayton(theta: float, n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    # frailty representation
    theta = float(theta)
    v = rng.gamma(shape=1.0 / theta, scale=1.0, size=n)
    e1 = rng.exponential(scale=1.0, size=n)
    e2 = rng.exponential(scale=1.0, size=n)
    u1 = (1.0 + e1 / v) ** (-1.0 / theta)
    u2 = (1.0 + e2 / v) ** (-1.0 / theta)
    return _clip_u(u1), _clip_u(u2)

def _sample_gumbel(theta: float, n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    # Standard Gumbel copula (UPPER-tail dependence).
    # Parameterization: theta >= 1, alpha = 1/theta in (0,1].
    # Frailty representation:
    #   V ~ positive-stable(alpha) with LT exp(-t^alpha)
    #   Ei ~ Exp(1)
    #   Ui = exp(-(Ei / V)^alpha)
    theta = float(theta)
    if theta <= 1.0 + 1e-12:
        u = rng.random(n)
        v = rng.random(n)
        return _clip_u(u), _clip_u(v)

    alpha = 1.0 / theta
    V = _positive_stable(alpha=alpha, n=n, rng=rng)
    E1 = rng.exponential(scale=1.0, size=n)
    E2 = rng.exponential(scale=1.0, size=n)

    u1 = np.exp(-((E1 / V) ** alpha))
    u2 = np.exp(-((E2 / V) ** alpha))
    return _clip_u(u1), _clip_u(u2)

def _clamp_rho(rho: float, rho_clamp: float = 1e-6) -> float:
    rho = float(rho)
    lo = -1.0 + float(rho_clamp)
    hi = 1.0 - float(rho_clamp)
    return float(np.clip(rho, lo, hi))


@dataclass(frozen=True)
class EmpiricalQuantile:
    """Fast empirical quantile function Q(u) built from train returns only.

    Implementation:
      - precompute quantiles on a fixed probability grid
      - interpolate (monotone) to evaluate Q(u) quickly for large N
    """

    p_grid: np.ndarray  # shape (M,)
    q_grid: np.ndarray  # shape (M,)

    def __call__(self, u: np.ndarray) -> np.ndarray:
        u = np.asarray(u, dtype=float)
        u = np.clip(u, 0.0, 1.0)
        return np.interp(u, self.p_grid, self.q_grid)


def build_empirical_quantile(
    train_returns: np.ndarray,
    *,
    grid_size: int = 2001,
    clip_eps: float = 1e-6,
) -> EmpiricalQuantile:
    x = np.asarray(train_returns, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 50:
        raise ValueError(f"Not enough finite train returns for empirical quantile: n={x.size}")
    # stable probability grid with small clipping away from 0/1
    p = np.linspace(0.0, 1.0, int(grid_size))
    p = np.clip(p, float(clip_eps), 1.0 - float(clip_eps))
    # np.quantile is vectorized and deterministic
    q = np.quantile(x, p, method="linear")
    return EmpiricalQuantile(p_grid=p.astype(float), q_grid=np.asarray(q, dtype=float))


def compute_var_es(losses: np.ndarray, alpha: float) -> tuple[float, float]:
    """VaR/ES on losses (L = -return). VaR_alpha is quantile alpha of L."""
    L = np.asarray(losses, dtype=float)
    L = L[np.isfinite(L)]
    if L.size == 0:
        return (float("nan"), float("nan"))
    a = float(alpha)
    if not (0.0 < a < 1.0):
        raise ValueError(f"alpha must be in (0,1), got {alpha}")
    var = float(np.quantile(L, a, method="linear"))
    tail = L[L >= var]
    es = float(np.mean(tail)) if tail.size > 0 else var
    return var, es


def sample_copula(
    family: CopulaFamily,
    theta: dict,
    n: int,
    rng: np.random.Generator,
    *,
    rho_clamp: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample (u,v) ~ copula(u,v; theta) with u,v in (0,1)."""
    n = int(n)
    if n <= 0:
        raise ValueError("n must be positive")
    if family == "indep":
        u = rng.random(n)
        v = rng.random(n)
        return _clip_u(u), _clip_u(v)
    if family == "clayton":
        th = float(theta["theta"])
        return _sample_clayton(th, n, rng)

    if family == "gumbel":
        th = float(theta["theta"])
        return _sample_gumbel(th, n, rng)
    if family == "gauss":
        rho = _clamp_rho(theta["rho"], rho_clamp=rho_clamp)
        # (Z1,Z2) ~ N(0, Sigma)
        z1 = rng.standard_normal(n)
        eps = rng.standard_normal(n)
        z2 = rho * z1 + np.sqrt(max(0.0, 1.0 - rho * rho)) * eps
        u = norm.cdf(z1)
        v = norm.cdf(z2)
        return _clip_u(u), _clip_u(v)

    if family == "t":
        rho = _clamp_rho(theta["rho"], rho_clamp=rho_clamp)
        nu = float(theta["nu"])
        if not np.isfinite(nu) or nu <= 2.0:
            raise ValueError(f"nu must be finite and > 2 for stable tails, got {nu}")
        # multivariate t via normal + chi2 scaling:
        # X ~ N(0,Sigma), s ~ chi2(nu), T = X / sqrt(s/nu)
        x1 = rng.standard_normal(n)
        eps = rng.standard_normal(n)
        x2 = rho * x1 + np.sqrt(max(0.0, 1.0 - rho * rho)) * eps
        s = rng.chisquare(df=nu, size=n)
        scale = np.sqrt(s / nu)
        t1 = x1 / scale
        t2 = x2 / scale
        u = student_t.cdf(t1, df=nu)
        v = student_t.cdf(t2, df=nu)
        return _clip_u(u), _clip_u(v)

    raise ValueError(f"Unknown family={family}")


def sample_mixture(
    pi: np.ndarray,
    thetas: list[dict],
    family: CopulaFamily,
    n: int,
    rng: np.random.Generator,
    *,
    rho_clamp: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample from mixture Σ_k pi[k] * Copula(theta_k)."""
    pi = np.asarray(pi, dtype=float)
    if pi.ndim != 1:
        raise ValueError("pi must be 1D")
    K = pi.size
    if K != len(thetas):
        raise ValueError(f"pi size {K} != len(thetas) {len(thetas)}")
    if np.any(~np.isfinite(pi)) or pi.sum() <= 0:
        raise ValueError(f"Invalid pi: {pi}")
    pi = pi / pi.sum()

    n = int(n)
    z = rng.choice(K, size=n, p=pi)
    u = np.empty(n, dtype=float)
    v = np.empty(n, dtype=float)
    for k in range(K):
        idx = np.where(z == k)[0]
        if idx.size == 0:
            continue
        uk, vk = sample_copula(family, thetas[k], idx.size, rng, rho_clamp=rho_clamp)
        u[idx] = uk
        v[idx] = vk
    return u, v


def sanity_check_var_es(var95: float, es95: float, var99: float, es99: float) -> None:
    vals = [var95, es95, var99, es99]
    if any(not np.isfinite(x) for x in vals):
        raise ValueError(f"VaR/ES contains non-finite values: {vals}")
    if var99 + 1e-12 < var95:
        raise ValueError(f"Monotonicity violated: VaR99={var99} < VaR95={var95}")
    if es95 + 1e-12 < var95 or es99 + 1e-12 < var99:
        raise ValueError(f"ES should be >= VaR. Got (VaR95,ES95,VaR99,ES99)=({var95},{es95},{var99},{es99})")