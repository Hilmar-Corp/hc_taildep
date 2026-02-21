from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

import numpy as np
from scipy import stats


def _clamp_rho(rho: float, delta: float = 1e-6) -> float:
    return float(np.clip(rho, -1.0 + delta, 1.0 - delta))


@dataclass(frozen=True)
class TCopulaParams:
    rho: float
    nu: float

    def __iter__(self):
        # Enables: rho, nu = fit(...)
        yield self.rho
        yield self.nu

    def __array__(self, dtype=None):
        # Enables: np.allclose(params1, params2)
        arr = np.asarray([self.rho, self.nu], dtype=float)
        return arr.astype(dtype) if dtype is not None else arr


def logpdf(
    u: np.ndarray,
    v: np.ndarray,
    rho: float,
    nu: float,
    *,
    rho_clamp: float = 1e-6,
) -> np.ndarray:
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)

    rho = _clamp_rho(float(rho), float(rho_clamp))
    nu = float(nu)

    eps = 1e-12
    u = np.clip(u, eps, 1 - eps)
    v = np.clip(v, eps, 1 - eps)

    x = stats.t.ppf(u, df=nu)
    y = stats.t.ppf(v, df=nu)

    shape = np.array([[1.0, rho], [rho, 1.0]], dtype=float)
    mv = stats.multivariate_t(loc=np.zeros(2), shape=shape, df=nu)

    xy = np.column_stack([x, y])
    log_t2 = mv.logpdf(xy)
    log_t1x = stats.t.logpdf(x, df=nu)
    log_t1y = stats.t.logpdf(y, df=nu)

    out = log_t2 - log_t1x - log_t1y
    return np.where(np.isfinite(out), out, -1e9)


def fit(
    u: np.ndarray,
    v: np.ndarray,
    *,
    nu_grid: Iterable[float] = (3, 4, 5, 7, 10, 15, 20, 30, 50),
    rho_clamp: float = 1e-6,
    nu_bounds: Tuple[float, float] = (2.1, 60.0),
) -> TCopulaParams:
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)

    # Filter non-finite early (important for toy tests and stability)
    m = np.isfinite(u) & np.isfinite(v)
    u = u[m]
    v = v[m]
    lo, hi = nu_bounds
    if u.size < 2:
        # Stable fallback (quasi-independence)
        nu0 = float(next(iter(nu_grid), 10.0))
        nu0 = float(np.clip(nu0, lo, hi))
        return TCopulaParams(rho=0.0, nu=nu0)

    best_ll = None
    best_params = None

    for nu0 in nu_grid:
        nu = float(np.clip(float(nu0), lo, hi))
        x = stats.t.ppf(u, df=nu)
        y = stats.t.ppf(v, df=nu)

        if x.size < 2:
            continue
        rho = float(np.corrcoef(x, y)[0, 1])
        if not np.isfinite(rho):
            rho = 0.0
        rho = _clamp_rho(rho, rho_clamp)

        ll = float(np.sum(logpdf(u, v, rho, nu, rho_clamp=rho_clamp)))
        if (best_ll is None) or (ll > best_ll):
            best_ll = ll
            best_params = (rho, nu)

    if best_params is None:
        # Deterministic fallback
        nu0 = float(next(iter(nu_grid), 10.0))
        nu0 = float(np.clip(nu0, lo, hi))
        return TCopulaParams(rho=0.0, nu=nu0)

    rho, nu = best_params
    return TCopulaParams(rho=rho, nu=nu)