from __future__ import annotations

from dataclasses import asdict
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import t as student_t_cdf

from hc_taildep.copulas import student_t as tcop


def t_copula_tail_lambda(rho: float, nu: float, *, rho_clamp: float = 1e-6) -> float:
    """
    For t-copula with params (rho, nu), tail dependence is symmetric:
      lambda_L = lambda_U = 2 * T_{nu+1}(- sqrt((nu+1)*(1-rho)/(1+rho)) )

    Returns lambda in [0,1].
    """
    rho = float(np.clip(float(rho), -1.0 + rho_clamp, 1.0 - rho_clamp))
    nu = float(nu)
    if nu <= 2.0:
        # we enforce nu>=2.1 elsewhere; guard here to avoid weirdness
        nu = 2.1

    arg = -np.sqrt((nu + 1.0) * (1.0 - rho) / (1.0 + rho))
    lam = 2.0 * float(student_t_cdf.cdf(arg, df=nu + 1.0))
    return float(np.clip(lam, 0.0, 1.0))


def empirical_taildep(u: np.ndarray, v: np.ndarray, qs: Iterable[float]) -> dict:
    """
    Empirical tail dependence diagnostics computed *within the provided sample*.

    Important: when we condition on a regime (calm/stress), U and V are generally
    NOT Uniform(0,1) anymore in that subset. Therefore, using absolute thresholds
    (e.g. U<=0.05) is not directly comparable across regimes.

    We instead use *conditional-quantile* thresholds inside the subset:
      qU = quantile(U, q), qV = quantile(V, q)

    Lower tail:
      lambda_L^subset(q) = P(U<=qU, V<=qV) / q

    Upper tail:
      qUu = quantile(U, 1-q), qVu = quantile(V, 1-q)
      lambda_U^subset(q) = P(U>=qUu, V>=qVu) / q

    This yields a regime-comparable diagnostic (still noisy for small n).
    """
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)
    m = np.isfinite(u) & np.isfinite(v)
    u = u[m]
    v = v[m]

    out: dict[str, float] = {}
    if u.size == 0:
        for q in qs:
            keyL = f"lambdaL_emp_q{str(q).replace('.','p')}"
            keyU = f"lambdaU_emp_q{str(q).replace('.','p')}"
            out[keyL] = np.nan
            out[keyU] = np.nan
        return out

    for q in qs:
        q = float(q)
        keyL = f"lambdaL_emp_q{str(q).replace('.','p')}"
        keyU = f"lambdaU_emp_q{str(q).replace('.','p')}"
        if not (0.0 < q < 0.5):
            out[keyL] = np.nan
            out[keyU] = np.nan
            continue

        # Conditional-quantile thresholds within the subset
        qU = float(np.quantile(u, q))
        qV = float(np.quantile(v, q))
        qUu = float(np.quantile(u, 1.0 - q))
        qVu = float(np.quantile(v, 1.0 - q))

        pL = float(np.mean((u <= qU) & (v <= qV)))
        pU = float(np.mean((u >= qUu) & (v >= qVu)))

        out[keyL] = pL / q
        out[keyU] = pU / q

    return out


def fit_tcopula_and_lambda(
    u: np.ndarray,
    v: np.ndarray,
    *,
    nu_grid: Iterable[float],
    nu_bounds: tuple[float, float],
    rho_clamp: float,
) -> dict:
    """
    Fit t-copula params via existing tcop.fit (grid over nu, rho from corr on t-ppf),
    then compute closed-form lambda.
    """
    params = tcop.fit(u, v, nu_grid=nu_grid, nu_bounds=nu_bounds, rho_clamp=rho_clamp)
    rho = float(params.rho)
    nu = float(params.nu)
    lam = t_copula_tail_lambda(rho, nu, rho_clamp=rho_clamp)
    return {"rho_hat": rho, "nu_hat": nu, "lambda_hat": lam, "fit_status": "ok", "params": asdict(params)}


def summarize_regime(
    df: pd.DataFrame,
    mask: pd.Series,
    *,
    u_col: str,
    v_col: str,
    nu_grid: Iterable[float],
    nu_bounds: tuple[float, float],
    rho_clamp: float,
    empirical_qs: Iterable[float],
    min_n: int = 200,
) -> dict:
    """
    Summarize tail dependence for a single regime subset.
    """
    m = mask.reindex(df.index).fillna(False).astype(bool)
    sub = df.loc[m, [u_col, v_col]]
    n = int(sub.shape[0])

    out = {"n_obs": n}
    if n < 2:
        out.update({"rho_hat": np.nan, "nu_hat": np.nan, "lambda_hat": np.nan, "fit_status": "fail_too_few"})
        out.update(empirical_taildep(np.array([]), np.array([]), empirical_qs))
        return out

    u = sub[u_col].to_numpy(dtype=float)
    v = sub[v_col].to_numpy(dtype=float)

    # Fit
    fit = fit_tcopula_and_lambda(u, v, nu_grid=nu_grid, nu_bounds=nu_bounds, rho_clamp=rho_clamp)
    out.update({k: fit[k] for k in ["rho_hat", "nu_hat", "lambda_hat", "fit_status"]})

    # Mark low power
    if n < min_n:
        out["fit_status"] = "low_power"

    # Empirical diagnostics
    out.update(empirical_taildep(u, v, empirical_qs))
    return out