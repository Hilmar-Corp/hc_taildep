from __future__ import annotations

import numpy as np
import pandas as pd


def realized_vol(r: pd.Series, window: int) -> pd.Series:
    """
    RV_t = sqrt(sum_{i=t-w+1..t} r_i^2) on a rolling window ending at t.
    """
    if window <= 1:
        raise ValueError("window must be > 1")
    x = pd.to_numeric(r, errors="coerce").astype(float)
    return np.sqrt((x * x).rolling(window=window, min_periods=window).sum())


def stress_by_rv(rv: pd.Series, *, stress_q: float = 0.90, calm_q: float = 0.50) -> tuple[pd.Series, pd.Series, dict]:
    """
    Stress: RV >= quantile(stress_q) on finite RV in analysis window.
    Calm  : RV <= quantile(calm_q) on finite RV in analysis window.
    """
    if not (0.0 < calm_q < 1.0) or not (0.0 < stress_q < 1.0):
        raise ValueError("quantiles must be in (0,1)")
    if calm_q >= stress_q:
        raise ValueError("calm_q must be < stress_q")

    rv = pd.to_numeric(rv, errors="coerce").astype(float)
    finite = rv.replace([np.inf, -np.inf], np.nan).dropna()
    if finite.empty:
        is_stress = pd.Series(False, index=rv.index)
        is_calm = pd.Series(False, index=rv.index)
        info = {"stress_threshold": np.nan, "calm_threshold": np.nan, "n_finite": 0}
        return is_stress, is_calm, info

    thr_stress = float(finite.quantile(stress_q))
    thr_calm = float(finite.quantile(calm_q))

    is_stress = (rv >= thr_stress) & rv.notna()
    is_calm = (rv <= thr_calm) & rv.notna()

    info = {
        "stress_threshold": thr_stress,
        "calm_threshold": thr_calm,
        "n_finite": int(finite.shape[0]),
    }
    return is_stress, is_calm, info


def stress_by_joint_downside(
    r_btc: pd.Series,
    r_eth: pd.Series,
    *,
    alpha: float = 0.10,
) -> tuple[pd.Series, pd.Series, dict]:
    """
    Stress: (r_btc < q_alpha(r_btc)) & (r_eth < q_alpha(r_eth)) computed on analysis window.
    Calm  : complement of stress (on dates where both returns are finite).
    """
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must be in (0,1)")

    a = pd.to_numeric(r_btc, errors="coerce").astype(float)
    b = pd.to_numeric(r_eth, errors="coerce").astype(float)
    finite = a.notna() & b.notna() & np.isfinite(a) & np.isfinite(b)
    if finite.sum() == 0:
        is_stress = pd.Series(False, index=a.index)
        is_calm = pd.Series(False, index=a.index)
        info = {"thr_btc": np.nan, "thr_eth": np.nan, "n_finite": 0}
        return is_stress, is_calm, info

    thr_btc = float(a[finite].quantile(alpha))
    thr_eth = float(b[finite].quantile(alpha))

    is_stress = finite & (a < thr_btc) & (b < thr_eth)
    is_calm = finite & (~is_stress)

    info = {"thr_btc": thr_btc, "thr_eth": thr_eth, "n_finite": int(finite.sum())}
    return is_stress, is_calm, info