# src/hc_taildep/impact/var_es_core.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import chi2

from hc_taildep.impact.empirical import EmpiricalQuantile, pseudo_obs_from_returns
from hc_taildep.impact import copulas


@dataclass(frozen=True)
class StressSpec:
    rv_window: int
    calm_q: float
    stress_q: float


def rolling_vol(x: np.ndarray, window: int) -> np.ndarray:
    # simple rolling std, aligned to current index (uses past window)
    out = np.full_like(x, np.nan, dtype=float)
    for i in range(window - 1, len(x)):
        w = x[i - window + 1 : i + 1]
        out[i] = float(np.std(w, ddof=1))
    return out


def bucketize(x_rv: np.ndarray, calm_thr: float, stress_thr: float) -> np.ndarray:
    b = np.full(x_rv.shape, "mid", dtype=object)
    b[x_rv <= calm_thr] = "calm"
    b[x_rv >= stress_thr] = "stress"
    return b


def compute_var_es(losses: np.ndarray, alpha: float) -> Tuple[float, float]:
    losses = np.asarray(losses, dtype=float)
    losses = losses[np.isfinite(losses)]
    if losses.size < 100:
        return (float("nan"), float("nan"))
    var = float(np.quantile(losses, alpha))
    tail = losses[losses >= var]
    es = float(tail.mean()) if tail.size else float("nan")
    return var, es


def kupiec_pof_test(hit_count: int, n: int, alpha: float) -> float:
    """
    Kupiec POF test p-value on exceedances:
      p0 = 1-alpha
      LR = -2 log( ( (1-p0)^(n-x) p0^x ) / ( (1-x/n)^(n-x) (x/n)^x ) )
    """
    if n <= 0:
        return float("nan")
    x = hit_count
    p0 = 1.0 - alpha
    phat = x / n if n > 0 else 0.0

    # handle edge cases
    if phat <= 0.0 or phat >= 1.0:
        return 0.0

    from math import log

    ll0 = (n - x) * log(1 - p0) + x * log(p0)
    ll1 = (n - x) * log(1 - phat) + x * log(phat)
    lr = -2.0 * (ll0 - ll1)
    pval = 1.0 - chi2.cdf(lr, df=1)
    return float(pval)


def _seed_for_timestamp(base_seed: int, ts: pd.Timestamp) -> int:
    # deterministic: base_seed XOR date hash
    h = int(ts.value % (2**31 - 1))
    return int((base_seed ^ h) % (2**31 - 1))


def run_var_es_for_pair(
    *,
    ts: pd.DatetimeIndex,
    r1: np.ndarray,
    r2: np.ndarray,
    refit_every: int,
    n_scenarios: int,
    alphas: List[float],
    stress: StressSpec,
    base_seed: int,
    models: List[str],
    nu_grid=(4, 6, 8, 12, 20, 30),
) -> pd.DataFrame:
    """
    Pure OOS block protocol.
    Fits copula params + marges on TRAIN of each block.
    Computes VaR/ES per date for each model.
    """
    assert len(ts) == len(r1) == len(r2)

    # portfolio for RV and realized loss
    r_port = 0.5 * r1 + 0.5 * r2
    loss_real = -r_port

    x_rv = rolling_vol(r_port, window=stress.rv_window)

    rows = []
    n = len(ts)
    # block starts at indices [refit_every, 2*refit_every, ...] BUT we need enough train
    starts = list(range(refit_every, n, refit_every))
    if starts and starts[-1] != n:
        starts.append(n)
    if not starts:
        starts = [n]

    prev = refit_every
    block_starts = list(range(refit_every, n, refit_every))
    if not block_starts:
        block_starts = [refit_every]

    # Iterate blocks
    for b_start in block_starts:
        train_end = b_start  # exclusive
        train_r1 = r1[:train_end]
        train_r2 = r2[:train_end]
        train_rp = r_port[:train_end]
        train_rv = x_rv[:train_end]

        # margins (train only)
        Q1 = EmpiricalQuantile(train_r1)
        Q2 = EmpiricalQuantile(train_r2)

        # stress thresholds (train only)
        train_rv_f = train_rv[np.isfinite(train_rv)]
        if train_rv_f.size < 50:
            calm_thr, stress_thr = np.nan, np.nan
        else:
            calm_thr = float(np.quantile(train_rv_f, stress.calm_q))
            stress_thr = float(np.quantile(train_rv_f, stress.stress_q))

        # pseudo obs u,v for copula fitting (train only)
        u = pseudo_obs_from_returns(train_r1)
        v = pseudo_obs_from_returns(train_r2)

        # fit params (static)
        gauss_static = copulas.fit_gauss_from_u(u, v)
        t_static = copulas.fit_t_from_u_grid(u, v, nu_grid=nu_grid)

        # threshold subsets based on TRAIN RV
        thr_gauss = {"calm": gauss_static, "stress": gauss_static, "mid": gauss_static}
        thr_t = {"calm": t_static, "stress": t_static, "mid": t_static}

        if np.isfinite(calm_thr) and np.isfinite(stress_thr):
            idx_calm = np.where(train_rv <= calm_thr)[0]
            idx_stress = np.where(train_rv >= stress_thr)[0]
            if idx_calm.size >= 100:
                u_c = pseudo_obs_from_returns(train_r1[idx_calm])
                v_c = pseudo_obs_from_returns(train_r2[idx_calm])
                thr_gauss["calm"] = copulas.fit_gauss_from_u(u_c, v_c)
                thr_t["calm"] = copulas.fit_t_from_u_grid(u_c, v_c, nu_grid=nu_grid)
            if idx_stress.size >= 100:
                u_s = pseudo_obs_from_returns(train_r1[idx_stress])
                v_s = pseudo_obs_from_returns(train_r2[idx_stress])
                thr_gauss["stress"] = copulas.fit_gauss_from_u(u_s, v_s)
                thr_t["stress"] = copulas.fit_t_from_u_grid(u_s, v_s, nu_grid=nu_grid)

        # loop dates in OOS block
        b_end = min(b_start + refit_every, n)
        for i in range(b_start, b_end):
            ts_i = ts[i]
            x_i = x_rv[i]
            bucket = "mid"
            if np.isfinite(calm_thr) and np.isfinite(stress_thr) and np.isfinite(x_i):
                if x_i <= calm_thr:
                    bucket = "calm"
                elif x_i >= stress_thr:
                    bucket = "stress"

            row = {
                "ts_utc": ts_i.isoformat(),
                "bucket": bucket,
                "x_rv": float(x_i) if np.isfinite(x_i) else np.nan,
                "r_port_real": float(r_port[i]),
                "loss_real": float(loss_real[i]),
                "calm_thr": calm_thr,
                "stress_thr": stress_thr,
            }

            for model in models:
                seed_i = _seed_for_timestamp(base_seed, ts_i)
                rng = np.random.default_rng(seed_i)

                if model == "indep":
                    u1, u2 = copulas.sample_indep(n_scenarios, rng)
                elif model == "static_gauss":
                    u1, u2 = copulas.sample_gauss(gauss_static, n_scenarios, rng)
                elif model == "static_t":
                    u1, u2 = copulas.sample_t(t_static, n_scenarios, rng)
                elif model == "thr_gauss":
                    u1, u2 = copulas.sample_gauss(thr_gauss[bucket], n_scenarios, rng)
                elif model == "thr_t":
                    u1, u2 = copulas.sample_t(thr_t[bucket], n_scenarios, rng)
                else:
                    raise ValueError(f"Unknown model: {model}")

                r1_sim = Q1(u1)
                r2_sim = Q2(u2)
                rport_sim = 0.5 * r1_sim + 0.5 * r2_sim
                loss_sim = -rport_sim

                for a in alphas:
                    var, es = compute_var_es(loss_sim, alpha=a)
                    row[f"VaR{int(a*100)}_{model}"] = var
                    row[f"ES{int(a*100)}_{model}"] = es
                    # exceedance on realized loss
                    row[f"exceed{int(a*100)}_{model}"] = bool(loss_real[i] > var) if np.isfinite(var) else False

            rows.append(row)

    return pd.DataFrame(rows)


def coverage_table(df: pd.DataFrame, models: List[str], alpha: float) -> pd.DataFrame:
    col_hit = f"exceed{int(alpha*100)}_{{model}}"
    rows = []
    for bucket in ["calm", "mid", "stress", "all"]:
        if bucket == "all":
            d = df
        else:
            d = df[df["bucket"] == bucket]
        n_obs = int(d.shape[0])
        for m in models:
            hits = int(d[col_hit.format(model=m)].sum()) if n_obs > 0 else 0
            hit_rate = hits / n_obs if n_obs else np.nan
            pval = kupiec_pof_test(hits, n_obs, alpha=alpha) if n_obs else np.nan
            rows.append({
                "bucket": bucket,
                "model": m,
                "alpha": alpha,
                "n_obs": n_obs,
                "hit_count": hits,
                "hit_rate": hit_rate,
                "target_rate": 1.0 - alpha,
                "kupiec_pvalue": pval,
            })
    return pd.DataFrame(rows)