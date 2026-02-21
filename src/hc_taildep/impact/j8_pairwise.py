# src/hc_taildep/impact/j8_pairwise.py
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import kendalltau

from hc_taildep.copulas import gaussian as gcop
from hc_taildep.copulas import student_t as tcop
from hc_taildep.markov.ms_copula import fit_ms_copula_train, score_ms_oos_forward
from hc_taildep.impact.var_es import build_empirical_quantile, compute_var_es, sample_copula, sample_mixture


# --------------------------
# utilities (deterministic)
# --------------------------

def seed_for_key(global_seed: int, key: str) -> int:
    h = hashlib.sha256((str(global_seed) + "|" + str(key)).encode("utf-8")).hexdigest()
    return int(h[:8], 16)


def sha12_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()[:12]


def sha12_file(path) -> str:
    import pathlib
    p = pathlib.Path(path)
    return hashlib.sha256(p.read_bytes()).hexdigest()[:12]


def ensure_dir(p) -> None:
    import pathlib
    pathlib.Path(p).mkdir(parents=True, exist_ok=True)


def detect_date_col(df: pd.DataFrame) -> str:
    for c in ["date", "Date", "datetime", "timestamp", "time", "ds", "ts_utc"]:
        if c in df.columns:
            return c
    for c in ["Unnamed: 0", "index"]:
        if c in df.columns:
            x = pd.to_datetime(df[c], errors="coerce", utc=True)
            if float(x.notna().mean()) >= 0.90:
                return c
    raise ValueError(f"Cannot find date column in columns={list(df.columns)[:20]}")


def normalize_date(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    out = df.copy()
    # IMPORTANT: keep full timestamp for intraday; keep ISO string with UTC
    t = pd.to_datetime(out[date_col], errors="coerce", utc=True)
    if t.isna().any():
        raise ValueError("Some dates could not be parsed.")
    out["date"] = t.dt.strftime("%Y-%m-%d %H:%M:%S+00:00")
    if date_col != "date":
        out = out.drop(columns=[date_col])
    return out


def rolling_rv(r: np.ndarray, window: int) -> np.ndarray:
    r = np.asarray(r, dtype=float)
    out = np.full_like(r, np.nan, dtype=float)
    if window <= 1:
        return np.sqrt(r * r)
    rsq = r * r
    c = np.cumsum(np.where(np.isfinite(rsq), rsq, 0.0))
    for i in range(window - 1, r.size):
        s = c[i] - (c[i - window] if i >= window else 0.0)
        out[i] = np.sqrt(s)
    return out


def finite_quantile(x: np.ndarray, q: float) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan")
    return float(np.quantile(x, q, method="linear"))


def bucket_from_x(x: float, thr_calm: float, thr_stress: float) -> str:
    if not np.isfinite(x):
        return "mid"
    if x <= thr_calm:
        return "calm"
    if x >= thr_stress:
        return "stress"
    return "mid"


def pseudo_obs_from_returns(x: np.ndarray) -> np.ndarray:
    """Rank-based pseudo-observations u in (0,1) on TRAIN only."""
    x = np.asarray(x, dtype=float)
    mask = np.isfinite(x)
    y = x[mask]
    n = y.size
    if n < 10:
        # return all-nan same size
        out = np.full(x.shape, np.nan, dtype=float)
        return out
    order = np.argsort(y, kind="mergesort")
    ranks = np.empty(n, dtype=float)
    ranks[order] = np.arange(1, n + 1, dtype=float)
    u = ranks / (n + 1.0)
    u = np.clip(u, 1e-12, 1 - 1e-12)
    out = np.full(x.shape, np.nan, dtype=float)
    out[mask] = u
    return out


def rho_from_kendall_tau(tau: float) -> float:
    # elliptical copulas: rho = sin(pi/2 * tau)
    tau = float(np.clip(tau, -0.999, 0.999))
    return float(np.sin(np.pi * 0.5 * tau))


def kendall_tau(u: np.ndarray, v: np.ndarray) -> float:
    tau = kendalltau(u, v, nan_policy="omit").correlation
    if tau is None or (not np.isfinite(tau)):
        return 0.0
    return float(tau)


def fit_static_gauss_from_returns(r1_tr: np.ndarray, r2_tr: np.ndarray, rho_clamp: float) -> Dict[str, float]:
    u = pseudo_obs_from_returns(r1_tr)
    v = pseudo_obs_from_returns(r2_tr)
    m = np.isfinite(u) & np.isfinite(v)
    if m.sum() < 50:
        return {"rho": 0.0}
    tau = kendall_tau(u[m], v[m])
    rho = rho_from_kendall_tau(tau)
    rho = float(np.clip(rho, -1 + rho_clamp, 1 - rho_clamp))
    return {"rho": rho}


def fit_static_t_from_returns(
    r1_tr: np.ndarray,
    r2_tr: np.ndarray,
    nu_grid: List[float],
    nu_bounds: Tuple[float, float],
    rho_clamp: float,
) -> Dict[str, float]:
    u = pseudo_obs_from_returns(r1_tr)
    v = pseudo_obs_from_returns(r2_tr)
    m = np.isfinite(u) & np.isfinite(v)
    if m.sum() < 50:
        return {"rho": 0.0, "nu": float(max(4.0, nu_bounds[0] + 0.5))}
    tau = kendall_tau(u[m], v[m])
    rho0 = rho_from_kendall_tau(tau)
    rho0 = float(np.clip(rho0, -1 + rho_clamp, 1 - rho_clamp))

    # choose nu by grid via existing t-copula logpdf
    best_ll = -np.inf
    best_nu = float(nu_grid[0])
    for nu in nu_grid:
        nu = float(np.clip(nu, nu_bounds[0], nu_bounds[1]))
        ll = float(np.nansum(tcop.logpdf(u[m], v[m], rho0, nu)))
        if np.isfinite(ll) and ll > best_ll:
            best_ll = ll
            best_nu = nu
    return {"rho": rho0, "nu": float(best_nu)}


@dataclass(frozen=True)
class MSFit:
    fit_status: str
    A: np.ndarray
    pi0: np.ndarray
    theta: List[Dict[str, float]]
    n_eff: np.ndarray
    ll_train: float


def fit_ms_for_pair(
    u_tr: np.ndarray,
    v_tr: np.ndarray,
    x_tr: np.ndarray,
    *,
    family: str,
    K: int,
    init_A: np.ndarray,
    rho_clamp: float,
    nu_grid: List[float],
    nu_bounds: Tuple[float, float],
    calm_q: float,
    stress_q: float,
    max_iter: int,
    tol: float,
    min_state_eff_n: int,
    ordering_key: str,
    seed: int,
) -> MSFit:
    fit = fit_ms_copula_train(
        u_tr, v_tr, x_tr,
        family=family, K=K, init_A=init_A,
        rho_clamp=rho_clamp,
        nu_grid=nu_grid, nu_bounds=nu_bounds,
        calm_q=calm_q, stress_q=stress_q,
        max_iter=max_iter, tol=tol,
        min_state_eff_n=min_state_eff_n,
        ordering_key=ordering_key,
        seed=seed,
    )
    return MSFit(
        fit_status=str(getattr(fit, "fit_status", "unknown")),
        A=np.asarray(fit.A, dtype=float),
        pi0=np.asarray(fit.pi0, dtype=float),
        theta=list(fit.theta),
        n_eff=np.asarray(getattr(fit, "n_eff", np.full(K, np.nan)), dtype=float),
        ll_train=float(getattr(fit, "ll_train", np.nan)),
    )


def fit_ok_ms(f: MSFit, *, K: int, min_eff: int) -> bool:
    if f.fit_status != "ok":
        return False
    if (f.A.shape != (K, K)) or (not np.all(np.isfinite(f.A))):
        return False
    rs = f.A.sum(axis=1)
    if np.any(rs <= 0.0):
        return False
    if f.n_eff.shape != (K,) or np.any(~np.isfinite(f.n_eff)) or np.any(f.n_eff < float(min_eff)):
        return False
    if not np.isfinite(f.ll_train):
        return False
    return True


# --------------------------
# main pairwise runner core
# --------------------------

def run_pair_j7style(
    *,
    dates: np.ndarray,               # array of str date keys
    r1: np.ndarray,
    r2: np.ndarray,
    asset1: str,
    asset2: str,
    refit_every: int,
    rv_window: int,
    calm_q: float,
    stress_q: float,
    n_scenarios: int,
    alphas: List[float],
    seed: int,
    rho_clamp: float,
    nu_grid: List[float],
    nu_bounds: Tuple[float, float],
    models_base: List[str],
    ms_enable: bool,
    ms_models: List[str],
    ms_cfg: Dict[str, Any],
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Produces a J7-style dataframe for one pair.
    - Always computes: indep/static_gauss/static_t/thr_gauss/thr_t
    - Optionally computes: ms_gauss/ms_t (MS spot-check)
    """

    n = int(len(dates))
    r1 = np.asarray(r1, dtype=float)
    r2 = np.asarray(r2, dtype=float)
    r_port_real = 0.5 * r1 + 0.5 * r2
    loss_real = -r_port_real

    x_rv = rolling_rv(r_port_real, window=rv_window)

    # output frame
    models = list(models_base)
    if ms_enable:
        for m in ms_models:
            if m not in models:
                models.append(m)

    out_cols = ["date", "pair", "asset1", "asset2", "bucket", "x_rv", "r_port_real", "loss_real"]
    for m in models:
        for a in alphas:
            tag = int(round(a * 100))
            out_cols += [f"VaR{tag}_{m}", f"ES{tag}_{m}", f"exceed{tag}_{m}"]
    # optional pi audit columns (only meaningful when ms_enable)
    out_cols += [
        "used_ms_t",
        "used_ms_gauss",
        "pi_pred_state1",
        "pi_pred_state2",
        "pi_pred_gauss_state1",
        "pi_pred_gauss_state2",
    ]
    out = pd.DataFrame(index=np.arange(n), columns=out_cols)
    out["date"] = dates
    out["pair"] = f"{asset1}_{asset2}"
    out["asset1"] = asset1
    out["asset2"] = asset2
    out["x_rv"] = x_rv
    out["r_port_real"] = r_port_real
    out["loss_real"] = loss_real
    out["bucket"] = "mid"
    out["used_ms_t"] = 0
    out["used_ms_gauss"] = 0
    out["pi_pred_state1"] = np.nan
    out["pi_pred_state2"] = np.nan
    out["pi_pred_gauss_state1"] = np.nan
    out["pi_pred_gauss_state2"] = np.nan

    # block starts: like your J6/J7, we start scoring after enough train
    starts = list(range(0, n, refit_every))
    # enforce min train
    MIN_TRAIN = int(ms_cfg.get("min_train", 250))
    audit = {
        "pair": f"{asset1}_{asset2}",
        "n": n,
        "refit_every": refit_every,
        "rv_window": rv_window,
        "calm_q": calm_q,
        "stress_q": stress_q,
        "n_scenarios": n_scenarios,
        "alphas": alphas,
        "models": models,
        "ms_enable": bool(ms_enable),
    }

    # MS config
    K = int(ms_cfg.get("K", 2))
    ordering_key = str(ms_cfg.get("ordering_key", "rho"))
    init_A = np.asarray(ms_cfg.get("init_A", [[0.98, 0.02], [0.02, 0.98]]), dtype=float)
    max_iter = int(ms_cfg.get("max_iter", 50))
    tol = float(ms_cfg.get("tol", 1e-6))
    min_state_eff_n = int(ms_cfg.get("min_state_eff_n", 150))
    min_state_eff_n_t = int(ms_cfg.get("min_state_eff_n_t", 120))
    ms_seed = int(ms_cfg.get("seed", seed))

    # main loop blocks
    for b, start_i in enumerate(starts):
        train_end_i = start_i - 1
        if train_end_i < MIN_TRAIN:
            continue
        end_i = starts[b + 1] if (b + 1 < len(starts)) else n

        # TRAIN slices
        tr = slice(0, train_end_i + 1)
        r1_tr = r1[tr]
        r2_tr = r2[tr]
        x_tr = x_rv[tr]

        # thresholds on TRAIN only
        thr_stress = finite_quantile(x_tr, stress_q)
        thr_calm = finite_quantile(x_tr, calm_q)

        # empirical marginals on TRAIN only
        Q1 = build_empirical_quantile(r1_tr)
        Q2 = build_empirical_quantile(r2_tr)

        # fit static params on TRAIN only
        static_g = fit_static_gauss_from_returns(r1_tr, r2_tr, rho_clamp=rho_clamp)
        static_t = fit_static_t_from_returns(r1_tr, r2_tr, nu_grid=nu_grid, nu_bounds=nu_bounds, rho_clamp=rho_clamp)

        # fit threshold params on TRAIN subsets
        m_calm = np.isfinite(x_tr) & (x_tr <= thr_calm)
        m_stress = np.isfinite(x_tr) & (x_tr >= thr_stress)

        # default: fallback to static
        thr_g_calm = dict(static_g)
        thr_g_stress = dict(static_g)
        thr_t_calm = dict(static_t)
        thr_t_stress = dict(static_t)

        if int(m_calm.sum()) >= 100:
            thr_g_calm = fit_static_gauss_from_returns(r1_tr[m_calm], r2_tr[m_calm], rho_clamp=rho_clamp)
            thr_t_calm = fit_static_t_from_returns(r1_tr[m_calm], r2_tr[m_calm], nu_grid=nu_grid, nu_bounds=nu_bounds, rho_clamp=rho_clamp)
        if int(m_stress.sum()) >= 100:
            thr_g_stress = fit_static_gauss_from_returns(r1_tr[m_stress], r2_tr[m_stress], rho_clamp=rho_clamp)
            thr_t_stress = fit_static_t_from_returns(r1_tr[m_stress], r2_tr[m_stress], nu_grid=nu_grid, nu_bounds=nu_bounds, rho_clamp=rho_clamp)

        # MS fit (spot-check only)
        fit_g = None
        fit_t = None
        u_tr = pseudo_obs_from_returns(r1_tr)
        v_tr = pseudo_obs_from_returns(r2_tr)
        mm = np.isfinite(u_tr) & np.isfinite(v_tr) & np.isfinite(x_tr)

        if ms_enable and int(mm.sum()) >= max(300, MIN_TRAIN):
            uu = u_tr[mm]
            vv = v_tr[mm]
            xx = x_tr[mm]
            fit_g = fit_ms_for_pair(
                uu, vv, xx,
                family="gauss",
                K=K,
                init_A=init_A,
                rho_clamp=rho_clamp,
                nu_grid=nu_grid,
                nu_bounds=nu_bounds,
                calm_q=calm_q,
                stress_q=stress_q,
                max_iter=max_iter,
                tol=tol,
                min_state_eff_n=min_state_eff_n,
                ordering_key=ordering_key,
                seed=ms_seed,
            )
            fit_t = fit_ms_for_pair(
                uu, vv, xx,
                family="t",
                K=K,
                init_A=init_A,
                rho_clamp=rho_clamp,
                nu_grid=nu_grid,
                nu_bounds=nu_bounds,
                calm_q=calm_q,
                stress_q=stress_q,
                max_iter=max_iter,
                tol=tol,
                min_state_eff_n=min_state_eff_n_t,
                ordering_key=ordering_key,
                seed=ms_seed,
            )

        used_g = bool(ms_enable and (fit_g is not None) and fit_ok_ms(fit_g, K=K, min_eff=min_state_eff_n))
        used_t = bool(ms_enable and (fit_t is not None) and fit_ok_ms(fit_t, K=K, min_eff=min_state_eff_n_t))

        # Score inside block
        for i in range(start_i, end_i):
            di = str(dates[i])
            x_i = float(x_rv[i])
            bucket = bucket_from_x(x_i, thr_calm=thr_calm, thr_stress=thr_stress)
            out.loc[i, "bucket"] = bucket

            rng = np.random.default_rng(seed_for_key(seed, di))

            def losses_from_uv(u_s: np.ndarray, v_s: np.ndarray) -> np.ndarray:
                r1_s = Q1(u_s)
                r2_s = Q2(v_s)
                rp = 0.5 * r1_s + 0.5 * r2_s
                return -rp

            # base models
            for m in models_base:
                if m == "indep":
                    u_s, v_s = sample_copula("indep", {}, n_scenarios, rng, rho_clamp=rho_clamp)
                elif m == "static_gauss":
                    u_s, v_s = sample_copula("gauss", static_g, n_scenarios, rng, rho_clamp=rho_clamp)
                elif m == "static_t":
                    u_s, v_s = sample_copula("t", static_t, n_scenarios, rng, rho_clamp=rho_clamp)
                elif m == "thr_gauss":
                    theta = thr_g_stress if bucket == "stress" else thr_g_calm
                    u_s, v_s = sample_copula("gauss", theta, n_scenarios, rng, rho_clamp=rho_clamp)
                elif m == "thr_t":
                    theta = thr_t_stress if bucket == "stress" else thr_t_calm
                    u_s, v_s = sample_copula("t", theta, n_scenarios, rng, rho_clamp=rho_clamp)
                else:
                    raise ValueError(f"Unknown base model: {m}")

                losses = losses_from_uv(u_s, v_s)
                for a in alphas:
                    tag = int(round(a * 100))
                    var, es = compute_var_es(losses, a)
                    out.loc[i, f"VaR{tag}_{m}"] = var
                    out.loc[i, f"ES{tag}_{m}"] = es
                    lr = float(out.loc[i, "loss_real"])
                    out.loc[i, f"exceed{tag}_{m}"] = 1.0 if (np.isfinite(lr) and np.isfinite(var) and (lr > var)) else 0.0

            # MS models (spot-check)
            if ms_enable:
                # We need pi_pred(t|t-1) computed forward-only on the block's u/v sequence.
                # For strictness: we score pi using ONLY pseudo obs from returns in this block (no future).
                # We'll compute block-level pi_pred arrays once per block, then just index here.
                pass

        # If MS enabled and used, compute block-level pi_pred and then fill ms_* for that block.
        if ms_enable:
            blk = slice(start_i, end_i)
            # pseudo obs for OOS block (computed from returns in that block only for ranking)
            # NOTE: this is an ANNEX spot-check; we keep it causal by using PIT based on TRAIN empirical CDF.
            # Implementation: use TRAIN empirical quantile mapping via ranks is not directly invertible,
            # so we approximate by using ranks in TRAIN+current point would leak.
            # Safer: use normal scores from TRAIN ranks. Instead: we reuse pseudo-obs from TRAIN-only by:
            # mapping r_i through TRAIN empirical CDF: u = rank_train(r_i) / (n_train+1).
            # We'll implement that mapping here deterministically.

            # Build TRAIN empirical CDF for each margin from TRAIN returns (sorted).
            r1_tr_f = np.asarray(r1_tr, dtype=float)
            r2_tr_f = np.asarray(r2_tr, dtype=float)
            r1_tr_f = r1_tr_f[np.isfinite(r1_tr_f)]
            r2_tr_f = r2_tr_f[np.isfinite(r2_tr_f)]
            if r1_tr_f.size < 200 or r2_tr_f.size < 200:
                continue
            s1 = np.sort(r1_tr_f)
            s2 = np.sort(r2_tr_f)

            def ecdf_u(xv: np.ndarray, s: np.ndarray) -> np.ndarray:
                xv = np.asarray(xv, dtype=float)
                u = np.full_like(xv, np.nan, dtype=float)
                m = np.isfinite(xv)
                # rank = number of train samples <= x
                # use searchsorted (right) for tie-handling
                rnk = np.searchsorted(s, xv[m], side="right").astype(float)
                u[m] = (rnk + 1.0) / (s.size + 2.0)  # keep strictly in (0,1)
                return np.clip(u, 1e-10, 1 - 1e-10)

            u_blk = ecdf_u(r1[blk], s1)
            v_blk = ecdf_u(r2[blk], s2)
            m_ok = np.isfinite(u_blk) & np.isfinite(v_blk)
            if not m_ok.any():
                continue

            # ms_gauss
            if "ms_gauss" in ms_models:
                out.loc[blk, "used_ms_gauss"] = 1 if used_g else 0
                if used_g:
                    pi_pred_g, _, _ = score_ms_oos_forward(
                        u_blk, v_blk,
                        family="gauss",
                        A=fit_g.A,
                        pi0=fit_g.pi0,
                        theta=fit_g.theta,
                        rho_clamp=rho_clamp,
                    )
                    out.loc[blk, "pi_pred_gauss_state1"] = pi_pred_g[:, 0]
                    out.loc[blk, "pi_pred_gauss_state2"] = pi_pred_g[:, 1]

                    # simulate losses with mixture each date
                    for i in range(start_i, end_i):
                        di = str(dates[i])
                        rng = np.random.default_rng(seed_for_key(seed + 17, di))
                        pi1 = float(out.loc[i, "pi_pred_gauss_state1"])
                        pi2 = float(out.loc[i, "pi_pred_gauss_state2"])
                        bucket = str(out.loc[i, "bucket"])
                        # if pi invalid, fallback to threshold
                        if (not np.isfinite(pi1)) or (not np.isfinite(pi2)) or (pi1 + pi2 <= 0):
                            theta = thr_g_stress if bucket == "stress" else thr_g_calm
                            u_s, v_s = sample_copula("gauss", theta, n_scenarios, rng, rho_clamp=rho_clamp)
                        else:
                            u_s, v_s = sample_mixture(np.array([pi1, pi2]), [fit_g.theta[0], fit_g.theta[1]], "gauss", n_scenarios, rng, rho_clamp=rho_clamp)

                        losses = -(0.5 * Q1(u_s) + 0.5 * Q2(v_s))
                        for a in alphas:
                            tag = int(round(a * 100))
                            var, es = compute_var_es(losses, a)
                            out.loc[i, f"VaR{tag}_ms_gauss"] = var
                            out.loc[i, f"ES{tag}_ms_gauss"] = es
                            lr = float(out.loc[i, "loss_real"])
                            out.loc[i, f"exceed{tag}_ms_gauss"] = 1.0 if (np.isfinite(lr) and np.isfinite(var) and (lr > var)) else 0.0
                else:
                    # fallback: copy thr_gauss
                    for i in range(start_i, end_i):
                        for a in alphas:
                            tag = int(round(a * 100))
                            out.loc[i, f"VaR{tag}_ms_gauss"] = out.loc[i, f"VaR{tag}_thr_gauss"]
                            out.loc[i, f"ES{tag}_ms_gauss"] = out.loc[i, f"ES{tag}_thr_gauss"]
                            out.loc[i, f"exceed{tag}_ms_gauss"] = out.loc[i, f"exceed{tag}_thr_gauss"]

            # ms_t
            if "ms_t" in ms_models:
                out.loc[blk, "used_ms_t"] = 1 if used_t else 0
                if used_t:
                    pi_pred_t, _, _ = score_ms_oos_forward(
                        u_blk, v_blk,
                        family="t",
                        A=fit_t.A,
                        pi0=fit_t.pi0,
                        theta=fit_t.theta,
                        rho_clamp=rho_clamp,
                    )
                    out.loc[blk, "pi_pred_state1"] = pi_pred_t[:, 0]
                    out.loc[blk, "pi_pred_state2"] = pi_pred_t[:, 1]

                    for i in range(start_i, end_i):
                        di = str(dates[i])
                        rng = np.random.default_rng(seed_for_key(seed + 31, di))
                        pi1 = float(out.loc[i, "pi_pred_state1"])
                        pi2 = float(out.loc[i, "pi_pred_state2"])
                        bucket = str(out.loc[i, "bucket"])
                        if (not np.isfinite(pi1)) or (not np.isfinite(pi2)) or (pi1 + pi2 <= 0):
                            theta = thr_t_stress if bucket == "stress" else thr_t_calm
                            u_s, v_s = sample_copula("t", theta, n_scenarios, rng, rho_clamp=rho_clamp)
                        else:
                            u_s, v_s = sample_mixture(np.array([pi1, pi2]), [fit_t.theta[0], fit_t.theta[1]], "t", n_scenarios, rng, rho_clamp=rho_clamp)

                        losses = -(0.5 * Q1(u_s) + 0.5 * Q2(v_s))
                        for a in alphas:
                            tag = int(round(a * 100))
                            var, es = compute_var_es(losses, a)
                            out.loc[i, f"VaR{tag}_ms_t"] = var
                            out.loc[i, f"ES{tag}_ms_t"] = es
                            lr = float(out.loc[i, "loss_real"])
                            out.loc[i, f"exceed{tag}_ms_t"] = 1.0 if (np.isfinite(lr) and np.isfinite(var) and (lr > var)) else 0.0
                else:
                    # fallback: copy thr_t
                    for i in range(start_i, end_i):
                        for a in alphas:
                            tag = int(round(a * 100))
                            out.loc[i, f"VaR{tag}_ms_t"] = out.loc[i, f"VaR{tag}_thr_t"]
                            out.loc[i, f"ES{tag}_ms_t"] = out.loc[i, f"ES{tag}_thr_t"]
                            out.loc[i, f"exceed{tag}_ms_t"] = out.loc[i, f"exceed{tag}_thr_t"]

    # build audit summary
    audit["ms_gauss_used_rate"] = float((out["used_ms_gauss"].to_numpy(float) == 1).mean()) if ms_enable else 0.0
    audit["ms_t_used_rate"] = float((out["used_ms_t"].to_numpy(float) == 1).mean()) if ms_enable else 0.0
    return out, audit