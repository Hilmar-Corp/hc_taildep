from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
import matplotlib.pyplot as plt

from hc_taildep.copulas import gaussian as gcop
from hc_taildep.copulas import student_t as tcop
from hc_taildep.copulas.gating import (
    ThetaT,
    grid_fit_ab_for_tcopula,
    logistic_weights,
    fisher_mix_rho,
    mix_nu,
    robust_mad,
)
from hc_taildep.eval.dm_test import dm_test


def _sha12(path: Path) -> str:
    h = hashlib.sha256(path.read_bytes()).hexdigest()
    return h[:12]


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text())


def _write_yaml(path: Path, obj: dict[str, Any]) -> None:
    path.write_text(yaml.safe_dump(obj, sort_keys=False))


def _resolve_vars(cfg: dict[str, Any]) -> dict[str, Any]:
    # tiny resolver for ${dataset_version}
    s = yaml.safe_dump(cfg, sort_keys=False)
    dv = cfg.get("dataset_version", "")
    s = s.replace("${dataset_version}", str(dv))
    return yaml.safe_load(s)


def _detect_date_col(df: pd.DataFrame) -> str:
    # common names first
    for c in ["date", "Date", "datetime", "timestamp", "time", "ds"]:
        if c in df.columns:
            return c

    # common pandas CSV export of index
    for c in ["Unnamed: 0", "index"]:
        if c in df.columns:
            # accept if it parses as datetime for most rows
            x = pd.to_datetime(df[c], errors="coerce", utc=False)
            frac = float(x.notna().mean())
            if frac >= 0.90:
                return c

    raise ValueError(f"Cannot find date column in columns={list(df.columns)[:20]}")


def _norm_date(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col], utc=False).dt.date.astype(str)
    if date_col != "date":
        out = out.rename(columns={date_col: "date"})
    return out


def _quantile(x: np.ndarray, q: float) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan")
    return float(np.quantile(x, q))


def _rv_series(r: np.ndarray, window: int) -> np.ndarray:
    r = np.asarray(r, dtype=float)
    out = np.full_like(r, np.nan, dtype=float)
    if window <= 1:
        return np.sqrt(r * r)
    # rolling sqrt(sum r^2)
    rsq = r * r
    c = np.cumsum(np.where(np.isfinite(rsq), rsq, 0.0))
    for i in range(window - 1, r.size):
        s = c[i] - (c[i - window] if i >= window else 0.0)
        out[i] = np.sqrt(s)
    return out


def _fit_static(u_tr: np.ndarray, v_tr: np.ndarray, *, rho_clamp: float, nu_grid, nu_bounds) -> tuple[float, ThetaT]:
    rho_g = float(gcop.fit(u_tr, v_tr, rho_clamp=rho_clamp))
    p = tcop.fit(u_tr, v_tr, nu_grid=nu_grid, nu_bounds=tuple(nu_bounds), rho_clamp=rho_clamp)
    theta_t = ThetaT(rho=float(p.rho), nu=float(p.nu))
    return rho_g, theta_t


def _fit_regime_t(
    u_tr: np.ndarray,
    v_tr: np.ndarray,
    mask: np.ndarray,
    *,
    rho_clamp: float,
    nu_grid,
    nu_bounds,
    min_n: int,
    fallback: ThetaT,
) -> tuple[ThetaT, str]:
    m = np.asarray(mask, dtype=bool)
    if m.sum() < min_n:
        return fallback, "low_power_regime"
    p = tcop.fit(u_tr[m], v_tr[m], nu_grid=nu_grid, nu_bounds=tuple(nu_bounds), rho_clamp=rho_clamp)
    return ThetaT(rho=float(p.rho), nu=float(p.nu)), "ok"


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg_raw = _load_yaml(Path(args.config))
    cfg = _resolve_vars(cfg_raw)

    dataset_version = cfg["dataset_version"]
    inputs = cfg["inputs"]
    refit_every = int(cfg.get("refit_every", 20))
    seed = int(cfg.get("seed", 123))

    # stress
    stress = cfg["stress"]
    rv_window = int(stress["window"])
    stress_q = float(stress["stress_q"])
    calm_q = float(stress["calm_q"])
    on_asset = str(stress.get("on_asset", "BTC"))

    # copula fit
    tc_cfg = cfg["tcopula"]
    nu_grid = tc_cfg["nu_grid"]
    nu_bounds = tuple(tc_cfg["nu_bounds"])
    rho_clamp = float(tc_cfg.get("rho_clamp", 1e-6))

    # gating
    gating = cfg["gating"]
    thr_enabled = bool(gating["threshold"]["enabled"])
    min_regime_n = int(gating["threshold"].get("min_regime_n", 200))

    logit_enabled = bool(gating["logistic"]["enabled"])
    inner_val_days = int(gating["logistic"].get("inner_val_days", 252))
    a_grid = gating["logistic"]["a_grid"]
    b_grid = gating["logistic"]["b_grid"]

    # DM test
    dm_cfg = cfg.get("dm_test", {"enabled": True, "nw_lag_rule": "4*(n/100)^(2/9)"})
    dm_enabled = bool(dm_cfg.get("enabled", True))
    nw_lag_rule = str(dm_cfg.get("nw_lag_rule", "4*(n/100)^(2/9)"))

    # reporting
    rep = cfg.get("reporting", {})
    make_figs = bool(rep.get("figures", True))
    rolling_window = int(rep.get("rolling_window", 63))

    out_dir = Path(f"data/processed/{dataset_version}/copulas/conditional/j5_gating")
    _ensure_dir(out_dir)
    _ensure_dir(out_dir / "tables")
    _ensure_dir(out_dir / "figures")

    # Load inputs
    returns = pd.read_csv(inputs["returns_path"])
    u_series = pd.read_csv(inputs["u_series_path"])

    returns = _norm_date(returns, _detect_date_col(returns))
    u_series = _norm_date(u_series, _detect_date_col(u_series))

    # expected columns
    # returns: r_BTC, r_ETH (or returns_BTC etc). We'll try common variants.
    def _pick_ret(df: pd.DataFrame, asset: str) -> str:
        cands = [f"r_{asset}", f"ret_{asset}", f"return_{asset}", f"{asset}_ret", f"{asset}_return"]
        for c in cands:
            if c in df.columns:
                return c
        # fallback: if only 2 numeric columns besides date, assume order BTC,ETH
        num_cols = [c for c in df.columns if c != "date" and np.issubdtype(df[c].dtype, np.number)]
        if len(num_cols) >= 2:
            return num_cols[0] if asset == "BTC" else num_cols[1]
        raise ValueError(f"Cannot find returns column for {asset}. cols={list(df.columns)}")

    rbtc_col = _pick_ret(returns, "BTC")
    reth_col = _pick_ret(returns, "ETH")

    # u_series columns: u_BTC, u_ETH
    def _pick_u(df: pd.DataFrame, asset: str) -> str:
        cands = [f"u_{asset}", f"U_{asset}", asset]
        for c in cands:
            if c in df.columns:
                return c
        raise ValueError(f"Cannot find u column for {asset}. cols={list(df.columns)}")

    ubtc_col = _pick_u(u_series, "BTC")
    ueth_col = _pick_u(u_series, "ETH")

    # merge on date
    df = returns.merge(u_series[["date", ubtc_col, ueth_col]], on="date", how="inner", validate="one_to_one")
    df = df.sort_values("date").reset_index(drop=True)

    # scoring window: align with PIT validity automatically (we already inner-joined)
    # If you want, you can later enforce first_oos/last_oos from splits.json, but J2 PIT already aligned.
    u = df[ubtc_col].to_numpy(float)
    v = df[ueth_col].to_numpy(float)

    # stress x_t = RV on BTC (core)
    x = _rv_series(df[rbtc_col].to_numpy(float), window=rv_window)

    # storage
    n = df.shape[0]
    logc_static_t = np.full(n, np.nan)
    logc_static_g = np.full(n, np.nan)
    logc_thr_t = np.full(n, np.nan)
    logc_thr_g = np.full(n, np.nan)
    logc_logit_t = np.full(n, np.nan)
    logc_logit_g = np.full(n, np.nan)
    S_bin = np.full(n, np.nan)
    z = np.full(n, np.nan)
    w_logit = np.full(n, np.nan)

    params_rows: list[dict[str, Any]] = []

    rng = np.random.default_rng(seed)

    # refit schedule by index
    refit_indices = list(range(0, n, refit_every))
    for ridx in refit_indices:
        train_end = ridx - 1
        if train_end < 5:
            continue

        tr = slice(0, train_end + 1)
        u_tr = u[tr]
        v_tr = v[tr]
        x_tr = x[tr]

        # train thresholds (NO LEAKAGE)
        thr_stress = _quantile(x_tr, stress_q)
        thr_calm = _quantile(x_tr, calm_q)
        # train robust z-score (median/MAD)
        # Robust scaling for logistic gating (median/MAD), with explicit all-NaN guard.
        x_tr_finite = x_tr[np.isfinite(x_tr)]
        if x_tr_finite.size == 0:
            # Happens at very early refits when RV window not yet available.
            # Deterministic fallback: z_t = 0 -> w_t = sigmoid(a) (default a=0 => 0.5).
            med = 0.0
            mad = 1.0
        else:
            med = float(np.median(x_tr_finite))
            mad = float(np.median(np.abs(x_tr_finite - med)))
            if mad == 0.0:
                mad = 1.0

        # define train regime masks using train thresholds
        m_calm_tr = np.isfinite(x_tr) & (x_tr <= thr_calm)
        m_stress_tr = np.isfinite(x_tr) & (x_tr >= thr_stress)

        # fit static params on full train
        rho_g_static, theta_t_static = _fit_static(
            u_tr, v_tr, rho_clamp=rho_clamp, nu_grid=nu_grid, nu_bounds=nu_bounds
        )

        # gaussian regime params (cheap)
        def _fit_rho_g(mask: np.ndarray, fallback: float) -> tuple[float, str]:
            if mask.sum() < min_regime_n:
                return float(fallback), "low_power_regime"
            rho = float(gcop.fit(u_tr[mask], v_tr[mask], rho_clamp=rho_clamp))
            return rho, "ok"

        rho_g_calm, st_g_calm = _fit_rho_g(m_calm_tr, rho_g_static)
        rho_g_stress, st_g_stress = _fit_rho_g(m_stress_tr, rho_g_static)

        # t-copula regime params (use fallback=static)
        theta_t_calm, st_t_calm = _fit_regime_t(
            u_tr, v_tr, m_calm_tr, rho_clamp=rho_clamp, nu_grid=nu_grid, nu_bounds=nu_bounds, min_n=min_regime_n, fallback=theta_t_static
        )
        theta_t_stress, st_t_stress = _fit_regime_t(
            u_tr, v_tr, m_stress_tr, rho_clamp=rho_clamp, nu_grid=nu_grid, nu_bounds=nu_bounds, min_n=min_regime_n, fallback=theta_t_static
        )

        # logistic (a,b) fit on inner validation inside TRAIN (no leakage)
        a_best, b_best, best_sum = 0.0, 0.0, float("nan")
        if logit_enabled:
            # val = last inner_val_days of train
            val_start = max(0, (train_end + 1) - inner_val_days)
            u_val = u[val_start : train_end + 1]
            v_val = v[val_start : train_end + 1]
            x_val = x[val_start : train_end + 1]
            z_val = (x_val - med) / mad

            # if too short or non-finite -> fallback
            m_ok = np.isfinite(u_val) & np.isfinite(v_val) & np.isfinite(z_val)
            if int(m_ok.sum()) >= 50 and (train_end + 1 - val_start) >= min(50, inner_val_days):
                a_best, b_best, best_sum = grid_fit_ab_for_tcopula(
                    u_val[m_ok],
                    v_val[m_ok],
                    z_val[m_ok],
                    theta_calm=theta_t_calm,
                    theta_stress=theta_t_stress,
                    a_grid=a_grid,
                    b_grid=b_grid,
                    rho_clamp=rho_clamp,
                    nu_bounds=nu_bounds,
                )
            else:
                a_best, b_best = 0.0, 0.0

        # scoring block indices: from ridx to next refit-1
        next_ridx = min(n, ridx + refit_every)
        blk = slice(ridx, next_ridx)

        x_blk = x[blk]
        z_blk = (x_blk - med) / mad
        # S_t computed with TRAIN thresholds (no leakage) + observable x_t
        S_blk = np.isfinite(x_blk) & (x_blk >= thr_stress)

        # store z and S
        z[blk] = z_blk
        S_bin[blk] = S_blk.astype(float)

        # compute scores for each date in block
        for i in range(ridx, next_ridx):
            if not (np.isfinite(u[i]) and np.isfinite(v[i]) and np.isfinite(x[i])):
                continue

            # static
            logc_static_g[i] = float(gcop.logpdf(np.array([u[i]]), np.array([v[i]]), rho_g_static, rho_clamp=rho_clamp)[0])
            logc_static_t[i] = float(tcop.logpdf(np.array([u[i]]), np.array([v[i]]), theta_t_static.rho, theta_t_static.nu)[0])

            # threshold
            if thr_enabled:
                if bool(S_blk[i - ridx]):
                    rho_g = rho_g_stress
                    th_t = theta_t_stress
                else:
                    rho_g = rho_g_calm
                    th_t = theta_t_calm
                logc_thr_g[i] = float(gcop.logpdf(np.array([u[i]]), np.array([v[i]]), rho_g, rho_clamp=rho_clamp)[0])
                logc_thr_t[i] = float(tcop.logpdf(np.array([u[i]]), np.array([v[i]]), th_t.rho, th_t.nu)[0])

            # logistic
            if logit_enabled and np.isfinite(z_blk[i - ridx]):
                w = float(logistic_weights(np.array([z_blk[i - ridx]]), a_best, b_best)[0])
                w_logit[i] = w

                # gaussian mix
                rho_t = float(fisher_mix_rho(rho_g_calm, rho_g_stress, np.array([w]), rho_clamp=rho_clamp)[0])
                logc_logit_g[i] = float(gcop.logpdf(np.array([u[i]]), np.array([v[i]]), rho_t, rho_clamp=rho_clamp)[0])

                # t-copula mix
                rho_mix = float(fisher_mix_rho(theta_t_calm.rho, theta_t_stress.rho, np.array([w]), rho_clamp=rho_clamp)[0])
                nu_mix = float(mix_nu(theta_t_calm.nu, theta_t_stress.nu, np.array([w]), nu_bounds=nu_bounds)[0])
                logc_logit_t[i] = float(tcop.logpdf(np.array([u[i]]), np.array([v[i]]), rho_mix, nu_mix)[0])

        params_rows.append(
            {
                "refit_index": int(ridx),
                "refit_date": df.loc[ridx, "date"],
                "train_end_date": df.loc[train_end, "date"],
                "n_train": int(train_end + 1),
                "thr_stress": float(thr_stress),
                "thr_calm": float(thr_calm),
                "x_med": float(med),
                "x_mad": float(mad),
                "rho_g_static": float(rho_g_static),
                "rho_g_calm": float(rho_g_calm),
                "rho_g_stress": float(rho_g_stress),
                "theta_t_static_rho": float(theta_t_static.rho),
                "theta_t_static_nu": float(theta_t_static.nu),
                "theta_t_calm_rho": float(theta_t_calm.rho),
                "theta_t_calm_nu": float(theta_t_calm.nu),
                "theta_t_stress_rho": float(theta_t_stress.rho),
                "theta_t_stress_nu": float(theta_t_stress.nu),
                "status_g_calm": st_g_calm,
                "status_g_stress": st_g_stress,
                "status_t_calm": st_t_calm,
                "status_t_stress": st_t_stress,
                "a_best": float(a_best),
                "b_best": float(b_best),
                "logit_val_sum": float(best_sum) if np.isfinite(best_sum) else np.nan,
            }
        )

    # Build predictions df
    out = pd.DataFrame(
        {
            "date": df["date"].astype(str),
            "u_BTC": u,
            "u_ETH": v,
            "x_rv": x,
            "z_rv": z,
            "S_bin": S_bin,
            "w_logit": w_logit,
            "logc_static_gauss": logc_static_g,
            "logc_static_t": logc_static_t,
            "logc_thr_gauss": logc_thr_g,
            "logc_thr_t": logc_thr_t,
            "logc_logit_gauss": logc_logit_g,
            "logc_logit_t": logc_logit_t,
        }
    )

    pred_path = out_dir / "predictions.csv"
    out.to_csv(pred_path, index=False)

    params_df = pd.DataFrame(params_rows)
    params_path = out_dir / "tables" / "params_summary.csv"
    params_df.to_csv(params_path, index=False)

    # Summaries
    def _summ(xarr: np.ndarray) -> dict[str, float]:
        xarr = np.asarray(xarr, dtype=float)
        xarr = xarr[np.isfinite(xarr)]
        if xarr.size == 0:
            return {"n_obs": 0, "sum": np.nan, "mean": np.nan, "std": np.nan, "min": np.nan, "max": np.nan}
        return {
            "n_obs": int(xarr.size),
            "sum": float(np.sum(xarr)),
            "mean": float(np.mean(xarr)),
            "std": float(np.std(xarr, ddof=1)) if xarr.size > 1 else 0.0,
            "min": float(np.min(xarr)),
            "max": float(np.max(xarr)),
        }

    metrics = {
        "dataset_version": dataset_version,
        "refit_every": refit_every,
        "stress": {"type": "RV", "on_asset": on_asset, "window": rv_window, "stress_q": stress_q, "calm_q": calm_q, "zscore": "MAD"},
        "tcopula": {"nu_grid": nu_grid, "nu_bounds": list(nu_bounds), "rho_clamp": rho_clamp},
        "gating": {"threshold": {"enabled": thr_enabled, "min_regime_n": min_regime_n}, "logistic": {"enabled": logit_enabled, "inner_val_days": inner_val_days}},
        "per_model": {
            "static_gauss": _summ(logc_static_g),
            "static_t": _summ(logc_static_t),
            "thr_gauss": _summ(logc_thr_g),
            "thr_t": _summ(logc_thr_t),
            "logit_gauss": _summ(logc_logit_g),
            "logit_t": _summ(logc_logit_t),
        },
        "hashes": {"predictions_csv": _sha12(pred_path), "params_summary_csv": _sha12(params_path)},
    }

    # DM tests
    dm_out = {"schema_version": "j5_dm_v1", "nw_lag_rule": nw_lag_rule, "comparisons": [], "results": []}
    dm_rows = []
    if dm_enabled:
        def _do_dm(name: str, a: np.ndarray, b: np.ndarray) -> None:
            d = a - b
            res = dm_test(d, nw_lag_rule=nw_lag_rule, alternative="two-sided")
            row = {"comparison": name, **asdict(res)}
            dm_out["results"].append(row)
            dm_out["comparisons"].append({"name": name, **asdict(res)})
            dm_rows.append(
                {"name": name, "n_obs": res.n_obs, "nw_lag": res.nw_lag, "mean_delta": res.mean_delta, "std_delta": res.std_delta, "dm_stat": res.dm_stat, "pvalue": res.pvalue}
            )

        _do_dm("thr_t_vs_static_t", logc_thr_t, logc_static_t)
        _do_dm("logit_t_vs_static_t", logc_logit_t, logc_static_t)
        _do_dm("thr_gauss_vs_static_gauss", logc_thr_g, logc_static_g)
        _do_dm("logit_gauss_vs_static_gauss", logc_logit_g, logc_static_g)

    dm_path = out_dir / "dm_test.json"
    dm_path.write_text(json.dumps(dm_out, indent=2))

    dm_df = pd.DataFrame(dm_rows)
    dm_summary_path = out_dir / "tables" / "dm_summary.csv"
    dm_df.to_csv(dm_summary_path, index=False)

    scores_summary = pd.DataFrame(
        [
            {"model": k, **v}
            for k, v in metrics["per_model"].items()
        ]
    )
    scores_summary_path = out_dir / "tables" / "scores_summary.csv"
    scores_summary.to_csv(scores_summary_path, index=False)

    metrics_path = out_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))

    # figures
    if make_figs:
        # delta cumul (logit_t vs static_t)
        d = (logc_logit_t - logc_static_t).copy()
        d[~np.isfinite(d)] = 0.0
        cum = np.cumsum(d)

        plt.figure()
        plt.plot(pd.to_datetime(out["date"]), cum)
        plt.title("Cumulative Δ logscore (logit_t - static_t)")
        plt.tight_layout()
        plt.savefig(out_dir / "figures" / "delta_cum.png", dpi=150)
        plt.close()

        # hist
        dd = (logc_logit_t - logc_static_t)
        dd = dd[np.isfinite(dd)]
        plt.figure()
        plt.hist(dd, bins=50)
        plt.title("Δ logscore distribution (logit_t - static_t)")
        plt.tight_layout()
        plt.savefig(out_dir / "figures" / "delta_hist.png", dpi=150)
        plt.close()

        # w timeseries
        ww = out["w_logit"].to_numpy(float)
        plt.figure()
        plt.plot(pd.to_datetime(out["date"]), ww)
        plt.title("Logistic weight w_t")
        plt.tight_layout()
        plt.savefig(out_dir / "figures" / "w_timeseries.png", dpi=150)
        plt.close()

    # report (minimal, auditable)
    report = []
    report.append(f"# J5 — Conditional copulas (threshold + logistic gating) — {dataset_version}\n")
    report.append("## Protocol OOS\n")
    report.append(f"- Refit every {refit_every} days.\n")
    report.append("- At each refit date R_k: thresholds (stress/calm) + robust z-score params computed on train ≤ R_k-1 (no leakage).\n")
    report.append("- Scoring uses PIT u_BTC/u_ETH at date t with params fitted on ≤ R_k-1.\n")
    report.append("\n## Stress definition\n")
    report.append(f"- x_t = RV_t on {on_asset}, window={rv_window}\n")
    report.append(f"- Threshold stress: x_t >= q{stress_q} (computed on train)\n")
    report.append(f"- Calm subset for regime fit: x_t <= q{calm_q} (computed on train)\n")
    report.append("\n## Results (logscore summaries)\n")
    report.append(scores_summary.to_string(index=False))
    report.append("\n\n## DM tests (HAC Newey–West)\n")
    report.append(dm_df.to_string(index=False) if dm_enabled else "DM disabled")
    report.append("\n\n## Files\n")
    report.append(f"- predictions.csv hash={_sha12(pred_path)}\n")
    report.append(f"- metrics.json hash={_sha12(metrics_path)}\n")
    report.append(f"- dm_test.json hash={_sha12(dm_path)}\n")
    (out_dir / "report.md").write_text("\n".join(report))

    # config.resolved + provenance
    _write_yaml(out_dir / "config.resolved.yaml", cfg)
    prov = {
        "inputs": {k: str(v) for k, v in inputs.items()},
        "outputs": {
            "predictions.csv": str(pred_path),
            "metrics.json": str(metrics_path),
            "dm_test.json": str(dm_path),
        },
        "hashes": {
            "predictions.csv": _sha12(pred_path),
            "metrics.json": _sha12(metrics_path),
            "dm_test.json": _sha12(dm_path),
            "scores_summary.csv": _sha12(scores_summary_path),
            "dm_summary.csv": _sha12(dm_summary_path),
            "params_summary.csv": _sha12(params_path),
        },
    }
    (out_dir / "provenance.json").write_text(json.dumps(prov, indent=2))

    print(f"[OK] built J5 conditional copulas: {dataset_version}")
    print(f"[OK] out_dir: {out_dir.resolve()}")
    print(f"[OK] predictions: {pred_path.name} hash={_sha12(pred_path)}")
    print(f"[OK] metrics: metrics.json hash={_sha12(metrics_path)}")
    if dm_enabled:
        print(f"[OK] dm_test: dm_test.json hash={_sha12(dm_path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
