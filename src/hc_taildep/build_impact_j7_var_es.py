# src/hc_taildep/build_impact_j7_var_es.py
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

# headless-safe backend (same pattern as J6; prevents VSCode/Mac segfaults)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from hc_taildep.impact.var_es import (
    build_empirical_quantile,
    compute_var_es,
    sample_copula,
    sample_mixture,
    sanity_check_var_es,
)


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
    s = yaml.safe_dump(cfg, sort_keys=False)
    dv = cfg.get("dataset_version", "")
    s = s.replace("${dataset_version}", str(dv))
    return yaml.safe_load(s)


def _detect_date_col(df: pd.DataFrame) -> str:
    for c in ["date", "Date", "datetime", "timestamp", "time", "ds", "ts_utc", "ts", "time_utc"]:
        if c in df.columns:
            return c
    for c in ["Unnamed: 0", "index"]:
        if c in df.columns:
            x = pd.to_datetime(df[c], errors="coerce", utc=False)
            if float(x.notna().mean()) >= 0.90:
                return c
    raise ValueError(f"Cannot find date column in columns={list(df.columns)[:20]}")


def _norm_date(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    """Normalize a datetime column into a deterministic merge key.

    - Daily datasets: keep YYYY-MM-DD
    - Intraday datasets (e.g. 4h): keep full UTC timestamp (YYYY-MM-DD HH:MM:SS+00:00)

    This avoids collisions (multiple rows per day) that would break one-to-one merges.
    """
    out = df.copy()
    t = pd.to_datetime(out[date_col], errors="coerce", utc=True)

    if t.isna().any():
        bad = int(t.isna().sum())
        raise ValueError(f"Could not parse {bad} timestamps in column={date_col}")

    # Intraday if any time component is not midnight.
    is_intraday = ((t.dt.hour != 0) | (t.dt.minute != 0) | (t.dt.second != 0)).any()

    if is_intraday:
        out[date_col] = t.dt.strftime("%Y-%m-%d %H:%M:%S+00:00")
    else:
        out[date_col] = t.dt.strftime("%Y-%m-%d")

    if date_col != "date":
        out = out.rename(columns={date_col: "date"})
    return out


def _quantile(x: np.ndarray, q: float) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan")
    return float(np.quantile(x, q, method="linear"))


def _rv_series(r: np.ndarray, window: int) -> np.ndarray:
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


def _pick_ret(df: pd.DataFrame, asset: str) -> str:
    cands = [f"r_{asset}", f"ret_{asset}", f"return_{asset}", f"{asset}_ret", f"{asset}_return", asset]
    for c in cands:
        if c in df.columns and c != "date":
            return c
    num_cols = [c for c in df.columns if c != "date" and np.issubdtype(df[c].dtype, np.number)]
    if len(num_cols) >= 2:
        return num_cols[0] if asset == "BTC" else num_cols[1]
    raise ValueError(f"Cannot find returns column for {asset}. cols={list(df.columns)}")


def _seed_for_date(global_seed: int, date_str: str) -> int:
    # deterministic audit-friendly per-date seed
    h = hashlib.sha256((str(global_seed) + "|" + str(date_str)).encode("utf-8")).hexdigest()
    return int(h[:8], 16)


def _bucket(x: float, thr_calm: float, thr_stress: float) -> str:
    if not np.isfinite(x):
        return "mid"
    if x <= thr_calm:
        return "calm"
    if x >= thr_stress:
        return "stress"
    return "mid"


def kupiec_pvalue(n: int, x: int, p: float) -> float:
    """Kupiec unconditional coverage test for VaR exceedances.

    n: number of observations
    x: number of exceedances (hits)
    p: target exceedance probability (e.g. 0.05 for VaR95)

    Returns: p-value (Chi-square df=1 approximation).
    """
    if n <= 0:
        return float("nan")
    if x < 0 or x > n:
        return float("nan")
    p = float(p)
    if not (0.0 < p < 1.0):
        return float("nan")

    phat = x / n

    # avoid log(0)
    eps = 1e-12
    phat = min(max(phat, eps), 1.0 - eps)
    p = min(max(p, eps), 1.0 - eps)

    # LR_uc = -2 ln( L(p) / L(phat) )
    lr = -2.0 * (
        (n - x) * math.log(1.0 - p) + x * math.log(p)
        - (n - x) * math.log(1.0 - phat) - x * math.log(phat)
    )

    # Chi2_1 survival function: P(Chi2_1 >= lr) = erfc(sqrt(lr/2))
    return float(math.erfc(math.sqrt(max(lr, 0.0) / 2.0)))


def _qstats(x: np.ndarray) -> dict[str, float]:
    """Simple distribution stats for deltas (mean + upper quantiles)."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {"n": 0.0, "mean": float("nan"), "p50": float("nan"), "p75": float("nan"), "p90": float("nan"), "p95": float("nan")}
    return {
        "n": float(x.size),
        "mean": float(np.mean(x)),
        "p50": float(np.quantile(x, 0.50, method="linear")),
        "p75": float(np.quantile(x, 0.75, method="linear")),
        "p90": float(np.quantile(x, 0.90, method="linear")),
        "p95": float(np.quantile(x, 0.95, method="linear")),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg_raw = _load_yaml(Path(args.config))
    cfg = _resolve_vars(cfg_raw)

    dataset_version = cfg["dataset_version"]
    seed = int(cfg.get("seed", 123))
    refit_every = int(cfg.get("refit_every", 40))
    n_scenarios = int(cfg.get("n_scenarios", 20000))
    alphas = [float(a) for a in cfg.get("alphas", [0.95, 0.99])]
    weights = [float(w) for w in cfg.get("weights", [0.5, 0.5])]
    if len(weights) != 2:
        raise ValueError("weights must be length 2 (BTC, ETH)")
    w_btc, w_eth = weights

    inputs = cfg["inputs"]
    returns_path = str(inputs["returns_path"])

    copula_base = Path(str(inputs["copula_markov_base_dir"]))
    copula_run = str(cfg["copula_run_name"])
    copula_dir = copula_base / copula_run
    params_path = copula_dir / "tables" / "params_summary.csv"
    preds_path = copula_dir / "predictions.csv"
    metrics_path = copula_dir / "metrics.json"

    if not params_path.exists():
        raise FileNotFoundError(f"Missing params_summary.csv: {params_path}")
    if not preds_path.exists():
        raise FileNotFoundError(f"Missing predictions.csv: {preds_path}")
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing metrics.json: {metrics_path}")

    # models (what you compare)
    models = [str(m) for m in cfg.get("models_to_compare", [
        "indep",
        "static_gauss",
        "static_t",
        "thr_gauss",
        "thr_t",
        "ms_gauss",
        "ms_t",
    ])]

    # copula numeric safety (should match J6 u_clip_eps for consistent PIT usage)
    rho_clamp = float(cfg.get("rho_clamp", 1e-6))

    stress = cfg["stress"]
    rv_window = int(stress.get("window", 30))
    stress_q = float(stress.get("stress_q", 0.9))
    calm_q = float(stress.get("calm_q", 0.5))

    out_name = str(cfg.get("out_name", f"j7_var_es_{copula_run}_N{n_scenarios}")).strip()
    out_dir = Path(f"data/processed/{dataset_version}/impact/j7_var_es/{out_name}")
    _ensure_dir(out_dir)
    _ensure_dir(out_dir / "tables")
    _ensure_dir(out_dir / "figures")

    # load data
    returns = pd.read_csv(returns_path)
    returns = _norm_date(returns, _detect_date_col(returns))
    returns = returns.sort_values("date").reset_index(drop=True)

    rbtc_col = _pick_ret(returns, "BTC")
    reth_col = _pick_ret(returns, "ETH")

    # load J6 params & predictions
    params = pd.read_csv(params_path)
    preds = pd.read_csv(preds_path)
    if "date" not in preds.columns:
        raise ValueError("predictions.csv must contain a 'date' column")
    preds["date"] = preds["date"].astype(str)
    preds = preds.sort_values("date").reset_index(drop=True)

    # join date index on returns
    df = returns.merge(preds[[
        "date",
        "x_rv",
        "used_ms_gauss",
        "used_ms_t",
        "used_ms_t_mode",
        "pi_pred_state1",
        "pi_pred_state2",
        # If you apply the optional J6 patch, these columns will exist:
        "pi_pred_gauss_state1" if "pi_pred_gauss_state1" in preds.columns else "date",
        "pi_pred_gauss_state2" if "pi_pred_gauss_state2" in preds.columns else "date",
    ]], on="date", how="inner", validate="one_to_one")

    # clean merge artifact if patch columns absent
    if "pi_pred_gauss_state1" not in df.columns:
        df["pi_pred_gauss_state1"] = np.nan
    if "pi_pred_gauss_state2" not in df.columns:
        df["pi_pred_gauss_state2"] = np.nan

    df = df.sort_values("date").reset_index(drop=True)
    n = df.shape[0]

    # Audit: availability of π predictions and MS usage rates (helps interpret ms_gauss vs ms_t)
    gauss_pi_available_rate = float(np.isfinite(df["pi_pred_gauss_state2"].to_numpy(float)).mean())
    t_pi_available_rate = float(np.isfinite(df["pi_pred_state2"].to_numpy(float)).mean())
    used_ms_gauss_rate = float((df["used_ms_gauss"].to_numpy(float) == 1).mean())
    used_ms_t_rate = float((df["used_ms_t"].to_numpy(float) == 1).mean())
    ms_gauss_mode = "gauss_pi" if gauss_pi_available_rate > 0 else "proxy_t_pi_or_thr"

    # Bucketing RV: use the SAME stress indicator as J6 (predictions.csv:x_rv)
    # This keeps calm/stress definitions aligned with the threshold/MS gating from J6.
    if "x_rv" not in df.columns:
        raise ValueError("Merged df must contain x_rv from J6 predictions.csv")
    df["x_rv_j7"] = df["x_rv"].to_numpy(float)

    # params rows must be aligned to refit blocks; in J6, params_summary has refit_index + train_end_date etc.
    # We'll build a list of blocks in chronological order.
    required_cols = ["refit_index", "refit_date", "train_end_date", "n_train"]
    for c in required_cols:
        if c not in params.columns:
            raise ValueError(f"params_summary.csv missing column: {c}")

    params = params.sort_values("refit_index").reset_index(drop=True)

    # Build mapping date -> block_id using refit_date boundaries
    refit_dates = params["refit_date"].astype(str).tolist()
    refit_indices = params["refit_index"].astype(int).tolist()

    # sanity: refit_date should exist in df dates
    date_to_i = {d: i for i, d in enumerate(df["date"].astype(str).tolist())}
    block_starts = []
    for rd in refit_dates:
        if rd not in date_to_i:
            # This can happen if merge clipped dates; keep deterministic by skipping
            continue
        block_starts.append(date_to_i[rd])
    block_starts = sorted(set(block_starts))
    if len(block_starts) == 0:
        raise ValueError("No refit_date from params_summary found in merged df dates.")

    # Pre-alloc output (pandas requires an explicit index when initializing from scalars)
    out_cols = ["date", "bucket", "x_rv", "r_port_real", "loss_real"]
    for model in models:
        for a in alphas:
            tag = int(round(a * 100))
            out_cols += [f"VaR{tag}_{model}", f"ES{tag}_{model}", f"exceed{tag}_{model}"]

    out_df = pd.DataFrame(index=np.arange(n), columns=out_cols)

    # Fill core columns
    out_df["date"] = df["date"].astype(str).to_numpy()
    out_df["bucket"] = "mid"  # will be overwritten later
    out_df["x_rv"] = df["x_rv_j7"].to_numpy(float)

    r_port_real = w_btc * df[rbtc_col].to_numpy(float) + w_eth * df[reth_col].to_numpy(float)
    loss_real = -r_port_real
    out_df["r_port_real"] = r_port_real
    out_df["loss_real"] = loss_real

    # Initialize exceedance flags to 0.0 (optional but avoids NaNs when a date is skipped)
    for model in models:
        for a in alphas:
            tag = int(round(a * 100))
            out_df[f"exceed{tag}_{model}"] = 0.0

    # iterate blocks (like J6)
    # define blocks as [start_i, next_start_i) on df index
    for b, start_i in enumerate(block_starts):
        end_i = block_starts[b + 1] if b + 1 < len(block_starts) else n
        if start_i <= 0:
            continue
        train_end_i = start_i - 1
        if train_end_i < 250:
            continue

        # train slice
        tr = slice(0, train_end_i + 1)
        rbtc_tr = df.loc[tr, rbtc_col].to_numpy(float)
        reth_tr = df.loc[tr, reth_col].to_numpy(float)
        x_tr = out_df.loc[tr, "x_rv"].to_numpy(float)

        # thresholds computed train-only
        thr_stress = _quantile(x_tr, stress_q)
        thr_calm = _quantile(x_tr, calm_q)

        # empirical marginals (train-only)
        Q_btc = build_empirical_quantile(rbtc_tr)
        Q_eth = build_empirical_quantile(reth_tr)

        # pick the params row that matches this start refit_index best:
        # We try exact match on refit_date; fallback to nearest by date order.
        refit_date = df.loc[start_i, "date"]
        p_row = None
        # exact match
        hits = params.index[params["refit_date"].astype(str) == str(refit_date)].tolist()
        if hits:
            p_row = params.loc[hits[0]]
        else:
            # fallback: choose row with refit_index closest to start_i
            params["_absdiff"] = (params["refit_index"].astype(int) - int(start_i)).abs()
            p_row = params.sort_values("_absdiff").iloc[0]
            params = params.drop(columns=["_absdiff"], errors="ignore")

        # extract parameters needed for models
        # static
        static_gauss = {"rho": float(p_row["static_rho_g"])}
        static_t = {"rho": float(p_row["static_t_rho"]), "nu": float(p_row["static_t_nu"])}

        # threshold
        thr_gauss_calm = {"rho": float(p_row["thr_rho_g_calm"])}
        thr_gauss_stress = {"rho": float(p_row["thr_rho_g_stress"])}
        thr_t_calm = {"rho": float(p_row["thr_t_rho_calm"]), "nu": float(p_row["thr_t_nu_calm"])}
        thr_t_stress = {"rho": float(p_row["thr_t_rho_stress"]), "nu": float(p_row["thr_t_nu_stress"])}

        # ms states (t)
        ms_t_theta1 = {"rho": float(p_row["ms_t_rho1"]), "nu": float(p_row["ms_t_nu1"])}
        ms_t_theta2 = {"rho": float(p_row["ms_t_rho2"]), "nu": float(p_row["ms_t_nu2"])}

        # ms states (gauss)
        ms_g_theta1 = {"rho": float(p_row["ms_gauss_rho1"])}
        ms_g_theta2 = {"rho": float(p_row["ms_gauss_rho2"])}

        # process dates in this block
        for i in range(start_i, end_i):
            date_i = str(df.loc[i, "date"])
            x_i = float(out_df.loc[i, "x_rv"])
            bucket = _bucket(x_i, thr_calm=thr_calm, thr_stress=thr_stress)
            out_df.loc[i, "bucket"] = bucket

            # per-date RNG
            rng = np.random.default_rng(_seed_for_date(seed, date_i))

            # helper: simulate portfolio losses given u-samples
            def _losses_from_uv(u_s: np.ndarray, v_s: np.ndarray) -> np.ndarray:
                r1 = Q_btc(u_s)
                r2 = Q_eth(v_s)
                rp = w_btc * r1 + w_eth * r2
                return -rp

            # build all model VaR/ES
            for model in models:
                model = str(model)

                # decide copula sampler per model
                if model == "indep":
                    u_s, v_s = sample_copula("indep", {}, n_scenarios, rng, rho_clamp=rho_clamp)

                elif model == "static_gauss":
                    u_s, v_s = sample_copula("gauss", static_gauss, n_scenarios, rng, rho_clamp=rho_clamp)

                elif model == "static_t":
                    u_s, v_s = sample_copula("t", static_t, n_scenarios, rng, rho_clamp=rho_clamp)

                elif model == "thr_gauss":
                    theta = thr_gauss_stress if bucket == "stress" else thr_gauss_calm
                    u_s, v_s = sample_copula("gauss", theta, n_scenarios, rng, rho_clamp=rho_clamp)

                elif model == "thr_t":
                    theta = thr_t_stress if bucket == "stress" else thr_t_calm
                    u_s, v_s = sample_copula("t", theta, n_scenarios, rng, rho_clamp=rho_clamp)

                elif model == "ms_t":
                    # STRICT one-step-ahead: use pi_pred from predictions.csv (t MS)
                    pi2 = float(df.loc[i, "pi_pred_state2"])
                    pi1 = float(df.loc[i, "pi_pred_state1"])
                    used_ms_t = int(df.loc[i, "used_ms_t"]) if np.isfinite(df.loc[i, "used_ms_t"]) else 0
                    if used_ms_t == 1 and np.isfinite(pi1) and np.isfinite(pi2) and (pi1 + pi2) > 0:
                        u_s, v_s = sample_mixture(np.array([pi1, pi2]), [ms_t_theta1, ms_t_theta2], "t", n_scenarios, rng, rho_clamp=rho_clamp)
                    else:
                        # policy: MS2 if healthy else THR (same as J6 runner)
                        theta = thr_t_stress if bucket == "stress" else thr_t_calm
                        u_s, v_s = sample_copula("t", theta, n_scenarios, rng, rho_clamp=rho_clamp)

                elif model == "ms_gauss":
                    # If J6 patch exists: use pi_pred_gauss_*.
                    pi2g = float(df.loc[i, "pi_pred_gauss_state2"])
                    pi1g = float(df.loc[i, "pi_pred_gauss_state1"])
                    if np.isfinite(pi1g) and np.isfinite(pi2g) and (pi1g + pi2g) > 0:
                        u_s, v_s = sample_mixture(np.array([pi1g, pi2g]), [ms_g_theta1, ms_g_theta2], "gauss", n_scenarios, rng, rho_clamp=rho_clamp)
                    else:
                        # Proxy (as requested): use pi_pred from t-model when gauss pi is not stored.
                        pi2 = float(df.loc[i, "pi_pred_state2"])
                        pi1 = float(df.loc[i, "pi_pred_state1"])
                        used_ms_g = int(df.loc[i, "used_ms_gauss"]) if np.isfinite(df.loc[i, "used_ms_gauss"]) else 0
                        if used_ms_g == 1 and np.isfinite(pi1) and np.isfinite(pi2) and (pi1 + pi2) > 0:
                            u_s, v_s = sample_mixture(np.array([pi1, pi2]), [ms_g_theta1, ms_g_theta2], "gauss", n_scenarios, rng, rho_clamp=rho_clamp)
                        else:
                            theta = thr_gauss_stress if bucket == "stress" else thr_gauss_calm
                            u_s, v_s = sample_copula("gauss", theta, n_scenarios, rng, rho_clamp=rho_clamp)

                else:
                    raise ValueError(f"Unknown model: {model}")

                losses = _losses_from_uv(u_s, v_s)

                # fill VaR/ES + exceedance for each alpha
                for a in alphas:
                    tag = int(round(a * 100))
                    var, es = compute_var_es(losses, a)
                    # store
                    out_df.loc[i, f"VaR{tag}_{model}"] = var
                    out_df.loc[i, f"ES{tag}_{model}"] = es
                    # exceedance on real loss
                    lr = float(out_df.loc[i, "loss_real"])
                    out_df.loc[i, f"exceed{tag}_{model}"] = 1.0 if (np.isfinite(lr) and np.isfinite(var) and (lr > var)) else 0.0

                # sanity checks (cheap, per model per date)
                if (0.95 in alphas) and (0.99 in alphas):
                    sanity_check_var_es(
                        float(out_df.loc[i, f"VaR95_{model}"]),
                        float(out_df.loc[i, f"ES95_{model}"]),
                        float(out_df.loc[i, f"VaR99_{model}"]),
                        float(out_df.loc[i, f"ES99_{model}"]),
                    )

    # write predictions
    pred_path = out_dir / "var_es_predictions.csv"
    out_df.to_csv(pred_path, index=False)

    # summaries
    # var_es_summary: mean VaR/ES by bucket/model
    rows = []
    for model in models:
        for a in alphas:
            tag = int(round(a * 100))
            for bucket in ["calm", "mid", "stress"]:
                msk = (out_df["bucket"].astype(str) == bucket)
                v = out_df.loc[msk, f"VaR{tag}_{model}"].to_numpy(float)
                e = out_df.loc[msk, f"ES{tag}_{model}"].to_numpy(float)
                v = v[np.isfinite(v)]
                e = e[np.isfinite(e)]
                rows.append({
                    "bucket": bucket,
                    "model": model,
                    "alpha": a,
                    "n_obs": int(msk.sum()),
                    "VaR_mean": float(np.mean(v)) if v.size else np.nan,
                    "VaR_median": float(np.median(v)) if v.size else np.nan,
                    "ES_mean": float(np.mean(e)) if e.size else np.nan,
                    "ES_median": float(np.median(e)) if e.size else np.nan,
                })
    var_es_summary = pd.DataFrame(rows)
    var_es_summary_path = out_dir / "tables" / "var_es_summary.csv"
    var_es_summary.to_csv(var_es_summary_path, index=False)

    # exceedance summary
    rows = []
    for model in models:
        for a in alphas:
            tag = int(round(a * 100))
            for bucket in ["calm", "mid", "stress", "all"]:
                if bucket == "all":
                    msk = np.ones(out_df.shape[0], dtype=bool)
                else:
                    msk = (out_df["bucket"].astype(str) == bucket).to_numpy(bool)
                ex = out_df.loc[msk, f"exceed{tag}_{model}"].to_numpy(float)
                ex = ex[np.isfinite(ex)]
                rows.append({
                    "bucket": bucket,
                    "model": model,
                    "alpha": a,
                    "n_obs": int(msk.sum()),
                    "exceed_rate": float(np.mean(ex)) if ex.size else np.nan,
                    "target_rate": float(1.0 - a),
                })
    exceed_summary = pd.DataFrame(rows)
    exceed_summary_path = out_dir / "tables" / "exceedance_summary.csv"
    exceed_summary.to_csv(exceed_summary_path, index=False)

    # coverage tests (Kupiec) per bucket/model/alpha
    rows = []
    for model in models:
        for a in alphas:
            tag = int(round(a * 100))
            p_exc = float(1.0 - a)
            for bucket in ["calm", "mid", "stress", "all"]:
                if bucket == "all":
                    msk = np.ones(out_df.shape[0], dtype=bool)
                else:
                    msk = (out_df["bucket"].astype(str) == bucket).to_numpy(bool)

                ex = out_df.loc[msk, f"exceed{tag}_{model}"].to_numpy(float)
                ex = ex[np.isfinite(ex)]
                nobs = int(ex.size)
                xhits = int(np.sum(ex > 0.5)) if nobs > 0 else 0
                rate = (xhits / nobs) if nobs > 0 else np.nan
                pval = kupiec_pvalue(nobs, xhits, p_exc) if nobs > 0 else np.nan

                rows.append({
                    "bucket": bucket,
                    "model": model,
                    "alpha": a,
                    "n_obs": nobs,
                    "hit_count": xhits,
                    "hit_rate": rate,
                    "target_rate": p_exc,
                    "kupiec_pvalue": pval,
                })

    coverage_tests = pd.DataFrame(rows)
    coverage_tests_path = out_dir / "tables" / "coverage_tests.csv"
    coverage_tests.to_csv(coverage_tests_path, index=False)

    # deltas summary vs a baseline (default thr_t)
    baseline = str(cfg.get("baseline_model", "thr_t"))
    rows = []
    for model in models:
        if model == baseline:
            continue
        for a in alphas:
            tag = int(round(a * 100))
            for bucket in ["calm", "mid", "stress", "all"]:
                if bucket == "all":
                    msk = np.ones(out_df.shape[0], dtype=bool)
                else:
                    msk = (out_df["bucket"].astype(str) == bucket).to_numpy(bool)
                v_m = out_df.loc[msk, f"VaR{tag}_{model}"].to_numpy(float)
                v_b = out_df.loc[msk, f"VaR{tag}_{baseline}"].to_numpy(float)
                e_m = out_df.loc[msk, f"ES{tag}_{model}"].to_numpy(float)
                e_b = out_df.loc[msk, f"ES{tag}_{baseline}"].to_numpy(float)
                dv = (v_m - v_b)
                de = (e_m - e_b)
                dv = dv[np.isfinite(dv)]
                de = de[np.isfinite(de)]
                rows.append({
                    "bucket": bucket,
                    "alpha": a,
                    "model": model,
                    "baseline": baseline,
                    "dVaR_mean": float(np.mean(dv)) if dv.size else np.nan,
                    "dES_mean": float(np.mean(de)) if de.size else np.nan,
                })
    deltas = pd.DataFrame(rows)
    deltas_path = out_dir / "tables" / "deltas_summary.csv"
    deltas.to_csv(deltas_path, index=False)

    # deltas quantiles vs baseline (tail-focused summary for H3)
    rows = []
    for model in models:
        if model == baseline:
            continue
        for a in alphas:
            tag = int(round(a * 100))
            for bucket in ["calm", "mid", "stress", "all"]:
                if bucket == "all":
                    msk = np.ones(out_df.shape[0], dtype=bool)
                else:
                    msk = (out_df["bucket"].astype(str) == bucket).to_numpy(bool)

                dv = out_df.loc[msk, f"VaR{tag}_{model}"].to_numpy(float) - out_df.loc[msk, f"VaR{tag}_{baseline}"].to_numpy(float)
                de = out_df.loc[msk, f"ES{tag}_{model}"].to_numpy(float) - out_df.loc[msk, f"ES{tag}_{baseline}"].to_numpy(float)

                sv = _qstats(dv)
                se = _qstats(de)

                rows.append({
                    "bucket": bucket,
                    "alpha": a,
                    "model": model,
                    "baseline": baseline,
                    "dVaR_n": int(sv["n"]),
                    "dVaR_mean": sv["mean"],
                    "dVaR_p50": sv["p50"],
                    "dVaR_p75": sv["p75"],
                    "dVaR_p90": sv["p90"],
                    "dVaR_p95": sv["p95"],
                    "dES_n": int(se["n"]),
                    "dES_mean": se["mean"],
                    "dES_p50": se["p50"],
                    "dES_p75": se["p75"],
                    "dES_p90": se["p90"],
                    "dES_p95": se["p95"],
                })

    deltas_quantiles = pd.DataFrame(rows)
    deltas_quantiles_path = out_dir / "tables" / "deltas_quantiles.csv"
    deltas_quantiles.to_csv(deltas_quantiles_path, index=False)

    # figures
    make_figs = bool(cfg.get("reporting", {}).get("figures", True))
    if make_figs:
        tdates = pd.to_datetime(out_df["date"])

        for a in alphas:
            tag = int(round(a * 100))
            # VaR
            plt.figure()
            for model in models:
                plt.plot(tdates, out_df[f"VaR{tag}_{model}"].to_numpy(float), label=model)
            plt.title(f"J7 VaR{tag} (loss units) — {out_name}")
            plt.tight_layout()
            plt.legend(loc="best", fontsize=8)
            plt.savefig(out_dir / "figures" / f"var_time_series_{tag}.png", dpi=150)
            plt.close()

            # ES
            plt.figure()
            for model in models:
                plt.plot(tdates, out_df[f"ES{tag}_{model}"].to_numpy(float), label=model)
            plt.title(f"J7 ES{tag} (loss units) — {out_name}")
            plt.tight_layout()
            plt.legend(loc="best", fontsize=8)
            plt.savefig(out_dir / "figures" / f"es_time_series_{tag}.png", dpi=150)
            plt.close()

        # delta example: ms_t vs thr_t on VaR99 if present
        if ("ms_t" in models) and ("thr_t" in models) and (0.99 in alphas):
            plt.figure()
            d = out_df["VaR99_ms_t"].to_numpy(float) - out_df["VaR99_thr_t"].to_numpy(float)
            plt.plot(tdates, d)
            plt.title("Δ VaR99 (ms_t - thr_t)")
            plt.tight_layout()
            plt.savefig(out_dir / "figures" / "delta_var99_ms_t_vs_thr_t.png", dpi=150)
            plt.close()

    # report.md (minimal, audit-friendly)
    rep = []
    rep.append(f"# J7 — Impact risk (VaR/ES) — {dataset_version}\n")
    rep.append("## Protocol\n")
    rep.append(f"- copula_run_name: {copula_run}\n")
    rep.append(f"- refit_every: {refit_every}\n")
    rep.append(f"- n_scenarios: {n_scenarios}\n")
    rep.append(f"- alphas: {alphas}\n")
    rep.append(f"- weights: BTC={w_btc}, ETH={w_eth}\n")
    rep.append(f"- stress indicator: x_rv from J6 predictions.csv; stress_q={stress_q}, calm_q={calm_q} (thresholds computed on TRAIN only)\n")
    rep.append("\n## Key tables\n")
    rep.append(f"- tables/var_es_summary.csv\n- tables/exceedance_summary.csv\n- tables/coverage_tests.csv\n- tables/deltas_summary.csv\n- tables/deltas_quantiles.csv\n")
    rep.append("\n## Notes\n")
    rep.append("- ms_t uses π_pred from J6 predictions (one-step-ahead). If MS not used or π unavailable -> THR fallback.\n")
    rep.append("- ms_gauss uses gauss π_pred if present; otherwise uses t π_pred as proxy (approximation), else THR fallback.\n")

    rep.append("\n## Calibration\n")
    rep.append("- See tables/coverage_tests.csv (Kupiec p-values). Very small p-values indicate VaR miscalibration (too conservative or too aggressive).\n")

    rep.append("\n## H3 (stress impact)\n")
    rep.append("- See tables/deltas_quantiles.csv. H3 expects stress buckets and upper quantiles (p90/p95) to show larger ΔVaR/ΔES versus baseline than calm/mid.\n")
    rep.append("\n## Audit\n")
    rep.append(f"- gauss_pi_available_rate: {gauss_pi_available_rate:.6f}\n")
    rep.append(f"- t_pi_available_rate: {t_pi_available_rate:.6f}\n")
    rep.append(f"- used_ms_gauss_rate: {used_ms_gauss_rate:.6f}\n")
    rep.append(f"- used_ms_t_rate: {used_ms_t_rate:.6f}\n")
    rep.append(f"- ms_gauss_mode: {ms_gauss_mode}\n")
    rep_path = out_dir / "report.md"
    rep_path.write_text("".join(rep))

    # provenance
    prov = {
        "config_path": str(Path(args.config)),
        "dataset_version": dataset_version,
        "copula_source": {
            "copula_markov_base_dir": str(copula_base),
            "copula_run_name": copula_run,
            "params_summary_csv": str(params_path),
            "predictions_csv": str(preds_path),
            "metrics_json": str(metrics_path),
        },
        "inputs": {
            "returns_path": returns_path,
        },
        "outputs": {
            "var_es_predictions.csv": str(pred_path),
            "report.md": str(rep_path),
            "coverage_tests.csv": str(coverage_tests_path),
            "deltas_quantiles.csv": str(deltas_quantiles_path),
        },
        "hashes": {
            "var_es_predictions.csv": _sha12(pred_path),
            "var_es_summary.csv": _sha12(var_es_summary_path),
            "exceedance_summary.csv": _sha12(exceed_summary_path),
            "coverage_tests.csv": _sha12(coverage_tests_path),
            "deltas_summary.csv": _sha12(deltas_path),
            "deltas_quantiles.csv": _sha12(deltas_quantiles_path),
            "report.md": _sha12(rep_path),
        },
    }
    prov["audit"] = {
        "gauss_pi_available_rate": gauss_pi_available_rate,
        "t_pi_available_rate": t_pi_available_rate,
        "used_ms_gauss_rate": used_ms_gauss_rate,
        "used_ms_t_rate": used_ms_t_rate,
        "ms_gauss_mode": ms_gauss_mode,
    }
    (out_dir / "provenance.json").write_text(json.dumps(prov, indent=2))

    # resolved config
    _write_yaml(out_dir / "config.resolved.yaml", cfg)

    print(f"[OK] J7 built impact risk VaR/ES: {dataset_version}")
    print(f"[OK] out_dir: {out_dir.resolve()}")
    print(f"[OK] var_es_predictions.csv hash={_sha12(pred_path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())