from __future__ import annotations

# NOTE: macOS + numpy/scipy BLAS/OpenMP stacks can segfault when too many threads are spawned
# (especially under VS Code / forked terminals). Set conservative defaults *before* importing numpy/scipy.
import os

for k, v in {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}.items():
    os.environ.setdefault(k, v)

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

# Headless-safe matplotlib backend (prevents Mac/VSCode backend segfaults, e.g. exit 139)
import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt

from hc_taildep.copulas import gaussian as gcop
from hc_taildep.copulas import student_t as tcop
from hc_taildep.eval.dm_test import dm_test

from hc_taildep.markov.ms_copula import (
    fit_ms_copula_train,
    score_ms_oos_forward,
    ms_diagnostics,
)


def _sha12(path: Path) -> str:
    h = hashlib.sha256(path.read_bytes()).hexdigest()
    return h[:12]



def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


# Helper: find first existing path (for auto-discovery)
def _first_existing(paths: list[str]) -> str:
    for p in paths:
        if Path(p).exists():
            return p
    raise FileNotFoundError("None of the candidate paths exist:\n" + "\n".join(paths))


def _discover_u_series_path(dataset_version: str, candidates: list[str]) -> str:
    """Deterministic discovery for u_series.

    Order:
      1) explicit shortlist `candidates`
      2) glob search under data/processed/<dataset_version>/ for '*u_series*.csv'
         (lexicographically sorted to be deterministic)

    Raises FileNotFoundError with an audit-friendly message.
    """
    # 1) shortlist
    for p in candidates:
        if Path(p).exists():
            return p

    # 2) fallback glob (deterministic)
    base = Path(f"data/processed/{dataset_version}")
    if base.exists():
        hits = sorted([str(pp) for pp in base.rglob("*u_series*.csv")])
        if hits:
            return hits[0]

        # Secondary, slightly broader pattern (still deterministic)
        hits2 = sorted([str(pp) for pp in base.rglob("*u_*.csv") if "pit" in str(pp).lower()])
        if hits2:
            return hits2[0]

        raise FileNotFoundError(
            "Cannot locate u_series CSV. Tried shortlist candidates and glob fallback.\n"
            "Shortlist candidates:\n"
            + "\n".join(candidates)
            + "\n\nGlob attempted under: "
            + str(base)
            + "\nPatterns: '*u_series*.csv' then '*u_*.csv' restricted to paths containing 'pit'."
        )

    raise FileNotFoundError(
        "Cannot locate u_series CSV because base directory does not exist: "
        + str(base)
        + "\nShortlist candidates:\n"
        + "\n".join(candidates)
    )


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
    for c in ["date", "Date", "datetime", "timestamp", "time", "ds", "ts_utc"]:
        if c in df.columns:
            return c
    for c in ["Unnamed: 0", "index"]:
        if c in df.columns:
            x = pd.to_datetime(df[c], errors="coerce", utc=False)
            if float(x.notna().mean()) >= 0.90:
                return c
    raise ValueError(f"Cannot find date column in columns={list(df.columns)[:20]}")


def _norm_date(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    """Merge-safe normalization.

    - For intraday data (e.g. 4h): keep full UTC timestamp string
      -> "YYYY-MM-DD HH:MM:SS+00:00"
    - For daily data: keep date only
      -> "YYYY-MM-DD"

    This prevents collapsing multiple bars from the same day into one key.
    """
    out = df.copy()
    t = pd.to_datetime(out[date_col], errors="coerce", utc=True)

    if t.isna().any():
        bad = int(t.isna().sum())
        raise ValueError(f"Cannot parse {bad} timestamps in column={date_col}")

    intraday = ((t.dt.hour != 0) | (t.dt.minute != 0) | (t.dt.second != 0)).any()

    if intraday:
        out["date"] = t.dt.strftime("%Y-%m-%d %H:%M:%S+00:00")
    else:
        out["date"] = t.dt.strftime("%Y-%m-%d")

    if date_col != "date":
        out = out.drop(columns=[date_col])

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
    rsq = r * r
    c = np.cumsum(np.where(np.isfinite(rsq), rsq, 0.0))
    for i in range(window - 1, r.size):
        s = c[i] - (c[i - window] if i >= window else 0.0)
        out[i] = np.sqrt(s)
    return out


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg_raw = _load_yaml(Path(args.config))
    cfg = _resolve_vars(cfg_raw)

    dataset_version = cfg["dataset_version"]
    inputs = cfg["inputs"]
    seed = int(cfg.get("seed", 123))
    refit_every = int(cfg.get("refit_every", 20))

    # stress (for init + diagnostics)
    stress = cfg["stress"]
    rv_window = int(stress["window"])
    stress_q = float(stress["stress_q"])
    calm_q = float(stress["calm_q"])
    on_asset = str(stress.get("on_asset", "BTC"))

    # ms model
    ms = cfg["ms_model"]
    K = int(ms.get("K", 2))
    ordering_key = str(ms.get("ordering_key", "rho"))
    min_state_eff_n = int(ms.get("min_state_eff_n", 150))
    min_state_eff_n_t = int(ms.get("min_state_eff_n_t", min_state_eff_n))
    em = ms["em"]
    max_iter = int(em.get("max_iter", 50))
    tol = float(em.get("tol", 1e-6))
    init_A = np.asarray(em.get("init_A", [[0.98, 0.02], [0.02, 0.98]]), dtype=float)

    # copula fit
    tc_cfg = cfg["tcopula"]
    nu_grid = list(tc_cfg["nu_grid"])
    nu_bounds = tuple(tc_cfg["nu_bounds"])
    rho_clamp = float(tc_cfg.get("rho_clamp", 1e-6))

    dm_cfg = cfg.get("dm_test", {"enabled": True, "nw_lag_rule": "4*(n/100)^(2/9)"})
    dm_enabled = bool(dm_cfg.get("enabled", True))
    nw_lag_rule = str(dm_cfg.get("nw_lag_rule", "4*(n/100)^(2/9)"))

    rep = cfg.get("reporting", {})
    make_figs = bool(rep.get("figures", True))

    # Output directory tag (so sweeps don't overwrite each other)
    # Canonical key is `out_name` (preferred). Keep backward-compat with `out_tag`.
    out_name_raw = cfg.get("out_name", None)
    if out_name_raw is None:
        out_name_raw = cfg.get("out_tag", None)
    out_name = str(out_name_raw if out_name_raw is not None else "j6_ms2").strip()
    if not out_name:
        out_name = "j6_ms2"

    out_dir = Path(f"data/processed/{dataset_version}/copulas/markov/{out_name}")
    _ensure_dir(out_dir)
    _ensure_dir(out_dir / "tables")
    _ensure_dir(out_dir / "figures")

    # Load inputs
    returns_path = str(inputs["returns_path"])
    returns = pd.read_csv(returns_path)

    u_series_path = str(inputs.get("u_series_path", ""))
    if (not u_series_path) or (not Path(u_series_path).exists()):
        # Auto-discovery (audit-friendly deterministic shortlist)
        candidates = [
            f"data/processed/{dataset_version}/pit/u_series.csv",
            f"data/processed/{dataset_version}/pit/u_series_daily.csv",
            f"data/processed/{dataset_version}/pit/u_series_pit.csv",
            f"data/processed/{dataset_version}/u_series.csv",
            f"data/processed/{dataset_version}/pit/pit_u_series.csv",
        ]
        u_series_path = _discover_u_series_path(dataset_version, candidates)
        inputs["u_series_path"] = u_series_path  # persist resolved path for provenance/config

    u_series = pd.read_csv(u_series_path)

    returns = _norm_date(returns, _detect_date_col(returns))
    u_series = _norm_date(u_series, _detect_date_col(u_series))

    # pick columns
    def _pick_ret(df: pd.DataFrame, asset: str) -> str:
        cands = [f"r_{asset}", f"ret_{asset}", f"return_{asset}", f"{asset}_ret", f"{asset}_return", asset]
        for c in cands:
            if c in df.columns and c != "date":
                return c
        num_cols = [c for c in df.columns if c != "date" and np.issubdtype(df[c].dtype, np.number)]
        if len(num_cols) >= 2:
            return num_cols[0] if asset == "BTC" else num_cols[1]
        raise ValueError(f"Cannot find returns column for {asset}. cols={list(df.columns)}")

    def _pick_u(df: pd.DataFrame, asset: str) -> str:
        cands = [f"u_{asset}", f"U_{asset}", asset]
        for c in cands:
            if c in df.columns:
                return c
        raise ValueError(f"Cannot find u column for {asset}. cols={list(df.columns)}")

    rbtc_col = _pick_ret(returns, "BTC")
    reth_col = _pick_ret(returns, "ETH")
    ubtc_col = _pick_u(u_series, "BTC")
    ueth_col = _pick_u(u_series, "ETH")

    df = returns.merge(u_series[["date", ubtc_col, ueth_col]], on="date", how="inner", validate="one_to_one")
    df = df.sort_values("date").reset_index(drop=True)

    u = df[ubtc_col].to_numpy(float)
    v = df[ueth_col].to_numpy(float)

    # --- NUMERICAL SAFETY (copulas) ---
    # PIT values too close to 0/1 can make (especially) t-copula logpdf blow up.
    # Keep this configurable for audits.
    u_clip_eps = float(cfg.get("u_clip_eps", 1e-10))
    u = np.clip(u, u_clip_eps, 1.0 - u_clip_eps)
    v = np.clip(v, u_clip_eps, 1.0 - u_clip_eps)

    x = _rv_series(df[rbtc_col].to_numpy(float), window=rv_window)

    n = df.shape[0]

    # scores
    logc_static_t = np.full(n, np.nan)
    logc_static_g = np.full(n, np.nan)
    logc_thr_t = np.full(n, np.nan)
    logc_thr_g = np.full(n, np.nan)
    logc_ms_t = np.full(n, np.nan)
    logc_ms_g = np.full(n, np.nan)
    pi_pred_1 = np.full(n, np.nan)
    pi_pred_2 = np.full(n, np.nan)
    pi_filt_1 = np.full(n, np.nan)
    pi_filt_2 = np.full(n, np.nan)
    pi_pred_g1 = np.full(n, np.nan)
    pi_pred_g2 = np.full(n, np.nan)
    pi_filt_g1 = np.full(n, np.nan)
    pi_filt_g2 = np.full(n, np.nan)

    used_ms_gauss = np.zeros(n, dtype=int)  # 1 if MS gauss used, else fallback threshold
    used_ms_t = np.zeros(n, dtype=int)      # 1 if MS t used, else fallback threshold
    used_ms_t_mode = np.zeros(n, dtype=int)  # 2=MS2 used, 1=THR fallback, 0=unscored/invalid

    params_rows: list[dict[str, Any]] = []

    # refit schedule
    refit_indices = list(range(0, n, refit_every))
    for ridx in refit_indices:
        train_end = ridx - 1
        if train_end < 250:
            continue

        tr = slice(0, train_end + 1)
        u_tr = u[tr]
        v_tr = v[tr]
        x_tr = x[tr]

        # baseline static fit on train
        rho_g_static = float(gcop.fit(u_tr[np.isfinite(u_tr) & np.isfinite(v_tr)], v_tr[np.isfinite(u_tr) & np.isfinite(v_tr)], rho_clamp=rho_clamp))
        p_static = tcop.fit(
            u_tr[np.isfinite(u_tr) & np.isfinite(v_tr)],
            v_tr[np.isfinite(u_tr) & np.isfinite(v_tr)],
            nu_grid=nu_grid,
            nu_bounds=nu_bounds,
            rho_clamp=rho_clamp,
        )
        theta_t_static = {"rho": float(p_static.rho), "nu": float(p_static.nu)}

        # threshold “observable” (J5-style) for baseline comparison inside J6 runner
        thr_stress = _quantile(x_tr, stress_q)
        thr_calm = _quantile(x_tr, calm_q)
        m_calm_tr = np.isfinite(x_tr) & (x_tr <= thr_calm)
        m_stress_tr = np.isfinite(x_tr) & (x_tr >= thr_stress)

        # threshold gauss
        if m_calm_tr.sum() >= 200:
            rho_g_calm = float(gcop.fit(u_tr[m_calm_tr], v_tr[m_calm_tr], rho_clamp=rho_clamp))
        else:
            rho_g_calm = rho_g_static
        if m_stress_tr.sum() >= 200:
            rho_g_stress = float(gcop.fit(u_tr[m_stress_tr], v_tr[m_stress_tr], rho_clamp=rho_clamp))
        else:
            rho_g_stress = rho_g_static

        # threshold t
        if m_calm_tr.sum() >= 200:
            pc = tcop.fit(u_tr[m_calm_tr], v_tr[m_calm_tr], nu_grid=nu_grid, nu_bounds=nu_bounds, rho_clamp=rho_clamp)
            theta_t_calm = {"rho": float(pc.rho), "nu": float(pc.nu)}
        else:
            theta_t_calm = dict(theta_t_static)
        if m_stress_tr.sum() >= 200:
            ps = tcop.fit(u_tr[m_stress_tr], v_tr[m_stress_tr], nu_grid=nu_grid, nu_bounds=nu_bounds, rho_clamp=rho_clamp)
            theta_t_stress = {"rho": float(ps.rho), "nu": float(ps.nu)}
        else:
            theta_t_stress = dict(theta_t_static)

        # MS fit on train (gauss + t)
        fit_g = fit_ms_copula_train(
            u_tr, v_tr, x_tr,
            family="gauss", K=K, init_A=init_A,
            rho_clamp=rho_clamp,
            nu_grid=nu_grid, nu_bounds=nu_bounds,
            calm_q=calm_q, stress_q=stress_q,
            max_iter=max_iter, tol=tol,
            min_state_eff_n=min_state_eff_n,
            ordering_key=ordering_key,
            seed=seed,
        )
        fit_t = fit_ms_copula_train(
            u_tr, v_tr, x_tr,
            family="t", K=K, init_A=init_A,
            rho_clamp=rho_clamp,
            nu_grid=nu_grid, nu_bounds=nu_bounds,
            calm_q=calm_q, stress_q=stress_q,
            max_iter=max_iter, tol=tol,
            min_state_eff_n=min_state_eff_n_t,
            ordering_key=ordering_key,
            seed=seed,
        )

        # score block
        next_ridx = min(n, ridx + refit_every)
        blk = slice(ridx, next_ridx)

        u_blk = u[blk]
        v_blk = v[blk]
        x_blk = x[blk]
        S_blk = np.isfinite(x_blk) & (x_blk >= thr_stress)  # threshold computed on train

        # static scores
        m_ok = np.isfinite(u_blk) & np.isfinite(v_blk)

        # defaults (for audit row even if this block has no valid u/v)
        used_ms_gauss_blk = False
        used_ms_t_blk = False
        if m_ok.any():
            logc_static_g[blk] = gcop.logpdf(u_blk, v_blk, rho_g_static, rho_clamp=rho_clamp)
            logc_static_t[blk] = tcop.logpdf(u_blk, v_blk, theta_t_static["rho"], theta_t_static["nu"])

            # threshold gating
            # choose state based on S_blk
            out_thr_g = np.full(u_blk.size, np.nan)
            out_thr_t = np.full(u_blk.size, np.nan)
            for j in range(u_blk.size):
                if not (np.isfinite(u_blk[j]) and np.isfinite(v_blk[j]) and np.isfinite(x_blk[j])):
                    continue
                if bool(S_blk[j]):
                    out_thr_g[j] = gcop.logpdf(np.array([u_blk[j]]), np.array([v_blk[j]]), rho_g_stress, rho_clamp=rho_clamp)[0]
                    out_thr_t[j] = tcop.logpdf(np.array([u_blk[j]]), np.array([v_blk[j]]), theta_t_stress["rho"], theta_t_stress["nu"])[0]
                else:
                    out_thr_g[j] = gcop.logpdf(np.array([u_blk[j]]), np.array([v_blk[j]]), rho_g_calm, rho_clamp=rho_clamp)[0]
                    out_thr_t[j] = tcop.logpdf(np.array([u_blk[j]]), np.array([v_blk[j]]), theta_t_calm["rho"], theta_t_calm["nu"])[0]
            logc_thr_g[blk] = out_thr_g
            logc_thr_t[blk] = out_thr_t

            # --- MS forward-only mixture scores (within block) ---
            # Deterministic safety: if fit is degenerate, fallback to threshold scores.
            def _fit_ok(fit_obj, *, min_eff: int) -> bool:
                if getattr(fit_obj, "fit_status", None) != "ok":
                    return False
                ll = float(getattr(fit_obj, "ll_train", np.nan))
                if not np.isfinite(ll):
                    return False
                n_eff = np.asarray(getattr(fit_obj, "n_eff", []), dtype=float)
                if n_eff.size != K:
                    return False
                if np.any(~np.isfinite(n_eff)):
                    return False
                if np.any(n_eff < float(min_eff)):
                    return False
                A_ = np.asarray(getattr(fit_obj, "A", np.nan), dtype=float)
                if A_.shape != (K, K):
                    return False
                if not np.all(np.isfinite(A_)):
                    return False
                # Guard against near-zero rows that would explode logA
                row_sums = A_.sum(axis=1)
                if np.any(row_sums <= 0.0):
                    return False
                return True

            used_ms_gauss_blk = _fit_ok(fit_g, min_eff=min_state_eff_n)
            used_ms_t_blk = _fit_ok(fit_t, min_eff=min_state_eff_n_t)

            # NEW: write per-date flags for audit (block-level decision)
            # Rows with invalid u/v are still kept as 0; downstream masks already gate on finite scores.
            used_ms_gauss[blk] = 1 if used_ms_gauss_blk else 0
            used_ms_t[blk] = 1 if used_ms_t_blk else 0

            # t-mode audit: 2=MS2, 1=MS1 fallback (still fully OOS and deterministic)
            used_ms_t_mode[blk] = 2 if used_ms_t_blk else 1

            # GAUSS MS
            if used_ms_gauss_blk:
                pi_pred_g, pi_filt_g, logp_g = score_ms_oos_forward(
                    u_blk,
                    v_blk,
                    family="gauss",
                    A=fit_g.A,
                    pi0=fit_g.pi0,
                    theta=fit_g.theta,
                    rho_clamp=rho_clamp,
                )
                logc_ms_g[blk] = logp_g

                pi_pred_g1[blk] = pi_pred_g[:, 0]
                pi_pred_g2[blk] = pi_pred_g[:, 1]
                pi_filt_g1[blk] = pi_filt_g[:, 0]
                pi_filt_g2[blk] = pi_filt_g[:, 1]

            else:
                # fallback: same as threshold model (no claim of improvement)
                logc_ms_g[blk] = logc_thr_g[blk]


                pi_pred_g1[blk] = np.nan
                pi_pred_g2[blk] = np.nan
                pi_filt_g1[blk] = np.nan
                pi_filt_g2[blk] = np.nan

            # T MS
            if used_ms_t_blk:
                pi_pred_t, pi_filt_t, logp_t = score_ms_oos_forward(
                    u_blk,
                    v_blk,
                    family="t",
                    A=fit_t.A,
                    pi0=fit_t.pi0,
                    theta=fit_t.theta,
                    rho_clamp=rho_clamp,
                )
                logc_ms_t[blk] = logp_t

                # store pi (canonical: t MS)
                pi_pred_1[blk] = pi_pred_t[:, 0]
                pi_pred_2[blk] = pi_pred_t[:, 1]
                pi_filt_1[blk] = pi_filt_t[:, 0]
                pi_filt_2[blk] = pi_filt_t[:, 1]
            else:
                # fallback: use the threshold t-model (train-only thresholds).
                # Empirically, thr_t dominates static_t in these degenerate/short regimes.
                # Policy becomes: “MS2 if healthy else THR”, fully deterministic and OOS-safe.
                logc_ms_t[blk] = logc_thr_t[blk]

                # pi for 2-state MS is undefined in fallback mode
                pi_pred_1[blk] = np.nan
                pi_pred_2[blk] = np.nan
                pi_filt_1[blk] = np.nan
                pi_filt_2[blk] = np.nan

        params_rows.append(
            {
                "refit_index": int(ridx),
                "refit_date": df.loc[ridx, "date"],
                "train_end_date": df.loc[train_end, "date"],
                "n_train": int(train_end + 1),

                "thr_stress": float(thr_stress),
                "thr_calm": float(thr_calm),

                "static_rho_g": float(rho_g_static),
                "static_t_rho": float(theta_t_static["rho"]),
                "static_t_nu": float(theta_t_static["nu"]),

                "thr_rho_g_calm": float(rho_g_calm),
                "thr_rho_g_stress": float(rho_g_stress),
                "thr_t_rho_calm": float(theta_t_calm["rho"]),
                "thr_t_nu_calm": float(theta_t_calm["nu"]),
                "thr_t_rho_stress": float(theta_t_stress["rho"]),
                "thr_t_nu_stress": float(theta_t_stress["nu"]),

                "ms_gauss_fit_status": fit_g.fit_status,
                "ms_gauss_used": bool(used_ms_gauss_blk),
                "ms_t_fit_status": fit_t.fit_status,
                "ms_t_used": bool(used_ms_t_blk),
                "ms_t_mode": 2 if used_ms_t_blk else 1,

                "ms_gauss_A11": float(fit_g.A[0, 0]),
                "ms_gauss_A22": float(fit_g.A[1, 1]),
                "ms_gauss_rho1": float(fit_g.theta[0]["rho"]),
                "ms_gauss_rho2": float(fit_g.theta[1]["rho"]),
                "ms_gauss_n_eff1": float(fit_g.n_eff[0]),
                "ms_gauss_n_eff2": float(fit_g.n_eff[1]),
                "ms_gauss_ll": float(fit_g.ll_train),

                "ms_t_A11": float(fit_t.A[0, 0]),
                "ms_t_A22": float(fit_t.A[1, 1]),
                "ms_t_rho1": float(fit_t.theta[0]["rho"]),
                "ms_t_nu1": float(fit_t.theta[0]["nu"]),
                "ms_t_rho2": float(fit_t.theta[1]["rho"]),
                "ms_t_nu2": float(fit_t.theta[1]["nu"]),
                "ms_t_n_eff1": float(fit_t.n_eff[0]),
                "ms_t_n_eff2": float(fit_t.n_eff[1]),
                "ms_t_ll": float(fit_t.ll_train),

                "ordering_key": ordering_key,
            }
        )

    # predictions
    out = pd.DataFrame(
        {
            "date": df["date"].astype(str),
            "u_BTC": u,
            "u_ETH": v,
            "x_rv": x,

            "logc_static_gauss": logc_static_g,
            "logc_static_t": logc_static_t,
            "logc_thr_gauss": logc_thr_g,
            "logc_thr_t": logc_thr_t,
            "logc_ms_gauss": logc_ms_g,
            "logc_ms_t": logc_ms_t,

            "used_ms_gauss": used_ms_gauss,
            "used_ms_t": used_ms_t,
            "used_ms_t_mode": used_ms_t_mode,

            "pi_pred_state1": pi_pred_1,
            "pi_pred_state2": pi_pred_2,
            "pi_filt_state1": pi_filt_1,
            "pi_filt_state2": pi_filt_2,

            "pi_pred_gauss_state1": pi_pred_g1,
            "pi_pred_gauss_state2": pi_pred_g2,
            "pi_filt_gauss_state1": pi_filt_g1,
            "pi_filt_gauss_state2": pi_filt_g2,
        }
    )

    pred_path = out_dir / "predictions.csv"
    out.to_csv(pred_path, index=False)

    params_df = pd.DataFrame(params_rows)
    params_path = out_dir / "tables" / "params_summary.csv"
    params_df.to_csv(params_path, index=False)

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

    def _summ_mask(xarr: np.ndarray, mask: np.ndarray) -> dict[str, float]:
        xarr = np.asarray(xarr, dtype=float)
        mask = np.asarray(mask, dtype=bool)
        x = xarr[mask]
        x = x[np.isfinite(x)]
        if x.size == 0:
            return {"n_obs": 0, "sum": np.nan, "mean": np.nan, "std": np.nan, "min": np.nan, "max": np.nan}
        return {
            "n_obs": int(x.size),
            "sum": float(np.sum(x)),
            "mean": float(np.mean(x)),
            "std": float(np.std(x, ddof=1)) if x.size > 1 else 0.0,
            "min": float(np.min(x)),
            "max": float(np.max(x)),
        }

    # masks for interpretable comparisons
    mask_uv = np.isfinite(u) & np.isfinite(v)
    mask_all_models = (
        mask_uv
        & np.isfinite(logc_static_g)
        & np.isfinite(logc_static_t)
        & np.isfinite(logc_thr_g)
        & np.isfinite(logc_thr_t)
        & np.isfinite(logc_ms_g)
        & np.isfinite(logc_ms_t)
    )

    # policy usage masks
    mask_ms_gauss_used = mask_all_models & (used_ms_gauss.astype(bool))
    mask_ms_t_used = mask_all_models & (used_ms_t.astype(bool))
    mask_ms_t_ms1 = mask_all_models & (used_ms_t_mode == 1)

    # fallback rates computed on the intersection where things are scoreable
    denom = int(mask_all_models.sum())
    ms2 = float(mask_ms_t_used.sum() / denom) if denom > 0 else np.nan
    ms1 = float(mask_ms_t_ms1.sum() / denom) if denom > 0 else np.nan
    ms_unscored = float(1.0 - ms2 - ms1) if denom > 0 else np.nan
    fallback_rates = {
        "denom_n": denom,
        "ms_gauss_used_rate": float(mask_ms_gauss_used.sum() / denom) if denom > 0 else np.nan,
        "ms_gauss_fallback_rate": float(1.0 - (mask_ms_gauss_used.sum() / denom)) if denom > 0 else np.nan,
        "ms_t_ms2_used_rate": ms2,
        # backward-compat: ms_t_ms1_rate now means “thr_fallback” (mode==1)
        "ms_t_ms1_rate": ms1,
        "ms_t_thr_fallback_rate": ms1,
        "ms_t_unscored_rate": ms_unscored,
    }

    per_model_ms_only = {
        "ms_gauss_on_used": _summ_mask(logc_ms_g, mask_ms_gauss_used),
        "thr_gauss_on_ms_gauss_used": _summ_mask(logc_thr_g, mask_ms_gauss_used),
        "ms_t_on_used": _summ_mask(logc_ms_t, mask_ms_t_used),
        "thr_t_on_ms_t_used": _summ_mask(logc_thr_t, mask_ms_t_used),
        # mode==1 is “thr_fallback” (logc_ms_t copies thr_t)
        "ms_t_thr_fallback": _summ_mask(logc_ms_t, mask_ms_t_ms1),
        "thr_t_on_ms_t_thr_fallback": _summ_mask(logc_thr_t, mask_ms_t_ms1),
        "static_t_on_ms_t_thr_fallback": _summ_mask(logc_static_t, mask_ms_t_ms1),
        # backward-compat aliases
        "ms_t_ms1": _summ_mask(logc_ms_t, mask_ms_t_ms1),
        "thr_t_on_ms_t_ms1": _summ_mask(logc_thr_t, mask_ms_t_ms1),
        "static_t_on_ms_t_ms1": _summ_mask(logc_static_t, mask_ms_t_ms1),
    }

    metrics = {
        "dataset_version": dataset_version,
        "schema_version": "j6_ms_v1",
        "u_clip_eps": u_clip_eps,
        "refit_every": refit_every,
        "out_name": out_name,
        "out_tag": out_name,  # backward-compat alias
        "stress": {"type": "RV", "on_asset": on_asset, "window": rv_window, "stress_q": stress_q, "calm_q": calm_q},
        "ms_model": {
            "K": K,
            "ordering_key": ordering_key,
            "min_state_eff_n": min_state_eff_n,
            "min_state_eff_n_t": min_state_eff_n_t,
            "em": {"max_iter": max_iter, "tol": tol, "init_A": init_A.tolist()},
        },
        "tcopula": {"nu_grid": nu_grid, "nu_bounds": list(nu_bounds), "rho_clamp": rho_clamp},
        "per_model": {
            "static_gauss": _summ_mask(logc_static_g, mask_all_models),
            "static_t": _summ_mask(logc_static_t, mask_all_models),
            "thr_gauss": _summ_mask(logc_thr_g, mask_all_models),
            "thr_t": _summ_mask(logc_thr_t, mask_all_models),
            "ms_gauss": _summ_mask(logc_ms_g, mask_all_models),
            "ms_t": _summ_mask(logc_ms_t, mask_all_models),
        },
        "fallback_rates": fallback_rates,
        "per_model_ms_only": per_model_ms_only,
        "hashes": {"predictions_csv": _sha12(pred_path), "params_summary_csv": _sha12(params_path)},
    }

    # DM tests: core comparisons
    dm_out = {"schema_version": "j6_dm_v1", "nw_lag_rule": nw_lag_rule, "results": [], "comparisons": []}
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

        _do_dm("ms_t_vs_thr_t", logc_ms_t, logc_thr_t)
        _do_dm("ms_t_vs_static_t", logc_ms_t, logc_static_t)
        _do_dm("ms_gauss_vs_thr_gauss", logc_ms_g, logc_thr_g)
        _do_dm("ms_gauss_vs_static_gauss", logc_ms_g, logc_static_g)

        # NEW: DM tests restricted to periods where MS was actually used (MS-only)
        def _do_dm_masked(name: str, a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> None:
            mask = np.asarray(mask, dtype=bool)
            aa = np.asarray(a, dtype=float)[mask]
            bb = np.asarray(b, dtype=float)[mask]
            d = aa - bb
            res = dm_test(d, nw_lag_rule=nw_lag_rule, alternative="two-sided")
            row = {"comparison": name, **asdict(res)}
            dm_out["results"].append(row)
            dm_out["comparisons"].append({"name": name, **asdict(res)})
            dm_rows.append(
                {
                    "name": name,
                    "n_obs": res.n_obs,
                    "nw_lag": res.nw_lag,
                    "mean_delta": res.mean_delta,
                    "std_delta": res.std_delta,
                    "dm_stat": res.dm_stat,
                    "pvalue": res.pvalue,
                }
            )

        _do_dm_masked("ms_t_vs_thr_t__ms_only", logc_ms_t, logc_thr_t, mask_ms_t_used)
        _do_dm_masked("ms_t_vs_thr_t__thr_fallback", logc_ms_t, logc_thr_t, mask_ms_t_ms1)
        _do_dm_masked("ms_t_vs_thr_t__ms1", logc_ms_t, logc_thr_t, mask_ms_t_ms1)  # backward-compat
        _do_dm_masked("ms_gauss_vs_thr_gauss__ms_only", logc_ms_g, logc_thr_g, mask_ms_gauss_used)

    dm_path = out_dir / "dm_test.json"
    dm_path.write_text(json.dumps(dm_out, indent=2))

    dm_df = pd.DataFrame(dm_rows)
    dm_summary_path = out_dir / "tables" / "dm_summary.csv"
    dm_df.to_csv(dm_summary_path, index=False)

    scores_summary = pd.DataFrame([{"model": k, **v} for k, v in metrics["per_model"].items()])
    scores_summary_path = out_dir / "tables" / "scores_summary.csv"
    scores_summary.to_csv(scores_summary_path, index=False)

    metrics_path = out_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))

    # figures
    if make_figs:
        # cumulative delta vs thr (t)
        d = (logc_ms_t - logc_thr_t).copy()
        d[~np.isfinite(d)] = 0.0
        cum = np.cumsum(d)
        plt.figure()
        plt.plot(pd.to_datetime(out["date"]), cum)
        plt.title("Cumulative Δ logscore (ms_t - thr_t)")
        plt.tight_layout()
        plt.savefig(out_dir / "figures" / "delta_cum_ms_vs_thr_t.png", dpi=150)
        plt.close()

        # pi timeseries (pred)
        plt.figure()
        plt.plot(pd.to_datetime(out["date"]), out["pi_pred_state2"].to_numpy(float))
        plt.title("Predicted prob state2 (π_{t|t-1}) — t MS")
        plt.tight_layout()
        plt.savefig(out_dir / "figures" / "pi_pred_state2.png", dpi=150)
        plt.close()

        # hist delta
        dd = (logc_ms_t - logc_thr_t)
        dd = dd[np.isfinite(dd)]
        plt.figure()
        plt.hist(dd, bins=50)
        plt.title("Δ logscore distribution (ms_t - thr_t)")
        plt.tight_layout()
        plt.savefig(out_dir / "figures" / "delta_hist_ms_vs_thr_t.png", dpi=150)
        plt.close()

    # report
    report = []
    report.append(f"# J6 — Markov-switching copulas (K=2) — {dataset_version}\n")
    report.append("## Protocol OOS (anti-leakage)\n")
    report.append(f"- Refit every {refit_every} days.\n")
    report.append("- At refit R_k: fit on train dates ≤ R_k-1 only.\n")
    report.append("- OOS scoring: forward-only filtering within each block; NO smoothing (no future info).\n")
    report.append("- Initial π0 for each block: stationary distribution of fitted A (auditable, strict).\n")
    report.append("\n## Stress RV used only for initialization/diagnostics\n")
    report.append(f"- x_t = RV on {on_asset}, window={rv_window}\n")
    report.append(f"- Warm-start thresholds: calm <= q{calm_q}, stress >= q{stress_q} (train only)\n")
    report.append("\n## Results (logscore summaries)\n")
    report.append("\n\n## Policy audit (MS usage / fallback)\n")
    report.append(json.dumps(fallback_rates, indent=2))
    report.append("\n\n## MS-only summaries (restricted to dates where MS was used)\n")
    ms_only_df = pd.DataFrame([{"model": k, **v} for k, v in per_model_ms_only.items()])
    report.append(ms_only_df.to_string(index=False))
    report.append(scores_summary.to_string(index=False))
    report.append("\n\n## DM tests (HAC Newey–West)\n")
    report.append(dm_df.to_string(index=False) if dm_enabled else "DM disabled")
    report.append("\n\n## Files\n")
    report.append(f"- predictions.csv hash={_sha12(pred_path)}\n")
    report.append(f"- metrics.json hash={_sha12(metrics_path)}\n")
    report.append(f"- dm_test.json hash={_sha12(dm_path)}\n")
    (out_dir / "report.md").write_text("\n".join(report))

    _write_yaml(out_dir / "config.resolved.yaml", cfg)
    prov = {
        "config_path": str(Path(args.config)),
        "out_name": out_name,
        "out_tag": out_name,  # backward-compat alias
        "inputs": {k: str(v) for k, v in inputs.items()},
        "outputs": {"predictions.csv": str(pred_path), "metrics.json": str(metrics_path), "dm_test.json": str(dm_path)},
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

    print(f"[OK] built J6 Markov-switching copulas: {dataset_version}")
    print(f"[OK] out_dir: {out_dir.resolve()}")
    print(f"[OK] predictions: predictions.csv hash={_sha12(pred_path)}")
    print(f"[OK] metrics: metrics.json hash={_sha12(metrics_path)}")
    if dm_enabled:
        print(f"[OK] dm_test: dm_test.json hash={_sha12(dm_path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())