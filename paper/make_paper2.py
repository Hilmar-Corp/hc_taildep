# paper/make_paper2.py
from __future__ import annotations

import argparse
import json
import re
import hashlib
import platform
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt


class MissingArtifactError(RuntimeError):
    pass


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _read_text(p: Path) -> str:
    return p.read_text(encoding="utf-8")


# --- SHA256 and stable JSON helpers for reproducibility manifest ---
def _sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_json_stable(obj: dict[str, Any], out_path: Path) -> None:
    _ensure_dir(out_path.parent)
    out_path.write_text(
        json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _load_csv(p: Path) -> pd.DataFrame:
    return pd.read_csv(p)


def _write_csv_stable(df: pd.DataFrame, out_path: Path, *, float_fmt: str = "%.6g", table_round: int = 6) -> None:
    out = df.copy()
    for c in out.columns:
        if pd.api.types.is_numeric_dtype(out[c]):
            out[c] = np.round(out[c].astype(float), int(table_round))
    _ensure_dir(out_path.parent)
    out.to_csv(out_path, index=False, float_format=float_fmt)


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()



def _as_utc_ts(x: Any) -> pd.Timestamp:
    return pd.to_datetime(x, utc=True, errors="raise")


# --- Inserted: load_splits ---
def load_splits(splits_json: Path) -> dict[str, Any]:
    """Load dataset split boundaries.

    Expected keys (UTC timestamps or ISO dates):
      - train_start
      - first_oos
      - last_oos

    Returns the raw dict, but validates presence and UTC-parsability.
    """
    if not splits_json.exists():
        raise MissingArtifactError(f"Missing splits.json: {splits_json}")
    try:
        obj = json.loads(_read_text(splits_json))
    except Exception as e:
        raise MissingArtifactError(f"Failed to parse splits.json: {splits_json} ({e})")

    need = ["train_start", "first_oos", "last_oos"]
    miss = [k for k in need if k not in obj]
    if miss:
        raise MissingArtifactError(f"splits.json missing keys {miss}. Have={list(obj.keys())}")

    # Validate timestamps are parseable as UTC-aware
    for k in need:
        _ = _as_utc_ts(obj[k])

    return obj


def _set_matplotlib_style(fig_w: float, fig_h: float) -> None:
    plt.rcParams["figure.figsize"] = (float(fig_w), float(fig_h))
    plt.rcParams["savefig.dpi"] = 200


@dataclass(frozen=True)
class StressInputs:
    stress_csv: Path
    splits_json: Path


def _infer_default_outputs_from_paper_id(paper_id: str) -> tuple[Path, Path]:
    out_dir = Path("paper") / "out" / paper_id
    return (out_dir / "tables", out_dir / "figures")


def load_stress_df(stress_csv: Path) -> pd.DataFrame:
    if not stress_csv.exists():
        raise MissingArtifactError(f"Missing stress CSV: {stress_csv} (expected tab_FX_stress_mask.csv from make_paper.py)")
    df = _load_csv(stress_csv)
    need = ["date", "S_t"]
    miss = [c for c in need if c not in df.columns]
    if miss:
        raise MissingArtifactError(f"stress CSV missing columns {miss}. Have={list(df.columns)[:30]}")
    df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    df["S_t"] = pd.to_numeric(df["S_t"], errors="coerce").fillna(0).astype(int)
    df["S_t"] = (df["S_t"] != 0).astype(int)
    return df


def _coerce_regime_to_S_t(s: pd.Series) -> pd.Series:
    # accepts 'stress'/'calm' (case-insensitive) or 1/0 already
    ss = s.astype(str).str.lower().str.strip()
    out = pd.Series(np.zeros(len(ss), dtype=int), index=ss.index)
    out[ss.isin(["1", "stress", "true", "yes"])] = 1
    out[ss.isin(["0", "calm", "false", "no"])] = 0
    # if it was numeric-like but not in the above, fallback to numeric coercion
    try:
        num = pd.to_numeric(s, errors="coerce")
        m = num.notna()
        out.loc[m] = (num.loc[m] != 0).astype(int)
    except Exception:
        pass
    return out


def load_regimes_df(regimes_csv: Path, *, regime_col: Optional[str] = None) -> pd.DataFrame:
    """\
    regimes.csv is the canonical daily regime sequence for M1 threshold gating.

    Expected columns (any one of):
      - ['date','S_t'] where S_t in {0,1}
      - ['date','regime'] where regime in {'calm','stress'} (case-insensitive)
      - boolean flags such as is_stress_JDown / is_calm_JDown, is_stress_RV / is_calm_RV

    Returns a DataFrame with columns ['date','S_t'] where date is UTC-aware and S_t in {0,1}.
    """
    if not regimes_csv.exists():
        raise MissingArtifactError(f"Missing regimes CSV: {regimes_csv}")
    df = _load_csv(regimes_csv)
    if "date" not in df.columns:
        raise MissingArtifactError(f"regimes CSV missing column 'date'. Have={list(df.columns)[:30]}")
    df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    # 1) canonical columns
    if "S_t" in df.columns:
        df["S_t"] = pd.to_numeric(df["S_t"], errors="coerce").fillna(0).astype(int)
        df["S_t"] = (df["S_t"] != 0).astype(int)
        return df[["date", "S_t"]]

    if "regime" in df.columns:
        df["S_t"] = _coerce_regime_to_S_t(df["regime"]).astype(int)
        return df[["date", "S_t"]]

    # 2) support boolean flags: is_stress_*
    cols = list(df.columns)
    stress_cols = [c for c in cols if re.match(r"^is_stress_", str(c))]

    # pick stress flag
    pick: Optional[str] = None
    if regime_col is not None and str(regime_col).strip():
        rc = str(regime_col).strip()
        if rc not in df.columns:
            raise MissingArtifactError(
                f"--regime-col '{rc}' not found in regimes CSV. Have={list(df.columns)[:30]}"
            )
        pick = rc
    else:
        # sensible default: prefer JDown if present, else first available stress flag
        if "is_stress_JDown" in df.columns:
            pick = "is_stress_JDown"
        elif stress_cols:
            pick = stress_cols[0]

    if pick is None:
        raise MissingArtifactError(
            "regimes CSV must contain either 'S_t' or 'regime', or at least one 'is_stress_*' column. "
            f"Have={list(df.columns)[:30]}"
        )

    # build S_t from chosen stress flag
    s_flag = df[pick]
    if pd.api.types.is_bool_dtype(s_flag):
        df["S_t"] = s_flag.fillna(False).astype(bool).astype(int)
    else:
        # accept True/False strings or 0/1
        df["S_t"] = _coerce_regime_to_S_t(s_flag).astype(int)

    # optional integrity check if matching calm flag exists
    suffix = pick.replace("is_stress_", "")
    calm_match = f"is_calm_{suffix}"
    if calm_match in df.columns:
        c_flag = df[calm_match]
        if pd.api.types.is_bool_dtype(c_flag):
            c_bool = c_flag.fillna(False).astype(bool)
        else:
            c_bool = c_flag.astype(str).str.lower().str.strip().isin(["1", "true", "yes", "calm"]) | (
                pd.to_numeric(c_flag, errors="coerce").fillna(0) != 0
            )
        conflict = (df["S_t"].astype(int) == 1) & (c_bool.astype(bool))
        if bool(conflict.any()):
            bad = df.loc[conflict, ["date", pick, calm_match]].head(5)
            raise MissingArtifactError(
                f"regimes CSV has conflicting stress/calm flags for {suffix}. Examples:\n{bad.to_string(index=False)}"
            )

    return df[["date", "S_t"]]


def table_m1_regime_sizes_fallback_by_block(
    stress_df: pd.DataFrame,
    splits: dict[str, Any],
    *,
    refit_every: int = 63,
    n_min: int = 50,
) -> pd.DataFrame:
    """
    Reviewer-facing audit table:
      For each refit block:
        - TRAIN regime sizes up to t_refit (exclusive)
        - fallback_used = 1 if low_power (n_calm_train<n_min OR n_stress_train<n_min)
        - fallback_reason (string)
    NOTE: pass the same regime sequence used by M1 (preferred: regimes.csv).
    """
    if stress_df.empty:
        raise MissingArtifactError("stress_df empty")

    train_start = _as_utc_ts(splits["train_start"])
    first_oos = _as_utc_ts(splits["first_oos"])
    last_oos = _as_utc_ts(splits["last_oos"])

    df = stress_df[(stress_df["date"] >= train_start) & (stress_df["date"] <= last_oos)].reset_index(drop=True)
    if df.empty:
        raise MissingArtifactError("stress_df empty after clipping to [train_start,last_oos]")

    step = int(refit_every)
    if step < 1:
        step = 1
    nmin = int(n_min)
    if nmin < 1:
        nmin = 1

    # We need a tz-naive datetime64[ns] axis for np.searchsorted,
    # otherwise numpy/pandas will throw tz-naive vs tz-aware comparison errors.
    # Keep UTC-aware timestamps for all subsequent boolean masks.
    dates64 = df["date"].dt.tz_convert(None).to_numpy(dtype="datetime64[ns]")
    n = len(df)

    # first refit index: first_oos location in stress_df (or 1)
    first_oos64 = first_oos.tz_convert(None).to_datetime64()
    i0 = int(np.searchsorted(dates64, first_oos64, side="left"))
    if i0 < 1:
        i0 = 1

    refit_idx = list(range(i0, n, step))
    if not refit_idx:
        refit_idx = [n - 1]
    elif refit_idx[-1] != n - 1:
        refit_idx.append(n - 1)

    rows: list[dict[str, Any]] = []
    prev_i = i0
    block_id = 0

    for i in refit_idx:
        # Use the original UTC-aware timestamps for masks and reporting
        t_refit = df["date"].iloc[i]

        # TRAIN-only at refit: all obs strictly before t_refit
        m_train = df["date"] < t_refit
        s_train = df.loc[m_train, "S_t"].to_numpy(dtype=int)
        n_train = int(s_train.size)
        n_stress_train = int((s_train == 1).sum())
        n_calm_train = int((s_train == 0).sum())

        low_power = (n_stress_train < nmin) or (n_calm_train < nmin)
        reasons = []
        if n_stress_train < nmin:
            reasons.append("n_stress_train<n_min")
        if n_calm_train < nmin:
            reasons.append("n_calm_train<n_min")
        reason_s = ";".join(reasons)

        block_start = df["date"].iloc[prev_i]
        block_end = t_refit
        m_block = (df["date"] >= block_start) & (df["date"] <= block_end)
        n_block = int(m_block.sum())

        rows.append(
            {
                "block_id": block_id,
                "t_refit": t_refit.isoformat(),
                "block_start": block_start.isoformat(),
                "block_end": block_end.isoformat(),
                "refit_every_obs": step,
                "n_min": nmin,
                "n_block_obs": n_block,
                "n_train_obs": n_train,
                "n_calm_train": n_calm_train,
                "n_stress_train": n_stress_train,
                "fallback_used": int(low_power),
                "fallback_reason": reason_s,
            }
        )

        prev_i = i
        block_id += 1

    out = pd.DataFrame(rows)
    return out


def fig_m1_fallback_rate_by_block(
    m1_tbl: pd.DataFrame,
    out_path: Path,
    *,
    fig_w: float,
    fig_h: float,
    dpi: int,
) -> None:
    if m1_tbl.empty:
        raise MissingArtifactError("m1_tbl empty")

    t = pd.to_datetime(m1_tbl["t_refit"], utc=True, errors="coerce")
    y = pd.to_numeric(m1_tbl["fallback_used"], errors="coerce").fillna(0).to_numpy(dtype=float)

    _set_matplotlib_style(fig_w, fig_h)
    plt.figure(figsize=(max(8.5, float(fig_w)), max(3.2, float(fig_h) * 0.85)))
    plt.step(t, y, where="post")
    plt.ylim(-0.05, 1.05)
    plt.yticks([0, 1], ["0", "1"])
    plt.title("M1 — fallback_used by refit block")
    plt.xlabel("refit time (UTC)")
    plt.ylabel("fallback_used")
    plt.grid(axis="y", linestyle=":", linewidth=1.0)
    _ensure_dir(out_path.parent)
    plt.tight_layout()
    plt.savefig(out_path, dpi=int(dpi))
    plt.close()


def table_m2_params_per_refit(params_csv: Path, stress_df: pd.DataFrame, splits: dict[str, Any], *, n_min: int = 200) -> pd.DataFrame:
    """
    Reviewer audit table for M2 (logistic gating): for each refit, show parameter values, regime sizes, and fallback info.
    """
    if not params_csv.exists():
        raise MissingArtifactError(f"Missing M2 params CSV: {params_csv}")
    df = _load_csv(params_csv)
    need_cols = [
        "refit_date", "train_end_date", "n_train",
        "theta_t_calm_rho", "theta_t_calm_nu", "theta_t_stress_rho", "theta_t_stress_nu",
        "a_best", "b_best", "logit_val_sum",
        "status_t_calm", "status_t_stress", "status_g_calm", "status_g_stress"
    ]
    miss = [c for c in need_cols if c not in df.columns]
    if miss:
        raise MissingArtifactError(f"M2 params CSV missing columns {miss}. Have={list(df.columns)[:30]}")
    # Parse refit_date as UTC, drop NA
    df = df.copy()
    df["refit_date"] = pd.to_datetime(df["refit_date"], utc=True, errors="coerce")
    df = df.dropna(subset=["refit_date"]).reset_index(drop=True)
    # Prepare stress_df for regime counting
    train_start = _as_utc_ts(splits["train_start"])
    last_oos = _as_utc_ts(splits["last_oos"])
    stress = stress_df[(stress_df["date"] >= train_start) & (stress_df["date"] <= last_oos)].copy()
    if stress.empty:
        raise MissingArtifactError("stress_df empty after clipping to [train_start,last_oos]")
    # For each refit_date, compute train regime sizes
    nmin = int(n_min)
    rows = []
    for i, row in df.iterrows():
        refit_date = row["refit_date"]
        train_end = refit_date
        m_train = (stress["date"] < refit_date)
        n_train_obs = int(m_train.sum())
        n_calm_train = int((stress.loc[m_train, "S_t"] == 0).sum())
        n_stress_train = int((stress.loc[m_train, "S_t"] == 1).sum())
        fallback_regime_used = int((n_calm_train < nmin) or (n_stress_train < nmin))
        fallback_ab_used = int((float(row["a_best"]) == 0 and float(row["b_best"]) == 0))
        reasons = []
        if n_stress_train < nmin:
            reasons.append("n_stress_train<n_min")
        if n_calm_train < nmin:
            reasons.append("n_calm_train<n_min")
        if fallback_ab_used:
            reasons.append("a_b_zero")
        fallback_reason = ";".join(reasons)
        out_row = {
            "refit_index": i,
            "refit_date": refit_date.isoformat(),
            "train_end_date": row["train_end_date"],
            "n_train": row["n_train"],
            "rho_calm": row["theta_t_calm_rho"],
            "nu_calm": row["theta_t_calm_nu"],
            "rho_stress": row["theta_t_stress_rho"],
            "nu_stress": row["theta_t_stress_nu"],
            "a_best": row["a_best"],
            "b_best": row["b_best"],
            "logit_val_sum": row["logit_val_sum"],
            "n_train_obs": n_train_obs,
            "n_calm_train": n_calm_train,
            "n_stress_train": n_stress_train,
            "fallback_regime_used": fallback_regime_used,
            "fallback_ab_used": fallback_ab_used,
            "fallback_reason": fallback_reason,
            "status_g_calm": row["status_g_calm"],
            "status_g_stress": row["status_g_stress"],
            "status_t_calm": row["status_t_calm"],
            "status_t_stress": row["status_t_stress"],
        }
        rows.append(out_row)
    out_cols = [
        "refit_index","refit_date","train_end_date","n_train",
        "rho_calm","nu_calm","rho_stress","nu_stress",
        "a_best","b_best","logit_val_sum",
        "n_train_obs","n_calm_train","n_stress_train",
        "fallback_regime_used","fallback_ab_used","fallback_reason",
        "status_g_calm","status_g_stress","status_t_calm","status_t_stress"
    ]
    out = pd.DataFrame(rows)[out_cols]
    return out


def fig_m2_wt_zt_oos(pred_csv: Path, splits: dict[str, Any], out_path: Path, *, fig_w: float, fig_h: float, dpi: int) -> None:
    """
    Plot z_t (z_rv) and w_t (w_logit) vs date on OOS. Shade stress periods.
    """
    if not pred_csv.exists():
        raise MissingArtifactError(f"Missing M2 predictions CSV: {pred_csv}")
    df = _load_csv(pred_csv)
    need_cols = ["date", "z_rv", "w_logit", "S_bin"]
    miss = [c for c in need_cols if c not in df.columns]
    if miss:
        raise MissingArtifactError(f"M2 predictions CSV missing columns {miss}. Have={list(df.columns)[:30]}")
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
    df = df.dropna(subset=["date"]).reset_index(drop=True)
    df["z_rv"] = pd.to_numeric(df["z_rv"], errors="coerce")
    df["w_logit"] = pd.to_numeric(df["w_logit"], errors="coerce")
    df = df[df["z_rv"].notna() & df["w_logit"].notna()]
    if df.empty:
        raise MissingArtifactError("No valid z_rv and w_logit rows for M2 figure.")
    first_oos = _as_utc_ts(splits["first_oos"])
    last_oos = _as_utc_ts(splits["last_oos"])
    df = df[(df["date"] >= first_oos) & (df["date"] <= last_oos)].reset_index(drop=True)
    if df.empty:
        raise MissingArtifactError("No OOS rows for M2 figure after clipping to [first_oos,last_oos].")
    # Plot
    _set_matplotlib_style(fig_w, fig_h)
    fig, ax = plt.subplots(figsize=(float(fig_w), float(fig_h)))
    ax.plot(df["date"], df["z_rv"], label="z_t (z_rv)", color="tab:blue")
    ax.set_ylabel("z_t (z_rv)", color="tab:blue")
    ax.grid(axis="y", linestyle=":", linewidth=1.0)
    ax2 = ax.twinx()
    ax2.plot(df["date"], df["w_logit"], label="w_t (w_logit)", color="tab:orange")
    ax2.set_ylabel("w_t (w_logit)", color="tab:orange")
    # Shade stress segments (S_bin==1) contiguous
    s = df["S_bin"].astype(int).values
    dates = df["date"].values
    in_stress = False
    start = None
    for i in range(len(s)):
        if s[i] == 1 and not in_stress:
            in_stress = True
            start = dates[i]
        elif (s[i] == 0 or i == len(s)-1) and in_stress:
            end = dates[i] if s[i]==0 else dates[i]
            ax.axvspan(start, end, color="red", alpha=0.15)
            in_stress = False
    # Title and layout
    ax.set_title("M2 — z_t and w_t on OOS (stress marked)")
    _ensure_dir(out_path.parent)
    fig.tight_layout()
    fig.savefig(out_path, dpi=int(dpi))
    plt.close(fig)


# --- F8: cumulative deltas and histogram ---
def fig_f8_cumdelta_logscores(
    pred_csv: Path,
    splits: dict[str, Any],
    out_path: Path,
    *,
    fig_w: float,
    fig_h: float,
    dpi: int,
) -> None:
    """Figure F8: cumulative sum of per-day logscore deltas.

    Plots:
      - \\sum_t (logc_logit_t - logc_static_t)
      - \\sum_t (logc_thr_t   - logc_static_t)

    Uses OOS window [first_oos, last_oos].
    """
    if not pred_csv.exists():
        raise MissingArtifactError(f"Missing predictions CSV for F8: {pred_csv}")

    df = _load_csv(pred_csv)
    need = ["date", "logc_static_t", "logc_thr_t", "logc_logit_t"]
    miss = [c for c in need if c not in df.columns]
    if miss:
        raise MissingArtifactError(f"predictions CSV missing columns {miss} for F8. Have={list(df.columns)[:30]}")

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    for c in ["logc_static_t", "logc_thr_t", "logc_logit_t"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    first_oos = _as_utc_ts(splits["first_oos"])
    last_oos = _as_utc_ts(splits["last_oos"])
    df = df[(df["date"] >= first_oos) & (df["date"] <= last_oos)].reset_index(drop=True)

    df = df.dropna(subset=["logc_static_t", "logc_thr_t", "logc_logit_t"]).reset_index(drop=True)
    if df.empty:
        raise MissingArtifactError("No valid OOS rows with logc_* columns for F8.")

    d_logit = (df["logc_logit_t"] - df["logc_static_t"]).to_numpy(dtype=float)
    d_thr = (df["logc_thr_t"] - df["logc_static_t"]).to_numpy(dtype=float)

    # Guard against inf
    m = np.isfinite(d_logit) & np.isfinite(d_thr)
    df = df.loc[m].reset_index(drop=True)
    if df.empty:
        raise MissingArtifactError("No finite delta rows for F8.")

    d_logit = (df["logc_logit_t"] - df["logc_static_t"]).to_numpy(dtype=float)
    d_thr = (df["logc_thr_t"] - df["logc_static_t"]).to_numpy(dtype=float)

    c_logit = np.cumsum(d_logit)
    c_thr = np.cumsum(d_thr)

    _set_matplotlib_style(fig_w, fig_h)
    plt.figure(figsize=(float(fig_w), float(fig_h)))
    plt.plot(df["date"], c_logit, label="cum Δ logit_t − static_t")
    plt.plot(df["date"], c_thr, label="cum Δ thr_t − static_t")
    plt.axhline(0.0, linestyle=":", linewidth=1.0)
    plt.title("F8 — cumulative logscore deltas vs static (OOS)")
    plt.xlabel("date (UTC)")
    plt.ylabel("cumulative Δ logscore")
    plt.grid(axis="y", linestyle=":", linewidth=1.0)
    plt.legend(loc="best")
    _ensure_dir(out_path.parent)
    plt.tight_layout()
    plt.savefig(out_path, dpi=int(dpi))
    plt.close()


def fig_f8_hist_delta_logscores(
    pred_csv: Path,
    splits: dict[str, Any],
    out_path: Path,
    *,
    fig_w: float,
    fig_h: float,
    dpi: int,
    bins: int = 60,
) -> None:
    """Optional F8 histogram: distribution of per-day logscore deltas."""
    if not pred_csv.exists():
        raise MissingArtifactError(f"Missing predictions CSV for F8 hist: {pred_csv}")

    df = _load_csv(pred_csv)
    need = ["date", "logc_static_t", "logc_thr_t", "logc_logit_t"]
    miss = [c for c in need if c not in df.columns]
    if miss:
        raise MissingArtifactError(f"predictions CSV missing columns {miss} for F8 hist. Have={list(df.columns)[:30]}")

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    for c in ["logc_static_t", "logc_thr_t", "logc_logit_t"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    first_oos = _as_utc_ts(splits["first_oos"])
    last_oos = _as_utc_ts(splits["last_oos"])
    df = df[(df["date"] >= first_oos) & (df["date"] <= last_oos)].reset_index(drop=True)

    df = df.dropna(subset=["logc_static_t", "logc_thr_t", "logc_logit_t"]).reset_index(drop=True)
    if df.empty:
        raise MissingArtifactError("No valid OOS rows with logc_* columns for F8 hist.")

    d_logit = (df["logc_logit_t"] - df["logc_static_t"]).to_numpy(dtype=float)
    d_thr = (df["logc_thr_t"] - df["logc_static_t"]).to_numpy(dtype=float)
    m = np.isfinite(d_logit) & np.isfinite(d_thr)
    d_logit = d_logit[m]
    d_thr = d_thr[m]
    if d_logit.size == 0:
        raise MissingArtifactError("No finite delta rows for F8 hist.")

    _set_matplotlib_style(fig_w, fig_h)
    plt.figure(figsize=(float(fig_w), float(fig_h)))
    plt.hist(d_logit, bins=int(bins), alpha=0.6, label="Δ logit_t − static_t")
    plt.hist(d_thr, bins=int(bins), alpha=0.6, label="Δ thr_t − static_t")
    plt.axvline(0.0, linestyle=":", linewidth=1.0)
    plt.title("F8 — histogram of per-day logscore deltas vs static (OOS)")
    plt.xlabel("Δ logscore")
    plt.ylabel("count")
    plt.grid(axis="y", linestyle=":", linewidth=1.0)
    plt.legend(loc="best")
    _ensure_dir(out_path.parent)
    plt.tight_layout()
    plt.savefig(out_path, dpi=int(dpi))
    plt.close()


# --- J6 (MS2) policy usage tables ---

def table_j6_policy_usage_by_block(
    pred_csv: Path,
    splits: dict[str, Any],
    *,
    refit_every: int = 40,
) -> pd.DataFrame:
    """Reviewer audit table for production policy: MS2 if healthy, else THR.

    Uses daily `predictions.csv` from J6 and aggregates by refit blocks on OOS.

    Expected columns in predictions.csv:
      - date
      - used_ms_gauss (0/1)
      - used_ms_t (0/1)
      - used_ms_t_mode (typically 1 or 2; 0 if unused)
      - logc_ms_t (optional, for non-finite audit)
      - logc_thr_t (optional)
      - logc_ms_gauss (optional)

    Output columns:
      - block_id, t_refit, block_start, block_end, refit_every_obs, n_block_obs
      - ms_t_used_rate, ms_t_fallback_rate
      - ms_gauss_used_rate, ms_gauss_fallback_rate
      - ms_t_ms1_rate, ms_t_ms2_rate (shares over all obs; based on used_ms_t_mode)
      - ms_t_nonfinite_rate, ms_gauss_nonfinite_rate (if logc cols exist)
    """
    if not pred_csv.exists():
        raise MissingArtifactError(f"Missing J6 predictions CSV: {pred_csv}")

    df = _load_csv(pred_csv)
    if "date" not in df.columns:
        raise MissingArtifactError(f"J6 predictions CSV missing 'date'. Have={list(df.columns)[:30]}")

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    # Validate split boundaries
    first_oos = _as_utc_ts(splits["first_oos"])
    last_oos = _as_utc_ts(splits["last_oos"])

    # Clip to OOS window
    df = df[(df["date"] >= first_oos) & (df["date"] <= last_oos)].reset_index(drop=True)
    if df.empty:
        raise MissingArtifactError("J6 predictions empty after clipping to [first_oos,last_oos]")

    # Coerce usage flags if present
    for c in ["used_ms_gauss", "used_ms_t"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)
        else:
            # If column missing, treat as always unused
            df[c] = 0

    if "used_ms_t_mode" in df.columns:
        df["used_ms_t_mode"] = pd.to_numeric(df["used_ms_t_mode"], errors="coerce").fillna(0).astype(int)
    else:
        df["used_ms_t_mode"] = 0

    # Optional non-finite audit
    if "logc_ms_t" in df.columns:
        df["logc_ms_t"] = pd.to_numeric(df["logc_ms_t"], errors="coerce")
    if "logc_ms_gauss" in df.columns:
        df["logc_ms_gauss"] = pd.to_numeric(df["logc_ms_gauss"], errors="coerce")

    step = int(refit_every)
    if step < 1:
        step = 1

    dates64 = df["date"].dt.tz_convert(None).to_numpy(dtype="datetime64[ns]")
    n = len(df)

    first_oos64 = first_oos.tz_convert(None).to_datetime64()
    i0 = int(np.searchsorted(dates64, first_oos64, side="left"))
    if i0 < 1:
        i0 = 1

    refit_idx = list(range(i0, n, step))
    if not refit_idx:
        refit_idx = [n - 1]
    elif refit_idx[-1] != n - 1:
        refit_idx.append(n - 1)

    rows: list[dict[str, Any]] = []
    prev_i = i0
    block_id = 0

    for i in refit_idx:
        t_refit = df["date"].iloc[i]
        block_start = df["date"].iloc[prev_i]
        block_end = t_refit
        m_block = (df["date"] >= block_start) & (df["date"] <= block_end)
        n_block = int(m_block.sum())
        if n_block <= 0:
            prev_i = i
            block_id += 1
            continue

        used_t = df.loc[m_block, "used_ms_t"].to_numpy(dtype=int)
        used_g = df.loc[m_block, "used_ms_gauss"].to_numpy(dtype=int)
        mode = df.loc[m_block, "used_ms_t_mode"].to_numpy(dtype=int)

        ms_t_used_rate = float(np.mean(used_t))
        ms_gauss_used_rate = float(np.mean(used_g))

        # mode shares over all observations (not conditional), but only when MS was used
        ms_t_ms1_rate = float(np.mean((used_t == 1) & (mode == 1)))
        ms_t_ms2_rate = float(np.mean((used_t == 1) & (mode == 2)))

        # Non-finite rates (only meaningful if the columns exist)
        ms_t_nonfinite_rate = np.nan
        if "logc_ms_t" in df.columns:
            ms_t_nonfinite_rate = float(np.mean(~np.isfinite(df.loc[m_block, "logc_ms_t"].to_numpy(dtype=float))))

        ms_gauss_nonfinite_rate = np.nan
        if "logc_ms_gauss" in df.columns:
            ms_gauss_nonfinite_rate = float(np.mean(~np.isfinite(df.loc[m_block, "logc_ms_gauss"].to_numpy(dtype=float))))

        rows.append(
            {
                "block_id": block_id,
                "t_refit": t_refit.isoformat(),
                "block_start": block_start.isoformat(),
                "block_end": block_end.isoformat(),
                "refit_every_obs": step,
                "n_block_obs": n_block,
                "ms_t_used_rate": ms_t_used_rate,
                "ms_t_fallback_rate": float(1.0 - ms_t_used_rate),
                "ms_gauss_used_rate": ms_gauss_used_rate,
                "ms_gauss_fallback_rate": float(1.0 - ms_gauss_used_rate),
                "ms_t_ms1_rate": ms_t_ms1_rate,
                "ms_t_ms2_rate": ms_t_ms2_rate,
                "ms_t_nonfinite_rate": ms_t_nonfinite_rate,
                "ms_gauss_nonfinite_rate": ms_gauss_nonfinite_rate,
            }
        )

        prev_i = i
        block_id += 1

    return pd.DataFrame(rows)


def table_j6_health_by_refit(params_csv: Path) -> pd.DataFrame:
    """Optional helper: expose per-refit health inputs from params_summary.csv.

    This does NOT define 'healthy' (H_k) itself; it outputs the fields that can be used
    to define H_k in the paper (n_eff, A11/A22, fit_status, used flags, etc.).
    """
    if not params_csv.exists():
        raise MissingArtifactError(f"Missing J6 params CSV: {params_csv}")

    df = _load_csv(params_csv)
    if "refit_date" not in df.columns:
        raise MissingArtifactError(f"J6 params_summary missing 'refit_date'. Have={list(df.columns)[:30]}")

    out_cols = [
        "refit_index",
        "refit_date",
        "train_end_date",
        "n_train",
        "ms_gauss_fit_status",
        "ms_gauss_used",
        "ms_t_fit_status",
        "ms_t_used",
        "ms_t_mode",
        "ms_gauss_A11",
        "ms_gauss_A22",
        "ms_gauss_n_eff1",
        "ms_gauss_n_eff2",
        "ms_t_A11",
        "ms_t_A22",
        "ms_t_n_eff1",
        "ms_t_n_eff2",
        "ms_gauss_ll",
        "ms_t_ll",
    ]
    have = [c for c in out_cols if c in df.columns]
    if not have:
        raise MissingArtifactError(f"J6 params_summary has none of expected health columns. Have={list(df.columns)[:30]}")

    out = df[have].copy()
    out["refit_date"] = pd.to_datetime(out["refit_date"], utc=True, errors="coerce")
    out = out.dropna(subset=["refit_date"]).sort_values("refit_date").reset_index(drop=True)

    # Stabilize types
    for c in ["ms_gauss_used", "ms_t_used", "ms_t_mode"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0).astype(int)

    for c in [
        "ms_gauss_A11",
        "ms_gauss_A22",
        "ms_gauss_n_eff1",
        "ms_gauss_n_eff2",
        "ms_t_A11",
        "ms_t_A22",
        "ms_t_n_eff1",
        "ms_t_n_eff2",
        "ms_gauss_ll",
        "ms_t_ll",
    ]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")

    # ISO output for refit_date
    out["refit_date"] = out["refit_date"].apply(lambda x: x.isoformat() if pd.notna(x) else "")
    return out


# --- Inserted: table_j6_params_summary_policy ---

def table_j6_params_summary_policy(params_csv: Path) -> pd.DataFrame:
    """Paper-ready T6 table from J6 params_summary.csv.

    Adds a deterministic health flag and a simple fallback indicator.
    """
    if not params_csv.exists():
        raise MissingArtifactError(f"Missing J6 params CSV: {params_csv}")

    df = _load_csv(params_csv)
    if "refit_date" not in df.columns:
        raise MissingArtifactError(f"J6 params_summary missing 'refit_date'. Have={list(df.columns)[:30]}")

    df = df.copy()
    df["refit_date"] = pd.to_datetime(df["refit_date"], utc=True, errors="coerce")
    df = df.dropna(subset=["refit_date"]).sort_values("refit_date").reset_index(drop=True)

    # Stabilize types
    for c in ["ms_gauss_used", "ms_t_used", "ms_t_mode"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)

    for c in [
        "ms_t_A11", "ms_t_A22", "ms_t_n_eff1", "ms_t_n_eff2", "ms_t_ll",
        "ms_t_rho1", "ms_t_nu1", "ms_t_rho2", "ms_t_nu2",
        "ms_gauss_A11", "ms_gauss_A22", "ms_gauss_n_eff1", "ms_gauss_n_eff2", "ms_gauss_ll",
        "ms_gauss_rho1", "ms_gauss_rho2",
    ]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    def _is_ok(x: Any) -> bool:
        return str(x).strip().lower() == "ok"

    # Health: strict-but-simple, paper-defensible
    a11 = df.get("ms_t_A11")
    a22 = df.get("ms_t_A22")
    ne1 = df.get("ms_t_n_eff1")
    ne2 = df.get("ms_t_n_eff2")
    ll = df.get("ms_t_ll")

    cond = pd.Series(True, index=df.index)
    if a11 is not None:
        cond &= np.isfinite(a11) & (a11 > 0.0) & (a11 < 1.0)
    if a22 is not None:
        cond &= np.isfinite(a22) & (a22 > 0.0) & (a22 < 1.0)
    if ne1 is not None:
        cond &= np.isfinite(ne1) & (ne1 >= 1.0)
    if ne2 is not None:
        cond &= np.isfinite(ne2) & (ne2 >= 1.0)
    if ll is not None:
        cond &= np.isfinite(ll)

    df["healthy_ms_t"] = (
        df.get("ms_t_fit_status", "").map(_is_ok)
        & (df.get("ms_t_used", 0).astype(int) == 1)
        & cond
    ).astype(int)

    df["fallback_ms_t"] = (1 - df.get("ms_t_used", 0).astype(int)).astype(int)

    out_cols = [
        "refit_index", "refit_date", "train_end_date", "n_train",
        "ms_t_A11", "ms_t_A22", "ms_t_rho1", "ms_t_nu1", "ms_t_rho2", "ms_t_nu2",
        "ms_t_n_eff1", "ms_t_n_eff2", "ms_t_ll",
        "ms_t_fit_status", "ms_t_used", "ms_t_mode",
        "healthy_ms_t", "fallback_ms_t",
        "ms_gauss_A11", "ms_gauss_A22", "ms_gauss_rho1", "ms_gauss_rho2",
        "ms_gauss_n_eff1", "ms_gauss_n_eff2", "ms_gauss_ll",
        "ms_gauss_fit_status", "ms_gauss_used",
    ]
    have = [c for c in out_cols if c in df.columns]
    out = df[have].copy()
    out["refit_date"] = out["refit_date"].apply(lambda x: x.isoformat() if pd.notna(x) else "")
    return out


# --- Inserted: fig_j6_cumdelta_ms_vs_thr and fig_j6_hist_delta_ms_vs_thr ---

def fig_j6_cumdelta_ms_vs_thr(
    pred_csv: Path,
    splits: dict[str, Any],
    out_path: Path,
    *,
    fig_w: float,
    fig_h: float,
    dpi: int,
) -> None:
    """Figure F9: cumulative sum of policy delta (MS vs THR) on OOS."""
    if not pred_csv.exists():
        raise MissingArtifactError(f"Missing J6 predictions CSV: {pred_csv}")

    df = _load_csv(pred_csv)
    need = ["date", "logc_ms_t", "logc_thr_t", "used_ms_t"]
    miss = [c for c in need if c not in df.columns]
    if miss:
        raise MissingArtifactError(f"J6 predictions missing columns {miss} for F9. Have={list(df.columns)[:30]}")

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    for c in ["logc_ms_t", "logc_thr_t", "used_ms_t"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    first_oos = _as_utc_ts(splits["first_oos"])
    last_oos = _as_utc_ts(splits["last_oos"])
    df = df[(df["date"] >= first_oos) & (df["date"] <= last_oos)].reset_index(drop=True)

    df = df.dropna(subset=["logc_ms_t", "logc_thr_t", "used_ms_t"]).reset_index(drop=True)
    if df.empty:
        raise MissingArtifactError("No valid OOS rows for J6 F9.")

    used = (df["used_ms_t"].astype(float) > 0.5).astype(float).to_numpy()
    d = used * (df["logc_ms_t"].to_numpy(dtype=float) - df["logc_thr_t"].to_numpy(dtype=float))

    m = np.isfinite(d)
    df = df.loc[m].reset_index(drop=True)
    d = d[m]
    if d.size == 0:
        raise MissingArtifactError("No finite delta rows for J6 F9.")

    c = np.cumsum(d)

    _set_matplotlib_style(fig_w, fig_h)
    plt.figure(figsize=(float(fig_w), float(fig_h)))
    plt.plot(df["date"], c, label="cum Δ policy(ms_t) − thr_t")
    plt.axhline(0.0, linestyle=":", linewidth=1.0)
    plt.title("F9 — cumulative policy logscore delta vs THR (J6, OOS)")
    plt.xlabel("date (UTC)")
    plt.ylabel("cumulative Δ logscore")
    plt.grid(axis="y", linestyle=":", linewidth=1.0)
    plt.legend(loc="best")
    _ensure_dir(out_path.parent)
    plt.tight_layout()
    plt.savefig(out_path, dpi=int(dpi))
    plt.close()


def fig_j6_hist_delta_ms_vs_thr(
    pred_csv: Path,
    splits: dict[str, Any],
    out_path: Path,
    *,
    fig_w: float,
    fig_h: float,
    dpi: int,
    bins: int = 60,
) -> None:
    """Figure F10: histogram of per-day policy delta (MS vs THR) on OOS."""
    if not pred_csv.exists():
        raise MissingArtifactError(f"Missing J6 predictions CSV: {pred_csv}")

    df = _load_csv(pred_csv)
    need = ["date", "logc_ms_t", "logc_thr_t", "used_ms_t"]
    miss = [c for c in need if c not in df.columns]
    if miss:
        raise MissingArtifactError(f"J6 predictions missing columns {miss} for F10. Have={list(df.columns)[:30]}")

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    for c in ["logc_ms_t", "logc_thr_t", "used_ms_t"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    first_oos = _as_utc_ts(splits["first_oos"])
    last_oos = _as_utc_ts(splits["last_oos"])
    df = df[(df["date"] >= first_oos) & (df["date"] <= last_oos)].reset_index(drop=True)

    df = df.dropna(subset=["logc_ms_t", "logc_thr_t", "used_ms_t"]).reset_index(drop=True)
    if df.empty:
        raise MissingArtifactError("No valid OOS rows for J6 F10.")

    used = (df["used_ms_t"].astype(float) > 0.5).astype(float).to_numpy()
    d = used * (df["logc_ms_t"].to_numpy(dtype=float) - df["logc_thr_t"].to_numpy(dtype=float))
    d = d[np.isfinite(d)]
    if d.size == 0:
        raise MissingArtifactError("No finite delta rows for J6 F10.")

    _set_matplotlib_style(fig_w, fig_h)
    plt.figure(figsize=(float(fig_w), float(fig_h)))
    plt.hist(d, bins=int(bins), alpha=0.8, label="Δ policy(ms_t) − thr_t")
    plt.axvline(0.0, linestyle=":", linewidth=1.0)
    plt.title("F10 — histogram of per-day policy logscore delta vs THR (J6, OOS)")
    plt.xlabel("Δ logscore")
    plt.ylabel("count")
    plt.grid(axis="y", linestyle=":", linewidth=1.0)
    plt.legend(loc="best")
    _ensure_dir(out_path.parent)
    plt.tight_layout()
    plt.savefig(out_path, dpi=int(dpi))
    plt.close()


def _pick_first_existing(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _load_dm_summary(dm_csv: Path) -> pd.DataFrame:
    if not dm_csv.exists():
        raise MissingArtifactError(f"Missing DM summary CSV: {dm_csv}")
    df = _load_csv(dm_csv)
    if df.empty:
        raise MissingArtifactError(f"DM summary CSV empty: {dm_csv}")

    col_a = _pick_first_existing(df, ["model_a", "A", "a", "lhs_model", "lhs"])
    col_b = _pick_first_existing(df, ["model_b", "B", "b", "rhs_model", "rhs"])
    col_p = _pick_first_existing(df, ["dm_pvalue", "pvalue", "p_value", "pval", "p_val"])
    col_d = _pick_first_existing(df, ["delta_logscore_mean", "delta_mean", "d_mean", "mean_delta", "dm_mean"])

    if col_a is None or col_b is None:
        raise MissingArtifactError(
            f"DM CSV must contain model columns. Tried model_a/model_b variants. Have={list(df.columns)[:40]}"
        )
    if col_p is None:
        raise MissingArtifactError(
            f"DM CSV must contain a p-value column. Tried dm_pvalue/pvalue variants. Have={list(df.columns)[:40]}"
        )

    out = df.copy()
    out["model_a"] = out[col_a].astype(str)
    out["model_b"] = out[col_b].astype(str)
    out["dm_pvalue"] = pd.to_numeric(out[col_p], errors="coerce")
    out["delta_logscore_mean"] = pd.to_numeric(out[col_d], errors="coerce") if col_d is not None else np.nan
    out = out.dropna(subset=["model_a", "model_b"]).reset_index(drop=True)
    return out[["model_a", "model_b", "dm_pvalue", "delta_logscore_mean"]]


def _infer_model_to_es99_cols(var_df: pd.DataFrame) -> dict[str, str]:
    inferred = _infer_var_es_cols(var_df)
    es99_cols = inferred["es99"]
    m = {}
    for c in es99_cols:
        m[str(c)] = str(c)
    return m


def _infer_model_to_exceed99_cols(var_df: pd.DataFrame) -> dict[str, str]:
    exc99 = _pick_cols_by_regex(list(var_df.columns), r"^exceed99_")
    m = {}
    for c in exc99:
        model = c[len("exceed99_"):]
        m[model] = c
    return m


def table_t13_stat_vs_econ_summary(
    dm_csv: Path,
    var_es_pred_csv: Path,
    regimes_csv: Path,
    splits: dict[str, Any],
    *,
    regime_col: Optional[str],
    alpha: float = 0.05,
    es99_abs_threshold: float = 0.0,
) -> pd.DataFrame:
    dm = _load_dm_summary(dm_csv)

    var_df = load_var_es_predictions(var_es_pred_csv)
    stress_df = load_regimes_df(regimes_csv, regime_col=regime_col)
    var_df = var_df.merge(stress_df, on="date", how="left")
    var_df["S_t"] = pd.to_numeric(var_df["S_t"], errors="coerce").fillna(0).astype(int)
    var_df["bucket"] = np.where(var_df["S_t"].astype(int) == 1, "stress", "calm")

    first_oos = _as_utc_ts(splits["first_oos"])
    last_oos = _as_utc_ts(splits["last_oos"])
    var_df = var_df[(var_df["date"] >= first_oos) & (var_df["date"] <= last_oos)].reset_index(drop=True)
    if var_df.empty:
        raise MissingArtifactError("No OOS rows in var_es_predictions.csv for T13 after clipping to [first_oos,last_oos].")

    es99_map = _infer_model_to_es99_cols(var_df)
    exc99_map = _infer_model_to_exceed99_cols(var_df)

    def _mean_finite(x: pd.Series) -> float:
        v = pd.to_numeric(x, errors="coerce").to_numpy(dtype=float)
        v = v[np.isfinite(v)]
        return float(np.mean(v)) if v.size else float("nan")

    def _rate_flag(x: pd.Series) -> float:
        v = pd.to_numeric(x, errors="coerce").to_numpy(dtype=float)
        v = v[np.isfinite(v)]
        return float(np.mean(v > 0.5)) if v.size else float("nan")

    def _kupiec_from_flags(x: pd.Series, alpha: float) -> float:
        s = pd.to_numeric(x, errors="coerce").dropna()
        n = int(len(s))
        n_exc = int((s.astype(float) > 0.5).sum())
        lr, pval = _kupiec_pof(n, n_exc, float(alpha))
        return float(pval)

    rows = []
    for i, r in dm.iterrows():
        A = str(r["model_a"])
        B = str(r["model_b"])

        esA = es99_map.get(A, A)
        esB = es99_map.get(B, B)
        if esA not in var_df.columns or esB not in var_df.columns:
            continue

        for c in [esA, esB]:
            var_df[c] = pd.to_numeric(var_df[c], errors="coerce")

        excA = exc99_map.get(A)
        excB = exc99_map.get(B)

        es_over_A = _mean_finite(var_df[esA])
        es_over_B = _mean_finite(var_df[esB])
        es_st_A = _mean_finite(var_df.loc[var_df["bucket"] == "stress", esA])
        es_st_B = _mean_finite(var_df.loc[var_df["bucket"] == "stress", esB])

        exc_rate_A = _rate_flag(var_df[excA]) if excA and excA in var_df.columns else float("nan")
        exc_rate_B = _rate_flag(var_df[excB]) if excB and excB in var_df.columns else float("nan")

        kup_A = _kupiec_from_flags(var_df[excA], 0.01) if excA and excA in var_df.columns else float("nan")
        kup_B = _kupiec_from_flags(var_df[excB], 0.01) if excB and excB in var_df.columns else float("nan")

        d_es_over = es_over_A - es_over_B if np.isfinite(es_over_A) and np.isfinite(es_over_B) else float("nan")
        d_es_st = es_st_A - es_st_B if np.isfinite(es_st_A) and np.isfinite(es_st_B) else float("nan")
        d_exc = exc_rate_A - exc_rate_B if np.isfinite(exc_rate_A) and np.isfinite(exc_rate_B) else float("nan")

        pval = float(r["dm_pvalue"]) if np.isfinite(r["dm_pvalue"]) else float("nan")
        stat_sig = int(np.isfinite(pval) and (pval < float(alpha)))

        eco_mat = int(
            (np.isfinite(d_es_st) and abs(d_es_st) >= float(es99_abs_threshold))
            or (np.isfinite(d_es_over) and abs(d_es_over) >= float(es99_abs_threshold))
        )

        tag = ""
        if stat_sig and eco_mat:
            tag = "stat+ / eco+"
        elif stat_sig and not eco_mat:
            tag = "stat+ / eco0"
        elif (not stat_sig) and eco_mat:
            tag = "stat0 / eco+"
        else:
            tag = "stat0 / eco0"

        rows.append(
            {
                "comparison_id": f"cmp_{i}",
                "scope": "OOS",
                "model_A": A,
                "model_B": B,
                "n_oos": int(len(var_df)),
                "delta_logscore_mean": float(r["delta_logscore_mean"]) if np.isfinite(r["delta_logscore_mean"]) else float("nan"),
                "dm_pvalue": pval,
                "stat_sig": stat_sig,
                "delta_ES99_overall": d_es_over,
                "delta_ES99_stress": d_es_st,
                "delta_exceed99_rate": d_exc,
                "kupiec_pvalue_A": kup_A,
                "kupiec_pvalue_B": kup_B,
                "eco_mat": eco_mat,
                "conclusion_tag": tag,
                "note": "descriptive; no causal claim",
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        raise MissingArtifactError(
            "T13 produced 0 rows. Reason: model names in DM CSV did not match ES99 columns in var_es_predictions.csv. "
            "Fix by aligning DM model labels with ES99 column names (or rename via a pre-step)."
        )
    return out

def _pick_cols_by_regex(cols: list[str], pattern: str) -> list[str]:
    rx = re.compile(pattern, flags=re.IGNORECASE)
    return [c for c in cols if rx.search(c)]


def load_var_es_predictions(pred_csv: Path) -> pd.DataFrame:
    if not pred_csv.exists():
        raise MissingArtifactError(f"Missing var_es_predictions.csv: {pred_csv}")
    df = _load_csv(pred_csv)
    if "date" not in df.columns:
        raise MissingArtifactError(f"var_es_predictions.csv missing 'date'. Have={list(df.columns)[:30]}")
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    # Coerce numerics where possible
    for c in df.columns:
        if c == "date":
            continue
        if df[c].dtype == object:
            # keep strings (e.g. regime) if clearly non-numeric
            # but try numeric coercion quietly
            df[c] = pd.to_numeric(df[c], errors="ignore")
    return df


def _infer_var_es_cols(df: pd.DataFrame) -> dict[str, list[str]]:
    cols = list(df.columns)

    # very tolerant patterns (your schemas differ between J7/J8)
    es99 = _pick_cols_by_regex(cols, r"(?:^|_)es(?:_|)?99(?:$|_)|(?:^|_)es99(?:$|_)|(?:^|_)es_99(?:$|_)")
    var99 = _pick_cols_by_regex(cols, r"(?:^|_)var(?:_|)?99(?:$|_)|(?:^|_)var99(?:$|_)|(?:^|_)var_99(?:$|_)")
    es95 = _pick_cols_by_regex(cols, r"(?:^|_)es(?:_|)?95(?:$|_)|(?:^|_)es95(?:$|_)|(?:^|_)es_95(?:$|_)")
    var95 = _pick_cols_by_regex(cols, r"(?:^|_)var(?:_|)?95(?:$|_)|(?:^|_)var95(?:$|_)|(?:^|_)var_95(?:$|_)")

    return {"es99": es99, "var99": var99, "es95": es95, "var95": var95}


def fig_f11_es99_timeseries(
    var_es_pred_csv: Path,
    splits: dict[str, Any],
    out_path: Path,
    *,
    max_models: int = 4,
    es99_cols_csv: str = "",
    fig_w: float,
    fig_h: float,
    dpi: int,
) -> list[str]:
    """F11: plot selected ES99_t time series on OOS (few models only)."""
    df = load_var_es_predictions(var_es_pred_csv)
    first_oos = _as_utc_ts(splits["first_oos"])
    last_oos = _as_utc_ts(splits["last_oos"])
    df = df[(df["date"] >= first_oos) & (df["date"] <= last_oos)].reset_index(drop=True)
    if df.empty:
        raise MissingArtifactError("No OOS rows in var_es_predictions.csv for F11")

    inferred = _infer_var_es_cols(df)
    es99_cols = []
    if es99_cols_csv.strip():
        es99_cols = [c.strip() for c in es99_cols_csv.split(",") if c.strip()]
    else:
        es99_cols = inferred["es99"][: max(1, int(max_models))]

    # Validate + coerce
    es99_cols = [c for c in es99_cols if c in df.columns]
    if not es99_cols:
        raise MissingArtifactError(f"Could not infer ES99 columns for F11. Have={list(df.columns)[:30]}")

    for c in es99_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=es99_cols).reset_index(drop=True)
    if df.empty:
        raise MissingArtifactError("No valid numeric ES99 rows for F11 after dropping NaNs")

    _set_matplotlib_style(fig_w, fig_h)
    plt.figure(figsize=(float(fig_w), float(fig_h)))
    for c in es99_cols:
        plt.plot(df["date"], df[c], label=c)
    plt.title("F11 — ES99(t) on OOS (selected models)")
    plt.xlabel("date (UTC)")
    plt.ylabel("ES99")
    plt.grid(axis="y", linestyle=":", linewidth=1.0)
    plt.legend(loc="best", fontsize=8)
    _ensure_dir(out_path.parent)
    plt.tight_layout()
    plt.savefig(out_path, dpi=int(dpi))
    plt.close()
    return es99_cols


def fig_f12_stress_delta_es99(
    var_es_pred_csv: Path,
    regimes_csv: Path,
    splits: dict[str, Any],
    out_path: Path,
    *,
    regime_col: Optional[str],
    a_col: str = "",
    b_col: str = "",
    fig_w: float,
    fig_h: float,
    dpi: int,
) -> tuple[str, str]:
    """F12: diagnostic in stress: ΔES99^stress (A − B). A=MS, B=THR by convention."""
    df = load_var_es_predictions(var_es_pred_csv)

    # build S_t from regimes.csv (same helper as M1)
    stress_df = load_regimes_df(regimes_csv, regime_col=regime_col)
    df = df.merge(stress_df, on="date", how="left")
    df["S_t"] = pd.to_numeric(df["S_t"], errors="coerce").fillna(0).astype(int)

    first_oos = _as_utc_ts(splits["first_oos"])
    last_oos = _as_utc_ts(splits["last_oos"])
    df = df[(df["date"] >= first_oos) & (df["date"] <= last_oos)].reset_index(drop=True)
    if df.empty:
        raise MissingArtifactError("No OOS rows in var_es_predictions.csv for F12")

    inferred = _infer_var_es_cols(df)
    es99_cols = inferred["es99"]
    if not es99_cols:
        raise MissingArtifactError("Could not infer ES99 columns for F12")

    # choose A/B
    A = a_col.strip() if a_col.strip() else (es99_cols[0] if len(es99_cols) >= 1 else "")
    B = b_col.strip() if b_col.strip() else (es99_cols[1] if len(es99_cols) >= 2 else "")
    if A not in df.columns or B not in df.columns:
        raise MissingArtifactError(f"F12 needs two ES99 columns. Got A={A}, B={B}. Have ES99={es99_cols[:10]}")

    df[A] = pd.to_numeric(df[A], errors="coerce")
    df[B] = pd.to_numeric(df[B], errors="coerce")

    df = df.dropna(subset=[A, B, "S_t"]).reset_index(drop=True)
    df_stress = df[df["S_t"] == 1].reset_index(drop=True)

    if df_stress.empty:
        raise MissingArtifactError("No stress rows (S_t==1) after join for F12")

    delta = (df_stress[A] - df_stress[B]).to_numpy(dtype=float)
    m = np.isfinite(delta)
    df_stress = df_stress.loc[m].reset_index(drop=True)
    delta = (df_stress[A] - df_stress[B]).to_numpy(dtype=float)
    if df_stress.empty:
        raise MissingArtifactError("No finite stress deltas for F12")

    c = np.cumsum(delta)

    _set_matplotlib_style(fig_w, fig_h)
    plt.figure(figsize=(float(fig_w), float(fig_h)))
    plt.plot(df_stress["date"], c, label=f"cum ΔES99(stress): {A} − {B}")
    plt.axhline(0.0, linestyle=":", linewidth=1.0)
    plt.title("F12 — Stress impact diagnostic (cum ΔES99 in stress)")
    plt.xlabel("date (UTC)")
    plt.ylabel("cumulative ΔES99 (stress)")
    plt.grid(axis="y", linestyle=":", linewidth=1.0)
    plt.legend(loc="best", fontsize=8)
    _ensure_dir(out_path.parent)
    plt.tight_layout()
    plt.savefig(out_path, dpi=int(dpi))
    plt.close()

    return (A, B)


def table_t8_var_es_summary(
    var_es_pred_csv: Path,
    regimes_csv: Path,
    splits: dict[str, Any],
    *,
    regime_col: Optional[str],
) -> pd.DataFrame:
    """T8: var_es_summary by model x bucket (calm/stress).

    Outputs at minimum: n, VaR95, VaR99, ES95, ES99
    If a metric is missing in schema, it will be NaN for that model.
    """
    df = load_var_es_predictions(var_es_pred_csv)
    stress_df = load_regimes_df(regimes_csv, regime_col=regime_col)
    df = df.merge(stress_df, on="date", how="left")
    df["S_t"] = pd.to_numeric(df["S_t"], errors="coerce").fillna(0).astype(int)

    first_oos = _as_utc_ts(splits["first_oos"])
    last_oos = _as_utc_ts(splits["last_oos"])
    df = df[(df["date"] >= first_oos) & (df["date"] <= last_oos)].reset_index(drop=True)
    if df.empty:
        raise MissingArtifactError("No OOS rows in var_es_predictions.csv for T8")

    inferred = _infer_var_es_cols(df)

    # define models by ES99 columns (most reliable anchor)
    es99_cols = inferred["es99"]
    if not es99_cols:
        raise MissingArtifactError("Cannot build T8: no ES99 columns inferred.")

    # bucket mapping
    df["bucket"] = np.where(df["S_t"].astype(int) == 1, "stress", "calm")

    rows = []
    for es99_col in es99_cols:
        model = es99_col

        # best-effort mapping: try to find sister columns sharing a prefix
        # Example patterns: <prefix>_es99, <prefix>_var99, ...
        prefix = es99_col
        prefix = re.sub(r"(?i)(?:^|_)es(?:_|)?99(?:$|_).*", "", prefix).rstrip("_")
        # If prefix collapses, fallback to the column itself
        def _guess_sister(candidates: list[str], kind: str) -> str | None:
            # exact prefix match first
            for c in candidates:
                if prefix and c.lower().startswith(prefix.lower()):
                    return c
            return candidates[0] if candidates else None

        var99_col = _guess_sister(inferred["var99"], "var99")
        es95_col  = _guess_sister(inferred["es95"], "es95")
        var95_col = _guess_sister(inferred["var95"], "var95")

        # coerce used cols
        use_cols = [x for x in [var95_col, var99_col, es95_col, es99_col] if x and x in df.columns]
        for c in use_cols:
            df[c] = pd.to_numeric(df[c], errors="coerce")

        for bucket in ["calm", "stress"]:
            sub = df[df["bucket"] == bucket].copy()
            # n defined where ES99 exists
            sub = sub.dropna(subset=[es99_col])
            n = int(len(sub))

            def _mean(col: str | None) -> float:
                if not col or col not in sub.columns:
                    return float("nan")
                v = pd.to_numeric(sub[col], errors="coerce").to_numpy(dtype=float)
                v = v[np.isfinite(v)]
                return float(np.mean(v)) if v.size else float("nan")

            rows.append(
                {
                    "model": model,
                    "bucket": bucket,
                    "n": n,
                    "VaR95": _mean(var95_col),
                    "VaR99": _mean(var99_col),
                    "ES95":  _mean(es95_col),
                    "ES99":  _mean(es99_col),
                }
            )

    out = pd.DataFrame(rows)
    return out

def table_t9_exceedance_summary(
    var_es_pred_csv: Path,
    splits: dict[str, Any],
) -> pd.DataFrame:
    """T9: Exceedance summary (Kupiec POF) for VaR95/VaR99 (exceed95_*/exceed99_*) columns."""
    df = load_var_es_predictions(var_es_pred_csv)
    first_oos = _as_utc_ts(splits["first_oos"])
    last_oos = _as_utc_ts(splits["last_oos"])
    df = df[(df["date"] >= first_oos) & (df["date"] <= last_oos)].reset_index(drop=True)
    exc95 = _pick_cols_by_regex(list(df.columns), r"^exceed95_")
    exc99 = _pick_cols_by_regex(list(df.columns), r"^exceed99_")
    if not exc95 and not exc99:
        raise MissingArtifactError("No exceed95_*/exceed99_* columns found in var_es_predictions.csv")

    rows: list[dict[str, Any]] = []

    def _model_name_from_exc(col: str, prefix: str) -> str:
        # col like exceed95_static_gauss -> static_gauss
        return col[len(prefix):]

    # alpha=0.05 from exceed95_*
    for c in exc95:
        s = pd.to_numeric(df[c], errors="coerce")
        s = s.dropna()
        n = int(len(s))
        n_exc = int((s.astype(float) > 0.5).sum())
        rate_obs = (n_exc / n) if n > 0 else float("nan")
        lr, pval = _kupiec_pof(n, n_exc, 0.05)
        rows.append(
            {
                "model": _model_name_from_exc(c, "exceed95_"),
                "alpha": 0.05,
                "n": n,
                "n_exc": n_exc,
                "rate_obs": rate_obs,
                "rate_target": 0.05,
                "kupiec_lr_pof": lr,
                "pvalue": pval,
            }
        )

    # alpha=0.01 from exceed99_*
    for c in exc99:
        s = pd.to_numeric(df[c], errors="coerce")
        s = s.dropna()
        n = int(len(s))
        n_exc = int((s.astype(float) > 0.5).sum())
        rate_obs = (n_exc / n) if n > 0 else float("nan")
        lr, pval = _kupiec_pof(n, n_exc, 0.01)
        rows.append(
            {
                "model": _model_name_from_exc(c, "exceed99_"),
                "alpha": 0.01,
                "n": n,
                "n_exc": n_exc,
                "rate_obs": rate_obs,
                "rate_target": 0.01,
                "kupiec_lr_pof": lr,
                "pvalue": pval,
            }
        )

    out = pd.DataFrame(rows)
    # stable ordering: model then alpha
    if not out.empty:
        out = out.sort_values(["model", "alpha"]).reset_index(drop=True)
    return out

def _kupiec_pof(n: int, n_exc: int, alpha: float) -> tuple[float, float]:
    """Kupiec POF test statistic and p-value."""
    # Avoid division by zero
    if n == 0:
        return float("nan"), float("nan")
    pi = n_exc / n if n > 0 else 0.0
    if pi == 0.0 or pi == 1.0:
        lr = 0.0
    else:
        lr = -2 * (
            (n_exc * np.log(alpha) + (n - n_exc) * np.log(1 - alpha))
            - (n_exc * np.log(pi) + (n - n_exc) * np.log(1 - pi))
        )
    # p-value (1 dof)
    try:
        from scipy.stats import chi2
        pval = 1.0 - chi2.cdf(lr, df=1)
    except Exception:
        pval = float("nan")
    return lr, pval
# --- Appendix A: Top-8 daily pairwise panel (A.3) ---

def load_returns_panel(returns_csv: Path) -> pd.DataFrame:
    """Load returns.csv as a wide panel with a UTC `date` column.

    Supports schemas:
      - date,BTC,ETH,...
      - <unnamed first col>,BTC,ETH,... (pandas index saved to CSV)

    Returns columns: ['date', <asset1>, <asset2>, ...] with numeric returns.
    """
    if not returns_csv.exists():
        raise MissingArtifactError(f"Missing returns.csv: {returns_csv}")
    df = _load_csv(returns_csv)
    if df.empty:
        raise MissingArtifactError(f"returns.csv empty: {returns_csv}")

    # detect date column
    if "date" in df.columns:
        date_col = "date"
    else:
        date_col = str(df.columns[0])  # often unnamed index column

    df = df.copy()
    df.rename(columns={date_col: "date"}, inplace=True)
    df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    # coerce asset columns to numeric
    for c in df.columns:
        if c == "date":
            continue
        df[c] = pd.to_numeric(df[c], errors="coerce")

    return df


def table_a1_top8_universe(
    returns_csv: Path,
    splits: dict[str, Any],
    *,
    assets_csv: str = "",
    k: int = 8,
) -> tuple[pd.DataFrame, list[str]]:
    """Table A1 — top-k universe + coverage stats on [train_start, last_oos].

    Selection:
      - If assets_csv provided: use that ordered list (truncated to k).
      - Else: choose k assets with highest non-NaN coverage on the target window.
    """
    df = load_returns_panel(returns_csv)
    train_start = _as_utc_ts(splits["train_start"])
    last_oos = _as_utc_ts(splits["last_oos"])
    df = df[(df["date"] >= train_start) & (df["date"] <= last_oos)].reset_index(drop=True)
    if df.empty:
        raise MissingArtifactError("returns panel empty after clipping to [train_start,last_oos]")

    asset_cols = [c for c in df.columns if c != "date"]
    if not asset_cols:
        raise MissingArtifactError(f"No asset columns found in returns.csv: {returns_csv}")

    n_target = int(df["date"].nunique())
    if n_target <= 0:
        raise MissingArtifactError("No target dates for A1")

    if assets_csv.strip():
        chosen = [a.strip() for a in assets_csv.split(",") if a.strip()]
        missing = [a for a in chosen if a not in asset_cols]
        if missing:
            raise MissingArtifactError(f"A1 assets not in returns.csv: {missing}. Have={asset_cols[:30]}")
        chosen = chosen[: int(k)]
    else:
        cov = {a: float(df[a].notna().mean()) for a in asset_cols}
        chosen = [a for a, _ in sorted(cov.items(), key=lambda kv: (-kv[1], kv[0]))][: int(k)]

    rows = []
    for a in chosen:
        n_valid = int(df[a].notna().sum())
        rows.append(
            {
                "asset": a,
                "n_days_valid": n_valid,
                "n_days_target": n_target,
                "coverage_pct": 100.0 * (n_valid / n_target),
            }
        )

    return pd.DataFrame(rows), chosen


def table_a1b_pairwise_stats(
    returns_csv: Path,
    splits: dict[str, Any],
    *,
    assets: list[str],
) -> pd.DataFrame:
    """Optional A1b — summary stats of strict pairwise intersection counts."""
    if len(assets) < 2:
        raise MissingArtifactError("Need >=2 assets for pairwise stats")

    df = load_returns_panel(returns_csv)
    train_start = _as_utc_ts(splits["train_start"])
    last_oos = _as_utc_ts(splits["last_oos"])
    df = df[(df["date"] >= train_start) & (df["date"] <= last_oos)].reset_index(drop=True)

    miss = [a for a in assets if a not in df.columns]
    if miss:
        raise MissingArtifactError(f"Pairwise assets not in returns.csv: {miss}")

    n_ij = []
    pairs = 0
    for i in range(len(assets)):
        for j in range(i + 1, len(assets)):
            ai, aj = assets[i], assets[j]
            n = int((df[ai].notna() & df[aj].notna()).sum())
            n_ij.append(n)
            pairs += 1

    arr = np.asarray(n_ij, dtype=float)
    return pd.DataFrame([{
        "n_pairs": int(pairs),
        "min_n_ij": int(np.min(arr)),
        "median_n_ij": float(np.median(arr)),
        "max_n_ij": int(np.max(arr)),
    }])


# --- Appendix A: Top-8 pairwise (direct from pairwise artefacts, no returns.csv) ---

def _parse_pair_dir_name(pair_name: str) -> tuple[str, str]:
    """Parse pair folder name like 'BTCUSDT_ETHUSDT' into ('BTCUSDT','ETHUSDT')."""
    if "_" not in pair_name:
        raise MissingArtifactError(f"Bad pair folder name (expected A_B): {pair_name}")
    a, b = pair_name.split("_", 1)
    a = a.strip()
    b = b.strip()
    if not a or not b:
        raise MissingArtifactError(f"Bad pair folder name (empty leg): {pair_name}")
    return a, b


def _pairwise_var_es_paths(pairs_root: Path) -> list[Path]:
    """List pairwise var_es_predictions.csv files under a root that contains `pairs/<PAIR>/var_es_predictions.csv`."""
    if not pairs_root.exists():
        raise MissingArtifactError(f"Missing top8 pairwise root: {pairs_root}")
    pdir = pairs_root / "pairs"
    if not pdir.exists():
        raise MissingArtifactError(f"top8 pairwise root has no 'pairs' dir: {pairs_root}")
    paths = sorted(pdir.glob("*/var_es_predictions.csv"))
    if not paths:
        raise MissingArtifactError(f"No pairwise var_es_predictions.csv found under: {pdir}")
    return paths


def table_a1_top8_universe_from_pairwise(
    pairwise_root: Path,
) -> tuple[pd.DataFrame, list[str]]:
    """Table A1 — top-8 universe coverage inferred from pairwise artefacts.

    Uses only per-pair var_es_predictions.csv files. For each asset, reports:
      - n_pairs_involving_asset (should be 7 for top-8)
      - n_days_valid: min(n_rows) across pairs involving the asset
      - n_days_target: max(n_rows) across all pairs
      - coverage_pct = 100 * n_days_valid / n_days_target
      - min_start_utc / max_end_utc from per-pair time columns

    This is contract-style coverage (not returns-based), aligned with A.3 pairwise protocol.
    """
    paths = _pairwise_var_es_paths(pairwise_root)

    # Discover assets + per-pair lengths and time spans
    assets: dict[str, int] = {}
    per_asset_n: dict[str, list[int]] = {}
    per_asset_start: dict[str, list[pd.Timestamp]] = {}
    per_asset_end: dict[str, list[pd.Timestamp]] = {}

    pair_n: list[int] = []
    starts: list[pd.Timestamp] = []
    ends: list[pd.Timestamp] = []

    for p in paths:
        pair_name = p.parent.name  # e.g. BTCUSDT_ETHUSDT
        a, b = _parse_pair_dir_name(pair_name)

        # Read only the timestamp column to keep it fast
        cols = list(pd.read_csv(p, nrows=0).columns)
        ts_col = "ts_utc" if "ts_utc" in cols else ("date" if "date" in cols else None)
        if ts_col is None:
            raise MissingArtifactError(f"Pairwise CSV missing ts column (ts_utc/date): {p}. Have={cols[:30]}")

        d = pd.read_csv(p, usecols=[ts_col])
        d[ts_col] = pd.to_datetime(d[ts_col], utc=True, errors="raise")
        if d.empty:
            continue
        n_rows = int(len(d))
        t0 = pd.Timestamp(d[ts_col].min())
        t1 = pd.Timestamp(d[ts_col].max())

        pair_n.append(n_rows)
        starts.append(t0)
        ends.append(t1)

        for leg in [a, b]:
            assets[leg] = assets.get(leg, 0) + 1
            per_asset_n.setdefault(leg, []).append(n_rows)
            per_asset_start.setdefault(leg, []).append(t0)
            per_asset_end.setdefault(leg, []).append(t1)

    if not assets:
        raise MissingArtifactError(f"No assets discovered under: {pairwise_root}")
    if not pair_n:
        raise MissingArtifactError(f"No non-empty pairwise CSVs under: {pairwise_root}")

    all_assets = sorted(assets.keys())
    n_target = int(max(pair_n))

    rows: list[dict[str, Any]] = []
    for a in all_assets:
        n_pairs_involving = int(assets.get(a, 0))
        n_valid = int(min(per_asset_n.get(a, [0])))
        start_min = min(per_asset_start.get(a, [])) if per_asset_start.get(a) else None
        end_max = max(per_asset_end.get(a, [])) if per_asset_end.get(a) else None
        rows.append(
            {
                "asset": a,
                "n_pairs_involving_asset": n_pairs_involving,
                "n_days_valid": n_valid,
                "n_days_target": n_target,
                "coverage_pct": 100.0 * (n_valid / n_target) if n_target > 0 else float("nan"),
                "min_start_utc": start_min.isoformat() if start_min is not None else "",
                "max_end_utc": end_max.isoformat() if end_max is not None else "",
            }
        )

    out = pd.DataFrame(rows)
    # Deterministic ordering: most connected first, then coverage, then name
    out = out.sort_values(["n_pairs_involving_asset", "coverage_pct", "asset"], ascending=[False, False, True]).reset_index(drop=True)

    return out, all_assets



def table_a1b_pairwise_stats_from_pairwise(pairwise_root: Path) -> pd.DataFrame:
    """A1b — strict pairwise intersection stats inferred from pairwise artefacts."""
    paths = _pairwise_var_es_paths(pairwise_root)

    pair_n: list[int] = []
    starts: list[pd.Timestamp] = []
    ends: list[pd.Timestamp] = []

    for p in paths:
        cols = list(pd.read_csv(p, nrows=0).columns)
        ts_col = "ts_utc" if "ts_utc" in cols else ("date" if "date" in cols else None)
        if ts_col is None:
            raise MissingArtifactError(f"Pairwise CSV missing ts column (ts_utc/date): {p}. Have={cols[:30]}")
        d = pd.read_csv(p, usecols=[ts_col])
        d[ts_col] = pd.to_datetime(d[ts_col], utc=True, errors="raise")
        if d.empty:
            continue
        pair_n.append(int(len(d)))
        starts.append(pd.Timestamp(d[ts_col].min()))
        ends.append(pd.Timestamp(d[ts_col].max()))

    if not pair_n:
        raise MissingArtifactError(f"No non-empty pairwise CSVs under: {pairwise_root}")

    arr = pd.Series(pair_n, dtype=float)
    return pd.DataFrame([
        {
            "n_pairs": int(len(pair_n)),
            "min_n_ij": int(arr.min()),
            "median_n_ij": float(arr.median()),
            "max_n_ij": int(arr.max()),
            "min_start_utc": (min(starts).isoformat() if starts else ""),
            "max_end_utc": (max(ends).isoformat() if ends else ""),
        }
    ])

# --- Appendix A.4: Top-8 pairwise heatmaps (A1/A2/A3) + distribution table (A2) ---

def _pick_time_col(cols: list[str]) -> str:
    if "ts_utc" in cols:
        return "ts_utc"
    if "date" in cols:
        return "date"
    raise MissingArtifactError(f"Pairwise CSV missing time column (ts_utc/date). Have={cols[:30]}")


def _heatmap_square(
    mat: pd.DataFrame,
    title: str,
    out_path: Path,
    *,
    fig_w: float,
    fig_h: float,
    dpi: int,
) -> None:
    # NaNs are allowed (rendered as empty)
    _set_matplotlib_style(fig_w, fig_h)
    plt.figure(figsize=(float(max(8.0, fig_w)), float(max(6.5, fig_h))))
    plt.imshow(mat.values, aspect="auto")
    plt.xticks(range(len(mat.columns)), mat.columns, rotation=45, ha="right", fontsize=8)
    plt.yticks(range(len(mat.index)), mat.index, fontsize=8)
    plt.title(title)
    plt.colorbar()
    plt.tight_layout()
    _ensure_dir(out_path.parent)
    plt.savefig(out_path, dpi=int(dpi))
    plt.close()


def _a4_pairwise_metrics(
    pairwise_root: Path,
    *,
    es99_a_col: str,
    es99_b_col: str,
    es99_ratio_col: str,
    exc_col: str,
    eps: float = 1e-12,
) -> tuple[pd.DataFrame, list[str]]:
    """Compute per-pair A.4 metrics from pairwise `var_es_predictions.csv` artefacts.

    Metrics per pair (robust):
      - delta_es99_stress_med: median(ES99_A - ES99_B | bucket==stress)
      - ratio_es99_stress_calm_med: median(ES99_ratio|stress) / median(ES99_ratio|calm)
      - exceed_rate: mean(exceed flag)

    Returns:
      - per_pair_df with asset_i/asset_j columns
      - ordered asset list
    """
    paths = _pairwise_var_es_paths(pairwise_root)

    rows: list[dict[str, Any]] = []
    assets_set: set[str] = set()

    for p in paths:
        pair_name = p.parent.name
        a, b = _parse_pair_dir_name(pair_name)
        assets_set.add(a)
        assets_set.add(b)

        cols = list(pd.read_csv(p, nrows=0).columns)
        ts_col = _pick_time_col(cols)
        if "bucket" not in cols:
            raise MissingArtifactError(f"Pairwise CSV missing 'bucket': {p}")

        # Read needed columns only
        need_cols = [ts_col, "bucket", es99_a_col, es99_b_col, es99_ratio_col, exc_col]
        miss = [c for c in need_cols if c not in cols]
        if miss:
            raise MissingArtifactError(f"Pairwise CSV missing columns {miss}: {p}. Have={cols[:40]}")

        df = pd.read_csv(p, usecols=need_cols)
        df[ts_col] = pd.to_datetime(df[ts_col], utc=True, errors="raise")
        df["bucket"] = df["bucket"].astype(str).str.lower().str.strip()

        for c in [es99_a_col, es99_b_col, es99_ratio_col]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df[exc_col] = pd.to_numeric(df[exc_col], errors="coerce")

        stress = df[df["bucket"] == "stress"].copy()
        calm = df[df["bucket"] == "calm"].copy()

        ds = (pd.to_numeric(stress[es99_a_col], errors="coerce") - pd.to_numeric(stress[es99_b_col], errors="coerce")).to_numpy(dtype=float)
        ds = ds[np.isfinite(ds)]
        delta_es99_stress_med = float(np.median(ds)) if ds.size else float("nan")

        s = pd.to_numeric(stress[es99_ratio_col], errors="coerce").to_numpy(dtype=float)
        c = pd.to_numeric(calm[es99_ratio_col], errors="coerce").to_numpy(dtype=float)
        s = s[np.isfinite(s)]
        c = c[np.isfinite(c)]
        med_s = float(np.median(s)) if s.size else float("nan")
        med_c = float(np.median(c)) if c.size else float("nan")
        ratio_es99_stress_calm_med = (med_s / max(med_c, float(eps))) if np.isfinite(med_s) and np.isfinite(med_c) else float("nan")

        e = pd.to_numeric(df[exc_col], errors="coerce").to_numpy(dtype=float)
        e = e[np.isfinite(e)]
        exceed_rate = float(np.mean(e > 0.5)) if e.size else float("nan")

        rows.append(
            {
                "pair": pair_name,
                "asset_i": a,
                "asset_j": b,
                "delta_es99_stress_med": delta_es99_stress_med,
                "ratio_es99_stress_calm_med": ratio_es99_stress_calm_med,
                "exceed_rate": exceed_rate,
                "n_total": int(len(df)),
                "n_stress": int(len(stress)),
                "n_calm": int(len(calm)),
            }
        )

    if not rows:
        raise MissingArtifactError(f"No pairwise rows parsed under: {pairwise_root}")

    per_pair = pd.DataFrame(rows)
    assets = sorted(assets_set)
    return per_pair, assets


def _a4_build_symmetric_matrix(per_pair: pd.DataFrame, assets: list[str], value_col: str) -> pd.DataFrame:
    mat = pd.DataFrame(np.nan, index=assets, columns=assets, dtype=float)
    for _, r in per_pair.iterrows():
        i = str(r["asset_i"])
        j = str(r["asset_j"])
        v = r[value_col]
        if i in mat.index and j in mat.columns:
            mat.loc[i, j] = v
            mat.loc[j, i] = v
    # leave diagonal NaN
    return mat





def fig_a1_heatmap_delta_es99_stress(
    pairwise_root: Path,
    out_path: Path,
    *,
    es99_a_col: str,
    es99_b_col: str,
    es99_ratio_col: str,
    exc_col: str,
    eps: float = 1e-12,
    fig_w: float,
    fig_h: float,
    dpi: int,
) -> list[str]:
    
    per_pair, assets = _a4_pairwise_metrics(
        pairwise_root,
        es99_a_col=es99_a_col,
        es99_b_col=es99_b_col,
        es99_ratio_col=es99_ratio_col,
        exc_col=exc_col,
        eps=float(eps),
    )
    mat = _a4_build_symmetric_matrix(per_pair, assets, "delta_es99_stress_med")
    _heatmap_square(
        mat,
        f"A1 — heatmap ΔES99^stress (median per pair): {es99_a_col} − {es99_b_col}",
        out_path,
        fig_w=fig_w,
        fig_h=fig_h,
        dpi=dpi,
    )
    return assets


def fig_a2_heatmap_ratio_es99_stress_calm(
    pairwise_root: Path,
    out_path: Path,
    *,
    es99_a_col: str,
    es99_b_col: str,
    es99_ratio_col: str,
    exc_col: str,
    eps: float = 1e-12,
    fig_w: float,
    fig_h: float,
    dpi: int,
) -> None:
    per_pair, assets = _a4_pairwise_metrics(
        pairwise_root,
        es99_a_col=es99_a_col,
        es99_b_col=es99_b_col,
        es99_ratio_col=es99_ratio_col,
        exc_col=exc_col,
        eps=float(eps),
    )

    mat = _a4_build_symmetric_matrix(per_pair, assets, "ratio_es99_stress_calm_med")
    _heatmap_square(
        mat,
        f"A2 — heatmap ratio ES99 stress/calm (median): {es99_ratio_col}",
        out_path,
        fig_w=fig_w,
        fig_h=fig_h,
        dpi=dpi,
    )


def fig_a3_heatmap_exceedance_rate(
    pairwise_root: Path,
    out_path: Path,
    *,
    es99_a_col: str,
    es99_b_col: str,
    es99_ratio_col: str,
    exc_col: str,
    eps: float = 1e-12,
    fig_w: float,
    fig_h: float,
    dpi: int,
) -> None:
    per_pair, assets = _a4_pairwise_metrics(
        pairwise_root,
        es99_a_col=es99_a_col,
        es99_b_col=es99_b_col,
        es99_ratio_col=es99_ratio_col,
        exc_col=exc_col,
        eps=float(eps),
    )

    mat = _a4_build_symmetric_matrix(per_pair, assets, "exceed_rate")
    _heatmap_square(
        mat,
        f"A3 — heatmap exceedance rate (observed): {exc_col}",
        out_path,
        fig_w=fig_w,
        fig_h=fig_h,
        dpi=dpi,
    )


def table_a2_top8_pairwise_metric_distrib(
    pairwise_root: Path,
    *,
    es99_a_col: str,
    es99_b_col: str,
    es99_ratio_col: str,
    exc_col: str,
    eps: float = 1e-12,
) -> pd.DataFrame:
    """Table A2 — distribution summaries across the 28 pairs.

    Columns: metric, n_pairs_valid, min, q25, median, q75, max
    """
    per_pair, assets = _a4_pairwise_metrics(
        pairwise_root,
        es99_a_col=es99_a_col,
        es99_b_col=es99_b_col,
        es99_ratio_col=es99_ratio_col,
        exc_col=exc_col,
        eps=float(eps),
    )

    def _summarize(x: pd.Series) -> dict[str, Any]:
        v = pd.to_numeric(x, errors="coerce").to_numpy(dtype=float)
        v = v[np.isfinite(v)]
        if v.size == 0:
            return {"n_pairs_valid": 0, "min": np.nan, "q25": np.nan, "median": np.nan, "q75": np.nan, "max": np.nan}
        q25, q50, q75 = np.quantile(v, [0.25, 0.5, 0.75])
        return {
            "n_pairs_valid": int(v.size),
            "min": float(np.min(v)),
            "q25": float(q25),
            "median": float(q50),
            "q75": float(q75),
            "max": float(np.max(v)),
        }

    rows: list[dict[str, Any]] = []
    for metric, col in [
        ("delta_es99_stress_med", "delta_es99_stress_med"),
        ("ratio_es99_stress_calm_med", "ratio_es99_stress_calm_med"),
        ("exceed_rate", "exceed_rate"),
    ]:
        s = _summarize(per_pair[col])
        s["metric"] = metric
        rows.append(s)

    out = pd.DataFrame(rows)[["metric", "n_pairs_valid", "min", "q25", "median", "q75", "max"]]
    return out

def _median_safe(x: pd.Series) -> float:
    v = pd.to_numeric(x, errors="coerce").to_numpy(dtype=float)
    v = v[np.isfinite(v)]
    return float(np.median(v)) if v.size else float("nan")


def _mean_flag_rate(x: pd.Series) -> float:
    v = pd.to_numeric(x, errors="coerce").to_numpy(dtype=float)
    v = v[np.isfinite(v)]
    return float(np.mean(v > 0.5)) if v.size else float("nan")


def _sign_str(x: float) -> str:
    if not np.isfinite(x):
        return ""
    if x > 0:
        return "+"
    if x < 0:
        return "-"
    return "0"


def _load_var_es_with_bucket(pred_csv: Path) -> pd.DataFrame:
    """Load a var_es_predictions.csv that already contains a 'bucket' column."""
    if not pred_csv.exists():
        raise MissingArtifactError(f"Missing var_es_predictions.csv: {pred_csv}")
    df = _load_csv(pred_csv)
    need = ["date", "bucket"]
    miss = [c for c in need if c not in df.columns]
    if miss:
        raise MissingArtifactError(f"{pred_csv} missing columns {miss}. Have={list(df.columns)[:30]}")
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    df["bucket"] = df["bucket"].astype(str).str.lower().str.strip()
    return df



def table_a4_tau_theta_audit(theta_csv: Path) -> pd.DataFrame:
    if not theta_csv.exists():
        raise MissingArtifactError(f"Missing A4 theta CSV: {theta_csv}")

    df = _load_csv(theta_csv).copy()
    need = ["bucket", "family", "theta"]
    miss = [c for c in need if c not in df.columns]
    if miss:
        raise MissingArtifactError(f"A4 needs columns {miss}. Have={list(df.columns)[:40]}")

    df["bucket"] = df["bucket"].astype(str).str.lower().str.strip()
    df["family"] = df["family"].astype(str).str.lower().str.strip()
    df["theta"] = pd.to_numeric(df["theta"], errors="coerce")

    def tau_from_theta(fam: str, th: float) -> float:
        if not (th is not None and th == th):
            return float("nan")
        if fam == "clayton":
            return float(th / (th + 2.0)) if th > 0 else float("nan")
        if fam == "gumbel":
            return float(1.0 - 1.0 / th) if th >= 1.0 else float("nan")
        return float("nan")

    def theta_domain(fam: str) -> str:
        if fam == "clayton":
            return "(0, +inf)"
        if fam == "gumbel":
            return "[1, +inf)"
        return ""

    def admissible(fam: str, th: float) -> int:
        if not (th is not None and th == th):
            return 0
        if fam == "clayton":
            return int(th > 0.0)
        if fam == "gumbel":
            return int(th >= 1.0)
        return 0

    df["tau_implied"] = [tau_from_theta(f, float(t)) for f, t in zip(df["family"], df["theta"])]
    df["theta_domain"] = [theta_domain(f) for f in df["family"]]
    df["admissible"] = [admissible(f, float(t)) for f, t in zip(df["family"], df["theta"])]

    if "fit_status" in df.columns:
        df["fit_status"] = df["fit_status"].astype(str).str.lower().str.strip()
    else:
        df["fit_status"] = df["admissible"].map(lambda x: "ok" if int(x) == 1 else "invalid")

    keep = ["bucket", "family", "theta", "tau_implied", "theta_domain", "admissible", "fit_status"]
    for c in ["refit_index", "n_total", "n_stress", "n_calm"]:
        if c in df.columns:
            keep.insert(0, c)

    out = df[keep].copy()
    order_cols = [c for c in ["bucket", "family"] if c in out.columns]
    if order_cols:
        out = out.sort_values(order_cols).reset_index(drop=True)
    return out



def table_a3_pattern_summary_daily_vs_4h(
    daily_var_es_pred_csv: Path,
    h4_var_es_pred_csv: Path,
    *,
    es99_a_col: str = "ES99_indep",
    es99_b_col: str = "ES99_static_gauss",
    es99_ratio_col: str = "ES99_static_gauss",
    exceed_col: str = "exceed99_static_gauss",
    ratio_eps: float = 1e-12,
) -> pd.DataFrame:
    """Table A3 — pattern summary daily vs 4h (descriptive only).

    Metrics:
      - delta_es99_stress_med = median(ES99_A - ES99_B | bucket==stress)
      - ratio_es99_stress_calm_med = median(ES99_ratio|stress) / median(ES99_ratio|calm)
      - exceed_rate = mean(exceed_col) over all rows (flag rate)
    """
    d0 = _load_var_es_with_bucket(daily_var_es_pred_csv)
    d1 = _load_var_es_with_bucket(h4_var_es_pred_csv)

    for df, tag, p in [(d0, "daily", daily_var_es_pred_csv), (d1, "h4", h4_var_es_pred_csv)]:
        need_cols = [es99_a_col, es99_b_col, es99_ratio_col, exceed_col]
        miss = [c for c in need_cols if c not in df.columns]
        if miss:
            raise MissingArtifactError(f"{tag} {p} missing columns {miss}. Have={list(df.columns)[:30]}")
        for c in need_cols:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    def _metrics(df: pd.DataFrame) -> dict[str, float]:
        stress = df[df["bucket"] == "stress"]
        calm = df[df["bucket"] == "calm"]

        delta_stress = _median_safe(stress[es99_a_col] - stress[es99_b_col])

        med_stress = _median_safe(stress[es99_ratio_col])
        med_calm = _median_safe(calm[es99_ratio_col])
        ratio_sc = (
            float(med_stress / max(float(med_calm), float(ratio_eps)))
            if np.isfinite(med_stress) and np.isfinite(med_calm)
            else float("nan")
        )

        exc_rate = _mean_flag_rate(df[exceed_col])

        return {
            "delta_es99_stress_med": delta_stress,
            "ratio_es99_stress_calm_med": ratio_sc,
            "exceed_rate": exc_rate,
        }

    m_daily = _metrics(d0)
    m_h4 = _metrics(d1)

    rows: list[dict[str, Any]] = []
    for metric in ["delta_es99_stress_med", "ratio_es99_stress_calm_med", "exceed_rate"]:
        v_d = float(m_daily.get(metric, float("nan")))
        v_h = float(m_h4.get(metric, float("nan")))
        s_d = _sign_str(v_d)
        s_h = _sign_str(v_h)

        rows.append(
            {
                "metric": metric,
                "daily_value": v_d,
                "h4_value": v_h,
                "daily_sign": s_d,
                "h4_sign": s_h,
                "direction_agree": int((s_d != "") and (s_d == s_h)),
                "amplitude_ratio_h4_over_daily": (v_h / v_d) if np.isfinite(v_h) and np.isfinite(v_d) and v_d != 0 else float("nan"),
                "note": "descriptive; no causal claim",
            }
        )

    return pd.DataFrame(rows)

def fig_a4_overlay_daily_vs_4h(
    daily_var_es_pred_csv: Path,
    h4_var_es_pred_csv: Path,
    out_path: Path,
    *,
    es99_a_col: str = "ES99_indep",
    es99_b_col: str = "ES99_static_gauss",
    bucket: str = "stress",
    rolling: int = 14,
    fig_w: float,
    fig_h: float,
    dpi: int,
) -> None:
    """Figure A4 — overlay daily vs 4h on an aggregated metric (descriptive).

    We plot per-day median ΔES99_t = ES99_A - ES99_B for a chosen bucket (default: stress).
    For 4h data, we first aggregate to daily by flooring timestamps to UTC day.
    Optional rolling median smoothing (default: 14 days) improves readability.

    No causal claim: this is only a stability/robustness diagnostic.
    """
    d0 = _load_var_es_with_bucket(daily_var_es_pred_csv)
    d1 = _load_var_es_with_bucket(h4_var_es_pred_csv)

    for df, tag, p in [(d0, "daily", daily_var_es_pred_csv), (d1, "h4", h4_var_es_pred_csv)]:
        miss = [c for c in [es99_a_col, es99_b_col] if c not in df.columns]
        if miss:
            raise MissingArtifactError(f"{tag} {p} missing columns {miss}. Have={list(df.columns)[:30]}")
        df[es99_a_col] = pd.to_numeric(df[es99_a_col], errors="coerce")
        df[es99_b_col] = pd.to_numeric(df[es99_b_col], errors="coerce")

    def _daily_series(df: pd.DataFrame) -> pd.DataFrame:
        sub = df[df["bucket"] == str(bucket).lower().strip()].copy()
        if sub.empty:
            return pd.DataFrame({"day": [], "value": []})
        sub["day"] = sub["date"].dt.floor("D")
        sub["delta"] = sub[es99_a_col] - sub[es99_b_col]
        g = (
            sub.dropna(subset=["day", "delta"])
            .groupby("day", as_index=False)["delta"]
            .median()
            .rename(columns={"delta": "value"})
            .sort_values("day")
            .reset_index(drop=True)
        )
        if int(rolling) and int(rolling) > 1 and not g.empty:
            g["value"] = g["value"].rolling(int(rolling), min_periods=max(2, int(rolling) // 3)).median()
        return g

    s_daily = _daily_series(d0)
    s_h4 = _daily_series(d1)

    if s_daily.empty and s_h4.empty:
        raise MissingArtifactError(f"A4: no data for bucket='{bucket}' after aggregation (daily and 4h).")

    _set_matplotlib_style(fig_w, fig_h)
    plt.figure(figsize=(float(fig_w), float(fig_h)))
    if not s_daily.empty:
        plt.plot(s_daily["day"], s_daily["value"], label=f"daily (bucket={bucket})")
    if not s_h4.empty:
        plt.plot(s_h4["day"], s_h4["value"], label=f"4h→daily agg (bucket={bucket})")

    plt.axhline(0.0, linestyle=":", linewidth=1.0)
    ttl_roll = f", rolling={int(rolling)}d" if int(rolling) and int(rolling) > 1 else ""
    plt.title(f"A4 — daily vs 4h: median ΔES99 (A−B) in {bucket}{ttl_roll}")
    plt.xlabel("day (UTC)")
    plt.ylabel(f"median( {es99_a_col} − {es99_b_col} )")
    plt.grid(axis="y", linestyle=":", linewidth=1.0)
    plt.legend(loc="best", fontsize=9)
    _ensure_dir(out_path.parent)
    plt.tight_layout()
    plt.savefig(out_path, dpi=int(dpi))
    plt.close()


def fig_a5_lambda_ratio_barplot(
    lambda_csv: Path,
    out_path: Path,
    *,
    include_t: bool = False,
    eps: float = 1e-12,
    fig_w: float,
    fig_h: float,
    dpi: int,
) -> None:
    if not lambda_csv.exists():
        raise MissingArtifactError(f"Missing A5 lambda CSV: {lambda_csv}")

    df = _load_csv(lambda_csv)
    need = ["bucket", "copula", "method", "lambda_L", "lambda_U"]
    miss = [c for c in need if c not in df.columns]
    if miss:
        raise MissingArtifactError(f"A5 lambda CSV missing columns {miss}. Have={list(df.columns)[:30]}")

    df = df.copy()
    df["bucket"] = df["bucket"].astype(str).str.lower().str.strip()
    df["copula"] = df["copula"].astype(str).str.lower().str.strip()
    df["method"] = df["method"].astype(str).str.lower().str.strip()

    df["lambda_L"] = pd.to_numeric(df["lambda_L"], errors="coerce")
    df["lambda_U"] = pd.to_numeric(df["lambda_U"], errors="coerce")
    df = df.dropna(subset=["lambda_L", "lambda_U"]).reset_index(drop=True)

    allow = ["clayton", "gumbel"]
    if include_t:
        allow.append("t")
    df = df[df["copula"].isin(allow)].reset_index(drop=True)
    if df.empty:
        raise MissingArtifactError(f"A5: no rows after filtering copulas={allow}")

    buckets = ["calm", "stress"]
    cop_order = [c for c in ["clayton", "gumbel", "t"] if c in df["copula"].unique().tolist()]
    if not cop_order:
        cop_order = sorted(df["copula"].unique().tolist())

    def _val(bucket: str, copula: str, method: str, tail: str) -> float:
        sub = df[(df["bucket"] == bucket) & (df["copula"] == copula) & (df["method"] == method)]
        if sub.empty:
            return float("nan")
        vv = pd.to_numeric(sub[tail], errors="coerce").dropna()
        return float(vv.iloc[0]) if len(vv) else float("nan")

    _set_matplotlib_style(fig_w, fig_h)
    fig, axes = plt.subplots(1, 2, figsize=(max(11.0, float(fig_w) * 1.6), max(4.0, float(fig_h))), sharey=True)

    width = 0.18
    x = np.arange(len(cop_order), dtype=float)

    legend_labels = ["param: λ_L", "param: λ_U", "qcheck: λ_L", "qcheck: λ_U"]

    for ax, bucket in zip(axes, buckets):
        y_param_L = [_val(bucket, c, "param", "lambda_L") for c in cop_order]
        y_param_U = [_val(bucket, c, "param", "lambda_U") for c in cop_order]
        y_q_L     = [_val(bucket, c, "qcheck_mc", "lambda_L") for c in cop_order]
        y_q_U     = [_val(bucket, c, "qcheck_mc", "lambda_U") for c in cop_order]

        ax.bar(x - 1.5 * width, y_param_L, width=width, label=legend_labels[0])
        ax.bar(x - 0.5 * width, y_param_U, width=width, label=legend_labels[1])
        ax.bar(x + 0.5 * width, y_q_L,     width=width, label=legend_labels[2])
        ax.bar(x + 1.5 * width, y_q_U,     width=width, label=legend_labels[3])

        ax.set_xticks(x)
        ax.set_xticklabels(cop_order)
        ax.set_title(bucket)
        ax.grid(axis="y", linestyle=":", linewidth=1.0)

    axes[0].set_ylabel("tail dependence level")
    fig.suptitle("A5 — tail dependence levels λ_L and λ_U (param vs q-checks)")
    fig.legend(legend_labels, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=[0, 0.06, 1, 0.95])

    _ensure_dir(out_path.parent)
    fig.savefig(out_path, dpi=int(dpi))
    plt.close(fig)

def main() -> int:
    ap = argparse.ArgumentParser()

    ap.add_argument("--paper-id", required=True, help="paper_id used by make_paper.py outputs (e.g. hc_taildep_v0_camera_ready)")
    ap.add_argument("--stress-csv", default="", help="Override path to tab_FX_stress_mask.csv")
    ap.add_argument("--regimes-csv", default="", help="Preferred: canonical regimes.csv for M1 (date + regime or S_t)")
    ap.add_argument("--regime-col", default="", help="When --regimes-csv has is_stress_* flags, choose which column to use (e.g. is_stress_JDown)")
    ap.add_argument("--splits-json", default="", help="Override path to dataset splits.json (UTC)")
    ap.add_argument("--refit-every", type=int, default=63, help="Refit frequency in observations")
    ap.add_argument("--n-min", type=int, default=50, help="Min obs per regime in TRAIN to avoid fallback")

    ap.add_argument("--fig-w", type=float, default=7.0)
    ap.add_argument("--fig-h", type=float, default=4.0)
    ap.add_argument("--dpi", type=int, default=200)
    ap.add_argument("--float-fmt", type=str, default="%.6g")
    ap.add_argument("--table-round", type=int, default=6)

    ap.add_argument("--fallback-fig-threshold", type=float, default=0.0, help="Only write figure if mean(fallback_used) >= threshold")

    # --- M2 (logistic gating) arguments ---
    ap.add_argument("--m2-predictions-csv", default="", help="Path to j5_gating/predictions.csv")
    ap.add_argument("--m2-params-csv", default="", help="Path to j5_gating/tables/params_summary.csv")
    ap.add_argument("--m2-n-min", type=int, default=200, help="Min regime size for M2 audit table")
    ap.add_argument("--m2-enable", action="store_true", help="Enable M2 artifacts generation")
    ap.add_argument("--m2-fig-threshold", type=float, default=0.0, help="Threshold for M2 figure (ignored unless no data)")

    ap.add_argument("--f8-enable", action="store_true", help="Write Figure F8 (cum deltas logit vs static, thr vs static) using predictions.csv")
    ap.add_argument("--f8-hist", action="store_true", help="Also write optional histogram of per-day deltas for F8")
    ap.add_argument("--f8-bins", type=int, default=60, help="Histogram bins for --f8-hist")

    # --- J6 (MS2) policy usage artifacts ---
    ap.add_argument("--j6-enable", action="store_true", help="Enable J6 policy/usage audit tables")
    ap.add_argument("--j6-predictions-csv", default="", help="Path to j6_ms2_*/predictions.csv")
    ap.add_argument("--j6-params-csv", default="", help="Optional path to j6_ms2_*/tables/params_summary.csv")
    ap.add_argument("--j6-refit-every", type=int, default=40, help="Refit frequency (obs) for J6 block aggregation")
    ap.add_argument("--j6-dm-csv", default="", help="Optional path to j6_ms2_*/tables/dm_summary.csv")
    ap.add_argument("--j6-figs", action="store_true", help="Write J6 figures F9 (cum delta) and F10 (hist delta)")
    ap.add_argument("--j6-fig-bins", type=int, default=60, help="Histogram bins for J6 F10")

    # --- Impact (VAR/ES) artifacts: F11/F12/T8/T9 ---
    ap.add_argument("--impact-enable", action="store_true", help="Enable impact VAR/ES artifacts (F11/F12/T8/T9)")
    ap.add_argument("--var-es-predictions-csv", default="", help="Path to var_es_predictions.csv")
    ap.add_argument("--impact-regime-col", default="", help="Optional: column in regimes.csv to define stress (e.g. is_stress_JDown)")
    ap.add_argument("--f11-max-models", type=int, default=4, help="Max ES99 series to plot in F11")
    ap.add_argument("--f11-es99-cols", default="", help="Optional comma-separated ES99 column names to plot")
    ap.add_argument("--f12-a-col", default="", help="Optional ES99 column name for model A (MS)")
    ap.add_argument("--f12-b-col", default="", help="Optional ES99 column name for model B (THR)")

    # --- Appendix A (Top-8 pairwise) ---
    ap.add_argument("--top8-enable", action="store_true", help="Enable Appendix A (Top-8) artifacts")
    ap.add_argument("--top8-returns-csv", default="", help="Wide returns panel CSV (date + assets); optional if pairwise root is provided")
    ap.add_argument("--top8-assets", default="", help="Optional comma-separated asset list to force selection (only used with returns.csv path)")
    ap.add_argument("--top8-k", type=int, default=8, help="Top-k assets to select (only used with returns.csv path)")
    ap.add_argument("--top8-pairwise-root", default="", help="Path to a run dir containing pairs/*/var_es_predictions.csv")
    ap.add_argument("--top8-heatmaps", action="store_true", help="Write A.4 heatmaps + Table A2 (requires --top8-pairwise-root)")
    ap.add_argument("--top8-es99-a-col", default="ES99_indep", help="ES99 column for model A (used in ΔES99 stress)")
    ap.add_argument("--top8-es99-b-col", default="ES99_static_gauss", help="ES99 column for model B (used in ΔES99 stress)")
    ap.add_argument("--top8-es99-ratio-col", default="ES99_static_gauss", help="ES99 column used for stress/calm ratio")
    ap.add_argument("--top8-exceed-col", default="exceed99_static_gauss", help="Exceedance flag column used for calibration heatmap")
    ap.add_argument("--top8-ratio-eps", type=float, default=1e-12, help="Epsilon floor for ratio denominator")

    # --- Appendix A.6 (Sensitivity): daily vs 4h pattern summary ---

    ap.add_argument("--a4-enable", action="store_true", help="Enable optional Table A4 (tau<->theta audit)")
    ap.add_argument("--a4-theta-csv", default="", help="CSV with columns: bucket,family,theta (optionally fit_status/refit_index)")

    ap.add_argument("--a6-enable", action="store_true", help="Enable Appendix A.6 sensitivity artifacts (daily vs 4h)")
    ap.add_argument("--a6-daily-var-es-csv", default="", help="Daily var_es_predictions.csv (must include 'bucket')")
    ap.add_argument("--a6-h4-var-es-csv", default="", help="4h var_es_predictions.csv (must include 'bucket')")
    ap.add_argument("--a6-es99-a-col", default="ES99_indep")
    ap.add_argument("--a6-es99-b-col", default="ES99_static_gauss")
    ap.add_argument("--a6-es99-ratio-col", default="ES99_static_gauss")
    ap.add_argument("--a6-exceed-col", default="exceed99_static_gauss")
    ap.add_argument("--a6-ratio-eps", type=float, default=1e-12)
    ap.add_argument("--a6-fig-a4", action="store_true", help="Write Figure A4 overlay daily vs 4h (aggregated ΔES99 in bucket)")
    ap.add_argument("--a6-a4-bucket", default="stress", help="Bucket for A4 overlay (stress/calm)")
    ap.add_argument("--a6-a4-rolling", type=int, default=14, help="Rolling median window (days) for A4; set 0/1 to disable")

    # --- Appendix A.5: Tail-dependence ratio λ_L/λ_U (Clayton vs Gumbel; optional t) ---
    ap.add_argument("--a5-enable", action="store_true", help="Enable Appendix A.5 Figure A5 (λ_L/λ_U barplot)")
    ap.add_argument("--a5-lambda-csv", default="", help="CSV with columns: copula, method, lambda_L, lambda_U")
    ap.add_argument("--a5-include-t", action="store_true", help="Include copula 't' as a reference in A5")
    ap.add_argument("--a5-eps", type=float, default=1e-12, help="Epsilon floor for lambda_U in ratio λ_L/λ_U")

    ap.add_argument("--t13-enable", action="store_true", help="Enable Table T13: Statistical vs economic significance summary")
    ap.add_argument("--t13-dm-csv", default="", help="Path to DM summary CSV (must contain model_a/model_b and pvalue)")
    ap.add_argument("--t13-var-es-predictions-csv", default="", help="Path to var_es_predictions.csv for economic metrics")
    ap.add_argument("--t13-regimes-csv", default="", help="Path to regimes.csv for stress definition (date + S_t/regime/is_stress_*)")
    ap.add_argument("--t13-regime-col", default="", help="Optional stress flag col in regimes.csv (e.g. is_stress_JDown)")
    ap.add_argument("--t13-alpha", type=float, default=0.05, help="Alpha threshold for statistical significance")
    ap.add_argument("--t13-es99-abs-threshold", type=float, default=0.0, help="Absolute threshold on |ΔES99| to flag economic materiality")

    args = ap.parse_args()

    # --- Outputs ---
    tab_out, fig_out = _infer_default_outputs_from_paper_id(args.paper_id)
    _ensure_dir(tab_out)
    _ensure_dir(fig_out)

    written: list[Path] = []

    # --- Splits ---
    splits_json = Path(args.splits_json) if str(args.splits_json).strip() else None
    if splits_json is None or not splits_json.exists():
        raise MissingArtifactError("--splits-json is required and must exist.")
    splits = load_splits(splits_json)

    # --- Regimes / stress sequence for M1 + impact ---
    regimes_csv = Path(args.regimes_csv) if str(args.regimes_csv).strip() else None
    stress_csv = Path(args.stress_csv) if str(args.stress_csv).strip() else None

    stress_df = None
    if regimes_csv is not None and regimes_csv.exists():
        stress_df = load_regimes_df(regimes_csv, regime_col=(args.regime_col or None))
    elif stress_csv is not None and stress_csv.exists():
        stress_df = load_stress_df(stress_csv)

    # --- M1 audit artifacts (fallback by block) ---
    # Optional: only run if we have a stress/regime sequence available.
    if stress_df is not None:
        m1_tbl = table_m1_regime_sizes_fallback_by_block(
            stress_df,
            splits,
            refit_every=int(args.refit_every),
            n_min=int(args.n_min),
        )
        out_m1 = tab_out / "tab_M1_regime_sizes_fallback_by_block.csv"
        _write_csv_stable(m1_tbl, out_m1, float_fmt=str(args.float_fmt), table_round=int(args.table_round))
        written.append(out_m1)
        print("[OK]", out_m1)

        fb_rate = float(m1_tbl["fallback_used"].mean()) if len(m1_tbl) else 0.0
        if fb_rate >= float(args.fallback_fig_threshold):
            out_m1_fig = fig_out / "fig_M1_fallback_rate_by_block.png"
            fig_m1_fallback_rate_by_block(
                m1_tbl,
                out_m1_fig,
                fig_w=float(args.fig_w),
                fig_h=float(args.fig_h),
                dpi=int(args.dpi),
            )
            written.append(out_m1_fig)
            print("[OK]", out_m1_fig, "fallback_rate=", fb_rate)
    else:
        print("[SKIP] M1 audit: provide --regimes-csv or --stress-csv to generate M1 fallback artifacts.")

    # --- J6 (MS2) policy artifacts ---
    if args.j6_enable:
        if not str(args.j6_predictions_csv).strip():
            raise MissingArtifactError("--j6-enable requires --j6-predictions-csv")
        j6_pred = Path(args.j6_predictions_csv)

        j6_usage = table_j6_policy_usage_by_block(j6_pred, splits, refit_every=int(args.j6_refit_every))
        out_j6_usage = tab_out / "tab_J6_policy_usage_by_block.csv"
        _write_csv_stable(j6_usage, out_j6_usage, float_fmt=str(args.float_fmt), table_round=int(args.table_round))
        written.append(out_j6_usage)
        print("[OK]", out_j6_usage)

        if str(args.j6_params_csv).strip():
            j6_params = Path(args.j6_params_csv)
            t6 = table_j6_params_summary_policy(j6_params)
            out_t6 = tab_out / "tab_T6_params_summary.csv"
            _write_csv_stable(t6, out_t6, float_fmt=str(args.float_fmt), table_round=int(args.table_round))
            written.append(out_t6)
            print("[OK]", out_t6)

        if args.j6_figs:
            out_f9 = fig_out / "fig_F9_delta_logscore_cum_policy_vs_thr.png"
            fig_j6_cumdelta_ms_vs_thr(j6_pred, splits, out_f9, fig_w=float(args.fig_w), fig_h=float(args.fig_h), dpi=int(args.dpi))
            written.append(out_f9)
            print("[OK]", out_f9)

            out_f10 = fig_out / "fig_F10_delta_logscore_hist_policy_vs_thr.png"
            fig_j6_hist_delta_ms_vs_thr(j6_pred, splits, out_f10, fig_w=float(args.fig_w), fig_h=float(args.fig_h), dpi=int(args.dpi), bins=int(args.j6_fig_bins))
            written.append(out_f10)
            print("[OK]", out_f10)

    # --- Impact risk artifacts (F11/F12/T8/T9) ---
    if args.impact_enable:
        if regimes_csv is None or not regimes_csv.exists():
            raise MissingArtifactError("--impact-enable requires --regimes-csv (for stress definition used by F12/T8).")
        if not str(args.var_es_predictions_csv).strip():
            raise MissingArtifactError("--impact-enable requires --var-es-predictions-csv")
        var_csv = Path(args.var_es_predictions_csv)

        t8 = table_t8_var_es_summary(
            var_csv,
            regimes_csv if (regimes_csv is not None and regimes_csv.exists()) else (stress_csv if (stress_csv is not None and stress_csv.exists()) else Path("")),
            splits,
            regime_col=(args.impact_regime_col or None),
        )
        out_t8 = tab_out / "tab_T8_var_es_summary.csv"
        _write_csv_stable(t8, out_t8, float_fmt=str(args.float_fmt), table_round=int(args.table_round))
        written.append(out_t8)
        print("[OK]", out_t8)

        out_f11 = fig_out / "fig_F11_es99_timeseries.png"
        es99_cols = fig_f11_es99_timeseries(
            var_csv,
            splits,
            out_f11,
            max_models=int(args.f11_max_models),
            es99_cols_csv=str(args.f11_es99_cols),
            fig_w=float(args.fig_w),
            fig_h=float(args.fig_h),
            dpi=int(args.dpi),
        )
        written.append(out_f11)
        print("[OK]", out_f11, "es99_cols=", es99_cols)

        if regimes_csv is None or not regimes_csv.exists():
            raise MissingArtifactError("F12 requires --regimes-csv (preferred) to define stress buckets consistently.")
        out_f12 = fig_out / "fig_F12_stress_delta_es99.png"
        A, B = fig_f12_stress_delta_es99(
            var_csv,
            regimes_csv,
            splits,
            out_f12,
            regime_col=(args.impact_regime_col or None),
            a_col=str(args.f12_a_col),
            b_col=str(args.f12_b_col),
            fig_w=float(args.fig_w),
            fig_h=float(args.fig_h),
            dpi=int(args.dpi),
        )
        written.append(out_f12)
        print("[OK]", out_f12, "A=", A, "B=", B)

        t9 = table_t9_exceedance_summary(var_csv, splits)
        out_t9 = tab_out / "tab_T9_exceedance_summary.csv"
        _write_csv_stable(t9, out_t9, float_fmt=str(args.float_fmt), table_round=int(args.table_round))
        written.append(out_t9)
        print("[OK]", out_t9)

    # --- Appendix A (Top-8) ---
    if args.top8_enable:
        pairwise_root = Path(args.top8_pairwise_root) if str(args.top8_pairwise_root).strip() else None
        returns_csv = Path(args.top8_returns_csv) if str(args.top8_returns_csv).strip() else None

        if pairwise_root is not None and pairwise_root.exists():
            a1_tbl, chosen_assets = table_a1_top8_universe_from_pairwise(pairwise_root)
            a1b_tbl = table_a1b_pairwise_stats_from_pairwise(pairwise_root)
        else:
            if returns_csv is None or not returns_csv.exists():
                raise MissingArtifactError("Top8 enabled but neither --top8-pairwise-root exists nor a valid --top8-returns-csv was provided.")
            a1_tbl, chosen_assets = table_a1_top8_universe(
                returns_csv,
                splits,
                assets_csv=str(args.top8_assets),
                k=int(args.top8_k),
            )
            a1b_tbl = table_a1b_pairwise_stats(returns_csv, splits, assets=chosen_assets)

        out_a1 = tab_out / "tab_A1_top8_universe_coverage.csv"
        _write_csv_stable(a1_tbl, out_a1, float_fmt=str(args.float_fmt), table_round=int(args.table_round))
        written.append(out_a1)
        print("[OK]", out_a1, "assets=", chosen_assets)

        out_a1b = tab_out / "tab_A1b_top8_pairwise_stats.csv"
        _write_csv_stable(a1b_tbl, out_a1b, float_fmt=str(args.float_fmt), table_round=int(args.table_round))
        written.append(out_a1b)
        print("[OK]", out_a1b)

        if args.top8_heatmaps:
            if pairwise_root is None or not pairwise_root.exists():
                raise MissingArtifactError("--top8-heatmaps requires --top8-pairwise-root pointing to a directory with pairs/*/var_es_predictions.csv")

            eps = float(args.top8_ratio_eps)
            es99_a = str(args.top8_es99_a_col)
            es99_b = str(args.top8_es99_b_col)
            es99_ratio = str(args.top8_es99_ratio_col)
            exc_col = str(args.top8_exceed_col)

            out_A1 = fig_out / "fig_A1_heatmap_delta_es99_stress.png"
            fig_a1_heatmap_delta_es99_stress(
                pairwise_root,
                out_A1,
                es99_a_col=es99_a,
                es99_b_col=es99_b,
                es99_ratio_col=es99_ratio,
                exc_col=exc_col,
                eps=eps,
                fig_w=float(args.fig_w),
                fig_h=float(args.fig_h),
                dpi=int(args.dpi),
            )
            written.append(out_A1)
            print("[OK]", out_A1)

            out_A2 = fig_out / "fig_A2_heatmap_ratio_es99_stress_calm.png"
            fig_a2_heatmap_ratio_es99_stress_calm(
                pairwise_root,
                out_A2,
                es99_a_col=es99_a,
                es99_b_col=es99_b,
                es99_ratio_col=es99_ratio,
                exc_col=exc_col,
                eps=eps,
                fig_w=float(args.fig_w),
                fig_h=float(args.fig_h),
                dpi=int(args.dpi),
            )
            written.append(out_A2)
            print("[OK]", out_A2)

            out_A3 = fig_out / "fig_A3_heatmap_exceedance_rate.png"
            fig_a3_heatmap_exceedance_rate(
                pairwise_root,
                out_A3,
                es99_a_col=es99_a,
                es99_b_col=es99_b,
                es99_ratio_col=es99_ratio,
                exc_col=exc_col,
                eps=eps,
                fig_w=float(args.fig_w),
                fig_h=float(args.fig_h),
                dpi=int(args.dpi),
            )
            written.append(out_A3)
            print("[OK]", out_A3)

            a2_tbl = table_a2_top8_pairwise_metric_distrib(
                pairwise_root,
                es99_a_col=es99_a,
                es99_b_col=es99_b,
                es99_ratio_col=es99_ratio,
                exc_col=exc_col,
                eps=eps,
            )
            out_a2 = tab_out / "tab_A2_top8_pairwise_metric_distrib.csv"
            _write_csv_stable(a2_tbl, out_a2, float_fmt=str(args.float_fmt), table_round=int(args.table_round))
            written.append(out_a2)
            print("[OK]", out_a2)


    if getattr(args, "a4_enable", False):
        if not str(getattr(args, "a4_theta_csv", "")).strip():
            raise MissingArtifactError("--a4-enable requires --a4-theta-csv")
        src = Path(getattr(args, "a4_theta_csv"))
        a4_tbl = table_a4_tau_theta_audit(src)
        out_a4 = tab_out / "tab_A4_tau_theta_audit.csv"
        _write_csv_stable(a4_tbl, out_a4, float_fmt=str(args.float_fmt), table_round=int(args.table_round))
        written.append(out_a4)
        print("[OK]", out_a4)

    # --- Appendix A.6: Sensitivity daily vs 4h (Table A3) ---
    if args.a6_enable:
        if not str(args.a6_daily_var_es_csv).strip():
            raise MissingArtifactError("--a6-enable requires --a6-daily-var-es-csv")
        if not str(args.a6_h4_var_es_csv).strip():
            raise MissingArtifactError("--a6-enable requires --a6-h4-var-es-csv")

        daily_csv = Path(args.a6_daily_var_es_csv)
        h4_csv = Path(args.a6_h4_var_es_csv)

        a3_tbl = table_a3_pattern_summary_daily_vs_4h(
            daily_csv,
            h4_csv,
            es99_a_col=str(args.a6_es99_a_col),
            es99_b_col=str(args.a6_es99_b_col),
            es99_ratio_col=str(args.a6_es99_ratio_col),
            exceed_col=str(args.a6_exceed_col),
            ratio_eps=float(args.a6_ratio_eps),
        )
        out_a3 = tab_out / "tab_A3_pattern_summary_daily_vs_4h.csv"
        _write_csv_stable(a3_tbl, out_a3, float_fmt=str(args.float_fmt), table_round=int(args.table_round))
        written.append(out_a3)
        print("[OK]", out_a3)


        if args.a6_fig_a4:
            out_a4 = fig_out / f"fig_A4_overlay_daily_vs_4h_delta_es99_{str(args.a6_a4_bucket).lower().strip()}.png"
            fig_a4_overlay_daily_vs_4h(
                daily_csv,
                h4_csv,
                out_a4,
                es99_a_col=str(args.a6_es99_a_col),
                es99_b_col=str(args.a6_es99_b_col),
                bucket=str(args.a6_a4_bucket),
                rolling=int(args.a6_a4_rolling),
                fig_w=float(args.fig_w),
                fig_h=float(args.fig_h),
                dpi=int(args.dpi),
            )
            written.append(out_a4)
            print("[OK]", out_a4)



    # --- Appendix A.5: Tail-dependence ratio λ_L/λ_U (Figure A5) ---
    if args.a5_enable:
        if not str(args.a5_lambda_csv).strip():
            raise MissingArtifactError("--a5-enable requires --a5-lambda-csv")
        a5_csv = Path(args.a5_lambda_csv)

        out_a5 = fig_out / "fig_A5_taildep_levels_lambdaL_lambdaU.png"
        fig_a5_lambda_ratio_barplot(
            a5_csv,
            out_a5,
            eps=float(args.a5_eps),
            include_t=bool(args.a5_include_t),
            fig_w=float(args.fig_w),
            fig_h=float(args.fig_h),
            dpi=int(args.dpi),
        )
        written.append(out_a5)
        print("[OK]", out_a5)

    if args.t13_enable:
        if not str(args.t13_dm_csv).strip():
            raise MissingArtifactError("--t13-enable requires --t13-dm-csv")
        if not str(args.t13_var_es_predictions_csv).strip():
            raise MissingArtifactError("--t13-enable requires --t13-var-es-predictions-csv")
        if not str(args.t13_regimes_csv).strip():
            raise MissingArtifactError("--t13-enable requires --t13-regimes-csv")

        t13 = table_t13_stat_vs_econ_summary(
            Path(args.t13_dm_csv),
            Path(args.t13_var_es_predictions_csv),
            Path(args.t13_regimes_csv),
            splits,
            regime_col=(args.t13_regime_col or None),
            alpha=float(args.t13_alpha),
            es99_abs_threshold=float(args.t13_es99_abs_threshold),
        )
        out_t13 = tab_out / "tab_T13_stat_vs_econ_summary.csv"
        _write_csv_stable(t13, out_t13, float_fmt=str(args.float_fmt), table_round=int(args.table_round))
        written.append(out_t13)
        print("[OK]", out_t13)

    # --- Reproducibility manifest (hashes + provenance) ---
    def _hash_if_exists(path_str: str) -> str | None:
        p = Path(path_str)
        if p.exists() and p.is_file():
            return _sha256_file(p)
        return None

    inputs: dict[str, str] = {
        "splits_json": str(splits_json),
    }
    if regimes_csv is not None and regimes_csv.exists():
        inputs["regimes_csv"] = str(regimes_csv)
        if str(args.regime_col).strip():
            inputs["regime_col"] = str(args.regime_col).strip()
    elif stress_csv is not None and stress_csv.exists():
        inputs["stress_csv"] = str(stress_csv)

    if args.impact_enable and str(args.var_es_predictions_csv).strip():
        inputs["var_es_predictions_csv"] = str(Path(args.var_es_predictions_csv))
        inputs["impact_regime_col"] = str(args.impact_regime_col).strip() if str(args.impact_regime_col).strip() else ""

    if args.top8_enable:
        if str(args.top8_pairwise_root).strip():
            inputs["top8_pairwise_root"] = str(Path(args.top8_pairwise_root))
        if str(args.top8_returns_csv).strip():
            inputs["top8_returns_csv"] = str(Path(args.top8_returns_csv))
        if str(args.top8_assets).strip():
            inputs["top8_assets"] = str(args.top8_assets).strip()
        inputs["top8_k"] = str(int(args.top8_k))
        if args.top8_heatmaps:
            inputs["top8_heatmaps"] = "1"
            inputs["top8_es99_a_col"] = str(args.top8_es99_a_col)
            inputs["top8_es99_b_col"] = str(args.top8_es99_b_col)
            inputs["top8_es99_ratio_col"] = str(args.top8_es99_ratio_col)
            inputs["top8_exceed_col"] = str(args.top8_exceed_col)
            inputs["top8_ratio_eps"] = str(float(args.top8_ratio_eps))

    input_hashes: dict[str, str] = {}
    for k, v in inputs.items():
        h = _hash_if_exists(v)
        if h is not None:
            input_hashes[k] = h

    if "top8_pairwise_root" in inputs:
        try:
            top8_root = Path(inputs["top8_pairwise_root"])
            for pp in _pairwise_var_es_paths(top8_root):
                pair_name = pp.parent.name
                input_hashes[f"top8_pairwise/{pair_name}/var_es_predictions.csv"] = _sha256_file(pp)
        except Exception:
            pass

    if args.a6_enable:
        inputs["a6_enable"] = "1"
        inputs["a6_daily_var_es_csv"] = str(Path(args.a6_daily_var_es_csv))
        inputs["a6_h4_var_es_csv"] = str(Path(args.a6_h4_var_es_csv))
        inputs["a6_es99_a_col"] = str(args.a6_es99_a_col)
        inputs["a6_es99_b_col"] = str(args.a6_es99_b_col)
        inputs["a6_es99_ratio_col"] = str(args.a6_es99_ratio_col)
        inputs["a6_exceed_col"] = str(args.a6_exceed_col)
        inputs["a6_ratio_eps"] = str(float(args.a6_ratio_eps))
        inputs["a6_fig_a4"] = "1" if args.a6_fig_a4 else "0"
        inputs["a6_a4_bucket"] = str(args.a6_a4_bucket)
        inputs["a6_a4_rolling"] = str(int(args.a6_a4_rolling))
        # A5
        inputs["a5_enable"] = "1" if args.a5_enable else "0"
        if args.a5_enable:
            inputs["a5_lambda_csv"] = str(Path(args.a5_lambda_csv))
            inputs["a5_include_t"] = "1" if args.a5_include_t else "0"
            inputs["a5_eps"] = str(float(args.a5_eps))



    output_hashes: dict[str, str] = {}
    for w in written:
        if w.exists() and w.is_file():
            output_hashes[str(w)] = _sha256_file(w)

    manifest = {
        "paper_id": args.paper_id,
        "created_utc": _now_utc_iso(),
        "inputs": inputs,
        "input_hashes": input_hashes,
        "outputs": {"written": [str(x) for x in written]},
        "output_hashes": output_hashes,
        "python": {
            "version": sys.version.split()[0],
            "platform": platform.platform(),
        },
    }

    out_manifest = Path("paper") / "out" / args.paper_id / "manifest.json"
    _write_json_stable(manifest, out_manifest)
    print("[OK]", out_manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
