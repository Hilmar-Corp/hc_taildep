from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
import matplotlib.pyplot as plt

from hc_taildep.stress.definitions import realized_vol, stress_by_joint_downside, stress_by_rv
from hc_taildep.eval.taildep import summarize_regime, fit_tcopula_and_lambda
from hc_taildep.eval.bootstrap import bootstrap_delta_lambda


def _sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _load_yaml(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _write_yaml(path: Path, obj: dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, sort_keys=False)


def _write_json(path: Path, obj: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=False)


def _write_csv_stable(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False, float_format="%.10g")


def _resolve_vars(cfg: dict[str, Any]) -> dict[str, Any]:
    # minimal ${dataset_version} expansion for your config style
    ds = cfg.get("dataset_version", "")
    def repl(x: Any) -> Any:
        if isinstance(x, str):
            return x.replace("${dataset_version}", ds)
        if isinstance(x, dict):
            return {k: repl(v) for k, v in x.items()}
        if isinstance(x, list):
            return [repl(v) for v in x]
        return x

    return repl(cfg)



def _window_from_splits(df: pd.DataFrame, splits: dict[str, Any], start_at: str, end_at: str) -> pd.DataFrame:
    def pick(which: str) -> str:
        if which in ("first_oos", "last_oos", "train_start"):
            return splits[which]
        return which  # allow explicit date string

    start = pick(start_at)
    end = pick(end_at)
    m = (df["date"] >= start) & (df["date"] <= end)
    return df.loc[m].copy()


# Helper to ensure PIT u_series has a date column
def _ensure_date_column(df: pd.DataFrame, *, preferred: str = "date") -> pd.DataFrame:
    """Ensure a dataframe has a `date` column.

    Supports common cases:
    - date stored as the index (with or without index name)
    - first column is an unnamed index column (e.g. 'Unnamed: 0')
    - first column contains ISO date strings but is named differently
    """
    if preferred in df.columns:
        return df

    # Helper: only treat object/string-like as "date candidates"
    def _is_string_like(s: pd.Series) -> bool:
        return pd.api.types.is_object_dtype(s.dtype) or pd.api.types.is_string_dtype(s.dtype)

    # Case 1: date is the index (but avoid numeric RangeIndex / numeric indices)
    if df.index is not None and df.index.nlevels == 1:
        idx = df.index
        # Guard: if the index is numeric-like (e.g. RangeIndex), do not try to interpret as dates
        try:
            if isinstance(idx, pd.RangeIndex) or pd.api.types.is_numeric_dtype(idx.dtype):
                idx_is_date_candidate = False
            else:
                idx_is_date_candidate = True
        except Exception:
            idx_is_date_candidate = True

        if idx_is_date_candidate:
            try:
                parsed = pd.to_datetime(idx, errors="coerce")
                if parsed.notna().mean() > 0.95:
                    out = df.reset_index().rename(columns={df.index.name or "index": preferred})
                    return out
            except Exception:
                pass

    # Case 2: an unnamed first column written by pandas when index=True
    if len(df.columns) >= 1 and str(df.columns[0]).startswith("Unnamed"):
        out = df.rename(columns={df.columns[0]: preferred})
        return out

    # Case 3: first column contains date-like strings (avoid numeric columns: pandas can coerce numbers to datetimes)
    if len(df.columns) >= 1:
        c0 = df.columns[0]
        if _is_string_like(df[c0]):
            try:
                parsed = pd.to_datetime(df[c0], errors="coerce")
                if parsed.notna().mean() > 0.95:
                    out = df.rename(columns={c0: preferred})
                    return out
            except Exception:
                pass

    raise KeyError(
        f"Could not find/construct a '{preferred}' column in dataframe. Columns={list(df.columns)}"
    )


def _plot_rv(df: pd.DataFrame, out_path: Path, stress_thr: float, calm_thr: float) -> None:
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(pd.to_datetime(df["date"]), df["RV"].to_numpy(dtype=float))
    ax.axhline(stress_thr, linestyle="--")
    ax.axhline(calm_thr, linestyle="--")
    ax.set_title("Realized Volatility (RV) with thresholds")
    ax.set_xlabel("Date")
    ax.set_ylabel("RV")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _plot_lambda_bar(summary_df: pd.DataFrame, out_path: Path) -> None:
    # expect rows per (definition, regime), we plot lambda_hat
    fig = plt.figure()
    ax = fig.add_subplot(111)
    labels = (summary_df["definition"] + "/" + summary_df["regime"]).tolist()
    y = summary_df["lambda_hat"].to_numpy(dtype=float)
    ax.bar(np.arange(len(labels)), y)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("lambda_hat (t-copula closed-form)")
    ax.set_title("Tail dependence by regime/definition")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _plot_bootstrap_hist(deltas: np.ndarray, out_path: Path, delta_hat: float) -> None:
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.hist(deltas, bins=50)
    ax.axvline(delta_hat, linestyle="--")
    ax.set_title("Bootstrap distribution of Δλ (stress - calm)")
    ax.set_xlabel("Δλ")
    ax.set_ylabel("count")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg0 = _load_yaml(args.config)
    cfg = _resolve_vars(cfg0)

    ds = cfg["dataset_version"]
    inputs = cfg["inputs"]
    window_cfg = cfg["window"]
    stress_cfg = cfg["stress_defs"]
    fit_cfg = cfg["t_copula_fit"]
    tail_cfg = cfg["tail_lambda"]
    boot_cfg = cfg["bootstrap"]
    rep_cfg = cfg.get("reporting", {})

    # Load inputs
    returns = pd.read_csv(inputs["returns_path"])
    returns = _ensure_date_column(returns, preferred="date")

    u_series = pd.read_csv(inputs["u_series_path"])
    u_series = _ensure_date_column(u_series, preferred="date")

    # Normalize date columns to ISO date strings (UTC day) for safe merging/comparisons
    returns["date"] = pd.to_datetime(returns["date"], utc=True, errors="raise").dt.strftime("%Y-%m-%d")
    u_series["date"] = pd.to_datetime(u_series["date"], utc=True, errors="raise").dt.strftime("%Y-%m-%d")
    with open(inputs["splits_path"], "r", encoding="utf-8") as f:
        splits = json.load(f)
    with open(inputs["pit_metrics_path"], "r", encoding="utf-8") as f:
        pitm = json.load(f)

    # Join on date
    if returns.duplicated(subset=["date"]).any():
        ndup = int(returns.duplicated(subset=["date"]).sum())
        raise SystemExit(f"Duplicate dates in returns.csv: {ndup}")
    if u_series.duplicated(subset=["date"]).any():
        ndup = int(u_series.duplicated(subset=["date"]).sum())
        raise SystemExit(f"Duplicate dates in u_series.csv: {ndup}")
    df = returns.merge(u_series, on="date", how="inner", validate="one_to_one")

    # Filter analysis window
    df = _window_from_splits(df, splits, window_cfg["start_at"], window_cfg["end_at"])

    # Hard checks
    need_cols = ["date", "BTC", "ETH", "u_BTC", "u_ETH"]
    missing = [c for c in need_cols if c not in df.columns]
    if missing:
        raise SystemExit(f"Missing required columns in merged df: {missing}")
    # Extra sanity: ensure date is a string column (YYYY-MM-DD)
    if df["date"].dtype != object:
        df["date"] = df["date"].astype(str)

    # Ensure PIT valid (no NaN) on window
    if df[["u_BTC", "u_ETH"]].isna().any().any():
        # we want no silent shrink: fail hard so we fix upstream
        n_bad = int(df[["u_BTC", "u_ETH"]].isna().any(axis=1).sum())
        raise SystemExit(f"Found NaN in PIT columns inside analysis window: {n_bad} rows")

    # Output dirs
    out_dir = Path(f"data/processed/{ds}/taildep/j4_calm_stress")
    fig_dir = out_dir / "figures"
    tab_dir = out_dir / "tables"
    _ensure_dir(fig_dir)
    _ensure_dir(tab_dir)

    # Save resolved config
    cfg_resolved = cfg
    _write_yaml(out_dir / "config.resolved.yaml", cfg_resolved)

    # Stress definitions
    regimes = df[["date", "BTC", "ETH", "u_BTC", "u_ETH"]].copy()

    # RV definition
    rv_info = {}
    if stress_cfg["rv"]["enabled"]:
        asset = stress_cfg["rv"]["on_asset"]
        w = int(stress_cfg["rv"]["window"])
        regimes["RV"] = realized_vol(df[asset], window=w).to_numpy(dtype=float)
        is_stress, is_calm, info = stress_by_rv(
            regimes["RV"],
            stress_q=float(stress_cfg["rv"]["stress_q"]),
            calm_q=float(stress_cfg["rv"]["calm_q"]),
        )
        regimes["is_stress_RV"] = is_stress.to_numpy(dtype=bool)
        regimes["is_calm_RV"] = is_calm.to_numpy(dtype=bool)
        rv_info = info
    else:
        regimes["RV"] = np.nan
        regimes["is_stress_RV"] = False
        regimes["is_calm_RV"] = False

    # Joint downside definition
    jd_info = {}
    if stress_cfg["joint_downside"]["enabled"]:
        is_stress, is_calm, info = stress_by_joint_downside(
            regimes["BTC"],
            regimes["ETH"],
            alpha=float(stress_cfg["joint_downside"]["alpha"]),
        )
        regimes["is_stress_JDown"] = is_stress.to_numpy(dtype=bool)
        regimes["is_calm_JDown"] = is_calm.to_numpy(dtype=bool)
        jd_info = info
    else:
        regimes["is_stress_JDown"] = False
        regimes["is_calm_JDown"] = False

    # Write regimes.csv
    regimes_path = out_dir / "regimes.csv"
    _write_csv_stable(regimes, regimes_path)
    regimes_hash = _sha256_file(regimes_path)

    # Taildep summaries
    nu_grid = fit_cfg["nu_grid"]
    nu_bounds = tuple(fit_cfg["nu_bounds"])
    rho_clamp = float(fit_cfg["rho_clamp"])
    empirical_qs = tail_cfg.get("empirical_qs", [0.05, 0.10])

    rows = []

    def add_summary(defn: str, calm_mask_col: str, stress_mask_col: str) -> None:
        calm = summarize_regime(
            regimes,
            regimes[calm_mask_col],
            u_col="u_BTC",
            v_col="u_ETH",
            nu_grid=nu_grid,
            nu_bounds=nu_bounds,
            rho_clamp=rho_clamp,
            empirical_qs=empirical_qs,
        )
        stress = summarize_regime(
            regimes,
            regimes[stress_mask_col],
            u_col="u_BTC",
            v_col="u_ETH",
            nu_grid=nu_grid,
            nu_bounds=nu_bounds,
            rho_clamp=rho_clamp,
            empirical_qs=empirical_qs,
        )

        calm_row = {"definition": defn, "regime": "calm", **calm}
        stress_row = {"definition": defn, "regime": "stress", **stress}
        rows.append(calm_row)
        rows.append(stress_row)

    if stress_cfg["rv"]["enabled"]:
        add_summary("RV", "is_calm_RV", "is_stress_RV")
    if stress_cfg["joint_downside"]["enabled"]:
        add_summary("JDown", "is_calm_JDown", "is_stress_JDown")

    summary_df = pd.DataFrame(rows)
    summary_path = out_dir / "taildep_summary.csv"
    _write_csv_stable(summary_df, summary_path)
    summary_hash = _sha256_file(summary_path)

    # Bootstrap (core: delta lambda from closed-form t-copula via fit)
    boot_out = []
    boot_json_path = out_dir / "taildep_bootstrap.json"

    if boot_cfg.get("enabled", False):
        B = int(boot_cfg["B"])
        L = int(boot_cfg["block_len"])
        seed = int(boot_cfg["seed"])

        def fit_lambda(u_sub: np.ndarray, v_sub: np.ndarray) -> float:
            fit = fit_tcopula_and_lambda(
                u_sub,
                v_sub,
                nu_grid=nu_grid,
                nu_bounds=nu_bounds,
                rho_clamp=rho_clamp,
            )
            return float(fit["lambda_hat"])

        # Do bootstrap per stress definition separately
        for defn in (["RV"] if stress_cfg["rv"]["enabled"] else []) + (["JDown"] if stress_cfg["joint_downside"]["enabled"] else []):
            calm_col = "is_calm_RV" if defn == "RV" else "is_calm_JDown"
            stress_col = "is_stress_RV" if defn == "RV" else "is_stress_JDown"

            u = regimes["u_BTC"].to_numpy(dtype=float)
            v = regimes["u_ETH"].to_numpy(dtype=float)
            mask_c = regimes[calm_col].to_numpy(dtype=bool)
            mask_s = regimes[stress_col].to_numpy(dtype=bool)

            res, deltas = bootstrap_delta_lambda(
                u, v, mask_c, mask_s, fit_lambda_fn=fit_lambda, B=B, block_len=L, seed=seed
            )

            # Persist bootstrap draws for paper-level histogram (Figure F9)
            deltas = np.asarray(deltas, dtype=float)
            deltas = deltas[np.isfinite(deltas)]
            samples_csv = tab_dir / f"delta_lambda_samples_{defn}.csv"
            if deltas.size:
                _write_csv_stable(pd.DataFrame({"delta_lambda": deltas}), samples_csv)
                samples_csv_hash = _sha256_file(samples_csv)
            else:
                # still write an empty file for determinism
                _write_csv_stable(pd.DataFrame({"delta_lambda": []}), samples_csv)
                samples_csv_hash = _sha256_file(samples_csv)

            boot_item = {
                "definition": defn,
                "bootstrap": {"B": res.B, "block_len": res.block_len, "seed": res.seed},
                "n_obs": {"calm": res.n_calm, "stress": res.n_stress},
                "delta_lambda_hat": res.delta_lambda_hat,
                "delta_lambda_ci95": [res.ci95[0], res.ci95[1]],
                "pvalue_two_sided": res.pvalue_two_sided,
                "delta_lambda_samples": [float(x) for x in deltas.tolist()],
                "delta_lambda_samples_hash_sha256": samples_csv_hash,
                "delta_lambda_samples_csv": str(samples_csv),
            }
            boot_out.append(boot_item)

            if rep_cfg.get("make_figures", True) and rep_cfg.get("bootstrap_hist", True) and deltas.size > 0:
                _plot_bootstrap_hist(deltas, fig_dir / f"delta_lambda_bootstrap_hist_{defn}.png", res.delta_lambda_hat)

        _write_json(boot_json_path, {"dataset_version": ds, "results": boot_out})
    else:
        _write_json(boot_json_path, {"dataset_version": ds, "results": []})

    boot_hash = _sha256_file(boot_json_path)

    # Figures
    if rep_cfg.get("make_figures", True):
        if stress_cfg["rv"]["enabled"] and rep_cfg.get("rv_plot", True):
            _plot_rv(regimes.dropna(subset=["RV"]), fig_dir / "rv_timeseries.png", rv_info.get("stress_threshold", np.nan), rv_info.get("calm_threshold", np.nan))
        _plot_lambda_bar(summary_df, fig_dir / "lambda_barplot.png")

    # Report
    report_lines = []
    report_lines.append("# J4 — Tail dependence (Calm vs Stress)\n")
    report_lines.append("## Inputs\n")
    report_lines.append(f"- dataset_version: `{ds}`\n")
    report_lines.append(f"- splits: first_oos={splits.get('first_oos')} last_oos={splits.get('last_oos')}\n")
    report_lines.append(f"- pit_u_hash_ref: {pitm.get('u_series_hash_sha256')}\n")
    report_lines.append("\n## Stress definitions\n")
    if stress_cfg["rv"]["enabled"]:
        report_lines.append(f"- RV: asset={stress_cfg['rv']['on_asset']} window={stress_cfg['rv']['window']} stress_q={stress_cfg['rv']['stress_q']} calm_q={stress_cfg['rv']['calm_q']}\n")
        report_lines.append(f"  - thresholds: stress>= {rv_info.get('stress_threshold')} calm<= {rv_info.get('calm_threshold')}\n")
        report_lines.append(f"  - n_stress={int(regimes['is_stress_RV'].sum())} n_calm={int(regimes['is_calm_RV'].sum())}\n")
    if stress_cfg["joint_downside"]["enabled"]:
        report_lines.append(f"- JointDown: alpha={stress_cfg['joint_downside']['alpha']}\n")
        report_lines.append(f"  - thresholds: BTC<{jd_info.get('thr_btc')} ETH<{jd_info.get('thr_eth')}\n")
        report_lines.append(f"  - n_stress={int(regimes['is_stress_JDown'].sum())} n_calm={int(regimes['is_calm_JDown'].sum())}\n")

    report_lines.append("\n## Results (t-copula params and closed-form lambda)\n")
    report_lines.append("See `taildep_summary.csv`.\n")

    if boot_cfg.get("enabled", False):
        report_lines.append("\n## Bootstrap (block)\n")
        report_lines.append(f"- B={boot_cfg['B']} block_len={boot_cfg['block_len']} seed={boot_cfg['seed']}\n")
        report_lines.append("See `taildep_bootstrap.json`.\n")

    report_lines.append("\n## Figures\n")
    report_lines.append("- `figures/lambda_barplot.png`\n")
    if stress_cfg["rv"]["enabled"]:
        report_lines.append("- `figures/rv_timeseries.png`\n")
    if boot_cfg.get("enabled", False):
        report_lines.append("- `figures/delta_lambda_bootstrap_hist_*.png` (annexe)\n")

    report_lines.append("\n## Notes / limitations\n")
    report_lines.append("- t-copula has symmetric tail dependence (λL=λU). Asymmetry requires asymmetric copulas (annexe later).\n")
    report_lines.append("- KS on PIT is diagnostic only; not a KPI.\n")
    report_lines.append("- Low power regimes are flagged via `fit_status=low_power`.\n")

    report_path = out_dir / "report.md"
    report_path.write_text("".join(report_lines), encoding="utf-8")
    report_hash = _sha256_file(report_path)

    # Provenance
    prov = {
        "dataset_version": ds,
        "inputs": inputs,
        "hash_refs": {
            "splits_hash_sha256": _sha256_file(Path(inputs["splits_path"])),
            "returns_hash_sha256": _sha256_file(Path(inputs["returns_path"])),
            "u_series_hash_sha256": pitm.get("u_series_hash_sha256"),
            "pit_metrics_hash_sha256": _sha256_file(Path(inputs["pit_metrics_path"])),
        },
        "outputs": {
            "regimes_csv": str(regimes_path),
            "regimes_hash_sha256": regimes_hash,
            "taildep_summary_csv": str(summary_path),
            "taildep_summary_hash_sha256": summary_hash,
            "taildep_bootstrap_json": str(boot_json_path),
            "taildep_bootstrap_hash_sha256": boot_hash,
            "bootstrap_samples_dir": str(tab_dir),
            "report_md": str(report_path),
            "report_hash_sha256": report_hash,
        },
        "stress_info": {"rv": rv_info, "joint_downside": jd_info},
        "config_path": args.config,
        "config_resolved_path": str(out_dir / "config.resolved.yaml"),
        "pid": os.getpid(),
    }
    _write_json(out_dir / "provenance.json", prov)

    print(f"[OK] built J4 taildep: {ds}")
    print(f"[OK] out_dir: {out_dir.resolve()}")
    print(f"[OK] regimes: regimes.csv hash={regimes_hash[:12]}")
    print(f"[OK] summary: taildep_summary.csv hash={summary_hash[:12]}")
    print(f"[OK] bootstrap: taildep_bootstrap.json hash={boot_hash[:12]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())