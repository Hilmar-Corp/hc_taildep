from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import yaml

from scipy import stats
import matplotlib.pyplot as plt

from hc_taildep.margins.ecdf_expanding import pit_ecdf_expanding_midrank
from hc_taildep.utils.hashing import sha256_bytes
from hc_taildep.utils.paths import ensure_dir


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _resolve_cfg_vars(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Resolve simple ${...} placeholders in YAML configs.

    Supported:
      - ${pit.dataset_version}
      - ${dataset_version}

    Note: build_copula_markov_j6 has a similar resolver; PIT runner previously did not.
    """
    s = yaml.safe_dump(cfg, sort_keys=False)
    pit = (cfg.get("pit") or {})
    dv = pit.get("dataset_version", cfg.get("dataset_version", ""))
    s = s.replace("${pit.dataset_version}", str(dv))
    s = s.replace("${dataset_version}", str(dv))
    return yaml.safe_load(s)


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def dump_json(path: Path, obj: Dict[str, Any]) -> None:
    path.write_text(json.dumps(obj, indent=2, sort_keys=False), encoding="utf-8")


def dump_text(path: Path, s: str) -> None:
    path.write_text(s.rstrip() + "\n", encoding="utf-8")


def _read_returns_from_path(path: Path) -> pd.DataFrame:
    """Read returns from a file path.

    Supports:
      - parquet with DatetimeIndex
      - csv where either:
          * a `date` column exists (preferred)
          * or the first column is an index-like datetime (legacy)

    Returns a DataFrame indexed by pandas datetime.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing returns file: {path}")

    if path.suffix.lower() in {".parquet", ".pq"}:
        df = pd.read_parquet(path)
        if not isinstance(df.index, pd.DatetimeIndex):
            # try common column name
            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True)
                df = df.set_index("date")
            else:
                raise ValueError(f"returns parquet must have DatetimeIndex or 'date' column: {path}")
        return df.sort_index()

    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True)
            if df["date"].isna().any():
                raise ValueError(f"Some dates could not be parsed in returns CSV: {path}")
            df = df.set_index("date")
        else:
            # legacy: first column is index
            df = pd.read_csv(path, index_col=0)
            df.index = pd.to_datetime(df.index, errors="coerce", utc=True)
            if df.index.isna().any():
                raise ValueError(f"Some index dates could not be parsed in returns CSV: {path}")
        return df.sort_index()

    raise ValueError(f"Unsupported returns file type: {path}")


def _read_returns(processed_dir: Path) -> pd.DataFrame:
    p_parq = processed_dir / "returns.parquet"
    p_csv = processed_dir / "returns.csv"
    if p_parq.exists():
        return _read_returns_from_path(p_parq)
    if p_csv.exists():
        return _read_returns_from_path(p_csv)
    raise FileNotFoundError(f"Missing returns.parquet/csv in {processed_dir}")


def _acf(x: np.ndarray, lags: List[int]) -> Dict[int, float]:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 5:
        return {int(k): float("nan") for k in lags}
    x = x - x.mean()
    denom = float(np.dot(x, x))
    out = {}
    for k in lags:
        k = int(k)
        if k <= 0 or k >= x.size:
            out[k] = float("nan")
            continue
        out[k] = float(np.dot(x[:-k], x[k:]) / denom)
    return out


def _save_hist(u: np.ndarray, bins: int, path: Path, title: str) -> None:
    ensure_dir(path.parent)
    plt.figure()
    plt.hist(u[np.isfinite(u)], bins=bins)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def _save_acf_plot(acf_map: Dict[int, float], path: Path, title: str) -> None:
    ensure_dir(path.parent)
    lags = sorted(acf_map.keys())
    vals = [acf_map[k] for k in lags]
    plt.figure()
    plt.bar(lags, vals)
    plt.title(title)
    plt.xlabel("lag")
    plt.ylabel("ACF")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def _save_timeseries(u: pd.Series, path: Path, title: str) -> None:
    ensure_dir(path.parent)
    plt.figure()
    plt.plot(u.index, u.values)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=str)
    args = ap.parse_args()

    t0 = time.time()
    config_path = Path(args.config).resolve()
    cfg = _resolve_cfg_vars(load_yaml(config_path))

    pit_cfg = cfg["pit"]
    diag_cfg = cfg["diagnostics"]

    dataset_version = pit_cfg["dataset_version"]
    processed_dir = Path("data/processed") / dataset_version
    pits_dir = processed_dir / pit_cfg.get("output_dirname", "pits")
    figs_dir = pits_dir / "figures"
    ensure_dir(pits_dir)
    ensure_dir(figs_dir)

    # Optional overrides (useful for 4h datasets / pit-ready CSVs)
    pit_inputs = pit_cfg.get("inputs", {}) or {}
    splits_path = Path(pit_inputs.get("splits_path", processed_dir / "splits.json"))

    # Load dataset-level references
    splits = load_json(splits_path)
    first_oos = pd.Timestamp(splits["first_oos"])
    train_start = pd.Timestamp(splits["train_start"])
    dataset_hash_ref = splits.get("dataset_hash_sha256", None)

    # Load returns (either explicit path or dataset default)
    returns_path = pit_inputs.get("returns_path", None)
    if returns_path is not None:
        ret_df = _read_returns_from_path(Path(returns_path))
    else:
        ret_df = _read_returns(processed_dir)

    # Required asset columns (configurable)
    assets = pit_cfg.get("assets", ["BTC", "ETH"])
    if not isinstance(assets, list) or len(assets) == 0:
        raise ValueError("pit.assets must be a non-empty list")
    assets = [str(a) for a in assets]

    for c in assets:
        if c not in ret_df.columns:
            raise ValueError(f"returns missing column: {c}")

    # Decide PIT start date
    start_at = pit_cfg.get("start_at", "first_oos")
    if start_at == "first_oos":
        start_date = first_oos
    elif start_at == "train_start":
        start_date = train_start
    else:
        raise ValueError("pit.start_at must be first_oos or train_start")

    # Determine start index in returns array
    if start_date not in ret_df.index:
        # If exact date not in index, take next available (should not happen if J1 is clean)
        start_i = int(np.searchsorted(ret_df.index.values, np.datetime64(start_date)))
    else:
        start_i = int(ret_df.index.get_loc(start_date))

    epsilon = float(pit_cfg["epsilon"])
    min_history = int(pit_cfg["min_history"])

    # Compute u series
    u_cols = {}
    ur_cols = {}
    clip_hits = {}

    for asset in assets:
        u, ur = pit_ecdf_expanding_midrank(
            ret_df[asset],
            min_history=min_history,
            epsilon=epsilon,
            start_index=start_i,
        )
        u_cols[f"u_{asset}"] = u
        ur_cols[f"u_raw_{asset}"] = ur
        clip_hits[asset] = ((u == epsilon) | (u == (1.0 - epsilon))).astype(float)

    # Build dataframe with all u + u_raw columns
    parts = []
    for asset in assets:
        parts.append(u_cols[f"u_{asset}"])
    for asset in assets:
        parts.append(ur_cols[f"u_raw_{asset}"])
    u_df = pd.concat(parts, axis=1)

    # Keep a "valid" mask aligned to all assets
    valid_mask = np.ones(len(u_df), dtype=bool)
    for asset in assets:
        valid_mask &= np.isfinite(u_df[f"u_{asset}"].to_numpy())
    u_valid = u_df.loc[valid_mask].copy()

    # Write u_series (CSV stable)
    u_path = pits_dir / "u_series.csv"
    float_fmt = pit_cfg.get("csv_float_format", "%.10g")
    u_df.to_csv(u_path, index=True, index_label="date", float_format=float_fmt)

    u_hash = sha256_bytes(u_path.read_bytes())

    # Metrics
    lags = [int(x) for x in diag_cfg.get("acf_lags", [1, 5, 10, 20])]
    bins = int(diag_cfg.get("hist_bins", 40))

    metrics: Dict[str, Any] = {
        "dataset_version": dataset_version,
        "dataset_hash_sha256_ref": dataset_hash_ref,
        "epsilon": epsilon,
        "min_history": min_history,
        "start_at": start_at,
        "date_start_u": str(u_valid.index.min().date()) if len(u_valid) else None,
        "n_u": int(len(u_valid)),
        "u_series_path": str(u_path),
        "u_series_hash_sha256": u_hash,
        "summary": {},
        "clip_rate": {},
        "acf": {},
        "ks_test": {},
    }

    for asset in assets:
        u_series = u_valid[f"u_{asset}"].to_numpy()
        metrics["summary"][asset] = {
            "mean": float(np.mean(u_series)),
            "std": float(np.std(u_series, ddof=1)) if u_series.size > 1 else float("nan"),
            "min": float(np.min(u_series)),
            "max": float(np.max(u_series)),
        }
        # clip rate computed on valid region for that asset
        u_all = u_df[f"u_{asset}"].to_numpy()
        valid_a = np.isfinite(u_all)
        if valid_a.any():
            hits = ((u_all[valid_a] == epsilon) | (u_all[valid_a] == (1.0 - epsilon))).mean()
        else:
            hits = float("nan")
        metrics["clip_rate"][asset] = float(hits)

        metrics["acf"][asset] = _acf(u_series, lags)

        if bool(diag_cfg.get("ks_test", True)):
            # KS test against Uniform(0,1)
            # Note: treat as diagnostic, not KPI
            res = stats.kstest(u_series, "uniform")
            metrics["ks_test"][asset] = {"stat": float(res.statistic), "pvalue": float(res.pvalue)}

    dump_json(pits_dir / "pit_metrics.json", metrics)

    # Figures
    for asset in assets:
        _save_hist(u_valid[f"u_{asset}"].to_numpy(), bins, figs_dir / f"pit_hist_{asset}.png", f"PIT histogram — {asset}")

    for asset in assets:
        _save_acf_plot(metrics["acf"][asset], figs_dir / f"pit_acf_{asset}.png", f"ACF(u) — {asset}")

    if bool(diag_cfg.get("make_timeseries_plots", True)):
        for asset in assets:
            _save_timeseries(u_df[f"u_{asset}"], figs_dir / f"pit_timeseries_{asset}.png", f"u_{asset} over time")

    # Report
    rep = []
    rep.append("# PIT Report — ECDF Expanding (Mid-rank)")
    rep.append("")
    rep.append("## Contract (anti-leakage)")
    rep.append("- For each date t: fit ECDF on history ≤ t-1, then transform r_t.")
    rep.append("- Notation: u_t = F_{t-1}(r_t).")
    rep.append("")
    rep.append("## Configuration")
    rep.append(f"- dataset_version: `{dataset_version}`")
    rep.append(f"- start_at: `{start_at}` (start_date = {str(start_date.date())})")
    rep.append(f"- min_history: `{min_history}`")
    rep.append(f"- epsilon (clipping): `{epsilon}`")
    rep.append("")
    rep.append("## Outputs")
    rep.append(f"- u_series: `u_series.csv` (sha256: `{u_hash[:12]}`)")
    rep.append("- metrics: `pit_metrics.json`")
    rep.append("- figures: `figures/`")
    rep.append("")
    rep.append("## Diagnostics (read as signals, not KPIs)")
    for asset in assets:
        rep.append(f"### {asset}")
        rep.append(f"- clip_rate: {metrics['clip_rate'][asset]:.6g}")
        rep.append(f"- summary: mean={metrics['summary'][asset]['mean']:.6g}, std={metrics['summary'][asset]['std']:.6g}, min={metrics['summary'][asset]['min']:.6g}, max={metrics['summary'][asset]['max']:.6g}")
        if metrics["ks_test"].get(asset):
            rep.append(f"- KS(uniform): stat={metrics['ks_test'][asset]['stat']:.6g}, p={metrics['ks_test'][asset]['pvalue']:.6g}")
        rep.append(f"- ACF lags {lags}: {', '.join([f'{k}:{metrics['acf'][asset][k]:.4g}' for k in lags])}")
        rep.append("")
    rep.append("## Figures")
    rep.append("- pit_hist_<ASSET>.png")
    rep.append("- pit_acf_<ASSET>.png")
    rep.append("- pit_timeseries_<ASSET>.png")
    rep.append("")
    rep.append("## Interpretation note")
    rep.append("- KS p-values are diagnostics; with large samples, rejection can occur even with acceptable PIT.")
    rep.append("- If clip_rate is high or histograms show masses at bounds, suspect window/leakage/bug.")
    dump_text(pits_dir / "pit_report.md", "\n".join(rep))

    # Provenance (dataset-level PIT)
    prov = {
        "created_utc": utc_now_iso(),
        "config_path": str(config_path),
        "env": {
            "python_version": sys.version.split()[0],
            "platform": platform.platform(),
        },
        "inputs": {
            "processed_dir": str(processed_dir),
            "returns_path": str(Path(returns_path)) if returns_path is not None else str(processed_dir / "returns.parquet" if (processed_dir / "returns.parquet").exists() else processed_dir / "returns.csv"),
            "splits_path": str(splits_path),
            "dataset_hash_sha256_ref": dataset_hash_ref,
        },
        "pit": {
            "method": pit_cfg["method"],
            "epsilon": epsilon,
            "min_history": min_history,
            "start_at": start_at,
            "start_date": str(start_date.date()),
            "assets": assets,
        },
        "outputs": {
            "pits_dir": str(pits_dir),
            "u_series_path": str(u_path),
            "u_series_hash_sha256": u_hash,
            "metrics_path": str(pits_dir / "pit_metrics.json"),
            "report_path": str(pits_dir / "pit_report.md"),
            "figures_dir": str(figs_dir),
        },
        "runtime_sec": float(time.time() - t0),
    }
    dump_json(pits_dir / "provenance.json", prov)

    print(f"[OK] built PIT: {dataset_version}")
    print(f"[OK] pits_dir: {pits_dir.resolve()}")
    print(f"[OK] u_series: {u_path.name} hash={u_hash[:12]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())