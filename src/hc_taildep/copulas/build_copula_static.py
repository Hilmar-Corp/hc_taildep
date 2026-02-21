from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Tuple, List

import numpy as np
import pandas as pd
import yaml
import matplotlib.pyplot as plt

from hc_taildep.utils.hashing import sha256_bytes
from hc_taildep.utils.paths import ensure_dir

from hc_taildep.copulas import indep
from hc_taildep.copulas import gaussian
from hc_taildep.copulas import student_t


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def dump_json(path: Path, obj: Dict[str, Any]) -> None:
    path.write_text(json.dumps(obj, indent=2, sort_keys=False), encoding="utf-8")


def dump_text(path: Path, s: str) -> None:
    path.write_text(s.rstrip() + "\n", encoding="utf-8")


def _read_u_series(pits_dir: Path, u_file: str) -> pd.DataFrame:
    p = pits_dir / u_file
    if not p.exists():
        raise FileNotFoundError(p)
    df = pd.read_csv(p, index_col=0)
    df.index = pd.to_datetime(df.index)
    # Expect u_BTC/u_ETH
    need = ["u_BTC", "u_ETH"]
    for c in need:
        if c not in df.columns:
            raise ValueError(f"Missing column {c} in {p}")
    return df


def _rolling_mean(x: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return x.copy()
    s = pd.Series(x)
    return s.rolling(window=window, min_periods=window).mean().to_numpy()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=str)
    args = ap.parse_args()

    t0 = time.time()
    config_path = Path(args.config).resolve()
    cfg = load_yaml(config_path)["copula_static"]

    dataset_version = cfg["dataset_version"]
    processed_dir = Path("data/processed") / dataset_version
    splits = load_json(processed_dir / cfg["inputs"].get("splits_file", "splits.json"))
    first_oos = pd.Timestamp(splits["first_oos"])
    last_oos = pd.Timestamp(splits["last_oos"])

    pits_dir = processed_dir / cfg["inputs"].get("pits_subdir", "pits")
    u_df = _read_u_series(pits_dir, cfg["inputs"].get("u_file", "u_series.csv"))

    # Valid region: both u finite
    mask = np.isfinite(u_df["u_BTC"].to_numpy()) & np.isfinite(u_df["u_ETH"].to_numpy())
    u_df = u_df.loc[mask].copy()

    # Restrict to [first_oos, last_oos]
    u_df = u_df.loc[(u_df.index >= first_oos) & (u_df.index <= last_oos)].copy()
    if u_df.empty:
        raise ValueError("No valid PIT observations in OOS window.")

    # Output dirs
    out_dir = processed_dir / "copulas" / "static"
    tables_dir = out_dir / "tables"
    figs_dir = out_dir / "figures"
    ensure_dir(out_dir)
    ensure_dir(tables_dir)
    ensure_dir(figs_dir)

    # Resolved config (freeze)
    cfg_resolved = {
        "dataset_version": dataset_version,
        "inputs": cfg["inputs"],
        "oos": cfg["oos"],
        "student_t": cfg["student_t"],
        "reporting": cfg["reporting"],
        "created_utc": utc_now_iso(),
    }
    (out_dir / "config.resolved.yaml").write_text(yaml.safe_dump(cfg_resolved, sort_keys=False), encoding="utf-8")

    rho_clamp = float(cfg["oos"].get("rho_clamp", 1e-6))
    refit_every = int(cfg["oos"].get("refit_every", 20))
    roll_w = int(cfg["reporting"].get("rolling_window", 63))
    float_fmt = cfg["reporting"].get("csv_float_format", "%.10g")

    nu_grid = [int(x) for x in cfg["student_t"]["nu_grid"]]
    nu_bounds = (float(cfg["student_t"]["nu_bounds"][0]), float(cfg["student_t"]["nu_bounds"][1]))

    dates = u_df.index.to_list()
    u = u_df["u_BTC"].to_numpy(dtype=float)
    v = u_df["u_ETH"].to_numpy(dtype=float)

    # Safety: values must be strictly inside (0,1)
    if not (np.all((u > 0.0) & (u < 1.0)) and np.all((v > 0.0) & (v < 1.0))):
        raise ValueError("u values must be in (0,1); check PIT clipping.")

    logc_ind = np.zeros_like(u)
    logc_g = np.full_like(u, np.nan)
    logc_t = np.full_like(u, np.nan)

    # Store refit params for params_summary
    refit_rows: List[Dict[str, Any]] = []

    rho_g = 0.0
    rho_t = 0.0
    nu_t = 10.0
    last_refit_k = None

    # OOS strict:
    # For point k (date t_k), training set is indices [0 .. k-1]
    # Refit happens when (k == 0?) no: first point has no train, but in our J2 start_at=first_oos ensures there is plenty history before first_oos in returns,
    # however PIT series begins at first_oos: so training for first scored point would be empty. We must therefore:
    # - Use a "warmup" inside J3 as well: require at least some train points within PIT window.
    # Simpler & correct: start scoring from k=1 (first point has no in-window train).
    start_k = 1

    for k in range(start_k, len(u)):
        need_refit = False
        if last_refit_k is None:
            need_refit = True
        else:
            if (k - last_refit_k) >= refit_every:
                need_refit = True

        if need_refit:
            u_tr = u[:k]
            v_tr = v[:k]
            # Gaussian
            rho_g = gaussian.fit(u_tr, v_tr, rho_clamp=rho_clamp)
            # Student-t
            rho_t, nu_t = student_t.fit(u_tr, v_tr, nu_grid=nu_grid, nu_bounds=nu_bounds, rho_clamp=rho_clamp)

            refit_rows.append(
                {
                    "refit_date": str(dates[k].date()),
                    "k": int(k),
                    "n_train": int(k),
                    "rho_gauss": float(rho_g),
                    "rho_t": float(rho_t),
                    "nu_t": float(nu_t),
                }
            )
            last_refit_k = k

        # score at k with params fit on <= k-1 (we used u[:k], v[:k])
        logc_g[k] = float(gaussian.logpdf(np.array([u[k]]), np.array([v[k]]), rho_g, rho_clamp=rho_clamp)[0])
        logc_t[k] = float(student_t.logpdf(np.array([u[k]]), np.array([v[k]]), rho_t, nu_t, rho_clamp=rho_clamp)[0])

    # Drop the first unscored point (k=0)
    out = pd.DataFrame(
        {
            "date": pd.to_datetime(dates),
            "logc_indep": logc_ind,
            "logc_gauss": logc_g,
            "logc_t": logc_t,
        }
    ).set_index("date")

    out = out.loc[out.index >= out.index[start_k]].copy()

    # sanity: finite
    if not np.isfinite(out["logc_gauss"].to_numpy()).all():
        raise ValueError("NaN/inf found in gaussian log-scores; check implementation / u bounds.")
    if not np.isfinite(out["logc_t"].to_numpy()).all():
        raise ValueError("NaN/inf found in t log-scores; check implementation / u bounds / multivariate_t.")

    # cum + rolling mean
    for col in ["logc_indep", "logc_gauss", "logc_t"]:
        out[f"cum_{col}"] = out[col].cumsum()
        out[f"rollmean_{col}"] = _rolling_mean(out[col].to_numpy(), roll_w)

    # Write predictions
    pred_path = out_dir / "predictions.csv"
    out.to_csv(pred_path, index=True, float_format=float_fmt)
    pred_hash = sha256_bytes(pred_path.read_bytes())

    # Summary metrics
    def summarize(x: np.ndarray) -> Dict[str, float]:
        return {
            "n_obs": float(len(x)),
            "sum_logscore": float(np.sum(x)),
            "mean_logscore": float(np.mean(x)),
            "std_logscore": float(np.std(x, ddof=1)) if len(x) > 1 else float("nan"),
            "min_logscore": float(np.min(x)),
            "max_logscore": float(np.max(x)),
        }

    m_ind = summarize(out["logc_indep"].to_numpy())
    m_g = summarize(out["logc_gauss"].to_numpy())
    m_t = summarize(out["logc_t"].to_numpy())

    warnings = []
    mean_diff_t_g = m_t["mean_logscore"] - m_g["mean_logscore"]
    mean_diff_g_i = m_g["mean_logscore"] - m_ind["mean_logscore"]

    if m_ind["mean_logscore"] > m_g["mean_logscore"] and m_ind["mean_logscore"] > m_t["mean_logscore"]:
        warnings.append("Independence mean logscore > Gaussian and Student-t (check copula logpdf implementation / fitting).")

    metrics = {
        "dataset_version": dataset_version,
        "dataset_hash_sha256_ref": splits.get("dataset_hash_sha256"),
        "pit_u_hash_ref": load_json(pits_dir / "pit_metrics.json").get("u_series_hash_sha256"),
        "evaluation_window": {
            "first_scored_date": str(out.index.min().date()),
            "last_scored_date": str(out.index.max().date()),
        },
        "oos": {
            "refit_every": refit_every,
            "rho_clamp": rho_clamp,
            "oos_convention": "fit_to_t_minus_1_score_at_t",
        },
        "student_t": {"nu_grid": nu_grid, "nu_bounds": list(nu_bounds)},
        "per_model": {"indep": m_ind, "gauss": m_g, "t": m_t},
        "ordering_check": {
            "t_vs_gauss_mean_diff": float(mean_diff_t_g),
            "gauss_vs_indep_mean_diff": float(mean_diff_g_i),
            "warnings": warnings,
        },
        "artifacts": {"predictions_csv": str(pred_path), "predictions_hash_sha256": pred_hash},
        "runtime_sec": float(time.time() - t0),
    }
    dump_json(out_dir / "metrics.json", metrics)

    # Tables
    scores_tbl = pd.DataFrame(
        [
            {"model": "indep", **m_ind},
            {"model": "gauss", **m_g},
            {"model": "t", **m_t},
        ]
    )
    scores_tbl.to_csv(tables_dir / "scores_summary.csv", index=False, float_format=float_fmt)

    params_tbl = pd.DataFrame(refit_rows)
    params_tbl.to_csv(tables_dir / "params_summary.csv", index=False, float_format=float_fmt)

    # Figures (cum)
    plt.figure()
    plt.plot(out.index, out["cum_logc_indep"], label="indep")
    plt.plot(out.index, out["cum_logc_gauss"], label="gauss")
    plt.plot(out.index, out["cum_logc_t"], label="t")
    plt.title("Cumulative log predictive score")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figs_dir / "logscore_cum.png", dpi=160)
    plt.close()

    # Figures (rolling mean)
    plt.figure()
    plt.plot(out.index, out["rollmean_logc_indep"], label="indep")
    plt.plot(out.index, out["rollmean_logc_gauss"], label="gauss")
    plt.plot(out.index, out["rollmean_logc_t"], label="t")
    plt.title(f"Rolling mean logscore (window={roll_w})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figs_dir / "logscore_rolling_mean.png", dpi=160)
    plt.close()

    # Report
    rep = []
    rep.append("# Static Copulas — OOS Log Predictive Score")
    rep.append("")
    rep.append("## Contract (anti-leakage)")
    rep.append("- For each scored date t: parameters are fit using observations with dates ≤ t-1.")
    rep.append("- Scoring: log c(u_t, v_t | θ_{t-1}).")
    rep.append("")
    rep.append("## Configuration")
    rep.append(f"- dataset_version: `{dataset_version}`")
    rep.append(f"- refit_every: `{refit_every}`")
    rep.append(f"- rho_clamp: `{rho_clamp}`")
    rep.append(f"- nu_grid: `{nu_grid}`")
    rep.append(f"- nu_bounds: `{list(nu_bounds)}`")
    rep.append("")
    rep.append("## Results (summary)")
    rep.append(scores_tbl.to_string(index=False))
    rep.append("")
    rep.append("## Ordering checks (diagnostic)")
    rep.append(f"- mean(t) - mean(gauss): {mean_diff_t_g:.6g}")
    rep.append(f"- mean(gauss) - mean(indep): {mean_diff_g_i:.6g}")
    if warnings:
        rep.append("- Warnings:")
        for w in warnings:
            rep.append(f"  - {w}")
    rep.append("")
    rep.append("## Figures")
    rep.append("- figures/logscore_cum.png")
    rep.append("- figures/logscore_rolling_mean.png")
    rep.append("")
    rep.append("## Failure modes")
    rep.append("- If independence beats both models: suspect incorrect logpdf formula, wrong PIT columns, or parameter fit not aligned with OOS slicing.")
    rep.append("- If NaN/inf: suspect u at 0/1, rho too close to ±1, or multivariate_t numerical issues.")
    dump_text(out_dir / "report.md", "\n".join(rep))

    # Provenance
    prov = {
        "created_utc": utc_now_iso(),
        "config_path": str(config_path),
        "env": {"python_version": sys.version.split()[0], "platform": platform.platform()},
        "inputs": {
            "splits_path": str(processed_dir / "splits.json"),
            "pits_dir": str(pits_dir),
            "u_series_path": str(pits_dir / cfg["inputs"].get("u_file", "u_series.csv")),
        },
        "outputs": {
            "out_dir": str(out_dir),
            "metrics_path": str(out_dir / "metrics.json"),
            "predictions_path": str(pred_path),
            "tables_dir": str(tables_dir),
            "figures_dir": str(figs_dir),
            "report_path": str(out_dir / "report.md"),
        },
        "hashes": {"predictions_sha256": pred_hash},
        "runtime_sec": float(time.time() - t0),
    }
    dump_json(out_dir / "provenance.json", prov)

    print(f"[OK] built static copulas: {dataset_version}")
    print(f"[OK] out_dir: {out_dir.resolve()}")
    print(f"[OK] predictions: {pred_path.name} hash={pred_hash[:12]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())