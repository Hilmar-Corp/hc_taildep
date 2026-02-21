# src/hc_taildep/build_impact_j8_top8_pairwise.py
from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from hc_taildep.utils.io import ensure_dir, read_yaml, resolve_config, write_json, write_yaml, write_text, build_provenance
from hc_taildep.impact.var_es_core import StressSpec, run_var_es_for_pair, coverage_table
from hc_taildep.impact.heatmaps import _mat_from_pairs, save_heatmap_png


def _load_returns_csv(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["ts_utc"] = pd.to_datetime(df["ts_utc"], utc=True)
    df = df.set_index("ts_utc").sort_index()
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = read_yaml(args.config)
    cfg = resolve_config(cfg)

    dataset_dir = Path(cfg["dataset_dir"])
    returns_path = dataset_dir / "returns.csv"
    df_ret = _load_returns_csv(returns_path)

    assets: List[str] = cfg["assets"]  # e.g. ["BTCUSDT",...]
    # returns columns match symbols, else allow mapping
    cols = cfg.get("returns_columns", assets)
    missing = [c for c in cols if c not in df_ret.columns]
    if missing:
        raise RuntimeError(f"returns.csv missing columns: {missing}")

    out_root = Path(cfg["out_root"])
    run_name = cfg["run_name"]
    out_dir = ensure_dir(out_root / run_name)
    ensure_dir(out_dir / "pairs")
    ensure_dir(out_dir / "tables")
    ensure_dir(out_dir / "figures")

    # save resolved config
    write_yaml(out_dir / "config.resolved.yaml", cfg)

    stress = StressSpec(
        rv_window=int(cfg["stress"]["rv_window"]),
        calm_q=float(cfg["stress"]["calm_q"]),
        stress_q=float(cfg["stress"]["stress_q"]),
    )

    models: List[str] = cfg["models"]
    alphas: List[float] = [float(a) for a in cfg["alphas"]]
    refit_every = int(cfg["refit_every"])
    n_scenarios = int(cfg["n_scenarios"])
    base_seed = int(cfg["seed"])

    # store per-pair summary KPIs for heatmaps
    # KPI examples:
    # 1) delta ES99 stress vs baseline thr_t
    baseline = cfg.get("baseline_model", "thr_t")
    kpi_delta_es99_stress: Dict[Tuple[str, str], float] = {}
    kpi_ratio_stress_calm: Dict[Tuple[str, str], float] = {}
    kpi_exceed99_all: Dict[Tuple[str, str], float] = {}

    # loop pairs
    for (a, b), (ca, cb) in zip(combinations(assets, 2), combinations(cols, 2)):
        pair_name = f"{a}_{b}"
        pair_dir = ensure_dir(out_dir / "pairs" / pair_name)
        ensure_dir(pair_dir / "tables")
        ensure_dir(pair_dir / "figures")

        ts = df_ret.index
        r1 = df_ret[ca].to_numpy(dtype=float)
        r2 = df_ret[cb].to_numpy(dtype=float)

        pred = run_var_es_for_pair(
            ts=ts,
            r1=r1,
            r2=r2,
            refit_every=refit_every,
            n_scenarios=n_scenarios,
            alphas=alphas,
            stress=stress,
            base_seed=base_seed,
            models=models,
        )
        pred_path = pair_dir / "var_es_predictions.csv"
        pred.to_csv(pred_path, index=False)

        # coverage tables
        cov_all = []
        for a_ in alphas:
            cov = coverage_table(pred, models=models, alpha=a_)
            cov_all.append(cov)
        cov_df = pd.concat(cov_all, axis=0).reset_index(drop=True)
        cov_path = pair_dir / "tables" / "coverage_tests.csv"
        cov_df.to_csv(cov_path, index=False)

        # simple summary for KPI extraction
        # ES99 stress averages
        alpha99 = 0.99 if 0.99 in alphas else max(alphas)
        es_col = f"ES{int(alpha99*100)}_{{m}}"
        d_stress = pred[pred["bucket"] == "stress"]
        d_calm = pred[pred["bucket"] == "calm"]
        d_all = pred

        def _mean(df: pd.DataFrame, col: str) -> float:
            if df.shape[0] == 0:
                return float("nan")
            x = df[col].to_numpy(dtype=float)
            x = x[np.isfinite(x)]
            return float(np.mean(x)) if x.size else float("nan")

        es_stress_m = _mean(d_stress, es_col.format(m=models[-1]))  # just a placeholder
        # choose one "focus model" for heatmaps
        focus_model = cfg.get("focus_model", "thr_t")
        if focus_model not in models:
            focus_model = models[-1]

        es_stress_focus = _mean(d_stress, es_col.format(m=focus_model))
        es_stress_base = _mean(d_stress, es_col.format(m=baseline))
        delta = es_stress_focus - es_stress_base
        kpi_delta_es99_stress[(a, b)] = delta

        es_calm_focus = _mean(d_calm, es_col.format(m=focus_model))
        ratio = (es_stress_focus / es_calm_focus) if (np.isfinite(es_stress_focus) and np.isfinite(es_calm_focus) and es_calm_focus != 0) else float("nan")
        kpi_ratio_stress_calm[(a, b)] = ratio

        # exceedance rate 99 all for baseline (calibration grossière)
        ex_col = f"exceed{int(alpha99*100)}_{baseline}"
        if ex_col in pred.columns and pred.shape[0] > 0:
            kpi_exceed99_all[(a, b)] = float(pred[ex_col].mean())
        else:
            kpi_exceed99_all[(a, b)] = float("nan")

        # per-pair report stub
        write_text(
            pair_dir / "report.md",
            f"""# ANNEXE — J8 Pairwise VaR/ES — {pair_name}

This is an ANNEX robustness run (pairwise, daily). It does not change core claims.

- Models: {models}
- Baseline: {baseline}
- Focus model (heatmaps): {focus_model}
- n_scenarios: {n_scenarios}
- refit_every: {refit_every}
- Stress RV window: {stress.rv_window} (train-only thresholds)

Key KPI (stress, ES{int(alpha99*100)}): Δ = {delta:.6g}
""",
        )

        # provenance per pair
        prov = build_provenance(
            config_path=args.config,
            config_resolved=cfg,
            inputs={"returns.csv": returns_path},
            outputs={"var_es_predictions.csv": pred_path, "coverage_tests.csv": cov_path},
        )
        write_json(pair_dir / "provenance.json", prov)

    # build global heatmaps
    assets_order = assets
    mat1 = _mat_from_pairs(assets_order, kpi_delta_es99_stress)
    mat2 = _mat_from_pairs(assets_order, kpi_ratio_stress_calm)
    mat3 = _mat_from_pairs(assets_order, kpi_exceed99_all)

    mat1.to_csv(out_dir / "tables" / "heatmap_delta_es99_stress.csv")
    mat2.to_csv(out_dir / "tables" / "heatmap_ratio_stress_calm_es99.csv")
    mat3.to_csv(out_dir / "tables" / "heatmap_exceed99_all_baseline.csv")

    save_heatmap_png(mat1, out_dir / "figures" / "heatmap_delta_es99_stress.png", "ΔES99(stress): focus - baseline")
    save_heatmap_png(mat2, out_dir / "figures" / "heatmap_ratio_stress_calm_es99.png", "ES99 stress/calm ratio (focus)")
    save_heatmap_png(mat3, out_dir / "figures" / "heatmap_exceed99_all_baseline.png", "Exceedance rate 99% (baseline)")

    # global report
    write_text(
        out_dir / "report.md",
        f"""# ANNEXE — J8 Top-8 Pairwise Heatmaps (Daily)

This is an ANNEX robustness check. It does not modify core narrative.

Dataset: {dataset_dir}
Pairs: {len(list(combinations(assets,2)))}
Models: {models}
Baseline: {baseline}
Focus model: {cfg.get("focus_model","thr_t")}
alphas: {alphas}
n_scenarios: {n_scenarios}
refit_every: {refit_every}

Produced:
- tables/heatmap_delta_es99_stress.csv
- tables/heatmap_ratio_stress_calm_es99.csv
- tables/heatmap_exceed99_all_baseline.csv
- figures/*.png
""",
    )

    # provenance global
    prov = build_provenance(
        config_path=args.config,
        config_resolved=cfg,
        inputs={"returns.csv": returns_path, "meta.json": dataset_dir / "meta.json"},
        outputs={"report.md": out_dir / "report.md"},
    )
    write_json(out_dir / "provenance.json", prov)

    print(f"[OK] J8 top8 pairwise done: out_dir={out_dir}")


if __name__ == "__main__":
    main()