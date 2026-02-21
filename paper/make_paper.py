# paper/make_paper.py
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

# headless-safe
import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt


# ----------------------------
# Low-level utils (deterministic)
# ----------------------------

def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _write_text(path: Path, s: str) -> None:
    _ensure_dir(path.parent)
    path.write_text(s, encoding="utf-8")


def _write_json(path: Path, obj: Any) -> None:
    _ensure_dir(path.parent)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")


def _read_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(_read_text(path))


_VAR_PAT = re.compile(r"\$\{([^}]+)\}")


def _resolve_vars(obj: Any, vars_map: Dict[str, str]) -> Any:
    """
    Minimal ${var} substitution for YAML strings.
    Only supports keys in vars_map, deterministic.
    """
    if isinstance(obj, dict):
        return {k: _resolve_vars(v, vars_map) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_resolve_vars(v, vars_map) for v in obj]
    if isinstance(obj, str):
        def repl(m):
            key = m.group(1).strip()
            return vars_map.get(key, m.group(0))
        return _VAR_PAT.sub(repl, obj)
    return obj


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _git_info(repo_root: Path) -> dict[str, Any]:
    """
    Best-effort git info without external deps.
    If git is not available, we still produce a stable manifest.
    """
    import subprocess

    def run(cmd: List[str]) -> Tuple[int, str]:
        try:
            p = subprocess.run(cmd, cwd=str(repo_root), capture_output=True, text=True)
            out = (p.stdout or "").strip()
            return p.returncode, out
        except Exception:
            return 1, ""

    rc, commit = run(["git", "rev-parse", "HEAD"])
    rc2, status = run(["git", "status", "--porcelain"])
    dirty = bool(status) if rc2 == 0 else None
    rc3, branch = run(["git", "rev-parse", "--abbrev-ref", "HEAD"])
    return {
        "commit": commit if rc == 0 else None,
        "dirty": dirty,
        "branch": branch if rc3 == 0 else None,
    }


def _load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def _save_csv_stable(df: pd.DataFrame, path: Path, float_fmt: str, table_round: int) -> None:
    out = df.copy()

    # stable column order
    out = out.reindex(columns=list(out.columns))

    # deterministic rounding for numeric columns
    for c in out.columns:
        if pd.api.types.is_numeric_dtype(out[c]):
            out[c] = np.round(out[c].astype(float), int(table_round))

    _ensure_dir(path.parent)
    out.to_csv(path, index=False, float_format=float_fmt)


def _set_matplotlib_style(fig_w: float, fig_h: float) -> None:
    # Keep default colors; just enforce deterministic sizing
    plt.rcParams["figure.figsize"] = (float(fig_w), float(fig_h))
    plt.rcParams["savefig.dpi"] = 200


# ----------------------------
# Checks
# ----------------------------

class MissingArtifactError(RuntimeError):
    pass


def _require(path: Path, what: str) -> None:
    if not path.exists():
        raise MissingArtifactError(f"Missing required {what}: {path}")


def _maybe(path: Path) -> Optional[Path]:
    return path if path.exists() else None


# ----------------------------
# Extractors
# ----------------------------

@dataclass(frozen=True)
class RunRef:
    tag: str
    out_dir: Path
    kind: str  # "j6" | "j7" | "j8_top8" | "j8_asym"
    is_annex: bool


def _validate_run_dir(run: RunRef, required_files: List[str], optional_files: List[str], fail_on_missing: bool) -> dict[str, Any]:
    info: dict[str, Any] = {"tag": run.tag, "kind": run.kind, "out_dir": str(run.out_dir), "files": {}}
    for rel in required_files:
        p = run.out_dir / rel
        if not p.exists():
            if fail_on_missing:
                raise MissingArtifactError(f"[{run.tag}] missing required file: {p}")
            info["files"][rel] = {"exists": False}
        else:
            info["files"][rel] = {"exists": True, "sha256": _sha256_file(p)}
    for rel in optional_files:
        p = run.out_dir / rel
        info["files"][rel] = {"exists": p.exists(), "sha256": _sha256_file(p) if p.exists() else None}
    return info


@dataclass(frozen=True)
class J6Bundle:
    out_dir: Path
    predictions: pd.DataFrame
    scores_summary: pd.DataFrame
    dm_summary: pd.DataFrame
    metrics: Optional[dict[str, Any]]


def load_j6(out_dir: Path) -> J6Bundle:
    pred_p = out_dir / "predictions.csv"
    ss_p = out_dir / "tables" / "scores_summary.csv"
    dm_p = out_dir / "tables" / "dm_summary.csv"
    met_p = out_dir / "metrics.json"

    _require(pred_p, "J6 predictions.csv")
    _require(ss_p, "J6 tables/scores_summary.csv")
    _require(dm_p, "J6 tables/dm_summary.csv")

    predictions = _load_csv(pred_p)
    scores_summary = _load_csv(ss_p)
    dm_summary = _load_csv(dm_p)

    metrics = None
    if met_p.exists():
        metrics = json.loads(_read_text(met_p))

    return J6Bundle(out_dir=out_dir, predictions=predictions, scores_summary=scores_summary, dm_summary=dm_summary, metrics=metrics)


@dataclass(frozen=True)
class J7Bundle:
    out_dir: Path
    var_es_predictions: pd.DataFrame
    var_es_summary: pd.DataFrame
    coverage_tests: pd.DataFrame
    deltas_summary: Optional[pd.DataFrame]


def load_j7(out_dir: Path) -> J7Bundle:
    pred_p = out_dir / "var_es_predictions.csv"
    ve_p = out_dir / "tables" / "var_es_summary.csv"
    cov_p = out_dir / "tables" / "coverage_tests.csv"
    deltas_p = out_dir / "tables" / "deltas_summary.csv"

    _require(pred_p, "J7 var_es_predictions.csv")
    _require(ve_p, "J7 tables/var_es_summary.csv")
    _require(cov_p, "J7 tables/coverage_tests.csv")

    var_es_predictions = _load_csv(pred_p)
    var_es_summary = _load_csv(ve_p)
    coverage_tests = _load_csv(cov_p)
    deltas_summary = _load_csv(deltas_p) if deltas_p.exists() else None

    return J7Bundle(
        out_dir=out_dir,
        var_es_predictions=var_es_predictions,
        var_es_summary=var_es_summary,
        coverage_tests=coverage_tests,
        deltas_summary=deltas_summary,
    )


@dataclass(frozen=True)
class J8Top8Bundle:
    out_dir: Path
    heatmap_delta_es99_stress: pd.DataFrame
    heatmap_ratio_stress_calm_es99: pd.DataFrame
    heatmap_exceed99_all_baseline: pd.DataFrame


def load_j8_top8(out_dir: Path) -> J8Top8Bundle:
    p1 = out_dir / "tables" / "heatmap_delta_es99_stress.csv"
    p2 = out_dir / "tables" / "heatmap_ratio_stress_calm_es99.csv"
    p3 = out_dir / "tables" / "heatmap_exceed99_all_baseline.csv"
    _require(p1, "J8 top8 tables/heatmap_delta_es99_stress.csv")
    _require(p2, "J8 top8 tables/heatmap_ratio_stress_calm_es99.csv")
    _require(p3, "J8 top8 tables/heatmap_exceed99_all_baseline.csv")

    return J8Top8Bundle(out_dir=out_dir, heatmap_delta_es99_stress=_load_csv(p1), heatmap_ratio_stress_calm_es99=_load_csv(p2), heatmap_exceed99_all_baseline=_load_csv(p3))


@dataclass(frozen=True)
class J8AsymBundle:
    out_dir: Path
    tail_dep: pd.DataFrame


def load_j8_asym(out_dir: Path) -> J8AsymBundle:
    p = out_dir / "tables" / "tail_dependence_mc.csv"
    _require(p, "J8 asym tables/tail_dependence_mc.csv")
    return J8AsymBundle(out_dir=out_dir, tail_dep=_load_csv(p))


# ----------------------------
# Figure builders (camera-ready)
# ----------------------------

def _parse_dates_maybe(s: pd.Series) -> pd.DatetimeIndex:
    # accept ISO strings, "YYYY-MM-DD", etc.
    return pd.to_datetime(s, utc=True, errors="coerce")


def fig_delta_logscore_cum(
    j6: J6Bundle,
    out_path: Path,
    *,
    fig_w: float,
    fig_h: float,
    dpi: int,
) -> None:
    df = j6.predictions.copy()

    # Expected columns in J6 predictions.csv
    # - date
    # - logc_ms_t
    # - logc_thr_t
    for c in ["date", "logc_ms_t", "logc_thr_t"]:
        _require(Path("."), f"J6 predictions must contain column '{c}'") if c not in df.columns else None
    if any(c not in df.columns for c in ["date", "logc_ms_t", "logc_thr_t"]):
        raise MissingArtifactError(f"J6 predictions.csv missing required columns. Have: {list(df.columns)[:30]}")

    t = _parse_dates_maybe(df["date"])
    d = (
        pd.to_numeric(df["logc_ms_t"], errors="coerce")
        - pd.to_numeric(df["logc_thr_t"], errors="coerce")
    ).to_numpy(dtype=float, copy=True)
    d[~np.isfinite(d)] = 0.0
    cum = np.cumsum(d)

    _set_matplotlib_style(fig_w, fig_h)
    plt.figure()
    plt.plot(t, cum)
    plt.title("Cumulative Δ logscore (ms_t − thr_t)")
    plt.xlabel("time (UTC)")
    plt.ylabel("cumulative Δ logscore")
    plt.tight_layout()
    _ensure_dir(out_path.parent)
    plt.savefig(out_path, dpi=int(dpi))
    plt.close()


def fig_delta_logscore_hist(
    j6: J6Bundle,
    out_path: Path,
    *,
    fig_w: float,
    fig_h: float,
    dpi: int,
) -> None:
    df = j6.predictions.copy()
    for c in ["logc_ms_t", "logc_thr_t"]:
        if c not in df.columns:
            raise MissingArtifactError(f"J6 predictions.csv missing '{c}'")
    d = (
        pd.to_numeric(df["logc_ms_t"], errors="coerce")
        - pd.to_numeric(df["logc_thr_t"], errors="coerce")
    ).to_numpy(dtype=float, copy=True)
    d = d[np.isfinite(d)]
    _set_matplotlib_style(fig_w, fig_h)
    plt.figure()
    plt.hist(d, bins=60)
    plt.title("Δ logscore distribution (ms_t − thr_t)")
    plt.xlabel("Δ logscore")
    plt.ylabel("count")
    plt.tight_layout()
    _ensure_dir(out_path.parent)
    plt.savefig(out_path, dpi=int(dpi))
    plt.close()


def fig_es_timeseries(
    j7: J7Bundle,
    out_path: Path,
    *,
    alpha: float,
    models: List[str],
    fig_w: float,
    fig_h: float,
    dpi: int,
) -> None:
    df = j7.var_es_predictions.copy()
    if "date" not in df.columns:
        # some of your runs use "ts_utc"
        if "ts_utc" in df.columns:
            df = df.rename(columns={"ts_utc": "date"})
        else:
            raise MissingArtifactError(f"J7 var_es_predictions.csv missing date column. Have: {list(df.columns)[:30]}")

    t = _parse_dates_maybe(df["date"])
    k = int(round(alpha * 100))
    _set_matplotlib_style(fig_w, fig_h)
    plt.figure()

    plotted = 0
    for m in models:
        col = f"ES{k}_{m}"
        if col not in df.columns:
            continue
        y = pd.to_numeric(df[col], errors="coerce").to_numpy(float)
        plt.plot(t, y, label=m)
        plotted += 1

    if plotted == 0:
        raise MissingArtifactError(f"No ES columns found for alpha={alpha} among models={models}. Example needed: ES{k}_thr_t")

    plt.title(f"ES{int(alpha*100)} time series (loss units)")
    plt.xlabel("time (UTC)")
    plt.ylabel(f"ES{int(alpha*100)}")
    plt.legend(loc="best", fontsize=8)
    plt.tight_layout()
    _ensure_dir(out_path.parent)
    plt.savefig(out_path, dpi=int(dpi))
    plt.close()


def fig_heatmap_from_csv_matrix(
    mat: pd.DataFrame,
    out_path: Path,
    title: str,
    *,
    fig_w: float,
    fig_h: float,
    dpi: int,
) -> None:
    # CSV matrices sometimes include unnamed index column.
    m = mat.copy()
    if m.columns.size > 0 and (m.columns[0].startswith("Unnamed") or m.columns[0] == ""):
        m = m.set_index(m.columns[0])
    # ensure numeric
    m = m.apply(pd.to_numeric, errors="coerce")

    _set_matplotlib_style(fig_w, fig_h)
    plt.figure(figsize=(max(6.0, fig_w), max(5.5, fig_h)))
    arr = m.to_numpy(dtype=float)
    plt.imshow(arr, aspect="auto")
    plt.title(title)
    plt.xticks(range(len(m.columns)), m.columns, rotation=45, ha="right")
    plt.yticks(range(len(m.index)), m.index)
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.tight_layout()
    _ensure_dir(out_path.parent)
    plt.savefig(out_path, dpi=int(dpi))
    plt.close()


def fig_asym_taildep_barplot(
    asym: J8AsymBundle,
    out_path: Path,
    *,
    q_label: str = "q=0.05 (finite-q)",
    fig_w: float,
    fig_h: float,
    dpi: int,
) -> None:
    df = asym.tail_dep.copy()
    for c in ["bucket", "family", "lambda_L", "lambda_U"]:
        if c not in df.columns:
            raise MissingArtifactError(f"tail_dependence_mc.csv missing '{c}'")

    # aggregate mean across refits (annex diagnostic)
    g = df.groupby(["bucket", "family"], as_index=False)[["lambda_L", "lambda_U"]].mean()

    # barplot in a simple deterministic order
    buckets = ["calm", "stress"]
    families = sorted(g["family"].unique().tolist())

    x_labels = []
    lamL = []
    lamU = []
    for b in buckets:
        for f in families:
            row = g[(g["bucket"] == b) & (g["family"] == f)]
            if row.empty:
                continue
            x_labels.append(f"{b}:{f}")
            lamL.append(float(row["lambda_L"].iloc[0]))
            lamU.append(float(row["lambda_U"].iloc[0]))

    x = np.arange(len(x_labels))
    _set_matplotlib_style(fig_w, fig_h)
    plt.figure(figsize=(max(7.0, fig_w), max(4.0, fig_h)))
    # no manual colors; default cycle
    plt.bar(x - 0.2, lamL, width=0.4, label="lambda_L")
    plt.bar(x + 0.2, lamU, width=0.4, label="lambda_U")
    plt.xticks(x, x_labels, rotation=45, ha="right")
    plt.title(f"Tail dependence (MC) — {q_label}")
    plt.ylabel("lambda (finite-q diagnostic)")
    plt.legend(loc="best", fontsize=8)
    plt.tight_layout()
    _ensure_dir(out_path.parent)
    plt.savefig(out_path, dpi=int(dpi))
    plt.close()


# ----------------------------
# Table builders
# ----------------------------

def table_scores_summary(j6: J6Bundle) -> pd.DataFrame:
    df = j6.scores_summary.copy()
    # stable sort if possible
    if "model" in df.columns:
        df = df.sort_values(["model"]).reset_index(drop=True)
    return df


def table_dm_summary(j6: J6Bundle) -> pd.DataFrame:
    df = j6.dm_summary.copy()
    if "name" in df.columns:
        df = df.sort_values(["name"]).reset_index(drop=True)
    elif "comparison" in df.columns:
        df = df.sort_values(["comparison"]).reset_index(drop=True)
    return df


def table_var_es_summary(j7: J7Bundle) -> pd.DataFrame:
    df = j7.var_es_summary.copy()
    # stable ordering
    sort_cols = [c for c in ["bucket", "model", "alpha"] if c in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols).reset_index(drop=True)
    return df


def table_coverage_tests(j7: J7Bundle) -> pd.DataFrame:
    df = j7.coverage_tests.copy()
    sort_cols = [c for c in ["bucket", "model", "alpha"] if c in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols).reset_index(drop=True)
    return df


# ----------------------------
# Report + Manifest
# ----------------------------

def _checklist_text() -> str:
    return "\n".join(
        [
            "# Checklist anti-boulette (reviewer-facing)",
            "",
            "- Scope: pas de trading, pas d’alpha, pas de “predict direction”.",
            "- OOS strict: à t, tout fit utilise uniquement ≤ t-1 ; scoring sur t (one-step-ahead).",
            "- PIT: jamais de fit global full-sample (u_t = F_{t-1}(r_t)).",
            "- Repro: seeds + hashes + dataset_version + commit ; outputs hashés.",
            "- Baselines d’abord: indep/gauss/t statique avant dynamique.",
            "- Si un modèle complexe gagne: DM (HAC/NW) + robustness ; sinon annexe.",
            "",
        ]
    )


def _write_paper_summary(
    out_dir: Path,
    *,
    paper_id: str,
    runs_info: List[dict[str, Any]],
    manifest: dict[str, Any],
) -> None:
    lines: List[str] = []
    lines.append(f"# {paper_id} — camera-ready pack")
    lines.append("")
    lines.append(_checklist_text())
    lines.append("## What this command produced")
    lines.append("")
    lines.append("- figures/: camera-ready figures (rebuilt from CSV/tables, not copied)")
    lines.append("- tables/: camera-ready tables (stable sort + rounding)")
    lines.append("- manifest.json: hashes + sources + git info")
    lines.append("")
    lines.append("## Runs (sources)")
    lines.append("")
    for r in runs_info:
        lines.append(f"- [{r['kind']}] {r['tag']}  →  `{r['out_dir']}`")
    lines.append("")
    lines.append("## Reproduce")
    lines.append("")
    lines.append("```bash")
    lines.append(f"python -m paper.make_paper --spec paper/paper_spec.yaml")
    lines.append("```")
    lines.append("")
    lines.append("## Manifest (summary)")
    lines.append("")
    lines.append(f"- manifest_sha256: `{manifest.get('manifest_sha256')}`")
    lines.append(f"- created_utc: `{manifest.get('created_utc')}`")
    lines.append(f"- git: {json.dumps(manifest.get('git', {}), sort_keys=True)}")
    lines.append("")
    _write_text(out_dir / "paper_summary.md", "\n".join(lines))


# ----------------------------
# Main orchestration
# ----------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True, help="paper/paper_spec.yaml")
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    spec_path = Path(args.spec)
    if not spec_path.exists():
        raise SystemExit(f"Spec not found: {spec_path}")

    spec_raw = _read_yaml(spec_path)

    paper_id = str(spec_raw.get("paper_id", "paper")).strip()
    vars_map = {"paper_id": paper_id}
    spec = _resolve_vars(spec_raw, vars_map)

    out_cfg = spec.get("outputs", {})
    out_dir = Path(out_cfg.get("out_dir", f"paper/out/{paper_id}"))
    formats = list(out_cfg.get("formats", ["png"]))
    dpi = int(out_cfg.get("dpi", 200))
    fig_w = float(out_cfg.get("fig_width_in", 7.0))
    fig_h = float(out_cfg.get("fig_height_in", 4.0))
    float_fmt = str(out_cfg.get("float_fmt", "%.6g"))
    table_round = int(out_cfg.get("table_round", 6))
    fail_on_missing = bool(out_cfg.get("fail_on_missing", True))

    checks = spec.get("checks", {})
    required_files = list(checks.get("required_files", ["config.resolved.yaml", "provenance.json"]))
    optional_files = list(checks.get("optional_files", ["report.md"]))

    # Prepare output structure
    fig_out = out_dir / "figures"
    tab_out = out_dir / "tables"
    _ensure_dir(fig_out)
    _ensure_dir(tab_out)

    # Collect run refs
    run_refs: List[RunRef] = []

    core = spec.get("core", {})
    if "j6" in core:
        run_refs.append(RunRef(tag="core_j6", kind="j6", out_dir=Path(core["j6"]["out_dir"]), is_annex=False))
    if "j7" in core:
        run_refs.append(RunRef(tag="core_j7", kind="j7", out_dir=Path(core["j7"]["out_dir"]), is_annex=False))

    annex = spec.get("annex", {})
    if annex.get("j8_top8", {}).get("enabled", False):
        run_refs.append(RunRef(tag="annex_j8_top8", kind="j8_top8", out_dir=Path(annex["j8_top8"]["out_dir"]), is_annex=True))
    if annex.get("j8_asym", {}).get("enabled", False):
        run_refs.append(RunRef(tag="annex_j8_asym", kind="j8_asym", out_dir=Path(annex["j8_asym"]["out_dir"]), is_annex=True))

    # Validate run dirs + collect provenance hashes
    runs_info: List[dict[str, Any]] = []
    for rr in run_refs:
        if not rr.out_dir.exists():
            if fail_on_missing:
                raise MissingArtifactError(f"Run dir missing: {rr.out_dir} ({rr.tag})")
            continue
        info = _validate_run_dir(rr, required_files, optional_files, fail_on_missing)
        runs_info.append(info)

    # Load bundles
    j6 = load_j6(Path(core["j6"]["out_dir"]))
    j7 = load_j7(Path(core["j7"]["out_dir"]))

    j8_top8 = None
    if annex.get("j8_top8", {}).get("enabled", False):
        j8_top8 = load_j8_top8(Path(annex["j8_top8"]["out_dir"]))

    j8_asym = None
    if annex.get("j8_asym", {}).get("enabled", False):
        j8_asym = load_j8_asym(Path(annex["j8_asym"]["out_dir"]))

    # ----------------------------
    # Build CORE figures/tables
    # ----------------------------

    # CORE figs
    fig_delta_logscore_cum(j6, fig_out / "fig_F1_delta_logscore_cum_ms_t_vs_thr_t.png", fig_w=fig_w, fig_h=fig_h, dpi=dpi)
    fig_delta_logscore_hist(j6, fig_out / "fig_F2_delta_logscore_hist_ms_t_vs_thr_t.png", fig_w=fig_w, fig_h=fig_h, dpi=dpi)

    # J7 ES timeseries (choose common models; skip missing columns gracefully inside builder)
    fig_es_timeseries(
        j7,
        fig_out / "fig_F3_es99_timeseries_models.png",
        alpha=0.99,
        models=["indep", "static_gauss", "static_t", "thr_gauss", "thr_t", "ms_gauss", "ms_t"],
        fig_w=fig_w,
        fig_h=fig_h,
        dpi=dpi,
    )

    # CORE tables (stable)
    t1 = table_scores_summary(j6)
    _save_csv_stable(t1, tab_out / "tab_T1_logscore_summary.csv", float_fmt=float_fmt, table_round=table_round)

    t2 = table_dm_summary(j6)
    _save_csv_stable(t2, tab_out / "tab_T2_dm_summary.csv", float_fmt=float_fmt, table_round=table_round)

    t3 = table_var_es_summary(j7)
    _save_csv_stable(t3, tab_out / "tab_T3_var_es_summary.csv", float_fmt=float_fmt, table_round=table_round)

    t4 = table_coverage_tests(j7)
    _save_csv_stable(t4, tab_out / "tab_T4_coverage_tests.csv", float_fmt=float_fmt, table_round=table_round)

    # ----------------------------
    # Build ANNEX figures/tables
    # ----------------------------
    if j8_top8 is not None:
        fig_heatmap_from_csv_matrix(
            j8_top8.heatmap_delta_es99_stress,
            fig_out / "fig_A1_heatmap_delta_es99_stress.png",
            "ANNEX — ΔES99(stress) (focus − baseline)",
            fig_w=max(fig_w, 7.5),
            fig_h=max(fig_h, 6.0),
            dpi=dpi,
        )
        fig_heatmap_from_csv_matrix(
            j8_top8.heatmap_ratio_stress_calm_es99,
            fig_out / "fig_A2_heatmap_ratio_stress_calm_es99.png",
            "ANNEX — ES99 stress/calm ratio (focus)",
            fig_w=max(fig_w, 7.5),
            fig_h=max(fig_h, 6.0),
            dpi=dpi,
        )
        fig_heatmap_from_csv_matrix(
            j8_top8.heatmap_exceed99_all_baseline,
            fig_out / "fig_A3_heatmap_exceed99_all_baseline.png",
            "ANNEX — Exceedance rate 99% (baseline)",
            fig_w=max(fig_w, 7.5),
            fig_h=max(fig_h, 6.0),
            dpi=dpi,
        )

        # also store the raw matrices in paper/tables
        _save_csv_stable(j8_top8.heatmap_delta_es99_stress, tab_out / "tab_A1_heatmap_delta_es99_stress.csv", float_fmt=float_fmt, table_round=table_round)
        _save_csv_stable(j8_top8.heatmap_ratio_stress_calm_es99, tab_out / "tab_A2_heatmap_ratio_stress_calm_es99.csv", float_fmt=float_fmt, table_round=table_round)
        _save_csv_stable(j8_top8.heatmap_exceed99_all_baseline, tab_out / "tab_A3_heatmap_exceed99_all_baseline.csv", float_fmt=float_fmt, table_round=table_round)

    if j8_asym is not None:
        # store tail dependence table + a simple barplot
        td = j8_asym.tail_dep.copy()
        # stable sort
        sort_cols = [c for c in ["refit_index", "bucket", "family"] if c in td.columns]
        if sort_cols:
            td = td.sort_values(sort_cols).reset_index(drop=True)
        _save_csv_stable(td, tab_out / "tab_A6_tail_dependence_mc.csv", float_fmt=float_fmt, table_round=table_round)
        fig_asym_taildep_barplot(j8_asym, fig_out / "fig_A5_tail_dependence_barplot.png", fig_w=fig_w, fig_h=fig_h, dpi=dpi)

    # ----------------------------
    # Manifest (inputs/outputs hashes)
    # ----------------------------
    git = _git_info(repo_root)

    produced_files: List[Path] = []
    for p in fig_out.rglob("*"):
        if p.is_file():
            produced_files.append(p)
    for p in tab_out.rglob("*"):
        if p.is_file():
            produced_files.append(p)

    produced = []
    for p in sorted(produced_files, key=lambda x: str(x)):
        produced.append({"path": str(p), "sha256": _sha256_file(p)})

    spec_bytes = json.dumps(spec, sort_keys=True).encode("utf-8")
    manifest = {
        "paper_id": paper_id,
        "created_utc": _now_utc_iso(),
        "git": git,
        "spec_path": str(spec_path),
        "spec_sha256": _sha256_bytes(spec_bytes),
        "runs": runs_info,
        "outputs": produced,
    }
    manifest_bytes = json.dumps(manifest, sort_keys=True).encode("utf-8")
    manifest["manifest_sha256"] = _sha256_bytes(manifest_bytes)

    _write_json(out_dir / "manifest.json", manifest)

    # Summary markdown
    _write_paper_summary(out_dir, paper_id=paper_id, runs_info=runs_info, manifest=manifest)

    print(f"[OK] J9 paper pack built: {out_dir}")
    print(f"[OK] figures: {fig_out}")
    print(f"[OK] tables: {tab_out}")
    print(f"[OK] manifest: {out_dir / 'manifest.json'} sha256={manifest['manifest_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())