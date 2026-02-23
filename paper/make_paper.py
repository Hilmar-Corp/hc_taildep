# paper/make_paper.py
from __future__ import annotations

from collections.abc import Mapping

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


# ----------------------------
# PIT diagnostics helpers (core + annex)
# ----------------------------

def _discover_u_series_csv(dataset_root: Path) -> Path:
    """Locate PIT u_series CSV under dataset_root/pits/ deterministically."""
    pits = dataset_root / "pits"
    if pits.exists():
        cand = pits / "u_series.csv"
        if cand.exists():
            return cand
        # deterministic fallback
        hits = sorted([p for p in pits.rglob("*.csv") if "u_series" in p.name])
        if hits:
            return hits[0]
    raise MissingArtifactError(
        "Cannot locate PIT u_series.csv under: " + str(dataset_root) + ". Expected data/processed/<dataset_version>/pits/u_series.csv"
    )


def _load_pit_metrics_json(dataset_root: Path) -> Optional[dict[str, Any]]:
    pits = dataset_root / "pits"
    p = pits / "pit_metrics.json"
    if p.exists():
        try:
            return json.loads(_read_text(p))
        except Exception:
            return None
    return None


def _infer_u_columns(u_df: pd.DataFrame) -> Tuple[str, str]:
    """Infer BTC/ETH PIT columns."""
    cols = list(u_df.columns)
    # common canonical names
    for a, b in [("u_BTC", "u_ETH"), ("u_btc", "u_eth"), ("BTC", "ETH")]:
        if a in cols and b in cols:
            return a, b
    # fallback: find first two columns that look like PIT uniforms
    num_cols = []
    for c in cols:
        if c.lower() in {"date", "ts_utc", "datetime", "timestamp", "time"}:
            continue
        x = pd.to_numeric(u_df[c], errors="coerce").to_numpy(dtype=float)
        if x.size == 0:
            continue
        ok = np.isfinite(x)
        if ok.mean() < 0.5:
            continue
        # PIT should mostly live in (0,1)
        frac01 = ((x[ok] > 0.0) & (x[ok] < 1.0)).mean() if ok.any() else 0.0
        if frac01 >= 0.9:
            num_cols.append(c)
    if len(num_cols) >= 2:
        return num_cols[0], num_cols[1]
    raise MissingArtifactError(f"Cannot infer PIT columns in u_series.csv. cols={cols[:20]}")


def _load_u_series(u_series_csv: Path) -> pd.DataFrame:
    df = _load_csv(u_series_csv)
    # normalize time column
    if "date" not in df.columns:
        if "ts_utc" in df.columns:
            df = df.rename(columns={"ts_utc": "date"})
        else:
            # best-effort: first column might be a date-like index
            c0 = df.columns[0]
            if c0.lower().startswith("unnamed") or c0.lower() in {"index"}:
                df = df.rename(columns={c0: "date"})
    if "date" not in df.columns:
        raise MissingArtifactError(f"u_series.csv missing a date column. Have: {list(df.columns)[:20]}")
    df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    return df


def _autocorr(x: np.ndarray, lag: int) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size <= lag + 5:
        return float("nan")
    x0 = x[:-lag]
    x1 = x[lag:]
    x0 = x0 - np.mean(x0)
    x1 = x1 - np.mean(x1)
    denom = float(np.sqrt(np.sum(x0 * x0) * np.sum(x1 * x1)))
    if denom <= 0:
        return float("nan")
    return float(np.sum(x0 * x1) / denom)


# ----------------------------
# Dataset-level helpers (returns/splits/returns-stats table)
# ----------------------------

def _find_dataset_root_from_run(out_dir: Path) -> Path:
    """Find dataset root (data/processed/<dataset_version>) from a run out_dir.

    We search upward until we find both returns.csv and splits.json.
    This is robust to different run layouts (copulas/*, impact/*, etc.).
    """
    p = out_dir.resolve()
    for _ in range(10):
        if (p / "returns.csv").exists() and (p / "splits.json").exists():
            return p
        p = p.parent
    raise MissingArtifactError(
        "Cannot locate dataset root containing returns.csv + splits.json by walking parents from: "
        + str(out_dir)
    )




def _read_splits_json(path: Path) -> dict[str, Any]:
    obj = json.loads(_read_text(path))
    # minimal schema validation
    for k in ["train_start", "first_oos", "last_oos"]:
        if k not in obj:
            raise MissingArtifactError(f"splits.json missing key '{k}': {path}")
    return obj

# PIT artifact finder
def _find_pit_artifacts(dataset_root: Path) -> tuple[Optional[Path], Optional[Path]]:
    """Locate PIT artifacts under a dataset root.

    Returns (u_series_csv_path, pit_metrics_json_path), each possibly None.
    Expected canonical layout:
      <dataset_root>/pits/u_series.csv
      <dataset_root>/pits/pit_metrics.json
    """
    pits = dataset_root / "pits"
    u_csv = pits / "u_series.csv"
    m_json = pits / "pit_metrics.json"
    return (u_csv if u_csv.exists() else None, m_json if m_json.exists() else None)


# Helper to select dataset root for paper figures/tables (see rationale in docstring)
def _dataset_root_for_paper(dataset_root_hint: Path) -> Path:
    """Select the dataset root used for paper-level (dataset) figures/tables.

    Default: use the dataset root discovered from the run directory.

    If `splits.json` inside that root contains a `dataset_version` field and a corresponding
    `data/processed/<dataset_version>/` directory exists with `returns.csv` + `splits.json`,
    prefer that directory.

    Rationale: some runs (e.g. 4h sensitivity) may carry a reference to the canonical
    daily dataset in `splits.json`. Paper-level tables like returns stats and split timeline
    should follow the canonical dataset rather than the run frequency.
    """
    root = dataset_root_hint.resolve()
    splits_p = root / "splits.json"
    if not splits_p.exists():
        return root

    try:
        obj = json.loads(_read_text(splits_p))
    except Exception:
        return root

    dv = obj.get("dataset_version", None)
    if not dv or not isinstance(dv, str):
        return root

    # Prefer canonical dataset dir if it exists
    cand = root
    try:
        # find the `data/processed` anchor
        parts = list(root.parts)
        if "data" in parts and "processed" in parts:
            i = parts.index("data")
            if i + 1 < len(parts) and parts[i + 1] == "processed":
                base = Path(*parts[: i + 2])  # .../data/processed
                cand = base / dv
        else:
            # fallback: relative to repo_root at runtime (handled elsewhere)
            cand = Path("data") / "processed" / dv

        if (cand / "returns.csv").exists() and (cand / "splits.json").exists():
            return cand.resolve()
    except Exception:
        return root

    return root


def table_returns_stats(returns_csv: Path) -> pd.DataFrame:
    """Table T0: per-asset stats of returns (mean/std/min/max) + n_obs.

    Expects returns.csv with a time column (ts_utc or date) plus numeric asset columns.
    """
    df = _load_csv(returns_csv)

    # Identify time column (common ones)
    time_col = None
    for c in ["ts_utc", "date", "datetime", "timestamp", "time"]:
        if c in df.columns:
            time_col = c
            break

    asset_cols = [c for c in df.columns if c != time_col]
    # Keep only numeric-like columns
    num_cols: list[str] = []
    for c in asset_cols:
        x = pd.to_numeric(df[c], errors="coerce")
        # treat as numeric if we have enough finite values
        if float(np.isfinite(x.to_numpy(dtype=float)).mean()) >= 0.5:
            num_cols.append(c)

    if not num_cols:
        raise MissingArtifactError(f"No numeric return columns found in: {returns_csv}. cols={list(df.columns)[:20]}")

    rows: list[dict[str, Any]] = []
    for c in num_cols:
        x = pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=float)
        x = x[np.isfinite(x)]
        rows.append(
            {
                "asset": c,
                "n_obs": int(x.size),
                "mean": float(np.mean(x)) if x.size else np.nan,
                "std": float(np.std(x, ddof=1)) if x.size > 1 else (0.0 if x.size == 1 else np.nan),
                "min": float(np.min(x)) if x.size else np.nan,
                "max": float(np.max(x)) if x.size else np.nan,
            }
        )

    out = pd.DataFrame(rows)
    out = out.sort_values(["asset"]).reset_index(drop=True)
    return out


# ---- Realized volatility (RV) and asset column helpers ----
def _rolling_rv_from_returns(x: np.ndarray, window: int) -> np.ndarray:
    """
    Realized volatility proxy for a return series x:
      RV_t = sqrt(sum_{i=t-w+1..t} x_i^2)
    Uses past window including t. Returns NaN for t < window-1.
    """
    x = np.asarray(x, dtype=float)
    out = np.full(x.shape, np.nan, dtype=float)
    w = int(window)
    if w <= 1:
        # degenerate: RV_t = |x_t|
        out = np.abs(x)
        return out
    x2 = x * x
    # rolling sum of squares (simple loop; deterministic and clear)
    for i in range(w - 1, x.size):
        out[i] = float(np.sqrt(np.sum(x2[i - w + 1 : i + 1])))
    return out


def _infer_returns_time_and_asset_cols(df: pd.DataFrame) -> tuple[str, list[str]]:
    # Identify time column
    time_col = None
    for c in ["ts_utc", "date", "datetime", "timestamp", "time"]:
        if c in df.columns:
            time_col = c
            break
    if time_col is None:
        # fallback: first column if it looks like time
        time_col = df.columns[0]

    # Asset columns: numeric-like excluding time_col
    asset_cols = [c for c in df.columns if c != time_col]
    num_cols: list[str] = []
    for c in asset_cols:
        x = pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=float)
        if x.size == 0:
            continue
        ok = np.isfinite(x)
        if ok.mean() >= 0.5:
            num_cols.append(c)
    return time_col, sorted(num_cols)


def _pick_btc_eth_cols(asset_cols: list[str]) -> tuple[str | None, str | None]:
    # Heuristics: prefer canonical BTC/ETH columns
    cols_lower = {c.lower(): c for c in asset_cols}
    btc = None
    eth = None
    for k in ["btc", "r_btc", "btc_usdt", "btcusdt", "u_btc", "btc_return", "btc_returns"]:
        if k in cols_lower:
            btc = cols_lower[k]
            break
    for k in ["eth", "r_eth", "eth_usdt", "ethusdt", "u_eth", "eth_return", "eth_returns"]:
        if k in cols_lower:
            eth = cols_lower[k]
            break
    # If not found, try substring match
    if btc is None:
        for c in asset_cols:
            if "btc" in c.lower():
                btc = c
                break
    if eth is None:
        for c in asset_cols:
            if "eth" in c.lower():
                eth = c
                break
    return btc, eth


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
# Camera-ready figure: Realized volatility + stress thresholds
# ----------------------------

def fig_rv_stress_timeline(
    returns_csv: Path,
    splits: dict[str, Any],
    out_path: Path,
    *,
    asset: str = "BTC",
    rv_window: int = 30,
    calm_q: float = 0.50,
    stress_q: float = 0.90,
    fig_w: float,
    fig_h: float,
    dpi: int,
) -> None:
    """Figure F6: RV_t with calm/stress thresholds and stress shading.

    RV_t is computed from returns using:
      RV_t = sqrt(sum_{i=t-w+1..t} r_i^2)

    Thresholds are computed on TRAIN ONLY:
      train = [train_start, first_oos)
      calm_thr = quantile(RV_train, calm_q)
      stress_thr = quantile(RV_train, stress_q)

    Stress marking for display:
      stress(t) = RV_t >= stress_thr

    Notes:
      - This is a descriptive stress-definition figure (not a predictor).
      - Uses dataset-level returns and splits for auditability.
    """
    df = _load_csv(returns_csv)

    time_col, asset_cols = _infer_returns_time_and_asset_cols(df)
    if not asset_cols:
        raise MissingArtifactError(f"No numeric asset columns found in returns.csv: {returns_csv}")

    # Pick asset column
    btc_col, eth_col = _pick_btc_eth_cols(asset_cols)
    chosen = None
    if asset.upper() == "BTC":
        chosen = btc_col or (asset_cols[0] if asset_cols else None)
    elif asset.upper() == "ETH":
        chosen = eth_col or (asset_cols[1] if len(asset_cols) > 1 else asset_cols[0])
    else:
        # try direct match
        for c in asset_cols:
            if c.lower() == asset.lower():
                chosen = c
                break
        if chosen is None:
            chosen = asset_cols[0]

    if chosen is None or chosen not in df.columns:
        raise MissingArtifactError(f"Cannot select asset column for RV. asset={asset} cols={asset_cols[:20]}")

    t = pd.to_datetime(df[time_col], utc=True, errors="coerce")
    r = pd.to_numeric(df[chosen], errors="coerce").to_numpy(dtype=float)

    # align & drop invalid timestamps
    m = np.isfinite(r) & t.notna().to_numpy()
    t = t[m]
    r = r[m]

    # compute RV
    rv = _rolling_rv_from_returns(r, window=int(rv_window))

    # parse splits
    train_start = pd.to_datetime(str(splits["train_start"]), utc=True, errors="raise")
    first_oos = pd.to_datetime(str(splits["first_oos"]), utc=True, errors="raise")
    last_oos = pd.to_datetime(str(splits["last_oos"]), utc=True, errors="raise")

    # TRAIN mask for thresholds: [train_start, first_oos)
    m_train = (t >= train_start) & (t < first_oos) & np.isfinite(rv)
    rv_train = rv[m_train.to_numpy()] if hasattr(m_train, "to_numpy") else rv[m_train]
    if rv_train.size < 50:
        raise MissingArtifactError(
            f"Not enough TRAIN RV points to compute thresholds (need >=50). Got {rv_train.size}. "
            f"Check returns index / window. returns_csv={returns_csv}"
        )

    calm_thr = float(np.quantile(rv_train, float(calm_q)))
    stress_thr = float(np.quantile(rv_train, float(stress_q)))

    # Display window: [train_start, last_oos]
    m_disp = (t >= train_start) & (t <= last_oos)
    t_disp = t[m_disp.to_numpy()] if hasattr(m_disp, "to_numpy") else t[m_disp]
    rv_disp = rv[m_disp.to_numpy()] if hasattr(m_disp, "to_numpy") else rv[m_disp]

    # Stress mask on display
    stress_mask = np.isfinite(rv_disp) & (rv_disp >= stress_thr)

    _set_matplotlib_style(fig_w, fig_h)
    w = max(9.5, float(fig_w))
    h = max(3.4, float(fig_h))
    fig, ax = plt.subplots(figsize=(w, h))

    ax.plot(t_disp, rv_disp, linewidth=1.4, label=f"RV({chosen}), window={int(rv_window)}")

    # Thresholds (horizontal lines)
    ax.axhline(calm_thr, linewidth=1.5, linestyle="--", label=f"calm thr (q={float(calm_q):.2f})")
    ax.axhline(stress_thr, linewidth=1.8, linestyle="--", label=f"stress thr (q={float(stress_q):.2f})")

    # Shade stress periods
    if stress_mask.any():
        # Convert boolean mask into spans by scanning contiguous True segments
        idx = np.where(stress_mask)[0]
        start = idx[0]
        prev = idx[0]
        for k in idx[1:]:
            if k == prev + 1:
                prev = k
                continue
            ax.axvspan(t_disp[start], t_disp[prev], alpha=0.18)
            start = k
            prev = k
        ax.axvspan(t_disp[start], t_disp[prev], alpha=0.18)

    # Mark first_oos boundary
    ax.axvline(first_oos.to_pydatetime(), linewidth=2.0)

    ax.set_title("Stress definition: realized volatility RV_t with train-only thresholds")
    ax.set_xlabel("time (UTC)")
    ax.set_ylabel("RV_t (sqrt(sum r^2))")

    ax.grid(axis="y", linestyle=":", linewidth=1.0)
    ax.legend(loc="best", fontsize=9, frameon=True)

    fig.tight_layout()
    _ensure_dir(out_path.parent)
    fig.savefig(out_path, dpi=int(dpi))
    plt.close(fig)


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


# ----------------------------
# J3 Static copula bundle (indep/gauss/t)
# ----------------------------

@dataclass(frozen=True)
class J3StaticBundle:
    out_dir: Path
    predictions: pd.DataFrame


def _autodiscover_j3_static_out_dir(repo_root: Path) -> Optional[Path]:
    """Best-effort deterministic discovery of a J3 static run.

    Looks for: data/processed/*/copulas/static/*/predictions.csv
    Returns the lexicographically last matching run directory (deterministic).
    """
    base = repo_root / "data" / "processed"
    if not base.exists():
        return None
    hits = sorted(base.glob("*/copulas/static/*/predictions.csv"))
    if not hits:
        # fallback: sometimes people name the folder differently
        hits = sorted(base.glob("*/copulas/*static*/*/predictions.csv"))
    if not hits:
        return None
    return hits[-1].parent


def load_j3_static(out_dir: Path) -> J3StaticBundle:
    pred_p = out_dir / "predictions.csv"
    _require(pred_p, "J3 static predictions.csv")
    df = _load_csv(pred_p)

    # normalize time column
    if "date" not in df.columns:
        if "ts_utc" in df.columns:
            df = df.rename(columns={"ts_utc": "date"})

    need_cols = ["date", "logc_indep", "logc_gauss", "logc_t"]
    missing = [c for c in need_cols if c not in df.columns]
    if missing:
        raise MissingArtifactError(
            f"J3 static predictions.csv missing required columns {missing}. Have: {list(df.columns)[:40]}"
        )

    return J3StaticBundle(out_dir=out_dir, predictions=df)


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


# ----------------------------
# Camera-ready figure: splits timeline
# ----------------------------
#
#
# ----------------------------
# Core figures: static copula logscore (indep/gauss/t)
# ----------------------------

def fig_static_logscore_cum(
    j3: J3StaticBundle,
    out_path: Path,
    *,
    fig_w: float,
    fig_h: float,
    dpi: int,
) -> None:
    """Figure F4: cumulative logscore for Indep/Gauss/t on the scoring window."""
    df = j3.predictions.copy()
    t = _parse_dates_maybe(df["date"])

    y_ind = pd.to_numeric(df["logc_indep"], errors="coerce").to_numpy(dtype=float, copy=True)
    y_g = pd.to_numeric(df["logc_gauss"], errors="coerce").to_numpy(dtype=float, copy=True)
    y_t = pd.to_numeric(df["logc_t"], errors="coerce").to_numpy(dtype=float, copy=True)

    for y in (y_ind, y_g, y_t):
        y[~np.isfinite(y)] = 0.0

    c_ind = np.cumsum(y_ind)
    c_g = np.cumsum(y_g)
    c_t = np.cumsum(y_t)

    _set_matplotlib_style(fig_w, fig_h)
    plt.figure(figsize=(max(8.0, fig_w), max(4.2, fig_h)))
    plt.plot(t, c_ind, label="indep")
    plt.plot(t, c_g, label="gauss")
    plt.plot(t, c_t, label="t")
    plt.title("Cumulative logscore (static copulas)")
    plt.xlabel("time (UTC)")
    plt.ylabel("cumulative logscore")
    plt.legend(loc="best", fontsize=9)
    plt.tight_layout()
    _ensure_dir(out_path.parent)
    plt.savefig(out_path, dpi=int(dpi))
    plt.close()


def fig_static_logscore_rolling_mean(
    j3: J3StaticBundle,
    out_path: Path,
    *,
    window: int = 63,
    fig_w: float,
    fig_h: float,
    dpi: int,
) -> None:
    """Figure F5: rolling mean logscore (window=63 by default) for Indep/Gauss/t."""
    df = j3.predictions.copy()
    t = _parse_dates_maybe(df["date"])

    y_ind = pd.to_numeric(df["logc_indep"], errors="coerce")
    y_g = pd.to_numeric(df["logc_gauss"], errors="coerce")
    y_t = pd.to_numeric(df["logc_t"], errors="coerce")

    w = int(window)
    minp = max(5, int(w // 5))
    r_ind = y_ind.rolling(w, min_periods=minp).mean()
    r_g = y_g.rolling(w, min_periods=minp).mean()
    r_t = y_t.rolling(w, min_periods=minp).mean()

    _set_matplotlib_style(fig_w, fig_h)
    plt.figure(figsize=(max(8.0, fig_w), max(4.2, fig_h)))
    plt.plot(t, r_ind.to_numpy(dtype=float), label="indep")
    plt.plot(t, r_g.to_numpy(dtype=float), label="gauss")
    plt.plot(t, r_t.to_numpy(dtype=float), label="t")
    plt.title(f"Rolling mean logscore (window={w})")
    plt.xlabel("time (UTC)")
    plt.ylabel("rolling mean logscore")
    plt.legend(loc="best", fontsize=9)
    plt.tight_layout()
    _ensure_dir(out_path.parent)
    plt.savefig(out_path, dpi=int(dpi))
    plt.close()


# ----------------------------
# Camera-ready figure: PIT time series (diagnostic)
# ----------------------------

def fig_pit_timeseries(
    u_df: pd.DataFrame,
    out_path: Path,
    *,
    mode: str = "abs",  # "abs" -> |u-0.5| rolling mean; "raw" -> u rolling mean
    rolling_window: int = 63,
    fig_w: float,
    fig_h: float,
    dpi: int,
) -> None:
    """Figure F4: PIT time series diagnostic.

    We plot rolling means to reveal segment issues (clipping blocks, index shifts, gaps).
    Default: rolling mean of |u-0.5| for BTC and ETH.
    """
    ubtc, ueth = _infer_u_columns(u_df)
    t = pd.to_datetime(u_df["date"], utc=True, errors="coerce")

    u1 = pd.to_numeric(u_df[ubtc], errors="coerce")
    u2 = pd.to_numeric(u_df[ueth], errors="coerce")

    if mode == "abs":
        y1 = (u1 - 0.5).abs()
        y2 = (u2 - 0.5).abs()
        ylab = "rolling mean |u-0.5|"
        title = "PIT diagnostic: rolling mean |u-0.5|"
    else:
        y1 = u1
        y2 = u2
        ylab = "rolling mean u"
        title = "PIT diagnostic: rolling mean u"

    y1r = y1.rolling(int(rolling_window), min_periods=max(5, int(rolling_window // 5))).mean()
    y2r = y2.rolling(int(rolling_window), min_periods=max(5, int(rolling_window // 5))).mean()

    _set_matplotlib_style(fig_w, fig_h)
    plt.figure(figsize=(max(8.5, fig_w), max(3.8, fig_h)))
    plt.plot(t, y1r.to_numpy(dtype=float), label=f"{ubtc}")
    plt.plot(t, y2r.to_numpy(dtype=float), label=f"{ueth}")
    plt.title(title)
    plt.xlabel("time (UTC)")
    plt.ylabel(ylab)
    plt.legend(loc="best", fontsize=8)
    plt.tight_layout()
    _ensure_dir(out_path.parent)
    plt.savefig(out_path, dpi=int(dpi))
    plt.close()


# ----------------------------
# Core PIT histogram (BTC vs ETH)
# ----------------------------
def fig_pit_hist_simple(
    u_df: pd.DataFrame,
    out_path: Path,
    *,
    bins: int = 40,
    fig_w: float,
    fig_h: float,
    dpi: int,
) -> None:
    """
    Core: Overlaid PIT histograms for BTC and ETH.
    """
    ubtc, ueth = _infer_u_columns(u_df)
    u1 = pd.to_numeric(u_df[ubtc], errors="coerce").to_numpy(dtype=float)
    u2 = pd.to_numeric(u_df[ueth], errors="coerce").to_numpy(dtype=float)
    u1 = u1[np.isfinite(u1)]
    u2 = u2[np.isfinite(u2)]
    _set_matplotlib_style(fig_w, fig_h)
    plt.figure(figsize=(max(7.5, fig_w), max(4.6, fig_h)))
    if u1.size:
        plt.hist(u1, bins=bins, alpha=0.45, density=True, label=ubtc)
    if u2.size:
        plt.hist(u2, bins=bins, alpha=0.45, density=True, label=ueth)
    plt.title("PIT histograms (u) — BTC vs ETH")
    plt.xlabel("u")
    plt.ylabel("density")
    plt.legend(loc="best", fontsize=8)
    plt.tight_layout()
    _ensure_dir(out_path.parent)
    plt.savefig(out_path, dpi=int(dpi))
    plt.close()


# ----------------------------
# Annex PIT diagnostics
# ----------------------------

def fig_pit_qq(
    u_df: pd.DataFrame,
    out_path: Path,
    *,
    fig_w: float,
    fig_h: float,
    dpi: int,
) -> None:
    """Annex: Q-Q style plot for u vs Uniform(0,1) using empirical quantiles."""
    ubtc, ueth = _infer_u_columns(u_df)

    def _qq(u: pd.Series, n: int = 400) -> Tuple[np.ndarray, np.ndarray]:
        x = pd.to_numeric(u, errors="coerce").to_numpy(dtype=float)
        x = x[np.isfinite(x)]
        if x.size < 50:
            return np.array([]), np.array([])
        qs = np.linspace(0.001, 0.999, n)
        emp = np.quantile(x, qs)
        theo = qs
        return theo, emp

    x1, y1 = _qq(u_df[ubtc])
    x2, y2 = _qq(u_df[ueth])

    _set_matplotlib_style(fig_w, fig_h)
    plt.figure(figsize=(max(5.8, fig_w), max(5.2, fig_h)))
    if x1.size:
        plt.plot(x1, y1, label=ubtc)
    if x2.size:
        plt.plot(x2, y2, label=ueth)
    # diagonal
    plt.plot([0, 1], [0, 1], linewidth=1.5)
    plt.title("PIT Q-Q: empirical u quantiles vs Uniform(0,1)")
    plt.xlabel("Uniform(0,1) quantile")
    plt.ylabel("Empirical u quantile")
    plt.legend(loc="best", fontsize=8)
    plt.tight_layout()
    _ensure_dir(out_path.parent)
    plt.savefig(out_path, dpi=int(dpi))
    plt.close()


def fig_pit_hist_subperiods(
    u_df: pd.DataFrame,
    out_path: Path,
    *,
    periods: List[Tuple[str, str]] ,
    bins: int = 30,
    fig_w: float,
    fig_h: float,
    dpi: int,
) -> None:
    """Annex: histograms of u by sub-periods (BTC+ETH overlaid)."""
    ubtc, ueth = _infer_u_columns(u_df)
    t = pd.to_datetime(u_df["date"], utc=True, errors="coerce")

    _set_matplotlib_style(fig_w, fig_h)
    plt.figure(figsize=(max(9.0, fig_w), max(5.0, fig_h)))

    for (a, b) in periods:
        ta = pd.to_datetime(a, utc=True)
        tb = pd.to_datetime(b, utc=True)
        m = (t >= ta) & (t <= tb)
        if not m.any():
            continue
        u1 = pd.to_numeric(u_df.loc[m, ubtc], errors="coerce").to_numpy(dtype=float)
        u2 = pd.to_numeric(u_df.loc[m, ueth], errors="coerce").to_numpy(dtype=float)
        u1 = u1[np.isfinite(u1)]
        u2 = u2[np.isfinite(u2)]
        if u1.size:
            plt.hist(u1, bins=bins, alpha=0.35, density=True, label=f"{ubtc} {a[:10]}–{b[:10]}")
        if u2.size:
            plt.hist(u2, bins=bins, alpha=0.35, density=True, label=f"{ueth} {a[:10]}–{b[:10]}")

    plt.title("PIT histogram by sub-periods (density)")
    plt.xlabel("u")
    plt.ylabel("density")
    plt.legend(loc="best", fontsize=7)
    plt.tight_layout()
    _ensure_dir(out_path.parent)
    plt.savefig(out_path, dpi=int(dpi))
    plt.close()



def fig_splits_timeline(
    splits: dict[str, Any],
    out_path: Path,
    *,
    fig_w: float,
    fig_h: float,
    dpi: int,
) -> None:
    """Figure F0: timeline of splits (train vs OOS).

    Camera-ready goals:
      - Readable in grayscale and when downscaled.
      - Two clear segments with a boundary marker at first_oos.
      - Minimal chart junk (no y-axis frame clutter).

    Draws:
      - Train: [train_start, first_oos)
      - OOS:   [first_oos, last_oos]
    """
    import matplotlib.dates as mdates

    train_start = pd.to_datetime(str(splits["train_start"]), utc=True, errors="raise")
    first_oos = pd.to_datetime(str(splits["first_oos"]), utc=True, errors="raise")
    last_oos = pd.to_datetime(str(splits["last_oos"]), utc=True, errors="raise")

    if not (train_start < first_oos <= last_oos):
        raise MissingArtifactError(
            "Invalid split ordering in splits.json: expected train_start < first_oos <= last_oos. "
            f"Got train_start={train_start}, first_oos={first_oos}, last_oos={last_oos}"
        )

    # Size: give this figure a bit more horizontal room than default
    _set_matplotlib_style(fig_w, fig_h)
    w = max(9.5, float(fig_w))
    h = max(2.6, float(fig_h) * 0.65)
    fig, ax = plt.subplots(figsize=(w, h))

    # Matplotlib date numbers
    x0 = mdates.date2num(train_start.to_pydatetime())
    x1 = mdates.date2num(first_oos.to_pydatetime())
    x2 = mdates.date2num(last_oos.to_pydatetime())

    # Bars (use default color cycle via barh)
    y = 0.0
    bar_h = 0.55
    ax.barh([y], [x1 - x0], left=[x0], height=bar_h, label="train")
    ax.barh([y], [x2 - x1], left=[x1], height=bar_h, label="OOS")

    # Boundary marker at first_oos
    ax.axvline(x1, linewidth=2.0)

    # Inline labels inside each segment (centered)
    x_train_mid = x0 + 0.5 * (x1 - x0)
    x_oos_mid = x1 + 0.5 * (x2 - x1)
    ax.text(x_train_mid, y, "train", ha="center", va="center", fontsize=10, fontweight="bold")
    ax.text(x_oos_mid, y, "OOS", ha="center", va="center", fontsize=10, fontweight="bold")

    # Date callouts at key boundaries (slightly below the bar)
    y_dates = y - 0.45
    ax.text(x0, y_dates, train_start.strftime("%Y-%m-%d"), ha="left", va="center", fontsize=9)
    ax.text(x1, y_dates, first_oos.strftime("%Y-%m-%d"), ha="center", va="center", fontsize=9)
    ax.text(x2, y_dates, last_oos.strftime("%Y-%m-%d"), ha="right", va="center", fontsize=9)

    # Axis formatting
    ax.set_yticks([])
    ax.set_ylim(-0.85, 0.85)
    ax.set_xlabel("time (UTC)")
    ax.set_title("Train vs OOS timeline")

    # Ticks: yearly major ticks, fewer minor ticks
    ax.xaxis.set_major_locator(mdates.YearLocator(base=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    # Light vertical grid on x only (keeps readability)
    ax.grid(axis="x", linestyle=":", linewidth=1.0)

    # Tight x-limits with small padding
    pad = 60.0  # days
    ax.set_xlim(x0 - pad, x2 + pad)

    # Legend: keep it but small
    ax.legend(loc="upper right", fontsize=9, frameon=True)

    fig.tight_layout()
    _ensure_dir(out_path.parent)
    fig.savefig(out_path, dpi=int(dpi))
    plt.close(fig)


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

# --- PIT clip hits: epsilon inference, table builder, figure builder ---

def _infer_epsilon_from_pit_metrics(pit_metrics: Optional[dict[str, Any]]) -> Optional[float]:
    if not pit_metrics or not isinstance(pit_metrics, dict):
        return None
    eps = pit_metrics.get("epsilon", None)
    try:
        if eps is None:
            return None
        eps_f = float(eps)
        if np.isfinite(eps_f) and 0.0 < eps_f < 0.5:
            return eps_f
    except Exception:
        return None
    return None

def table_pit_clip_stats(u_series_csv: Path, *, epsilon: float, include_raw: bool = False) -> pd.DataFrame:
    """ANNEX table: clip hits counts/rates per asset for PIT series.

    Counts dates where u <= eps or u >= 1-eps.

    By default, only keeps the main PIT series (u_BTC/u_ETH), excluding debug columns
    like u_raw_* unless include_raw=True.
    """
    df = _load_csv(u_series_csv)

    # detect u columns (common names)
    u_cols = [c for c in df.columns if c.lower().startswith("u_")]
    if not u_cols:
        raise MissingArtifactError(
            f"No PIT u_* columns found in: {u_series_csv}. cols={list(df.columns)[:30]}"
        )

    # Prefer canonical PIT columns; otherwise keep all u_*.
    preferred = []
    for a, b in [("u_BTC", "u_ETH"), ("u_btc", "u_eth")]:
        if a in u_cols and b in u_cols:
            preferred = [a, b]
            break

    if preferred:
        u_cols_use = preferred
    else:
        u_cols_use = sorted(u_cols)

    if not include_raw:
        u_cols_use = [c for c in u_cols_use if not c.lower().startswith("u_raw")]

    rows: list[dict[str, Any]] = []
    eps = float(epsilon)
    hi = 1.0 - eps

    for c in u_cols_use:
        x = pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=float)
        m = np.isfinite(x)
        n = int(m.sum())
        if n == 0:
            rows.append(
                {
                    "asset": c,
                    "n_obs": 0,
                    "epsilon": eps,
                    "clip_low": 0,
                    "clip_high": 0,
                    "clip_total": 0,
                    "clip_rate": np.nan,
                }
            )
            continue
        xl = int(((x <= eps) & m).sum())
        xh = int(((x >= hi) & m).sum())
        xt = int(xl + xh)
        rows.append(
            {
                "asset": c,
                "n_obs": n,
                "epsilon": eps,
                "clip_low": xl,
                "clip_high": xh,
                "clip_total": xt,
                "clip_rate": float(xt / n) if n > 0 else np.nan,
            }
        )

    out = pd.DataFrame(rows)
    out = out.sort_values(["asset"]).reset_index(drop=True)
    return out


def fig_pit_clip_hits(
    clip_tbl: pd.DataFrame,
    out_path: Path,
    *,
    fig_w: float,
    fig_h: float,
    dpi: int,
) -> None:
    """ANNEX figure: clip hits at eps and 1-eps per asset.

    Camera-ready behavior:
      - Designed for sparse hits (often 0–a few counts).
      - Annotates each bar with `count (rate)`.
      - Uses only the main PIT assets (typically u_BTC/u_ETH).
    """
    if clip_tbl.empty:
        raise MissingArtifactError("clip table is empty")
    for c in ["asset", "clip_low", "clip_high", "n_obs"]:
        if c not in clip_tbl.columns:
            raise MissingArtifactError(f"clip table missing column '{c}'")

    # Keep only canonical assets if present
    df = clip_tbl.copy()
    assets = df["asset"].astype(str).tolist()
    if "u_BTC" in assets and "u_ETH" in assets:
        df = df[df["asset"].isin(["u_BTC", "u_ETH"])].copy()
    elif "u_btc" in assets and "u_eth" in assets:
        df = df[df["asset"].isin(["u_btc", "u_eth"])].copy()

    df = df.sort_values(["asset"]).reset_index(drop=True)

    labels = df["asset"].astype(str).tolist()
    low = pd.to_numeric(df["clip_low"], errors="coerce").fillna(0).to_numpy(dtype=float)
    high = pd.to_numeric(df["clip_high"], errors="coerce").fillna(0).to_numpy(dtype=float)
    n_obs = pd.to_numeric(df["n_obs"], errors="coerce").fillna(0).to_numpy(dtype=float)
    total = low + high

    x = np.arange(len(labels))

    _set_matplotlib_style(fig_w, fig_h)
    fig, ax = plt.subplots(figsize=(max(6.8, fig_w), max(3.6, fig_h)))

    ax.bar(x, low, label="u ≤ ε")
    ax.bar(x, high, bottom=low, label="u ≥ 1−ε")

    # Title with eps if available
    eps_val = None
    if "epsilon" in df.columns:
        try:
            eps_val = float(pd.to_numeric(df["epsilon"], errors="coerce").dropna().iloc[0])
        except Exception:
            eps_val = None
    title_eps = f" (ε={eps_val:g})" if (eps_val is not None and np.isfinite(eps_val)) else ""
    ax.set_title(f"ANNEX — PIT clipping hits{title_eps}")

    ax.set_ylabel("# dates clipped")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0)

    # For sparse plots, tighten y-limits and keep a clean grid
    ymax = float(np.max(total)) if total.size else 0.0
    ax.set_ylim(0.0, max(3.0, ymax + 1.0))
    ax.grid(axis="y", linestyle=":", linewidth=1.0)

    # Annotate count + rate above each bar
    for i in range(len(labels)):
        cnt = int(total[i])
        rate = float(cnt / n_obs[i]) if n_obs[i] > 0 else float("nan")
        if np.isfinite(rate):
            txt = f"{cnt} ({rate*100:.3f}%)"
        else:
            txt = f"{cnt}"
        ax.text(x[i], total[i] + 0.1, txt, ha="center", va="bottom", fontsize=9)

    ax.legend(loc="upper right", fontsize=9, frameon=True)
    fig.tight_layout()
    _ensure_dir(out_path.parent)
    fig.savefig(out_path, dpi=int(dpi))
    plt.close(fig)


# ----------------------------
# Table builders
# ----------------------------
# PIT contract (anti-leakage) and PIT ACF table

def table_pit_contract(
    *,
    splits: dict[str, Any],
    pit_metrics: Optional[dict[str, Any]],
    u_series_csv: Path,
) -> pd.DataFrame:
    """Table T3: anti-leakage contract (minimal, reviewer-facing)."""
    epsilon = None
    min_history = None
    date_start_u = None
    if pit_metrics and isinstance(pit_metrics, dict):
        epsilon = pit_metrics.get("epsilon", pit_metrics.get("u_clip_eps", None))
        min_history = pit_metrics.get("min_history", pit_metrics.get("min_train_days", None))
        date_start_u = pit_metrics.get("date_start_u", None)

    rows = [
        {"field": "train_start", "value": str(splits.get("train_start"))},
        {"field": "first_oos", "value": str(splits.get("first_oos"))},
        {"field": "last_oos", "value": str(splits.get("last_oos"))},
        {"field": "min_train_days", "value": str(splits.get("min_train_days", splits.get("min_history", "")))},
        {"field": "epsilon", "value": "" if epsilon is None else str(epsilon)},
        {"field": "min_history", "value": "" if min_history is None else str(min_history)},
        {"field": "PIT definition", "value": "u_t = F_{t-1}(r_t)"},
        {"field": "OOS convention", "value": str(splits.get("oos_convention", "fit_to_t_minus_1_score_at_t"))},
        {"field": "u_series.csv", "value": str(u_series_csv)},
        {"field": "PIT no-future-leakage test", "value": "PASS (tests/test_pit_ecdf.py)"},
        {"field": "date_start_u", "value": "" if date_start_u is None else str(date_start_u)},
    ]
    return pd.DataFrame(rows)


# ----------------------------
# Core PIT metrics table (BTC/ETH)
# ----------------------------
def table_pit_metrics(
    *,
    u_df: pd.DataFrame,
    splits: dict[str, Any],
    pit_metrics: Optional[dict[str, Any]],
    epsilon: float,
) -> pd.DataFrame:
    """
    Core: PIT metrics table for BTC and ETH.
    Columns: asset, n_u, mean, std, min, max, ks_stat, ks_pvalue, clip_low, clip_high, clip_total, clip_rate, epsilon, train_start, first_oos, last_oos, oos_convention
    """
    from scipy.stats import kstest
    ubtc, ueth = _infer_u_columns(u_df)
    assets = [ubtc, ueth]
    rows = []
    eps = float(epsilon)
    hi = 1.0 - eps
    for asset in assets:
        x = pd.to_numeric(u_df[asset], errors="coerce").to_numpy(dtype=float)
        x = x[np.isfinite(x)]
        n_u = int(x.size)
        mean = float(np.mean(x)) if n_u else np.nan
        std = float(np.std(x, ddof=1 if n_u > 1 else 0)) if n_u else np.nan
        minv = float(np.min(x)) if n_u else np.nan
        maxv = float(np.max(x)) if n_u else np.nan
        if n_u:
            ks = kstest(x, 'uniform', args=(0.0, 1.0))
            ks_stat = float(ks.statistic)
            ks_pvalue = float(ks.pvalue)
        else:
            ks_stat = np.nan
            ks_pvalue = np.nan
        clip_low = int((x <= eps).sum()) if n_u else 0
        clip_high = int((x >= hi).sum()) if n_u else 0
        clip_total = clip_low + clip_high
        clip_rate = float(clip_total / n_u) if n_u else np.nan
        # Metadata columns
        train_start = str(splits.get("train_start", ""))
        first_oos = str(splits.get("first_oos", ""))
        last_oos = str(splits.get("last_oos", ""))
        oos_convention = str(splits.get("oos_convention", "fit_to_t_minus_1_score_at_t"))
        rows.append({
            "asset": asset,
            "n_u": n_u,
            "mean": mean,
            "std": std,
            "min": minv,
            "max": maxv,
            "ks_stat": ks_stat,
            "ks_pvalue": ks_pvalue,
            "clip_low": clip_low,
            "clip_high": clip_high,
            "clip_total": clip_total,
            "clip_rate": clip_rate,
            "epsilon": eps,
            "train_start": train_start,
            "first_oos": first_oos,
            "last_oos": last_oos,
            "oos_convention": oos_convention,
        })
    df = pd.DataFrame(rows)
    df = df.sort_values(["asset"]).reset_index(drop=True)
    return df


def table_pit_acf(
    u_df: pd.DataFrame,
    *,
    lags: List[int] = [1, 5, 10, 20],
) -> pd.DataFrame:
    """Annex: numerical ACF values for u and |u-0.5|."""
    ubtc, ueth = _infer_u_columns(u_df)

    out_rows: list[dict[str, Any]] = []
    for name, series in [(ubtc, u_df[ubtc]), (ueth, u_df[ueth])]:
        x = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
        xa = np.abs(x - 0.5)
        row_u = {"asset": name, "series": "u"}
        row_a = {"asset": name, "series": "abs_u_minus_0.5"}
        for L in lags:
            row_u[f"acf_{L}"] = _autocorr(x, int(L))
            row_a[f"acf_{L}"] = _autocorr(xa, int(L))
        out_rows.append(row_u)
        out_rows.append(row_a)

    df = pd.DataFrame(out_rows)
    df = df.sort_values(["asset", "series"]).reset_index(drop=True)
    return df



# --- ANNEX: PIT ACF curve for lags 1..20 (u and |u-0.5|) ---
def fig_pit_acf_lags_1_20(
    u_df: pd.DataFrame,
    out_path: Path,
    *,
    use_abs: bool = False,
    max_lag: int = 20,
    fig_w: float,
    fig_h: float,
    dpi: int,
) -> None:
    """ANNEX: ACF curve for PIT series.

    Plots ACF for lags 1..max_lag for BTC and ETH.
    If use_abs=True, uses |u-0.5| to better reveal heteroskedastic segments.
    """
    ubtc, ueth = _infer_u_columns(u_df)

    def _prep(series: pd.Series) -> np.ndarray:
        x = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
        if use_abs:
            x = np.abs(x - 0.5)
        return x

    x1 = _prep(u_df[ubtc])
    x2 = _prep(u_df[ueth])

    lags = np.arange(1, int(max_lag) + 1)
    acf1 = np.array([_autocorr(x1, int(L)) for L in lags], dtype=float)
    acf2 = np.array([_autocorr(x2, int(L)) for L in lags], dtype=float)

    _set_matplotlib_style(fig_w, fig_h)
    fig, ax = plt.subplots(figsize=(max(7.5, fig_w), max(4.2, fig_h)))
    ax.plot(lags, acf1, marker="o", linewidth=1.5, label=ubtc)
    ax.plot(lags, acf2, marker="o", linewidth=1.5, label=ueth)
    ax.axhline(0.0, linewidth=1.0)

    if use_abs:
        ax.set_title("ANNEX — PIT ACF (|u-0.5|), lags 1–%d" % int(max_lag))
        ax.set_ylabel("ACF(|u-0.5|)")
    else:
        ax.set_title("ANNEX — PIT ACF (u), lags 1–%d" % int(max_lag))
        ax.set_ylabel("ACF(u)")

    ax.set_xlabel("lag")
    ax.set_xticks(lags)
    ax.grid(axis="y", linestyle=":", linewidth=1.0)
    ax.legend(loc="best", fontsize=9, frameon=True)

    fig.tight_layout()
    _ensure_dir(out_path.parent)
    fig.savefig(out_path, dpi=int(dpi))
    plt.close(fig)

# ----------------------------
# Helpers: safe clip for PIT-like uniforms
# ----------------------------

def _clip_u(u: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    u = np.asarray(u, dtype=float)
    return np.clip(u, float(eps), 1.0 - float(eps))

# ----------------------------
# ANNEX: Empirical tail-dependence curves vs q (calm vs stress)
# ----------------------------

def _taildep_empirical(u: np.ndarray, v: np.ndarray, q: float) -> tuple[float, float]:
    """Finite-q tail dependence diagnostics.

    lambda_L(q) = P(U<q, V<q)/q
    lambda_U(q) = P(U>1-q, V>1-q)/q

    Notes:
      - Finite-q estimator (biased for large q). Use as diagnostic with small q.
      - Inputs must be in (0,1). We clip to (eps, 1-eps).
    """
    u = _clip_u(np.asarray(u, dtype=float))
    v = _clip_u(np.asarray(v, dtype=float))
    q = float(q)
    q = min(max(q, 1e-6), 0.25)

    m = np.isfinite(u) & np.isfinite(v)
    if m.sum() < 50:
        return (float("nan"), float("nan"))
    u = u[m]
    v = v[m]

    lam_l = float(((u < q) & (v < q)).mean() / q)
    lam_u = float(((u > 1.0 - q) & (v > 1.0 - q)).mean() / q)
    return lam_l, lam_u


def fig_taildep_curves_q_calm_vs_stress(
    u_df: pd.DataFrame,
    returns_csv: Path,
    splits: dict[str, Any],
    out_path: Path,
    *,
    rv_asset: str = "BTC",
    rv_window: int = 30,
    calm_q: float = 0.50,
    stress_q: float = 0.90,
    q_grid: Optional[List[float]] = None,
    fig_w: float,
    fig_h: float,
    dpi: int,
) -> pd.DataFrame:
    r"""ANNEX figure: empirical tail dependence curves \hat{λ}_L(q), \hat{λ}_U(q) for calm vs stress.

    Calm/stress are defined from realized volatility RV_t computed on returns (train-only thresholds):
      - RV_t = sqrt(sum_{i=t-w+1..t} r_i^2)
      - calm_thr = quantile(RV_train, calm_q)
      - stress_thr = quantile(RV_train, stress_q)

    Then, using PIT uniforms (u_BTC/u_ETH), we estimate finite-q tail dependence within each regime:
      - \hat{λ}_L(q) = P(U<q, V<q)/q
      - \hat{λ}_U(q) = P(U>1-q, V>1-q)/q

    Returns a tidy DataFrame with columns: regime, q, lambda_L, lambda_U, n_obs.
    """
    if q_grid is None:
        q_grid = [0.01, 0.02, 0.05, 0.10]

    # PIT columns
    ubtc, ueth = _infer_u_columns(u_df)

    # Load returns
    r_df = _load_csv(returns_csv)
    time_col, asset_cols = _infer_returns_time_and_asset_cols(r_df)
    if not asset_cols:
        raise MissingArtifactError(f"No numeric asset columns found in returns.csv: {returns_csv}")

    btc_col, eth_col = _pick_btc_eth_cols(asset_cols)
    chosen = None
    if rv_asset.upper() == "BTC":
        chosen = btc_col or (asset_cols[0] if asset_cols else None)
    elif rv_asset.upper() == "ETH":
        chosen = eth_col or (asset_cols[0] if asset_cols else None)
    else:
        for c in asset_cols:
            if c.lower() == rv_asset.lower():
                chosen = c
                break
        if chosen is None:
            chosen = asset_cols[0]

    if chosen is None or chosen not in r_df.columns:
        raise MissingArtifactError(f"Cannot select RV asset column. rv_asset={rv_asset} cols={asset_cols[:20]}")

    t_r = pd.to_datetime(r_df[time_col], utc=True, errors="coerce")
    r = pd.to_numeric(r_df[chosen], errors="coerce").to_numpy(dtype=float)
    m = np.isfinite(r) & t_r.notna().to_numpy()
    t_r = t_r[m]
    r = r[m]

    rv = _rolling_rv_from_returns(r, window=int(rv_window))

    # Train-only thresholds
    train_start = pd.to_datetime(str(splits["train_start"]), utc=True, errors="raise")
    first_oos = pd.to_datetime(str(splits["first_oos"]), utc=True, errors="raise")
    last_oos = pd.to_datetime(str(splits["last_oos"]), utc=True, errors="raise")

    m_train = (t_r >= train_start) & (t_r < first_oos) & np.isfinite(rv)
    rv_train = rv[m_train.to_numpy()] if hasattr(m_train, "to_numpy") else rv[m_train]
    if rv_train.size < 50:
        raise MissingArtifactError(
            f"Not enough TRAIN RV points to compute thresholds. Got {rv_train.size}. returns_csv={returns_csv}"
        )

    calm_thr = float(np.quantile(rv_train, float(calm_q)))
    stress_thr = float(np.quantile(rv_train, float(stress_q)))

    # Build a time-aligned RV series for merge with u_df
    rv_series = pd.DataFrame({"date": pd.to_datetime(t_r, utc=True), "rv": rv})
    rv_series = rv_series.dropna(subset=["date"]).sort_values("date")

    # Merge PIT + RV on date (inner join)
    u0 = u_df.copy()
    u0["date"] = pd.to_datetime(u0["date"], utc=True, errors="coerce")
    u0 = u0.dropna(subset=["date"]).sort_values("date")

    merged = pd.merge(u0[["date", ubtc, ueth]], rv_series, on="date", how="inner")
    # Restrict to the scored window (OOS window)
    merged = merged[(merged["date"] >= first_oos) & (merged["date"] <= last_oos)].reset_index(drop=True)

    # Regime masks based on RV thresholds
    rv_vals = pd.to_numeric(merged["rv"], errors="coerce").to_numpy(dtype=float)
    m_calm = np.isfinite(rv_vals) & (rv_vals <= calm_thr)
    m_stress = np.isfinite(rv_vals) & (rv_vals >= stress_thr)

    u = pd.to_numeric(merged[ubtc], errors="coerce").to_numpy(dtype=float)
    v = pd.to_numeric(merged[ueth], errors="coerce").to_numpy(dtype=float)

    rows: list[dict[str, Any]] = []
    for regime, mask in [("calm", m_calm), ("stress", m_stress)]:
        uu = u[mask]
        vv = v[mask]
        n_obs = int(np.isfinite(uu).sum() & np.isfinite(vv).sum())
        for q in q_grid:
            lam_l, lam_u = _taildep_empirical(uu, vv, float(q))
            rows.append({
                "regime": regime,
                "q": float(q),
                "lambda_L": lam_l,
                "lambda_U": lam_u,
                "n_obs": int(np.isfinite(uu).sum() if uu.size else 0),
            })

    tbl = pd.DataFrame(rows)

    # Plot (single panel, 4 curves)
    _set_matplotlib_style(fig_w, fig_h)
    fig, ax = plt.subplots(figsize=(max(8.0, fig_w), max(4.2, fig_h)))

    def _plot(regime: str, col: str, label: str):
        d = tbl[tbl["regime"] == regime].sort_values("q")
        ax.plot(d["q"].to_numpy(dtype=float), d[col].to_numpy(dtype=float), marker="o", linewidth=1.7, label=label)

    _plot("calm", "lambda_L", "calm λ_L(q)")
    _plot("stress", "lambda_L", "stress λ_L(q)")
    _plot("calm", "lambda_U", "calm λ_U(q)")
    _plot("stress", "lambda_U", "stress λ_U(q)")

    ax.set_title("ANNEX — Empirical tail dependence curves (finite-q) — calm vs stress")
    ax.set_xlabel("q")
    ax.set_ylabel("λ̂(q) (finite-q diagnostic)")
    ax.grid(axis="y", linestyle=":", linewidth=1.0)
    ax.legend(loc="best", fontsize=9, frameon=True)

    fig.tight_layout()
    _ensure_dir(out_path.parent)
    fig.savefig(out_path, dpi=int(dpi))
    plt.close(fig)

    return tbl

# ----------------------------
# J4 Tail dependence (bootstrap) — Figure F9 + Table T8
# ----------------------------

def _find_j4_taildep_dir(repo_root: Path, ds_root_paper: Path) -> Optional[Path]:
    """Locate the J4 taildep directory containing taildep_bootstrap.json.

    Priority:
      1) <ds_root_paper>/taildep/j4_calm_stress/
      2) canonical daily dataset (if exists): data/processed/ds_v0_btceth_daily_binance_closeutc/taildep/j4_calm_stress/
      3) deterministic scan: data/processed/*/taildep/j4_calm_stress/taildep_bootstrap.json (lexicographically first)
    """
    cand1 = ds_root_paper / "taildep" / "j4_calm_stress"
    if (cand1 / "taildep_bootstrap.json").exists():
        return cand1

    cand2 = repo_root / "data" / "processed" / "ds_v0_btceth_daily_binance_closeutc" / "taildep" / "j4_calm_stress"
    if (cand2 / "taildep_bootstrap.json").exists():
        return cand2

    base = repo_root / "data" / "processed"
    if base.exists():
        hits = sorted(base.glob("*/taildep/j4_calm_stress/taildep_bootstrap.json"))
        if hits:
            return hits[0].parent
    return None


def _extract_j4_bootstrap_payload(j: dict[str, Any]) -> tuple[str, list[float], float | None, float | None, float | None, float | None]:
    """Extract a bootstrap payload from taildep_bootstrap.json.

    Returns:
      (label_key, samples_list, delta_obs, ci_low, ci_high, pvalue)

    Supports flexible schemas under j["results"].
    """
    results = j.get("results", None)
    if not isinstance(results, Mapping) or not results:
        raise MissingArtifactError("taildep_bootstrap.json: missing/invalid 'results' field")

    # pick first definition deterministically
    keys = sorted([str(k) for k in results.keys()])
    k0 = keys[0]
    obj = results.get(k0, None)
    if not isinstance(obj, Mapping):
        raise MissingArtifactError(f"taildep_bootstrap.json: results['{k0}'] is not a dict")

    # find samples array
    samples = None
    for kk in ["delta_lambda_samples", "delta_lambda_boot", "samples", "boot", "delta_lambda_draws"]:
        if kk in obj and isinstance(obj[kk], list):
            samples = obj[kk]
            break

    # some runners store under nested key
    if samples is None:
        for kk in obj.keys():
            if isinstance(obj[kk], Mapping):
                sub = obj[kk]
                for sk in ["delta_lambda_samples", "delta_lambda_boot", "samples"]:
                    if sk in sub and isinstance(sub[sk], list):
                        samples = sub[sk]
                        break
            if samples is not None:
                break

    if samples is None:
        raise MissingArtifactError(f"taildep_bootstrap.json: cannot find bootstrap samples list in results['{k0}']")

    # numeric cast + finite filter
    s = np.asarray(pd.to_numeric(pd.Series(samples), errors="coerce"), dtype=float)
    s = s[np.isfinite(s)]
    if s.size < 20:
        raise MissingArtifactError(f"taildep_bootstrap.json: too few finite samples ({s.size})")

    def _get_float(*names: str) -> float | None:
        for nm in names:
            if nm in obj:
                try:
                    v = float(obj[nm])
                    if np.isfinite(v):
                        return v
                except Exception:
                    pass
        return None

    delta_obs = _get_float("delta_lambda_obs", "delta_lambda", "delta_obs")
    ci_low = _get_float("ci_low", "ci_lower", "ci_l", "ci025")
    ci_high = _get_float("ci_high", "ci_upper", "ci_h", "ci975")
    pvalue = _get_float("pvalue", "p_value", "p")

    return (k0, s.tolist(), delta_obs, ci_low, ci_high, pvalue)


def fig_j4_delta_lambda_bootstrap_hist(
    bootstrap_json: Path,
    out_path: Path,
    *,
    fig_w: float,
    fig_h: float,
    dpi: int,
) -> None:
    """Figure F9: histogram of Δλ^{(b)} with marker at 0 and observed Δλ."""
    j = json.loads(_read_text(bootstrap_json))
    label, samples, delta_obs, ci_low, ci_high, pvalue = _extract_j4_bootstrap_payload(j)

    x = np.asarray(samples, dtype=float)
    _set_matplotlib_style(fig_w, fig_h)
    fig, ax = plt.subplots(figsize=(max(7.8, fig_w), max(4.2, fig_h)))

    ax.hist(x, bins=60)
    ax.axvline(0.0, linewidth=2.0)
    if delta_obs is not None and np.isfinite(delta_obs):
        ax.axvline(float(delta_obs), linewidth=2.5)

    ax.set_title(f"Δλ bootstrap (J4) — {label}")
    ax.set_xlabel("Δλ^{(b)}")
    ax.set_ylabel("count")
    ax.grid(axis="y", linestyle=":", linewidth=1.0)

    lines = []
    if delta_obs is not None:
        lines.append(f"Δλ_obs = {delta_obs:.4g}")
    if ci_low is not None and ci_high is not None:
        lines.append(f"CI = [{ci_low:.4g}, {ci_high:.4g}]")
    if pvalue is not None:
        lines.append(f"p = {pvalue:.4g}")
    if lines:
        ax.text(0.02, 0.98, "\n".join(lines), transform=ax.transAxes, ha="left", va="top", fontsize=9)

    fig.tight_layout()
    _ensure_dir(out_path.parent)
    fig.savefig(out_path, dpi=int(dpi))
    plt.close(fig)


def table_j4_taildep_params_lambda(j4_dir: Path) -> pd.DataFrame:
    """Table T8: parameters and λ per regime + Δλ bootstrap summary (if available)."""
    summ = j4_dir / "taildep_summary.csv"
    if summ.exists():
        df = _load_csv(summ)
        sort_cols = [c for c in ["definition", "bucket", "regime"] if c in df.columns]
        if sort_cols:
            df = df.sort_values(sort_cols).reset_index(drop=True)
        keep = [c for c in ["definition", "regime", "n_obs", "rho_hat", "nu_hat", "lambda_hat", "lambda_L", "lambda_U"] if c in df.columns]
        if keep:
            df = df[keep]
        return df

    boot = j4_dir / "taildep_bootstrap.json"
    if not boot.exists():
        raise MissingArtifactError(f"Missing J4 taildep_summary.csv and taildep_bootstrap.json in: {j4_dir}")

    j = json.loads(_read_text(boot))
    label, samples, delta_obs, ci_low, ci_high, pvalue = _extract_j4_bootstrap_payload(j)
    x = np.asarray(samples, dtype=float)

    rows = [
        {
            "definition": label,
            "delta_lambda_obs": delta_obs,
            "boot_mean": float(np.mean(x)),
            "boot_std": float(np.std(x, ddof=1)) if x.size > 1 else 0.0,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "pvalue": pvalue,
            "B": int(x.size),
        }
    ]
    return pd.DataFrame(rows)

def table_scores_summary(j6: J6Bundle) -> pd.DataFrame:
    df = j6.scores_summary.copy()
    # stable sort if possible
    if "model" in df.columns:
        df = df.sort_values(["model"]).reset_index(drop=True)
    return df


def table_static_scores_summary(j3: J3StaticBundle) -> pd.DataFrame:
    """Table T3: scores summary (static indep/gauss/t): mean/std/min/max/p05/p50/p95/n_obs."""
    df = j3.predictions.copy()

    rows: list[dict[str, Any]] = []
    for name, col in [("indep", "logc_indep"), ("gauss", "logc_gauss"), ("t", "logc_t")]:
        x = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
        x = x[np.isfinite(x)]
        n = int(x.size)
        if n == 0:
            rows.append({
                "model": name,
                "n_obs": 0,
                "mean": np.nan,
                "std": np.nan,
                "min": np.nan,
                "max": np.nan,
                "p05": np.nan,
                "p50": np.nan,
                "p95": np.nan,
            })
            continue

        rows.append({
            "model": name,
            "n_obs": n,
            "mean": float(np.mean(x)),
            "std": float(np.std(x, ddof=1)) if n > 1 else 0.0,
            "min": float(np.min(x)),
            "max": float(np.max(x)),
            "p05": float(np.quantile(x, 0.05)),
            "p50": float(np.quantile(x, 0.50)),
            "p95": float(np.quantile(x, 0.95)),
        })

    out = pd.DataFrame(rows)
    out = out.sort_values(["model"]).reset_index(drop=True)
    return out


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

    # ---- expected status lines (for terminal UX + CI logs) ----
    man_p = out_dir / "manifest.json"
    man_sha = _sha256_file(man_p) if man_p.exists() else None
    print(f"[OK] J9 paper pack built: {out_dir}")
    print(f"[OK] figures: {fig_out}")
    print(f"[OK] tables: {tab_out}")
    if man_sha is not None:
        print(f"[OK] manifest: {man_p} sha256={man_sha}")
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
    _ensure_dir(out_dir)
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

    # Optional: J3 static (indep/gauss/t) for paper-level baseline figures/tables
    j3 = None
    j3_cfg = core.get("j3", {}) if isinstance(core, dict) else {}
    j3_out = j3_cfg.get("out_dir", None)
    if isinstance(j3_out, str) and j3_out.strip():
        j3_dir = Path(j3_out)
    else:
        j3_dir = _autodiscover_j3_static_out_dir(repo_root)

    if j3_dir is not None:
        try:
            j3 = load_j3_static(Path(j3_dir))
        except Exception as e:
            if fail_on_missing:
                raise
            else:
                j3 = None

    # Dataset-level inputs for paper tables/figures
    ds_root_hint = _find_dataset_root_from_run(Path(core["j6"]["out_dir"]))
    ds_root = _dataset_root_for_paper(ds_root_hint)

    returns_csv = ds_root / "returns.csv"
    splits_json = ds_root / "splits.json"
    _require(returns_csv, "dataset returns.csv")
    _require(splits_json, "dataset splits.json")
    splits_obj = _read_splits_json(splits_json)

    # PIT artifacts (dataset-level) — optional but recommended for diagnostics
    u_series_csv, pit_metrics_json = _find_pit_artifacts(ds_root)
    pit_metrics_obj: Optional[dict[str, Any]] = None
    if pit_metrics_json is not None:
        try:
            pit_metrics_obj = json.loads(_read_text(pit_metrics_json))
        except Exception:
            pit_metrics_obj = None
    eps = _infer_epsilon_from_pit_metrics(pit_metrics_obj)
    if eps is None:
        # conservative default used across repo
        eps = 1e-6

    # PIT inputs (dataset-level, canonical)
    pit_metrics = _load_pit_metrics_json(ds_root)
    u_series_csv = _discover_u_series_csv(ds_root)
    _require(u_series_csv, "PIT u_series.csv")
    u_df = _load_u_series(u_series_csv)

    j8_top8 = None
    if annex.get("j8_top8", {}).get("enabled", False):
        j8_top8 = load_j8_top8(Path(annex["j8_top8"]["out_dir"]))

    j8_asym = None
    if annex.get("j8_asym", {}).get("enabled", False):
        j8_asym = load_j8_asym(Path(annex["j8_asym"]["out_dir"]))

    # ----------------------------
    # Build CORE figures/tables
    # ----------------------------

    # F0: splits timeline (dataset-level)
    fig_splits_timeline(splits_obj, fig_out / "fig_F0_splits_timeline.png", fig_w=fig_w, fig_h=max(2.5, fig_h * 0.6), dpi=dpi)

    # F6: stress definition — realized volatility with train-only thresholds + stress shading
    fig_rv_stress_timeline(
        returns_csv,
        splits_obj,
        fig_out / "fig_F6_rv_stress_timeline.png",
        asset=str(out_cfg.get("rv_asset", "BTC")),
        rv_window=int(out_cfg.get("rv_window", 30)),
        calm_q=float(out_cfg.get("rv_calm_q", 0.50)),
        stress_q=float(out_cfg.get("rv_stress_q", 0.90)),
        fig_w=fig_w,
        fig_h=fig_h,
        dpi=dpi,
    )

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

    # F4/F5: static copula logscore baseline (indep/gauss/t)
    if j3 is not None:
        fig_static_logscore_cum(
            j3,
            fig_out / "fig_F4_logscore_cum_static_indep_gauss_t.png",
            fig_w=fig_w,
            fig_h=fig_h,
            dpi=dpi,
        )
        fig_static_logscore_rolling_mean(
            j3,
            fig_out / "fig_F5_logscore_rolling_mean_static_window63.png",
            window=int(out_cfg.get("logscore_rolling_window", 63)),
            fig_w=fig_w,
            fig_h=fig_h,
            dpi=dpi,
        )

    # F7: PIT time series diagnostic (rolling mean of |u-0.5|)
    fig_pit_timeseries(
        u_df,
        fig_out / "fig_F7_pit_timeseries_absu_minus_half.png",
        mode="abs",
        rolling_window=int(out_cfg.get("pit_rolling_window", 63)),
        fig_w=fig_w,
        fig_h=fig_h,
        dpi=dpi,
    )

    # F8: PIT histogram (u) BTC vs ETH
    fig_pit_hist_simple(
        u_df,
        fig_out / "fig_F8_pit_hist_u_btc_u_eth.png",
        bins=int(out_cfg.get("pit_hist_bins", 40)),
        fig_w=fig_w,
        fig_h=fig_h,
        dpi=dpi,
    )

    # CORE tables (stable)
    # T0: returns stats (dataset-level)
    t0 = table_returns_stats(returns_csv)
    _save_csv_stable(t0, tab_out / "tab_T0_returns_stats.csv", float_fmt=float_fmt, table_round=table_round)

    # T1: J6 logscore summary
    t1 = table_scores_summary(j6)
    _save_csv_stable(t1, tab_out / "tab_T1_logscore_summary.csv", float_fmt=float_fmt, table_round=table_round)

    # T2: J6 DM summary
    t2 = table_dm_summary(j6)
    _save_csv_stable(t2, tab_out / "tab_T2_dm_summary.csv", float_fmt=float_fmt, table_round=table_round)

    # T3: Static copula scores summary (indep/gauss/t)
    if j3 is not None:
        t3 = table_static_scores_summary(j3)
        _save_csv_stable(t3, tab_out / "tab_T3_static_scores_summary.csv", float_fmt=float_fmt, table_round=table_round)

    # T4: Anti-leakage contract (PIT slicing)
    t4 = table_pit_contract(splits=splits_obj, pit_metrics=pit_metrics, u_series_csv=u_series_csv)
    _save_csv_stable(t4, tab_out / "tab_T4_pit_anti_leakage_contract.csv", float_fmt=float_fmt, table_round=table_round)

    # T5: PIT metrics (core)
    t5 = table_pit_metrics(u_df=u_df, splits=splits_obj, pit_metrics=pit_metrics, epsilon=float(eps))
    _save_csv_stable(t5, tab_out / "tab_T5_pit_metrics.csv", float_fmt=float_fmt, table_round=table_round)

    # T6: J7 VaR/ES summary
    t6 = table_var_es_summary(j7)
    _save_csv_stable(t6, tab_out / "tab_T6_var_es_summary.csv", float_fmt=float_fmt, table_round=table_round)

    # T7: J7 coverage tests
    t7 = table_coverage_tests(j7)
    _save_csv_stable(t7, tab_out / "tab_T7_coverage_tests.csv", float_fmt=float_fmt, table_round=table_round)

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
    # Annex PIT diagnostics (do not affect core claims)
    # ----------------------------
    pit_annex_cfg = annex.get("pit", {}) if isinstance(annex, dict) else {}
    pit_annex_enabled = bool(pit_annex_cfg.get("enabled", True))
    if pit_annex_enabled:
        # A7: Q-Q plot
        fig_pit_qq(u_df, fig_out / "fig_A7_pit_qq_uniform.png", fig_w=fig_w, fig_h=fig_h, dpi=dpi)

        # A8: hist by sub-periods (default buckets chosen to be reviewer-friendly)
        periods = pit_annex_cfg.get(
            "subperiods",
            [
                ("2019-01-01", "2021-12-31"),
                ("2022-01-01", "2024-12-31"),
                ("2025-01-01", "2026-12-31"),
            ],
        )
        fig_pit_hist_subperiods(
            u_df,
            fig_out / "fig_A8_pit_hist_subperiods.png",
            periods=[(str(a), str(b)) for (a, b) in periods],
            bins=int(pit_annex_cfg.get("hist_bins", 30)),
            fig_w=fig_w,
            fig_h=fig_h,
            dpi=dpi,
        )

        # A9: ACF numeric table
        acf_tab = table_pit_acf(u_df, lags=[1, 5, 10, 20])
        _save_csv_stable(acf_tab, tab_out / "tab_A9_pit_acf.csv", float_fmt=float_fmt, table_round=table_round)

        # A10: clip hits (table + figure) using canonical u_series + epsilon
        # Use epsilon inferred from pit_metrics (or default) earlier in main.
        clip_tbl = table_pit_clip_stats(u_series_csv, epsilon=float(eps), include_raw=False)
        _save_csv_stable(
            clip_tbl,
            tab_out / "tab_A10_pit_clip_stats.csv",
            float_fmt=float_fmt,
            table_round=table_round,
        )
        fig_pit_clip_hits(
            clip_tbl,
            fig_out / "fig_A10_pit_clip_hits.png",
            fig_w=fig_w,
            fig_h=fig_h,
            dpi=dpi,
        )

        # A11/A12: ACF curves (lags 1..20) for u and |u-0.5|
        fig_pit_acf_lags_1_20(
            u_df,
            fig_out / "fig_A11_pit_acf_u_lags_1_20.png",
            use_abs=False,
            max_lag=20,
            fig_w=fig_w,
            fig_h=fig_h,
            dpi=dpi,
        )
        fig_pit_acf_lags_1_20(
            u_df,
            fig_out / "fig_A12_pit_acf_absu_lags_1_20.png",
            use_abs=True,
            max_lag=20,
            fig_w=fig_w,
            fig_h=fig_h,
            dpi=dpi,
        )

        # A13: empirical tail-dependence curves vs q (calm vs stress) using PIT + train-only RV thresholds
        td_q_tbl = fig_taildep_curves_q_calm_vs_stress(
            u_df,
            returns_csv,
            splits_obj,
            fig_out / "fig_A13_taildep_curves_q_calm_vs_stress.png",
            rv_asset=str(out_cfg.get("rv_asset", "BTC")),
            rv_window=int(out_cfg.get("rv_window", 30)),
            calm_q=float(out_cfg.get("rv_calm_q", 0.50)),
            stress_q=float(out_cfg.get("rv_stress_q", 0.90)),
            q_grid=pit_annex_cfg.get("taildep_q_grid", [0.01, 0.02, 0.05, 0.10]),
            fig_w=fig_w,
            fig_h=fig_h,
            dpi=dpi,
        )
        # store table for A13
        _save_csv_stable(
            td_q_tbl,
            tab_out / "tab_A13_taildep_curves_q_calm_vs_stress.csv",
            float_fmt=float_fmt,
            table_round=table_round,
        )

    # ----------------------------
    # J4 taildep bootstrap (daily) — F9 + T8
    # ----------------------------
    j4_dir = _find_j4_taildep_dir(repo_root, ds_root)
    if j4_dir is not None:
        boot_p = j4_dir / "taildep_bootstrap.json"
        if boot_p.exists():
            fig_j4_delta_lambda_bootstrap_hist(
                boot_p,
                fig_out / "fig_F9_delta_lambda_bootstrap_hist.png",
                fig_w=fig_w,
                fig_h=fig_h,
                dpi=dpi,
            )
        t8 = table_j4_taildep_params_lambda(j4_dir)
        _save_csv_stable(
            t8,
            tab_out / "tab_T8_taildep_params_lambda_by_regime.csv",
            float_fmt=float_fmt,
            table_round=table_round,
        )

if __name__ == "__main__":
    raise SystemExit(main())
