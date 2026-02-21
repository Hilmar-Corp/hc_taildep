from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _read_json(p: Path) -> dict[str, Any]:
    return json.loads(p.read_text())


def _get_dm(dm: dict[str, Any], key: str) -> dict[str, Any] | None:
    for r in dm.get("results", []):
        if r.get("comparison") == key:
            return r
    return None


def _fmt(x: Any) -> str:
    if x is None:
        return "NA"
    if isinstance(x, bool):
        return "1" if x else "0"
    if isinstance(x, int):
        return str(x)
    try:
        xf = float(x)
        if abs(xf) < 1e-3:
            return f"{xf:.3g}"
        return f"{xf:.6f}"
    except Exception:
        return str(x)


@dataclass
class Row:
    run: str
    out_dir: str
    refit_every: int | None
    min_eff_t: int | None

    ms_t_mean: float | None
    thr_t_mean: float | None
    delta_ms_t_thr_t: float | None

    dm_global_mean_delta: float | None
    dm_global_p: float | None
    dm_ms_only_mean_delta: float | None
    dm_ms_only_p: float | None

    ms2_used_rate: float | None
    thr_fallback_rate: float | None
    gauss_used_rate: float | None

    predictions_hash: str | None


def _row_to_dict(r: Row) -> dict[str, Any]:
    return {
        "run": r.run,
        "out_dir": r.out_dir,
        "refit_every": r.refit_every,
        "min_eff_t": r.min_eff_t,
        "ms_t_mean": r.ms_t_mean,
        "thr_t_mean": r.thr_t_mean,
        "delta_ms_t_thr_t": r.delta_ms_t_thr_t,
        "DM_global_mean_delta": r.dm_global_mean_delta,
        "DM_global_p": r.dm_global_p,
        "DM_ms_only_mean_delta": r.dm_ms_only_mean_delta,
        "DM_ms_only_p": r.dm_ms_only_p,
        "ms2_used_rate": r.ms2_used_rate,
        "thr_fallback_rate": r.thr_fallback_rate,
        "gauss_used_rate": r.gauss_used_rate,
        "predictions_hash": r.predictions_hash,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("")
        return
    cols = list(rows[0].keys())
    lines = [",".join(cols)]
    for row in rows:
        vals = []
        for c in cols:
            v = row.get(c)
            s = "" if v is None else str(v)
            if "," in s or "\n" in s or '"' in s:
                s = '"' + s.replace('"', '""') + '"'
            vals.append(s)
        lines.append(",".join(vals))
    path.write_text("\n".join(lines) + "\n")


def _write_md(path: Path, rows: list[dict[str, Any]], cols: list[str]) -> None:
    if not rows:
        path.write_text("")
        return
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    body = []
    for row in rows:
        body.append("| " + " | ".join(_fmt(row.get(c)) for c in cols) + " |")
    path.write_text("\n".join([header, sep] + body) + "\n")


def _print_table(rows: list[dict[str, Any]], cols: list[str]) -> None:
    if not rows:
        print("[empty]")
        return
    width = {c: max(len(c), max(len(_fmt(row.get(c))) for row in rows)) for c in cols}
    hdr = " | ".join(c.ljust(width[c]) for c in cols)
    sep = "-+-".join("-" * width[c] for c in cols)
    print(hdr)
    print(sep)
    for row in rows:
        print(" | ".join(_fmt(row.get(c)).ljust(width[c]) for c in cols))


def main() -> int:
    ap = argparse.ArgumentParser(description="Summarize J6 MS grid runs into a comparable table.")
    ap.add_argument(
        "--base",
        default="data/processed/ds_v0_btceth_daily_binance_closeutc/copulas/markov",
        help="Base directory containing j6_ms2_* run folders.",
    )
    ap.add_argument(
        "--runs",
        nargs="*",
        default=None,
        help="Explicit run folder names (e.g. j6_ms2_refit10 ...). If omitted, auto-discovers j6_ms2_*.",
    )
    ap.add_argument(
        "--pattern",
        default="j6_ms2_",
        help="Auto-discovery prefix pattern when --runs is omitted.",
    )
    ap.add_argument(
        "--outdir",
        default=None,
        help="Output directory for summary tables. Default: <base>/tables",
    )
    ap.add_argument(
        "--sort_by",
        default="delta_ms_t_thr_t",
        help="Sort key (descending). Use 'run' to keep lexical order.",
    )
    args = ap.parse_args()

    base = Path(args.base)
    if not base.exists():
        raise FileNotFoundError(f"Base not found: {base}")

    if args.runs:
        run_dirs = [base / r for r in args.runs]
    else:
        run_dirs = sorted([p for p in base.iterdir() if p.is_dir() and p.name.startswith(args.pattern)])

    rows: list[Row] = []
    for d in run_dirs:
        metrics_p = d / "metrics.json"
        dm_p = d / "dm_test.json"
        if not (metrics_p.exists() and dm_p.exists()):
            continue

        m = _read_json(metrics_p)
        dm = _read_json(dm_p)

        # Normalize types for stable reporting
        if m.get("refit_every") is not None:
            try:
                m["refit_every"] = int(m["refit_every"])
            except Exception:
                pass

        pm = m.get("per_model", {}) or {}
        fr = m.get("fallback_rates", {}) or {}

        dm_global = _get_dm(dm, "ms_t_vs_thr_t")
        dm_ms_only = _get_dm(dm, "ms_t_vs_thr_t__ms_only")

        ms_t_mean = (pm.get("ms_t") or {}).get("mean")
        thr_t_mean = (pm.get("thr_t") or {}).get("mean")
        delta = None
        if ms_t_mean is not None and thr_t_mean is not None:
            try:
                delta = float(ms_t_mean) - float(thr_t_mean)
            except Exception:
                delta = None

        ms_model = m.get("ms_model", {}) or {}

        # Prefer explicit field written by newer runners.
        min_eff_t = ms_model.get("min_state_eff_n_t", None)

        # If the folder name encodes mineffTXXX, treat it as authoritative for reporting.
        # This avoids incorrectly reporting the global default min_state_eff_n (often 150)
        # for runs whose only change was a T-specific threshold.
        m_ = re.search(r"mineffT(\d+)", d.name)
        if m_:
            try:
                min_eff_t = int(m_.group(1))
            except Exception:
                pass

        # Backward compatibility: if still absent, fall back to the generic min_state_eff_n.
        if min_eff_t is None:
            min_eff_t = ms_model.get("min_state_eff_n", None)

        rows.append(
            Row(
                run=d.name,
                out_dir=str(d),
                refit_every=m.get("refit_every"),
                min_eff_t=min_eff_t,
                ms_t_mean=ms_t_mean,
                thr_t_mean=thr_t_mean,
                delta_ms_t_thr_t=delta,
                dm_global_mean_delta=(dm_global.get("mean_delta") if dm_global else None),
                dm_global_p=(dm_global.get("pvalue") if dm_global else None),
                dm_ms_only_mean_delta=(dm_ms_only.get("mean_delta") if dm_ms_only else None),
                dm_ms_only_p=(dm_ms_only.get("pvalue") if dm_ms_only else None),
                ms2_used_rate=fr.get("ms_t_ms2_used_rate"),
                thr_fallback_rate=fr.get("ms_t_thr_fallback_rate", fr.get("ms_t_ms1_rate")),
                gauss_used_rate=fr.get("ms_gauss_used_rate"),
                predictions_hash=(m.get("hashes", {}) or {}).get("predictions_csv"),
            )
        )

    # dict rows
    drows = [_row_to_dict(r) for r in rows]

    # sort
    sb = args.sort_by
    if sb and sb != "run":
        def keyfun(x: dict[str, Any]) -> float:
            v = x.get(sb)
            try:
                return float(v)
            except Exception:
                return float("-inf")
        drows = sorted(drows, key=keyfun, reverse=True)
    else:
        drows = sorted(drows, key=lambda x: str(x.get("run")))

    # columns to print
    cols = [
        "run",
        "refit_every",
        "min_eff_t",
        "ms_t_mean",
        "thr_t_mean",
        "delta_ms_t_thr_t",
        "DM_global_p",
        "DM_ms_only_p",
        "ms2_used_rate",
        "thr_fallback_rate",
        "gauss_used_rate",
        "predictions_hash",
    ]

    _print_table(drows, cols)

    # write outputs
    outdir = Path(args.outdir) if args.outdir else (base / "tables")
    outdir.mkdir(parents=True, exist_ok=True)

    csv_p = outdir / "j6_grid_summary.csv"
    md_p = outdir / "j6_grid_summary.md"
    _write_csv(csv_p, drows)
    _write_md(md_p, drows, cols)

    print(f"\n[OK] wrote: {csv_p}")
    print(f"[OK] wrote: {md_p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())