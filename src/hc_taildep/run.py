from __future__ import annotations

import argparse
import os
import platform
import subprocess
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from hc_taildep.data.smoke_dataset import SmokeSpec, make_smoke_returns
from hc_taildep.reporting.bootstrap_report import write_json, write_report_md
from hc_taildep.utils.config import load_yaml, dump_yaml, resolve_config
from hc_taildep.utils.hashing import sha256_text, sha256_bytes
from hc_taildep.utils.paths import ensure_dir, is_relative_to
from hc_taildep.utils.seeds import derive_seeds


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def get_git_info(project_root: Path) -> Dict[str, Any]:
    def run(cmd: List[str]) -> str:
        # Silence git stderr (e.g., when not in a git repo)
        return subprocess.check_output(
            cmd,
            cwd=str(project_root),
            stderr=subprocess.DEVNULL,
        ).decode().strip()

    git = {"commit": None, "dirty": None, "branch": None}
    try:
        git["commit"] = run(["git", "rev-parse", "HEAD"])[:12]
        git["branch"] = run(["git", "rev-parse", "--abbrev-ref", "HEAD"])
        status = run(["git", "status", "--porcelain"])
        git["dirty"] = bool(status)
    except Exception:
        git["commit"] = "nogit"
        git["branch"] = None
        git["dirty"] = None
    return git


def pip_freeze() -> str:
    try:
        out = subprocess.check_output([sys.executable, "-m", "pip", "freeze"])
        return out.decode("utf-8", errors="replace")
    except Exception:
        return ""


def build_model_id(cfg: Dict[str, Any]) -> str:
    # model_id excludes seed/split/runtime. For J0, use only model block.
    model_block = cfg.get("model", {})
    h = sha256_text(str(model_block))[:12]
    return f"model_{h}"


def build_run_id(model_id: str, dataset_version: str, seed: int, git_commit: str) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")
    shortgit = (git_commit or "nogit")
    return f"{stamp}__{model_id}__{dataset_version}__s{seed}__{shortgit}"


def assert_write_sandbox(run_dir: Path, paths_written: List[Path]) -> Dict[str, bool]:
    # Enforce “no write outside run_dir”
    ok = True
    for p in paths_written:
        if not is_relative_to(p, run_dir):
            ok = False
            break
    return {"no_write_outside_run_ok": ok}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=str, help="Path to yaml config")
    args = ap.parse_args()

    t0 = time.time()
    start_utc = utc_now_iso()

    project_root = Path(".").resolve()
    config_path = Path(args.config).resolve()

    cfg_raw = load_yaml(config_path)
    cfg = resolve_config(cfg_raw, config_path)

    seed_global = int(cfg["run"]["seed_global"])
    seeds = derive_seeds(seed_global)

    git = get_git_info(project_root)
    model_id = build_model_id(cfg)

    dataset_version = cfg["dataset"]["dataset_version"]
    run_id = build_run_id(model_id, dataset_version, seed_global, git.get("commit"))

    runs_dir = (project_root / cfg["paths"]["runs_dir"]).resolve()
    run_dir = runs_dir / run_id

    # append-only: refuse if exists
    if run_dir.exists():
        raise RuntimeError(f"run_dir already exists (append-only): {run_dir}")

    # Create standard dirs
    ensure_dir(run_dir)
    ensure_dir(run_dir / "tables")
    ensure_dir(run_dir / "figures")

    paths_written: List[Path] = []

    # Dump resolved config
    resolved_path = run_dir / "config.resolved.yaml"
    dump_yaml(cfg, resolved_path)
    paths_written.append(resolved_path)

    config_hash = sha256_bytes(resolved_path.read_bytes())

    # Build smoke dataset deterministically
    ds = cfg["dataset"]
    spec = SmokeSpec(
        n_obs=int(ds["n_obs"]),
        start_date=str(ds["start_date"]),
        freq=str(ds["freq"]),
        assets=list(ds["assets"]),
        seed=seeds.data,
    )
    rets = make_smoke_returns(spec)

    # “Dummy PIT-like” transformation: rank-based in-sample (J0 only, not used later)
    u = rets.rank(pct=True).clip(1e-6, 1 - 1e-6)

    # Save a tiny artifact (optional)
    # Prefer Parquet, but fall back to CSV if no Parquet engine is installed.
    pred_path = run_dir / "predictions.parquet"
    try:
        u.to_parquet(pred_path)
    except ImportError:
        pred_path = run_dir / "predictions.csv"
        u.to_csv(pred_path, index=True)
    paths_written.append(pred_path)

    # dataset_hash: hash the bytes of the artifact we actually used in the run (here: predictions)
    dataset_hash = sha256_bytes(pred_path.read_bytes())

    # Checks
    checks = {}
    checks.update(assert_write_sandbox(run_dir, paths_written))
    checks["append_only_ok"] = True  # if we got here

    status = "ok" if all(checks.values()) else "fail"

    # Metrics
    metrics = {
        "status": status,
        "smoke": {
            "n_obs": int(rets.shape[0]),
            "n_assets": int(rets.shape[1]),
            "start_date": str(rets.index.min().date()),
            "end_date": str(rets.index.max().date()),
        },
        "runtime_sec": float(time.time() - t0),
        "checks": checks,
        "warnings": [],
    }
    metrics_path = run_dir / "metrics.json"
    write_json(metrics_path, metrics)
    paths_written.append(metrics_path)

    # Provenance
    prov = {
        "run_id": run_id,
        "model_id": model_id,
        "git": git,
        "time": {
            "start_utc": start_utc,
            "end_utc": utc_now_iso(),
            "duration_sec": float(time.time() - t0),
        },
        "env": {
            "python_version": sys.version.split()[0],
            "platform": platform.platform(),
            "os": os.name,
        },
        "packages": {
            "pip_freeze": pip_freeze(),
        },
        "dataset": {
            "dataset_version": dataset_version,
            "dataset_path": str(pred_path.relative_to(project_root)),
            "dataset_hash_sha256": dataset_hash,
        },
        "seeds": asdict(seeds),
        "config_hash_sha256": config_hash,
        "artifacts": [str(p.relative_to(project_root)) for p in paths_written],
    }
    prov_path = run_dir / "provenance.json"
    write_json(prov_path, prov)
    paths_written.append(prov_path)

    # Report.md
    report_lines = [
        "# hc_taildep — Smoke Run",
        "",
        f"- run_id: `{run_id}`",
        f"- model_id: `{model_id}`",
        f"- dataset_version: `{dataset_version}`",
        f"- git_commit: `{git.get('commit')}`",
        "",
        "## Inputs",
        f"- config: `{str(config_path.relative_to(project_root))}`",
        f"- seed_global: `{seed_global}`",
        "",
        "## Outputs",
        f"- config.resolved: `{str(resolved_path.relative_to(project_root))}`",
        f"- metrics: `{str(metrics_path.relative_to(project_root))}`",
        f"- provenance: `{str(prov_path.relative_to(project_root))}`",
        f"- predictions: `{str(pred_path.relative_to(project_root))}`",
        "",
        "## Checks",
    ]
    for k, v in checks.items():
        report_lines.append(f"- {k}: `{v}`")
    report_lines += [
        "",
        "## Reproduce",
        f"```bash\npython -m hc_taildep.run --config {str(config_path.relative_to(project_root))}\n```",
        "",
    ]
    report_path = run_dir / "report.md"
    write_report_md(report_path, report_lines)
    paths_written.append(report_path)

    # Final guard: ensure no_write_outside_run_ok still holds including late writes
    final_checks = assert_write_sandbox(run_dir, paths_written)
    if not final_checks["no_write_outside_run_ok"]:
        raise RuntimeError("Write outside run_dir detected.")

    print(f"[OK] run_dir: {run_dir}")
    print(f"[OK] status: {status}")
    return 0 if status == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())