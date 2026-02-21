# src/hc_taildep/utils/io.py
from __future__ import annotations

import dataclasses
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, Tuple

import yaml


def ensure_dir(p: str | Path) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def sha256_file(path: str | Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def write_text(path: str | Path, s: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(s, encoding="utf-8")


def write_json(path: str | Path, obj: Any, *, indent: int = 2) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=indent, sort_keys=True), encoding="utf-8")


def read_yaml(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def write_yaml(path: str | Path, obj: Any) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, sort_keys=False)


def resolve_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    # Minimal “resolved” config (freeze env vars if you used them)
    def _walk(x):
        if isinstance(x, dict):
            return {k: _walk(v) for k, v in x.items()}
        if isinstance(x, list):
            return [_walk(v) for v in x]
        if isinstance(x, str):
            return os.path.expandvars(x)
        return x

    return _walk(cfg)


def build_provenance(
    *,
    config_path: str | Path,
    config_resolved: Dict[str, Any],
    inputs: Dict[str, str | Path],
    outputs: Dict[str, str | Path],
) -> Dict[str, Any]:
    prov_inputs = {}
    for k, p in inputs.items():
        p = Path(p)
        prov_inputs[k] = {
            "path": str(p),
            "sha256": sha256_file(p) if p.exists() and p.is_file() else None,
            "exists": p.exists(),
        }
    prov_outputs = {}
    for k, p in outputs.items():
        p = Path(p)
        prov_outputs[k] = {
            "path": str(p),
            "sha256": sha256_file(p) if p.exists() and p.is_file() else None,
            "exists": p.exists(),
        }

    cfg_bytes = json.dumps(config_resolved, sort_keys=True).encode("utf-8")
    return {
        "config_path": str(config_path),
        "config_sha256": sha256_bytes(cfg_bytes),
        "config_resolved": config_resolved,
        "inputs": prov_inputs,
        "outputs": prov_outputs,
    }