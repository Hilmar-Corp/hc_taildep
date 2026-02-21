from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict
import yaml

@dataclass(frozen=True)
class Config:
    raw: Dict[str, Any]

def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def dump_yaml(obj: Dict[str, Any], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, sort_keys=False)

def resolve_config(cfg: Dict[str, Any], config_path: Path) -> Dict[str, Any]:
    # Minimal "resolution": add config_path and normalize project_root to absolute path
    cfg = dict(cfg)
    cfg["_meta"] = {
        "config_path": str(config_path.as_posix()),
    }
    return cfg