from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List
import json

def write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.write_text(json.dumps(obj, indent=2, sort_keys=False), encoding="utf-8")

def write_report_md(path: Path, lines: List[str]) -> None:
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")