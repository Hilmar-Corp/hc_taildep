from __future__ import annotations
import json
from pathlib import Path
import hashlib

def stable_hash(manifest: dict) -> str:
    m = dict(manifest)
    # remove volatile fields
    m.pop("created_utc", None)
    m.pop("manifest_sha256", None)

    # If you want to be extra strict, also ignore git dirty/branch
    if "git" in m and isinstance(m["git"], dict):
        g = dict(m["git"])
        # commit is fine, but dirty/branch can be volatile depending on local state
        g.pop("dirty", None)
        g.pop("branch", None)
        m["git"] = g

    b = json.dumps(m, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(b).hexdigest()

p = Path("paper/out/hc_taildep_v0_camera_ready/manifest.json")
manifest = json.loads(p.read_text())
print(stable_hash(manifest))
