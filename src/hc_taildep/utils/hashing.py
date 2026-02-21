from __future__ import annotations
import hashlib
from pathlib import Path

def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def sha256_text(s: str) -> str:
    return sha256_bytes(s.encode("utf-8"))

def stable_hash32(*parts: str) -> int:
    # deterministic 32-bit int from text parts
    h = hashlib.sha256(("||".join(parts)).encode("utf-8")).digest()
    return int.from_bytes(h[:4], "big", signed=False)