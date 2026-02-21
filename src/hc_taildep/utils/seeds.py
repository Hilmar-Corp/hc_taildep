from __future__ import annotations
from dataclasses import dataclass
from .hashing import stable_hash32

COMPONENTS = ["data", "margins", "copula", "eval", "reporting"]

@dataclass(frozen=True)
class Seeds:
    global_seed: int
    data: int
    margins: int
    copula: int
    eval: int
    reporting: int

def derive_seeds(seed_global: int) -> Seeds:
    def d(name: str) -> int:
        return stable_hash32(str(seed_global), name)
    return Seeds(
        global_seed=seed_global,
        data=d("data"),
        margins=d("margins"),
        copula=d("copula"),
        eval=d("eval"),
        reporting=d("reporting"),
    )