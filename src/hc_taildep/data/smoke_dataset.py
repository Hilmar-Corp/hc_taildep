from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd

@dataclass(frozen=True)
class SmokeSpec:
    n_obs: int
    start_date: str
    freq: str
    assets: list[str]
    seed: int

def make_smoke_returns(spec: SmokeSpec) -> pd.DataFrame:
    rng = np.random.default_rng(spec.seed)
    idx = pd.date_range(spec.start_date, periods=spec.n_obs, freq=spec.freq)
    # Simple correlated Gaussian returns (not a "model", just deterministic data)
    corr = 0.5
    z1 = rng.standard_normal(spec.n_obs)
    z2 = corr * z1 + (1 - corr**2) ** 0.5 * rng.standard_normal(spec.n_obs)
    data = np.vstack([z1, z2]).T * 0.01
    df = pd.DataFrame(data, index=idx, columns=spec.assets[:2])
    # If >2 assets requested, add independent noise
    for a in spec.assets[2:]:
        df[a] = rng.standard_normal(spec.n_obs) * 0.01
    return df