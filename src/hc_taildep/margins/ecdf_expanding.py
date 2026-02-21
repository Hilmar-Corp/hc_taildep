from __future__ import annotations

from bisect import bisect_left, bisect_right, insort
from typing import Optional, Tuple

import numpy as np
import pandas as pd


def pit_ecdf_expanding_midrank(
    returns: pd.Series,
    *,
    min_history: int,
    epsilon: float,
    start_index: int = 0,
) -> Tuple[pd.Series, pd.Series]:
    """
    Expanding ECDF PIT with mid-rank tie handling (anti-leakage):
      u_t = F_{t-1}(r_t), where F_{t-1} is ECDF built on returns[:t].

    Implementation details:
    - Maintains a sorted list of past returns for O(n log n).
    - mid-rank:
        rank = 1 + #{x < r_t} + 0.5 * #{x == r_t}
        u_raw = rank / (n + 1)
    - u = clip(u_raw, epsilon, 1-epsilon)

    Args:
      returns: pd.Series indexed by date, floats.
      min_history: minimum number of past observations required before producing u.
      epsilon: clipping parameter in (0, 0.5).
      start_index: compute u only for positions i >= max(start_index, min_history).

    Returns:
      u: pd.Series (clipped), same index as returns (NaN before activation)
      u_raw: pd.Series (unclipped), same index as returns (NaN before activation)
    """
    if not (0.0 < float(epsilon) < 0.5):
        raise ValueError("epsilon must be in (0, 0.5).")

    r = returns.astype(float).to_numpy()
    n_total = len(r)
    idx = returns.index

    u = np.full(n_total, np.nan, dtype=float)
    u_raw = np.full(n_total, np.nan, dtype=float)

    # Activation index: ensure enough history and respect start_index
    start_i = max(int(start_index), int(min_history))
    if start_i >= n_total:
        return pd.Series(u, index=idx, name=f"u_{returns.name}"), pd.Series(u_raw, index=idx, name=f"u_raw_{returns.name}")

    # Initialize sorted history with returns[:start_i]
    hist_sorted = []
    for x in r[:start_i]:
        insort(hist_sorted, float(x))

    # Iterate forward: for i, history is returns[:i] (size i)
    for i in range(start_i, n_total):
        x = float(r[i])
        n = i  # history size = i

        n_less = bisect_left(hist_sorted, x)
        n_leq = bisect_right(hist_sorted, x)
        n_eq = n_leq - n_less

        rank = 1.0 + n_less + 0.5 * n_eq
        ur = rank / (n + 1.0)

        u_raw[i] = ur
        u[i] = float(np.clip(ur, epsilon, 1.0 - epsilon))

        # update history AFTER scoring (anti-leakage)
        insort(hist_sorted, x)

    return (
        pd.Series(u, index=idx, name=f"u_{returns.name}"),
        pd.Series(u_raw, index=idx, name=f"u_raw_{returns.name}"),
    )