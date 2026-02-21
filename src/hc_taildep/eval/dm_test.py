from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.stats import norm


def _nw_lag_rule(n: int, rule: str) -> int:
    """
    Default NW lag rule:
      L = floor( 4 * (n/100)^(2/9) )
    """
    rule = (rule or "").strip()
    if rule in ("4*(n/100)^(2/9)", "4*(n/100)**(2/9)", "default", ""):
        if n <= 0:
            return 0
        L = int(math.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))
        return max(0, L)
    if rule == "n^(1/3)":
        if n <= 0:
            return 0
        L = int(math.floor(n ** (1.0 / 3.0)))
        return max(0, L)
    raise ValueError(f"Unknown nw_lag_rule: {rule}")


def newey_west_longrun_var(x: np.ndarray, L: int) -> float:
    """
    Long-run variance estimator (Newey-West / Bartlett kernel):
      gamma0 + 2 * sum_{k=1..L} w_k * gamma_k
    where gamma_k is sample autocov at lag k, and w_k = 1 - k/(L+1).
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = int(x.size)
    if n <= 1:
        return float("nan")
    x = x - np.mean(x)

    # gamma0
    gamma0 = float(np.dot(x, x) / n)
    if L <= 0:
        return gamma0

    out = gamma0
    for k in range(1, L + 1):
        w = 1.0 - (k / (L + 1.0))
        cov = float(np.dot(x[k:], x[:-k]) / n)
        out += 2.0 * w * cov
    # guard against negative numeric noise
    return float(max(out, 1e-18))


@dataclass(frozen=True)
class DMResult:
    n_obs: int
    nw_lag: int
    mean_delta: float
    std_delta: float
    dm_stat: float
    pvalue: float
    alternative: Literal["two-sided", "greater", "less"]


def dm_test(
    delta: np.ndarray,
    *,
    nw_lag_rule: str = "4*(n/100)^(2/9)",
    alternative: Literal["two-sided", "greater", "less"] = "two-sided",
) -> DMResult:
    """
    Diebold–Mariano test on series delta_t = lossA_t - lossB_t (here: logscore diffs).
    H0: E[delta]=0. Uses asymptotic normal with HAC (Newey-West) variance.

    alternative:
      - two-sided: H1 E[delta] != 0
      - greater:   H1 E[delta] > 0
      - less:      H1 E[delta] < 0
    """
    d = np.asarray(delta, dtype=float)
    d = d[np.isfinite(d)]
    n = int(d.size)
    if n < 5:
        return DMResult(
            n_obs=n,
            nw_lag=0,
            mean_delta=float(np.mean(d)) if n > 0 else float("nan"),
            std_delta=float(np.std(d, ddof=1)) if n > 1 else float("nan"),
            dm_stat=float("nan"),
            pvalue=float("nan"),
            alternative=alternative,
        )

    L = _nw_lag_rule(n, nw_lag_rule)
    mean_d = float(np.mean(d))
    std_d = float(np.std(d, ddof=1))
    lrvar = newey_west_longrun_var(d, L)
    # Var(mean) = lrvar / n
    se = math.sqrt(lrvar / n)
    dm = mean_d / se

    if alternative == "two-sided":
        p = 2.0 * float(norm.sf(abs(dm)))
    elif alternative == "greater":
        p = float(norm.sf(dm))
    elif alternative == "less":
        p = float(norm.cdf(dm))
    else:
        raise ValueError(f"Unknown alternative: {alternative}")

    return DMResult(
        n_obs=n,
        nw_lag=int(L),
        mean_delta=mean_d,
        std_delta=std_d,
        dm_stat=float(dm),
        pvalue=float(p),
        alternative=alternative,
    )
