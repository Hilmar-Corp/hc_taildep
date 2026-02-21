# src/hc_taildep/impact/__init__.py
from __future__ import annotations

from .var_es import (
    EmpiricalQuantile,
    build_empirical_quantile,
    compute_var_es,
    sample_copula,
    sample_mixture,
)