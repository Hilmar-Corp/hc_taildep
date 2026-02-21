# tests/test_j7_var_es.py
from __future__ import annotations

import numpy as np
import pytest

from hc_taildep.impact.var_es import (
    build_empirical_quantile,
    compute_var_es,
    sanity_check_var_es,
)


def test_compute_var_es_monotonic_basic():
    rng = np.random.default_rng(0)
    # losses ~ |N|
    L = np.abs(rng.standard_normal(50_000))
    var95, es95 = compute_var_es(L, 0.95)
    var99, es99 = compute_var_es(L, 0.99)

    assert np.isfinite(var95) and np.isfinite(es95)
    assert np.isfinite(var99) and np.isfinite(es99)
    assert var99 >= var95
    assert es95 >= var95
    assert es99 >= var99
    sanity_check_var_es(var95, es95, var99, es99)


def test_empirical_quantile_is_monotone_and_matches_distribution():
    rng = np.random.default_rng(1)
    x = rng.standard_normal(10_000)
    Q = build_empirical_quantile(x, grid_size=2001, clip_eps=1e-6)

    u = np.linspace(0.001, 0.999, 1000)
    q = Q(u)

    assert np.all(np.diff(q) >= -1e-12)  # monotone (allow tiny float noise)

    # crude distribution check: Q(U) should resemble x
    u_s = rng.random(50_000)
    y = Q(u_s)
    assert abs(np.mean(y) - np.mean(x)) < 0.05
    assert abs(np.std(y) - np.std(x)) < 0.05


def test_compute_var_es_bad_alpha():
    with pytest.raises(ValueError):
        compute_var_es(np.array([1.0, 2.0]), 1.0)