# tests/test_j7_sampling.py
from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr

from hc_taildep.impact.var_es import sample_copula, sample_mixture


def test_gauss_copula_correlation_order():
    rng = np.random.default_rng(0)
    n = 50_000
    rho = 0.6
    u, v = sample_copula("gauss", {"rho": rho}, n, rng)
    # check uniforms
    assert np.all((u > 0.0) & (u < 1.0))
    assert np.all((v > 0.0) & (v < 1.0))
    # monotone association should be positive
    r, _ = spearmanr(u, v)
    assert r > 0.4  # loose, but should hold


def test_t_copula_has_positive_association():
    rng = np.random.default_rng(1)
    n = 50_000
    u, v = sample_copula("t", {"rho": 0.5, "nu": 6.0}, n, rng)
    r, _ = spearmanr(u, v)
    assert r > 0.3


def test_mixture_shapes_and_bounds():
    rng = np.random.default_rng(2)
    n = 10_000
    u, v = sample_mixture(np.array([0.2, 0.8]), [{"rho": 0.1}, {"rho": 0.7}], "gauss", n, rng)
    assert u.shape == (n,)
    assert v.shape == (n,)
    assert np.all((u > 0.0) & (u < 1.0))
    assert np.all((v > 0.0) & (v < 1.0))