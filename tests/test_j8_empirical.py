# tests/test_j8_empirical.py
import numpy as np
from hc_taildep.impact.empirical import EmpiricalQuantile

def test_empirical_quantile_monotone():
    rng = np.random.default_rng(0)
    x = rng.standard_normal(2000)
    Q = EmpiricalQuantile(x)
    u = np.linspace(1e-6, 1-1e-6, 1000)
    y = Q(u)
    assert np.all(np.diff(y) >= -1e-12)