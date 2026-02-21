import numpy as np
from hc_taildep.impact.var_es import sample_copula

def test_clayton_gumbel_bounds():
    rng = np.random.default_rng(0)
    u1, u2 = sample_copula("clayton", {"theta": 2.0}, 5000, rng, rho_clamp=1e-6)
    assert np.all((u1 > 0) & (u1 < 1))
    assert np.all((u2 > 0) & (u2 < 1))

    rng = np.random.default_rng(1)
    u1, u2 = sample_copula("gumbel", {"theta": 2.0}, 5000, rng, rho_clamp=1e-6)
    assert np.all((u1 > 0) & (u1 < 1))
    assert np.all((u2 > 0) & (u2 < 1))