import numpy as np
from hc_taildep.impact.var_es import sample_copula 

def test_gumbel_marginals_uniformish():
    rng = np.random.default_rng(0)
    n = 200_000
    u, v = sample_copula("gumbel", {"theta": 1.8}, n, rng, rho_clamp=1e-6)

    # For uniform(0,1): mean ~ 0.5, P(u<0.05) ~ 0.05, P(u>0.95) ~ 0.05
    mu = float(u.mean())
    pv_lo = float((u < 0.05).mean())
    pv_hi = float((u > 0.95).mean())

    assert abs(mu - 0.5) < 0.01
    assert abs(pv_lo - 0.05) < 0.01
    assert abs(pv_hi - 0.05) < 0.01