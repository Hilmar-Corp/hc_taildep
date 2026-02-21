# tests/test_j8_copulas.py
import numpy as np
from hc_taildep.impact import copulas

def test_sample_bounds():
    rng = np.random.default_rng(0)
    u1,u2 = copulas.sample_gauss(copulas.GaussParams(rho=0.5), 5000, rng)
    assert np.all((u1 > 0) & (u1 < 1))
    assert np.all((u2 > 0) & (u2 < 1))

def test_gumbel_indep_limit():
    rng = np.random.default_rng(0)
    u1,u2 = copulas.sample_gumbel(copulas.GumbelParams(theta=1.0), 2000, rng)
    # theta=1 => indep => corr close to 0 (rough)
    assert abs(np.corrcoef(u1, u2)[0,1]) < 0.1