import numpy as np
from hc_taildep.copulas.gating import fisher_mix_rho, mix_nu, logistic_weights


def test_logistic_weights_bounds():
    z = np.array([-10, 0, 10], dtype=float)
    w = logistic_weights(z, a=0.0, b=1.0)
    assert np.all(w > 0.0) and np.all(w < 1.0)


def test_fisher_mix_rho_bounds():
    w = np.array([0.0, 0.5, 1.0])
    r = fisher_mix_rho(-0.9, 0.9, w, rho_clamp=1e-6)
    assert np.all(np.isfinite(r))
    assert np.all(r > -1.0) and np.all(r < 1.0)


def test_mix_nu_bounds():
    w = np.array([0.0, 0.5, 1.0])
    nu = mix_nu(3.0, 30.0, w, nu_bounds=(2.1, 60.0))
    assert np.all(np.isfinite(nu))
    assert np.all(nu >= 2.1) and np.all(nu <= 60.0)
