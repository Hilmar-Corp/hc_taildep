import numpy as np
from hc_taildep.markov.ms_copula import fit_ms_copula_train


def test_em_gauss_runs_and_returns_finite_ll():
    rng = np.random.default_rng(0)
    T = 1200
    # Build synthetic PIT-like data (rough)
    u = rng.uniform(0.01, 0.99, size=T)
    v = rng.uniform(0.01, 0.99, size=T)
    x = rng.normal(size=T)

    fit = fit_ms_copula_train(
        u, v, x,
        family="gauss",
        K=2,
        init_A=np.array([[0.98, 0.02],[0.02,0.98]]),
        rho_clamp=1e-6,
        nu_grid=[3,4,5,7,10],
        nu_bounds=(2.1, 60.0),
        calm_q=0.5,
        stress_q=0.9,
        max_iter=10,
        tol=1e-6,
        min_state_eff_n=50,
        ordering_key="rho",
        seed=123,
    )
    assert np.isfinite(fit.ll_train)
    assert fit.A.shape == (2,2)
    assert fit.pi0.shape == (2,)
    assert len(fit.theta) == 2