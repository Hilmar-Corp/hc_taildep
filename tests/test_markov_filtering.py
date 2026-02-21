import numpy as np
from hc_taildep.markov.filtering import forward_filter_log


def test_forward_filter_probs_sum_to_one():
    rng = np.random.default_rng(0)
    T, K = 200, 2
    # random emissions
    logf = rng.normal(size=(T, K))
    A = np.array([[0.95, 0.05], [0.05, 0.95]])
    pi0 = np.array([0.5, 0.5])

    pi_pred, pi_filt, logp = forward_filter_log(logf, A, pi0)

    assert np.all(np.isfinite(logp))
    assert np.allclose(pi_pred.sum(axis=1), 1.0, atol=1e-10)
    assert np.allclose(pi_filt.sum(axis=1), 1.0, atol=1e-10)
    assert np.all((pi_pred >= 0) & (pi_pred <= 1))
    assert np.all((pi_filt >= 0) & (pi_filt <= 1))