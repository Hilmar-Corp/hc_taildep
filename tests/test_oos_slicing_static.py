import numpy as np

from hc_taildep.copulas import gaussian


def test_oos_slicing_train_excludes_current_point():
    # toy: if we modify the "current" point only, fitted rho should not change for that point
    u = np.array([0.2, 0.25, 0.3, 0.35, 0.4])
    v = np.array([0.1, 0.15, 0.2, 0.25, 0.3])

    k = 4  # current point
    rho1 = gaussian.fit(u[:k], v[:k])

    u2 = u.copy()
    v2 = v.copy()
    u2[k] = 0.99
    v2[k] = 0.01
    rho2 = gaussian.fit(u2[:k], v2[:k])

    assert abs(rho1 - rho2) < 1e-12