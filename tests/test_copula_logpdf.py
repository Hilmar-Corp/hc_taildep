import numpy as np

from hc_taildep.copulas import indep, gaussian, student_t


def test_indep_zero():
    u = np.array([0.2, 0.7, 0.5])
    v = np.array([0.3, 0.6, 0.9])
    lp = indep.logpdf(u, v)
    assert np.allclose(lp, 0.0)


def test_gaussian_finite_and_symmetric():
    u = np.array([0.2, 0.7, 0.5, 0.9])
    v = np.array([0.3, 0.6, 0.9, 0.2])
    rho = gaussian.fit(u, v)
    lp1 = gaussian.logpdf(u, v, rho)
    lp2 = gaussian.logpdf(v, u, rho)
    assert np.isfinite(lp1).all()
    assert np.allclose(lp1, lp2, atol=1e-10)


def test_student_t_finite_and_symmetric():
    u = np.array([0.2, 0.7, 0.5, 0.9])
    v = np.array([0.3, 0.6, 0.9, 0.2])
    rho, nu = student_t.fit(u, v, nu_grid=[3, 5, 10], nu_bounds=(2.1, 60.0))
    lp1 = student_t.logpdf(u, v, rho, nu)
    lp2 = student_t.logpdf(v, u, rho, nu)
    assert np.isfinite(lp1).all()
    assert np.allclose(lp1, lp2, atol=1e-10)