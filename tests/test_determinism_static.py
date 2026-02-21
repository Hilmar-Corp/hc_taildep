import numpy as np
from hc_taildep.copulas import student_t


def test_student_t_fit_deterministic():
    u = np.linspace(0.05, 0.95, 50)
    v = np.linspace(0.06, 0.96, 50)[::-1]
    r1 = student_t.fit(u, v, nu_grid=[3, 4, 5, 7, 10], nu_bounds=(2.1, 60.0))
    r2 = student_t.fit(u, v, nu_grid=[3, 4, 5, 7, 10], nu_bounds=(2.1, 60.0))
    assert np.allclose(r1, r2)