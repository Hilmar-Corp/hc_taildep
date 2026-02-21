import numpy as np
from hc_taildep.impact.var_es import compute_var_es

def test_var_es_ordering():
    rng = np.random.default_rng(0)
    losses = rng.standard_normal(200000)
    var95, es95 = compute_var_es(losses, 0.95)
    var99, es99 = compute_var_es(losses, 0.99)
    assert var99 >= var95
    assert es95 >= var95
    assert es99 >= var99