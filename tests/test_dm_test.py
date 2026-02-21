import numpy as np
from hc_taildep.eval.dm_test import dm_test


def test_dm_zero_mean_not_always_significant():
    rng = np.random.default_rng(0)
    pvals = []
    for k in range(30):
        x = rng.normal(0.0, 1.0, size=500)
        res = dm_test(x)
        pvals.append(res.pvalue)
    # not too many false positives at 5%
    assert np.mean(np.array(pvals) < 0.05) < 0.25


def test_dm_positive_mean_significant():
    # make it deterministic: clearly positive mean, moderate noise
    rng = np.random.default_rng(1)
    x = 0.10 + rng.normal(0.0, 0.10, size=800)
    res = dm_test(x, alternative="greater")
    assert res.mean_delta > 0
    assert res.dm_stat > 0
    assert res.pvalue < 0.05
