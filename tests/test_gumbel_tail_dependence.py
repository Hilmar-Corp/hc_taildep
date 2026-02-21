# tests/test_gumbel_tail_dependence.py
import numpy as np
from hc_taildep.impact.copulas import sample_gumbel, GumbelParams, tail_dependence_mc


def test_gumbel_has_upper_tail_dependence():
    theta = 1.8
    rng = np.random.default_rng(0)
    u, v = sample_gumbel(GumbelParams(theta=theta), 200_000, rng)

    # Finite-q diagnostic (NOT the asymptotic coefficient).
    # For Gumbel, asymptotically: lambda_L -> 0, lambda_U > 0.
    # At q=0.05, lambda_L(q) can still be > 0 because dependence is positive overall.
    td = tail_dependence_mc(u, v, q=0.05)

    # Must show upper tail stronger than lower tail.
    assert td["lambda_U"] > 0.15
    assert td["lambda_U"] > td["lambda_L"] + 0.02

    # Lower tail should not look like "strong tail dependence" at this q.
    # (Loose bound: we only want to detect accidental survival-rotation bugs.)
    assert td["lambda_L"] < 0.35