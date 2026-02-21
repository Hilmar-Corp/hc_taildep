import numpy as np
import pandas as pd

from hc_taildep.margins.ecdf_expanding import pit_ecdf_expanding_midrank


def test_range_and_clipping():
    s = pd.Series([0.0, 0.1, -0.2, 0.3, 0.0, 0.05], index=pd.date_range("2020-01-01", periods=6, freq="D"), name="X")
    u, ur = pit_ecdf_expanding_midrank(s, min_history=2, epsilon=1e-6, start_index=0)
    u_valid = u.dropna().to_numpy()
    assert np.all(u_valid >= 1e-6) and np.all(u_valid <= 1.0 - 1e-6)


def test_no_future_leakage_toy():
    idx = pd.date_range("2020-01-01", periods=8, freq="D")
    s1 = pd.Series([0.0, 0.1, -0.2, 0.3, 0.0, 0.05, 0.2, -0.1], index=idx, name="X")
    u1, _ = pit_ecdf_expanding_midrank(s1, min_history=3, epsilon=1e-6, start_index=0)

    # Change a future value (last point)
    s2 = s1.copy()
    s2.iloc[-1] = 9.99
    u2, _ = pit_ecdf_expanding_midrank(s2, min_history=3, epsilon=1e-6, start_index=0)

    # All u except the last should match exactly
    assert np.allclose(u1.iloc[:-1].to_numpy(), u2.iloc[:-1].to_numpy(), equal_nan=True)


def test_determinism_small():
    idx = pd.date_range("2020-01-01", periods=10, freq="D")
    s = pd.Series(np.linspace(-1, 1, 10), index=idx, name="X")
    u1, _ = pit_ecdf_expanding_midrank(s, min_history=4, epsilon=1e-6, start_index=0)
    u2, _ = pit_ecdf_expanding_midrank(s, min_history=4, epsilon=1e-6, start_index=0)
    assert np.allclose(u1.to_numpy(), u2.to_numpy(), equal_nan=True)