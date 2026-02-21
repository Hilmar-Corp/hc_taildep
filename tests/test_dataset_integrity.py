import pandas as pd
import numpy as np
from hc_taildep.build_dataset import build_splits, detect_calendar_gaps

def test_calendar_gap_detector():
    idx = pd.date_range("2020-01-01", periods=5, freq="D")
    out = detect_calendar_gaps(pd.DatetimeIndex(idx))
    assert out["has_gaps"] is False

def test_splits_contract_fit_to_t_minus_1():
    idx = pd.date_range("2020-01-01", periods=1000, freq="D")
    s = build_splits(pd.DatetimeIndex(idx), min_train_days=10, first_oos_date=None, last_oos_date=None)
    assert pd.Timestamp(s["train_start"]) < pd.Timestamp(s["first_oos"]) <= pd.Timestamp(s["last_oos"])
    assert s["min_train_days"] == 10