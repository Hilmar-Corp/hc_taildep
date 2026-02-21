# src/hc_taildep/data/build_dataset_binance.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from hc_taildep.data.binance_klines import KlineRequest, fetch_klines, ms_from_utc, to_utc_index_from_close_time
from hc_taildep.utils.io import ensure_dir, write_json, write_text


@dataclass(frozen=True)
class DatasetSpec:
    dataset_id: str
    interval: str            # "1d" or "4h"
    symbols: List[str]       # ["BTCUSDT", ...]
    start_utc: str           # "2018-01-01"
    end_utc: str             # "2026-01-01"
    tz: str = "UTC"


def build_dataset_binance_closeutc(
    spec: DatasetSpec,
    out_root: str | Path,
) -> Path:
    """
    Writes:
      <out_root>/<dataset_id>/
        prices.csv  (close prices)
        returns.csv (log returns)
        meta.json
    """
    out_dir = ensure_dir(Path(out_root) / spec.dataset_id)

    start_ms = ms_from_utc(spec.start_utc)
    end_ms = ms_from_utc(spec.end_utc)

    price_frames = []
    for sym in spec.symbols:
        df = fetch_klines(KlineRequest(symbol=sym, interval=spec.interval, start_time_ms=start_ms, end_time_ms=end_ms))
        if df.empty:
            raise RuntimeError(f"No klines returned for {sym} interval={spec.interval}")

        idx = to_utc_index_from_close_time(df, freq=spec.interval)
        s = pd.Series(df["close"].to_numpy(dtype=float), index=idx, name=sym)
        s = s[~s.index.duplicated(keep="last")]
        price_frames.append(s)

    prices = pd.concat(price_frames, axis=1).sort_index()
    # align + drop rows with missing
    prices = prices.dropna(how="any")

    # log returns
    rets = np.log(prices).diff().dropna()

    prices_path = out_dir / "prices.csv"
    rets_path = out_dir / "returns.csv"
    prices.to_csv(prices_path, index_label="ts_utc")
    rets.to_csv(rets_path, index_label="ts_utc")

    meta = {
        "dataset_id": spec.dataset_id,
        "interval": spec.interval,
        "symbols": spec.symbols,
        "start_utc": spec.start_utc,
        "end_utc": spec.end_utc,
        "n_rows_prices": int(prices.shape[0]),
        "n_rows_returns": int(rets.shape[0]),
        "note": "Binance close prices, timestamp derived from kline close_time, normalized to UTC close (daily floored).",
    }
    write_json(out_dir / "meta.json", meta)
    write_text(out_dir / "STATUS.md", f"[OK] built dataset {spec.dataset_id}\n")
    return out_dir