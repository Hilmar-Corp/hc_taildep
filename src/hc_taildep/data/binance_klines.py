from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Iterable, Optional

import pandas as pd
import requests


BINANCE_BASE = "https://api.binance.com"


@dataclass(frozen=True)
class KlineRequest:
    symbol: str          # e.g. "BTCUSDT"
    interval: str        # "1d", "4h"
    start_time_ms: int   # inclusive
    end_time_ms: int     # inclusive (best-effort)
    limit: int = 1000


def _get(url: str, params: dict, timeout: int = 30) -> list:
    r = requests.get(url, params=params, timeout=timeout)
    r.raise_for_status()
    return r.json()


def fetch_klines(req: KlineRequest, *, sleep_s: float = 0.25) -> pd.DataFrame:
    """
    Fetch klines with pagination. Returns columns:
      open_time, open, high, low, close, volume, close_time, quote_asset_volume, trades, taker_buy_base, taker_buy_quote
    All times are ms since epoch.
    """
    url = f"{BINANCE_BASE}/api/v3/klines"

    rows = []
    start = req.start_time_ms
    while True:
        params = {
            "symbol": req.symbol,
            "interval": req.interval,
            "startTime": start,
            "endTime": req.end_time_ms,
            "limit": req.limit,
        }
        data = _get(url, params)
        if not data:
            break

        rows.extend(data)
        last_open_time = int(data[-1][0])
        next_start = last_open_time + 1

        if next_start <= start:
            break
        start = next_start

        if last_open_time >= req.end_time_ms:
            break

        time.sleep(sleep_s)

        # Stop if returned less than limit => reached end
        if len(data) < req.limit:
            break

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(
        rows,
        columns=[
            "open_time",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "close_time",
            "quote_asset_volume",
            "trades",
            "taker_buy_base",
            "taker_buy_quote",
            "ignore",
        ],
    )

    # numeric
    for c in ["open", "high", "low", "close", "volume", "quote_asset_volume", "taker_buy_base", "taker_buy_quote"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["open_time"] = df["open_time"].astype("int64")
    df["close_time"] = df["close_time"].astype("int64")
    df["trades"] = pd.to_numeric(df["trades"], errors="coerce").astype("Int64")

    df = df.drop(columns=["ignore"])
    return df


def ms_from_utc(ts: str) -> int:
    # ts like "2018-01-01T00:00:00Z" or "2018-01-01"
    t = pd.to_datetime(ts, utc=True)
    return int(t.value // 1_000_000)


def to_utc_index_from_close_time(df: pd.DataFrame, *, freq: str) -> pd.DatetimeIndex:
    """Build a UTC DatetimeIndex aligned to the candle *close* time.

    Notes:
      - `pd.to_datetime(..., utc=True)` returns a Series for a DataFrame column.
      - Series needs `.dt.floor(...)` (not `.floor(...)`).
      - We accept Binance-style frequency strings like "1d", "4h".
    """
    # close_time is ms since epoch
    t = pd.to_datetime(df["close_time"], unit="ms", utc=True, errors="coerce")

    # Normalize common Binance interval strings to pandas offset aliases
    f = str(freq).strip().lower()
    if f in {"1d", "1day", "d"}:
        floor_rule = "D"
    elif f.endswith("h") and f[:-1].isdigit():
        # pandas >= 3 prefers lowercase offset aliases ("h" not "H")
        floor_rule = f[:-1] + "h"  # e.g. "4h" -> "4h"
    elif f.endswith("m") and f[:-1].isdigit():
        # pandas >= 3 prefers "min" instead of legacy "T"
        floor_rule = f[:-1] + "min"  # e.g. "15m" -> "15min"
    else:
        # Fall back to whatever pandas can parse (best-effort)
        floor_rule = f

    # `t` is a Series -> floor via `.dt`
    t = t.dt.floor(floor_rule)

    # Convert to DatetimeIndex
    return pd.DatetimeIndex(t)