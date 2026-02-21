from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import requests
import yaml

from hc_taildep.utils.hashing import sha256_bytes, sha256_file
from hc_taildep.utils.paths import ensure_dir


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def dump_json(path: Path, obj: Dict[str, Any]) -> None:
    path.write_text(json.dumps(obj, indent=2, sort_keys=False), encoding="utf-8")


def dump_text(path: Path, s: str) -> None:
    path.write_text(s.rstrip() + "\n", encoding="utf-8")


def _binance_fetch_klines(
    base_url: str,
    endpoint: str,
    symbol: str,
    interval: str,
    start_ms: Optional[int],
    end_ms: Optional[int],
    limit: int = 1000,
) -> List[List[Any]]:
    params: Dict[str, Any] = {"symbol": symbol, "interval": interval, "limit": limit}
    if start_ms is not None:
        params["startTime"] = int(start_ms)
    if end_ms is not None:
        params["endTime"] = int(end_ms)

    url = base_url.rstrip("/") + endpoint
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    return r.json()


def download_binance_1d_close(
    base_url: str,
    endpoint: str,
    interval: str,
    symbol: str,
    start_date: str,
    end_date: Optional[str],
    raw_path: Path,
) -> Tuple[pd.Series, Dict[str, Any]]:
    """
    Returns a pd.Series of close prices indexed by UTC date (YYYY-MM-DD) for Binance 1d klines.
    Caches raw JSONL in raw_path (append-only rebuild, but deterministic if same time window).
    """
    ensure_dir(raw_path.parent)

    # Binance klines open time is ms since epoch UTC.
    # We'll paginate forward from start_date.
    start_ts = pd.Timestamp(start_date, tz="UTC")
    start_ms = int(start_ts.value // 10**6)

    end_ms = None
    if end_date is not None:
        # inclusive end-of-day UTC: end_date + 1 day - 1 ms
        end_ts = pd.Timestamp(end_date, tz="UTC") + pd.Timedelta(days=1) - pd.Timedelta(milliseconds=1)
        end_ms = int(end_ts.value // 10**6)

    all_rows: List[List[Any]] = []
    cur = start_ms
    loops = 0

    while True:
        loops += 1
        rows = _binance_fetch_klines(
            base_url=base_url,
            endpoint=endpoint,
            symbol=symbol,
            interval=interval,
            start_ms=cur,
            end_ms=end_ms,
            limit=1000,
        )
        if not rows:
            break

        all_rows.extend(rows)

        last_open_time_ms = int(rows[-1][0])
        next_cur = last_open_time_ms + 24 * 60 * 60 * 1000  # next day
        if next_cur <= cur:
            break
        cur = next_cur

        # stop if we are past requested end
        if end_ms is not None and cur > end_ms:
            break

        # safety to avoid infinite loops
        if loops > 5000:
            raise RuntimeError("Binance pagination exceeded safety limit.")

        # Binance sometimes returns last partial day; we keep and handle by end_date / alignment later.

    # Write raw cache deterministically (overwrite is acceptable for dataset build; dataset_version bump handles change)
    # JSONL lines: each row list
    with raw_path.open("w", encoding="utf-8") as f:
        for row in all_rows:
            f.write(json.dumps(row) + "\n")

    # Parse closes
    # Row format: [ open_time, open, high, low, close, volume, close_time, ... ]
    dates = []
    closes = []
    for row in all_rows:
        open_time_ms = int(row[0])
        close_str = row[4]
        dt = pd.Timestamp(open_time_ms, unit="ms", tz="UTC").date()  # date in UTC
        dates.append(pd.Timestamp(dt))
        closes.append(float(close_str))

    s = pd.Series(closes, index=pd.DatetimeIndex(dates, tz=None), name=symbol).sort_index()
    s = s[~s.index.duplicated(keep="last")]

    # strict date window
    s = s[s.index >= pd.Timestamp(start_date)]
    if end_date is not None:
        s = s[s.index <= pd.Timestamp(end_date)]

    meta = {
        "symbol": symbol,
        "interval": interval,
        "n_rows_raw": int(len(all_rows)),
        "n_rows_unique_dates": int(s.shape[0]),
        "first_date": str(s.index.min().date()) if len(s) else None,
        "last_date": str(s.index.max().date()) if len(s) else None,
    }
    return s, meta


def compute_log_returns(close: pd.Series) -> pd.Series:
    # close indexed by date (naive date index)
    r = np.log(close / close.shift(1))
    return r


def strict_intersection(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    before = df.shape[0]
    df2 = df.dropna(axis=0, how="any")
    after = df2.shape[0]
    removed = before - after
    info = {"rows_before": int(before), "rows_after": int(after), "rows_removed_due_to_missing": int(removed)}
    return df2, info


def detect_calendar_gaps(idx: pd.DatetimeIndex) -> Dict[str, Any]:
    if idx.empty:
        return {"has_gaps": False, "n_gaps": 0, "gaps": []}
    # Expect daily steps of 1 day
    diffs = idx.to_series().diff().dropna()
    gaps = diffs[diffs > pd.Timedelta(days=1)]
    out = []
    for t, dt in gaps.items():
        out.append({"date": str(t.date()), "gap_days": int(dt.days)})
    return {"has_gaps": bool(len(out) > 0), "n_gaps": int(len(out)), "gaps": out}


def build_splits(idx: pd.DatetimeIndex, min_train_days: int, first_oos_date: Optional[str], last_oos_date: Optional[str]) -> Dict[str, Any]:
    if idx.empty:
        raise ValueError("Empty index; cannot build splits.")
    train_start = str(idx.min().date())

    if first_oos_date is None:
        # first OOS after min_train_days of history (counting rows)
        if len(idx) <= min_train_days:
            raise ValueError("Not enough data for min_train_days.")
        first_oos = idx[min_train_days]
        first_oos_date = str(first_oos.date())

    if last_oos_date is None:
        last_oos_date = str(idx.max().date())

    # Basic sanity
    ts = pd.Timestamp(train_start)
    fo = pd.Timestamp(first_oos_date)
    lo = pd.Timestamp(last_oos_date)
    if not (ts < fo <= lo):
        raise ValueError("Invalid split dates ordering.")

    return {
        "train_start": train_start,
        "first_oos": first_oos_date,
        "last_oos": last_oos_date,
        "min_train_days": int(min_train_days),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=str)
    args = ap.parse_args()

    t0 = time.time()
    config_path = Path(args.config).resolve()
    cfg = load_yaml(config_path)

    ds = cfg["dataset"]
    paths = cfg["paths"]
    splits_cfg = cfg["splits"]

    dataset_version = ds["dataset_version"]
    processed_root = Path(paths["processed_root"])
    out_dir = processed_root / dataset_version
    ensure_dir(out_dir)

    raw_dir = Path(paths["raw_dir"])
    ensure_dir(raw_dir)

    # Download BTC/ETH closes
    src = ds["source"]
    interval = src["interval"]
    base_url = src["base_url"]
    endpoint = src["endpoint"]
    symbols = src["symbols"]
    start_date = ds["start_date"]
    end_date = ds["end_date"]

    # Freeze end_date to last fully closed UTC day if not provided (avoid mutable latest candle)
    end_date_effective = end_date
    if end_date_effective is None:
        end_date_effective = str((pd.Timestamp.now("UTC").floor("D") - pd.Timedelta(days=1)).date())

    series = {}
    raw_meta = {}
    raw_hashes = {}

    for asset, sym in symbols.items():
        raw_path = raw_dir / f"{sym}_{interval}_{start_date}_to_{end_date_effective}.jsonl"
        s_close, meta = download_binance_1d_close(
            base_url=base_url,
            endpoint=endpoint,
            interval=interval,
            symbol=sym,
            start_date=start_date,
            end_date=end_date_effective,
            raw_path=raw_path,
        )
        series[asset] = s_close.rename(asset)
        raw_meta[asset] = meta
        raw_hashes[asset] = sha256_file(raw_path)

    close_df = pd.concat([series["BTC"], series["ETH"]], axis=1).sort_index()

    # Compute log returns
    ret_df = close_df.apply(compute_log_returns)

    if ds.get("alignment", {}).get("drop_first_return", True):
        ret_df = ret_df.iloc[1:]

    # Strict intersection
    ret_df, inter_info = strict_intersection(ret_df)

    # Calendar checks
    idx = pd.DatetimeIndex(ret_df.index)
    cal = detect_calendar_gaps(idx)

    # Basic stats
    stats = {
        "n_obs": int(ret_df.shape[0]),
        "start_date": str(idx.min().date()),
        "end_date": str(idx.max().date()),
        "mean": {c: float(ret_df[c].mean()) for c in ret_df.columns},
        "std": {c: float(ret_df[c].std()) for c in ret_df.columns},
        "min": {c: float(ret_df[c].min()) for c in ret_df.columns},
        "max": {c: float(ret_df[c].max()) for c in ret_df.columns},
    }

    # Write returns (Parquet if possible else CSV)
    returns_parquet = out_dir / "returns.parquet"
    returns_csv = out_dir / "returns.csv"
    wrote = None
    try:
        ret_df.to_parquet(returns_parquet)
        wrote = returns_parquet
    except ImportError:
        ret_df.to_csv(returns_csv, index=True)
        wrote = returns_csv

    dataset_hash = sha256_bytes(wrote.read_bytes())

    # Splits
    split_obj = build_splits(
        idx=idx,
        min_train_days=int(splits_cfg["min_train_days"]),
        first_oos_date=splits_cfg.get("first_oos_date"),
        last_oos_date=splits_cfg.get("last_oos_date"),
    )
    split_obj["oos_convention"] = splits_cfg["oos_convention"]
    split_obj["dataset_version"] = dataset_version
    split_obj["dataset_hash_sha256"] = dataset_hash

    dump_json(out_dir / "splits.json", split_obj)

    # Dataset card
    card = []
    card.append("# Dataset Card — BTC/ETH Daily (Canonique)")
    card.append("")
    card.append("## Scope")
    card.append("- Core universe: BTC, ETH only")
    card.append("- Frequency: daily close-to-close (UTC)")
    card.append("- Returns: log-returns computed from close prices")
    card.append("- Alignment: strict intersection (keep only dates present for both assets)")
    card.append("")
    card.append("## Source")
    card.append(f"- Provider: Binance Spot REST API")
    card.append(f"- Endpoint: {src['base_url']}{src['endpoint']}")
    card.append(f"- Interval: {interval}")
    card.append(f"- Symbols (proxy via USDT): BTC={symbols['BTC']}, ETH={symbols['ETH']}")
    card.append("")
    card.append("## Conventions")
    card.append("- Timestamping: kline open time converted to UTC date; daily index uses that UTC date.")
    card.append("- Price field: close")
    card.append("- Returns: r_t = log(close_t / close_{t-1})")
    card.append("")
    card.append("## Cleaning / Missing policy")
    card.append(f"- Drop first return row after shift: {ds.get('alignment', {}).get('drop_first_return', True)}")
    card.append("- No forward-fill; strict intersection across BTC & ETH")
    card.append(f"- Rows removed due to missing after intersection: {inter_info['rows_removed_due_to_missing']}")
    card.append("")
    card.append("## Calendar diagnostics")
    card.append(f"- Has gaps (>1 day step): {cal['has_gaps']}")
    card.append(f"- Number of gaps: {cal['n_gaps']}")
    if cal["has_gaps"]:
        card.append("- Gaps (date, gap_days):")
        for g in cal["gaps"][:50]:
            card.append(f"  - {g['date']}: {g['gap_days']}")
        if len(cal["gaps"]) > 50:
            card.append("  - ... (truncated)")
    card.append("")
    card.append("## Summary stats (returns)")
    card.append(f"- n_obs: {stats['n_obs']}")
    card.append(f"- start_date: {stats['start_date']}")
    card.append(f"- end_date: {stats['end_date']}")
    dump_text(out_dir / "dataset_card.md", "\n".join(card))

    # Provenance (dataset-level)
    prov = {
        "dataset_version": dataset_version,
        "created_utc": utc_now_iso(),
        "config_path": str(config_path),
        "env": {
            "python_version": sys.version.split()[0],
            "platform": platform.platform(),
        },
        "source": {
            "name": src["name"],
            "base_url": base_url,
            "endpoint": endpoint,
            "interval": interval,
            "symbols": symbols,
            "start_date": start_date,
            "end_date": end_date_effective,
        },
        "raw": {
            "hashes_sha256": raw_hashes,
            "meta": raw_meta,
        },
        "processing": {
            "alignment": ds["alignment"],
            "intersection_info": inter_info,
            "calendar": cal,
        },
        "outputs": {
            "returns_path": str(wrote),
            "returns_format": "parquet" if wrote.suffix == ".parquet" else "csv",
            "dataset_hash_sha256": dataset_hash,
            "splits_path": str(out_dir / "splits.json"),
            "dataset_card_path": str(out_dir / "dataset_card.md"),
        },
        "runtime_sec": float(time.time() - t0),
    }
    dump_json(out_dir / "provenance.json", prov)

    print(f"[OK] built dataset: {dataset_version}")
    print(f"[OK] out_dir: {out_dir.resolve()}")
    print(f"[OK] returns: {wrote.name} hash={dataset_hash[:12]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())