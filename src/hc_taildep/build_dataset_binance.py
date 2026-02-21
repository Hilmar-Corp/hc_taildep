# src/hc_taildep/data/build_dataset_binance.py  (runner CLI addition)
from __future__ import annotations

import argparse
from pathlib import Path

from hc_taildep.data.build_dataset_binance import DatasetSpec, build_dataset_binance_closeutc

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--dataset_id", required=True)
    ap.add_argument("--interval", required=True, choices=["1d", "4h"])
    ap.add_argument("--symbols", required=True, help="comma-separated e.g. BTCUSDT,ETHUSDT")
    ap.add_argument("--start", required=True, help="YYYY-MM-DD")
    ap.add_argument("--end", required=True, help="YYYY-MM-DD")
    args = ap.parse_args()

    spec = DatasetSpec(
        dataset_id=args.dataset_id,
        interval=args.interval,
        symbols=[s.strip() for s in args.symbols.split(",") if s.strip()],
        start_utc=args.start,
        end_utc=args.end,
    )
    out_dir = build_dataset_binance_closeutc(spec, out_root=args.out_root)
    print(f"[OK] dataset built: {out_dir}")

if __name__ == "__main__":
    main()