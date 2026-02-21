import json
from pathlib import Path
import subprocess
import sys


def test_j5_runner_deterministic(tmp_path):
    # Run twice and compare predictions hash in metrics.json
    cfg = Path("configs/copulas/btceth_j5_gating.yaml")
    cmd = [sys.executable, "-m", "hc_taildep.build_copula_conditional_j5", "--config", str(cfg)]

    subprocess.check_call(cmd)
    out_dir = Path("data/processed/ds_v0_btceth_daily_binance_closeutc/copulas/conditional/j5_gating")
    m1 = json.loads((out_dir / "metrics.json").read_text())
    h1 = m1["hashes"]["predictions_csv"]

    subprocess.check_call(cmd)
    m2 = json.loads((out_dir / "metrics.json").read_text())
    h2 = m2["hashes"]["predictions_csv"]

    assert h1 == h2
