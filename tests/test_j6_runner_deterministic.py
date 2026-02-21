import json
import subprocess
import sys
from pathlib import Path


def test_j6_runner_deterministic(tmp_path):
    cfg = Path("configs/copulas/btceth_j6_ms.yaml")
    cmd = [sys.executable, "-m", "hc_taildep.build_copula_markov_j6", "--config", str(cfg)]

    subprocess.check_call(cmd)
    out_dir = Path("data/processed/ds_v0_btceth_daily_binance_closeutc/copulas/markov/j6_ms2")
    m1 = json.loads((out_dir / "metrics.json").read_text())
    h1 = m1["hashes"]["predictions_csv"]

    subprocess.check_call(cmd)
    m2 = json.loads((out_dir / "metrics.json").read_text())
    h2 = m2["hashes"]["predictions_csv"]

    assert h1 == h2