hc_taildep — Tail dependence (bootstrap)

Scope (core): quantify + test OOS tail dependence between crypto-assets; no trading, no alpha.

J0 objective: reproducible run factory (provenance, ids, append-only runs).

Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
python -m hc_taildep.run --config configs/runs/smoke.yaml