# hc_taildep_v0_camera_ready — camera-ready pack

# Checklist anti-boulette (reviewer-facing)

- Scope: pas de trading, pas d’alpha, pas de “predict direction”.
- OOS strict: à t, tout fit utilise uniquement ≤ t-1 ; scoring sur t (one-step-ahead).
- PIT: jamais de fit global full-sample (u_t = F_{t-1}(r_t)).
- Repro: seeds + hashes + dataset_version + commit ; outputs hashés.
- Baselines d’abord: indep/gauss/t statique avant dynamique.
- Si un modèle complexe gagne: DM (HAC/NW) + robustness ; sinon annexe.

## What this command produced

- figures/: camera-ready figures (rebuilt from CSV/tables, not copied)
- tables/: camera-ready tables (stable sort + rounding)
- manifest.json: hashes + sources + git info

## Runs (sources)

- [j6] core_j6  →  `data/processed/ds_v0_btceth_4h_binance_closeutc/copulas/markov/j6_ms2_gold_4h_refit240_mineffT120`
- [j7] core_j7  →  `data/processed/ds_v0_btceth_4h_binance_closeutc/impact/j7_var_es/j7_var_es_4h_sensitivity_N20000`
- [j8_top8] annex_j8_top8  →  `data/processed/ds_v0_top8_daily_binance_closeutc/impact/j8_top8_pairwise/j8_top8_pairwise_daily_refit40_N10000_ms_spot6`
- [j8_asym] annex_j8_asym  →  `data/processed/ds_v0_btceth_daily_binance_closeutc/impact/j8_asym_copulas/j8_asym_btceth_daily_refit40_N20000`

## Reproduce

```bash
python -m paper.make_paper --spec paper/paper_spec.yaml
```

## Manifest (summary)

- manifest_sha256: `7c88030f4dadc0145d7ec43928e3f6d6beb19282e07ad03beb82cd938116cbae`
- created_utc: `2026-02-23T20:19:00+00:00`
- git: {"branch": "main", "commit": "d50a1929c614711308e0b42eab8bb02ba20a2fba", "dirty": true}
