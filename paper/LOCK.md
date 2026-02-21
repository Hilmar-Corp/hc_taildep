# Paper lock (camera-ready)

Command:
  python -m paper.make_paper --spec paper/paper_spec.yaml

Output dir:
  paper/out/hc_taildep_v0_camera_ready/

Manifest:
  paper/out/hc_taildep_v0_camera_ready/manifest.json

Manifest sha256:
  9be6a83c734b879ab1ef2b6cd60f794c48b013077f2811db630158f5897fc66e

Run sources (paper_spec.yaml):
  - J6: data/processed/ds_v0_btceth_4h_binance_closeutc/copulas/markov/j6_ms2_gold_4h_refit240_mineffT120
  - J7: data/processed/ds_v0_btceth_4h_binance_closeutc/impact/j7_var_es/j7_var_es_4h_sensitivity_N20000
  - J8 top8: data/processed/ds_v0_top8_daily_binance_closeutc/impact/j8_top8_pairwise/j8_top8_pairwise_daily_refit40_N10000_ms_spot6
  - J8 asym: data/processed/ds_v0_btceth_daily_binance_closeutc/impact/j8_asym_copulas/j8_asym_btceth_daily_refit40_N20000

Stable lock hash (manifest without created_utc/manifest_sha256 and without git dirty/branch):
  <PASTE_OUTPUT_OF_hash_manifest_stable.py_HERE>

Stable lock hash (manifest normalized; ignores created_utc + manifest_sha256; ignores git.dirty + git.branch):
  d50ccc0bf734aa71e667b3ec40328696fa7be46101027b6c6e18b315494a5623
