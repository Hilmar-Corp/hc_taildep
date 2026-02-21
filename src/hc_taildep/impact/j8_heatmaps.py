# src/hc_taildep/impact/j8_heatmaps.py
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def mat_from_pairs(assets: List[str], values: Dict[Tuple[str, str], float]) -> pd.DataFrame:
    mat = pd.DataFrame(np.nan, index=assets, columns=assets, dtype=float)
    for a in assets:
        mat.loc[a, a] = 0.0
    for (x, y), v in values.items():
        mat.loc[x, y] = v
        mat.loc[y, x] = v
    return mat


def save_heatmap_png(mat: pd.DataFrame, out_path: Path, title: str) -> None:
    ensure_dir(out_path.parent)
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111)
    im = ax.imshow(mat.to_numpy(dtype=float), aspect="auto")
    ax.set_title(title)
    ax.set_xticks(range(len(mat.columns)))
    ax.set_yticks(range(len(mat.index)))
    ax.set_xticklabels(mat.columns, rotation=45, ha="right")
    ax.set_yticklabels(mat.index)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    