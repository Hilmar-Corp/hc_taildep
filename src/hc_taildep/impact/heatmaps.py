# src/hc_taildep/impact/heatmaps.py
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from hc_taildep.utils.io import ensure_dir


def _mat_from_pairs(
    assets: List[str],
    pair_metric: Dict[Tuple[str, str], float],
    fill: float = np.nan,
) -> pd.DataFrame:
    A = assets
    mat = pd.DataFrame(fill, index=A, columns=A, dtype=float)
    for i in range(len(A)):
        mat.iloc[i, i] = 0.0
    for (a, b), v in pair_metric.items():
        mat.loc[a, b] = v
        mat.loc[b, a] = v
    return mat


def save_heatmap_png(mat: pd.DataFrame, out_path: str | Path, title: str) -> None:
    out_path = Path(out_path)
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