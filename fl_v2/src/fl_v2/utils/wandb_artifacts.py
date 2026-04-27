"""One-shot artifact renderers for wandb (and as fallback PNGs in the exp dir).

These are intentionally cheap — single matplotlib calls — so they can run on
the server process without slowing training. The PNGs they produce are also
useful as standalone files in the experiment directory regardless of whether
wandb is enabled.
"""
from __future__ import annotations

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def render_label_histogram_panel(json_path: str, num_classes: int) -> str | None:
    """Render a client × class count heatmap and write it next to the JSON.

    Returns the PNG path, or None on any failure (caller treats None as "no
    image to upload").

    Layout:
      rows    = client ids (sorted ascending)
      columns = class labels [0, num_classes)
      color   = sample count

    Output path: ``<json_path without .json> + ".png"``.
    """
    if not os.path.exists(json_path):
        return None
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"[wandb_artifacts] failed to read {json_path}: {e!r}", flush=True)
        return None

    if not isinstance(data, dict) or not data:
        return None

    try:
        client_ids = sorted(int(k) for k in data.keys())
    except Exception:
        client_ids = sorted(data.keys())  # type: ignore[arg-type]

    matrix = np.zeros((len(client_ids), num_classes), dtype=np.int64)
    for row, cid in enumerate(client_ids):
        bucket = data[str(cid)] if str(cid) in data else data[cid]  # type: ignore[index]
        for k, v in bucket.items():
            try:
                col = int(k)
                if 0 <= col < num_classes:
                    matrix[row, col] = int(v)
            except Exception:
                continue

    fig_h = max(4.0, 0.18 * len(client_ids))
    fig_w = max(6.0, 0.18 * num_classes)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=110)
    im = ax.imshow(matrix, aspect="auto", cmap="viridis", interpolation="nearest")
    ax.set_xlabel("class label")
    ax.set_ylabel("client id")
    ax.set_title(
        f"Client × class sample counts ({len(client_ids)} clients × {num_classes} classes)"
    )
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("sample count")
    fig.tight_layout()

    out_path = os.path.splitext(json_path)[0] + ".png"
    try:
        fig.savefig(out_path)
    except Exception as e:
        print(f"[wandb_artifacts] failed to save {out_path}: {e!r}", flush=True)
        plt.close(fig)
        return None
    plt.close(fig)
    return out_path
