"""V3 fusion-feature renderers (clean) (fl_v3 T2) — into the ``fusion`` viz stage.

The clean V3 panels (the SPEC V3 gate: "fusion is active + localized"): ``camera_BEV_norm``,
``lidar_BEV_norm``, ``fused_BEV_norm``, plus the **side-by-side cam/LiDAR/fused
BEV-alignment overlay** — the same GT centers (single :mod:`bev_grid` convention) drawn
on all three so the reviewer can see camera-BEV and LiDAR-BEV land on the same objects
and the fusion responds there. *(The trigger-diff V3 panels — ``fused_triggered −
fused_clean`` etc. — are T5.)* Mirrors ``viz/calibration.py`` (Agg, sample_token-stable
names); does not rewrite ``writer.py``."""
from __future__ import annotations

from typing import List

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from fl_v3.models.fusion.bev_grid import BEVConfig, points_to_fine_grid


def _norm(bev: torch.Tensor) -> np.ndarray:
    return bev.detach().float().norm(dim=0).cpu().numpy()


def _gt_cells(boxes: torch.Tensor, cfg: BEVConfig):
    if boxes.numel() == 0:
        return np.zeros(0), np.zeros(0)
    col, row = points_to_fine_grid(boxes[:, :2].detach().cpu(), cfg)
    return col.numpy(), row.numpy()


@torch.no_grad()
def render_v3(writer, model, batch, cfg: BEVConfig, max_samples: int = 3) -> List[str]:
    """Clean V3 cam/LiDAR/fused BEV-norm alignment overlay for ``max_samples`` samples."""
    model.eval()
    out = model(batch, return_intermediates=True)
    paths: List[str] = []
    n = min(max_samples, len(batch["gt_boxes"]))
    for b in range(n):
        tok = batch["sample_token"][b]
        cam = _norm(out["_camera_bev"][b])
        lid = _norm(out["_lidar_bev"][b])
        fus = _norm(out["_fused_bev"][b])
        gc, gr = _gt_cells(batch["gt_boxes"][b], cfg)
        fig, axes = plt.subplots(1, 3, figsize=(13, 4.4), dpi=100)
        for ax, grid, name, cmap in zip(
            axes, (cam, lid, fus), ("camera_BEV_norm", "lidar_BEV_norm", "fused_BEV_norm"),
            ("viridis", "magma", "inferno"),
        ):
            ax.imshow(grid, origin="lower", cmap=cmap, aspect="equal")
            if len(gc) > 0:
                ax.scatter(gc, gr, s=22, facecolors="none", edgecolors="red", linewidths=1.0)
            ax.set_title(name, fontsize=8)
            ax.set_xlabel("col=x bin", fontsize=6); ax.set_ylabel("row=y bin", fontsize=6)
        fig.suptitle(f"V3 fusion BEV alignment (clean) | {tok} | GT=red○", fontsize=9)
        path = writer.figure_path("fusion", f"fusion_bev__{tok}")
        fig.savefig(path, bbox_inches="tight", pad_inches=0.1); plt.close(fig)
        paths.append(path)
    return paths


@torch.no_grad()
def render_v3_trigger(writer, model, sample_clean, placement, spec, cfg: BEVConfig,
                      target_box=None, name: str = "") -> str:
    """V3(trigger) (T5): clean-vs-triggered ``camera/lidar/fused`` BEV norms + the fused trigger diff,
    so the reviewer sees the trigger is fusion-relevant + localized to the target BEV cell."""
    from fl_v3.attacks import trigger as TR
    from fl_v3.attacks.fusion_ablation import _collate_one
    from fl_v3.models.fusion.bev_grid import points_to_fine_grid

    device = next(model.parameters()).device
    triggered = TR.apply_trigger(sample_clean, placement, spec)
    oc = model(_collate_one(sample_clean, device), return_intermediates=True)
    ot = model(_collate_one(triggered, device), return_intermediates=True)
    cam_c, lid_c, fus_c = _norm(oc["_camera_bev"][0]), _norm(oc["_lidar_bev"][0]), _norm(oc["_fused_bev"][0])
    fus_t = _norm(ot["_fused_bev"][0])
    diff = _norm(ot["_fused_bev"][0] - oc["_fused_bev"][0])
    tc = None
    if target_box is not None:
        col, row = points_to_fine_grid(torch.as_tensor(np.asarray(target_box)[:2]).reshape(1, 2), cfg)
        tc = (int(col[0]), int(row[0]))
    panels = [(cam_c, "camera_BEV (clean)", "viridis"), (lid_c, "lidar_BEV (clean)", "magma"),
              (fus_c, "fused_BEV (clean)", "inferno"), (fus_t, "fused_BEV (triggered)", "inferno"),
              (diff, "‖fused_trig − fused_clean‖", "inferno")]
    fig, axes = plt.subplots(1, 5, figsize=(22, 4.4), dpi=100)
    for ax, (grid, ttl, cmap) in zip(axes, panels):
        ax.imshow(grid, origin="lower", cmap=cmap, aspect="equal")
        if tc is not None:
            ax.scatter([tc[0]], [tc[1]], s=60, facecolors="none", edgecolors="red", linewidths=1.4)
        ax.set_title(ttl, fontsize=8)
    fig.suptitle(f"V3(trigger) | {sample_clean['sample_token']} | target cell=red○ | aligned={placement.aligned}", fontsize=9)
    path = writer.figure_path("fusion", f"fusion_trigger__{name or sample_clean['sample_token']}")
    fig.savefig(path, bbox_inches="tight", pad_inches=0.1); plt.close(fig)
    return path
