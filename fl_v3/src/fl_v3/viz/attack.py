"""V5 attack renderers (fl_v3 T5) — into the ``attack`` viz stage.

The SPEC V5 pre-trust gate ("trigger visualized before attack results trusted"): the trigger
placement must agree with the projected target before the ASR is trusted. Panels:

  * **original vs triggered** camera image (the chosen camera) + the projected target box + the
    projected enclosed LiDAR points + the patch rect (placement-vs-target agreement, the objective
    ``≤center_max_px`` check is reported in the title);
  * the **trigger mask**;
  * the **fused-BEV trigger diff** (``fused_triggered − fused_clean`` on the SAME poisoned model, +
    ``fused_poisoned_triggered − fused_clean`` vs the clean model) localized to the target BEV cell —
    from ``detector.forward(return_intermediates=True)`` on the SAME decode the ASR consumes (one
    decode, two consumers).

Reporting, not a gate (like V4) — never fails the eval run; mirrors ``viz/calibration.py`` (Agg,
``sample_token``-stable names); does not rewrite ``writer.py``.
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Polygon, Rectangle  # noqa: E402

from fl_v3.data.nuscenes import transforms as T
from fl_v3.viz.calibration import box7_to_corners, _draw_box_2d
from fl_v3.attacks import trigger as TR
from fl_v3.models.fusion.bev_grid import BEVConfig, points_to_fine_grid

_DPI = 100


def render_v5_image(writer, sample_clean: Dict[str, object], placement: TR.Placement,
                    spec: TR.TriggerSpec, target_box=None, name: str = "") -> str:
    """Original-vs-triggered camera image + trigger mask + placement/target agreement (one figure)."""
    triggered = TR.apply_trigger(sample_clean, placement, spec)
    cam = int(placement.cam)
    img0 = sample_clean["images"][cam].cpu().numpy().transpose(1, 2, 0)
    img1 = triggered["images"][cam].cpu().numpy().transpose(1, 2, 0)
    Himg, Wimg = img0.shape[:2]
    l2i = sample_clean["lidar2img"].cpu().numpy()[cam]
    x0, y0, x1, y1 = placement.patch_box
    off = float(np.hypot(placement.cx_px - placement.box_center_uv[0],
                         placement.cy_px - placement.box_center_uv[1]))

    fig, axes = plt.subplots(1, 3, figsize=(18, 4.6), dpi=_DPI)
    for ax, im, ttl in zip(axes[:2], (img0, img1), ("original", "triggered")):
        ax.imshow(im.astype(np.uint8))
        if target_box is not None:
            uv, depth = T.project_to_image(box7_to_corners(np.asarray(target_box, np.float64)), l2i)
            if np.any(depth > spec.depth_min):
                _draw_box_2d(ax, uv, depth, color="lime", lw=1.2)
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor="red", linewidth=1.6))
        ax.plot(placement.box_center_uv[0], placement.box_center_uv[1], "y+", markersize=10)
        ax.set_xlim(0, Wimg); ax.set_ylim(Himg, 0); ax.axis("off"); ax.set_title(ttl, fontsize=9)
    # trigger mask
    mask = np.zeros((Himg, Wimg), np.uint8); mask[y0:y1, x0:x1] = 255
    axes[2].imshow(mask, cmap="gray"); axes[2].set_title("trigger mask", fontsize=9); axes[2].axis("off")
    fig.suptitle(f"V5 attack | {sample_clean['sample_token']} | cam{cam} | aligned={placement.aligned} "
                 f"src={placement.source} | ctr↔box {off:.1f}px (≤{spec.center_max_px:.0f}) | "
                 f"area={placement.patch_area}/{placement.box2d_area:.0f}={placement.patch_area/max(1,placement.box2d_area):.2f}",
                 fontsize=9)
    path = writer.figure_path("attack", f"v5_image__{name or sample_clean['sample_token']}")
    fig.savefig(path, bbox_inches="tight", pad_inches=0.1); plt.close(fig)
    return path


def _bev_norm(bev: torch.Tensor) -> np.ndarray:
    return bev.detach().float().norm(dim=0).cpu().numpy()


@torch.no_grad()
def render_v5_fusion_diff(writer, poisoned_model, sample_clean, placement, spec: TR.TriggerSpec,
                          cfg: BEVConfig, target_box=None, clean_model=None, name: str = "") -> str:
    """``fused_triggered − fused_clean`` (poisoned model) + ``fused_poisoned_triggered − fused_clean``
    (vs the clean model) BEV diffs, with the target BEV cell marked (localization gate)."""
    from fl_v3.attacks.fusion_ablation import _collate_one
    device = next(poisoned_model.parameters()).device
    triggered = TR.apply_trigger(sample_clean, placement, spec)
    b_clean = _collate_one(sample_clean, device)
    b_trig = _collate_one(triggered, device)
    fused_clean_p = poisoned_model(b_clean, return_intermediates=True)["_fused_bev"][0]
    fused_trig_p = poisoned_model(b_trig, return_intermediates=True)["_fused_bev"][0]
    diff_pp = _bev_norm(fused_trig_p - fused_clean_p)
    panels = [(diff_pp, "‖fused_trig − fused_clean‖ (poisoned)")]
    if clean_model is not None:
        fused_clean_c = clean_model(b_clean, return_intermediates=True)["_fused_bev"][0]
        diff_pc = _bev_norm(fused_trig_p - fused_clean_c)
        panels.append((diff_pc, "‖fused_pois_trig − fused_clean(clean-model)‖"))

    # target BEV cell (fine grid)
    tc = None
    if target_box is not None:
        col, row = points_to_fine_grid(torch.as_tensor(np.asarray(target_box)[:2]).reshape(1, 2), cfg)
        tc = (int(col[0]), int(row[0]))

    fig, axes = plt.subplots(1, len(panels), figsize=(6.2 * len(panels), 5.4), dpi=_DPI, squeeze=False)
    for ax, (grid, ttl) in zip(axes[0], panels):
        im = ax.imshow(grid, origin="lower", cmap="inferno", aspect="equal")
        if tc is not None:
            ax.scatter([tc[0]], [tc[1]], s=80, facecolors="none", edgecolors="cyan", linewidths=1.8)
        ax.set_title(ttl, fontsize=8); fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle(f"V5 fused-BEV trigger diff | {sample_clean['sample_token']} | target cell=cyan○", fontsize=9)
    path = writer.figure_path("attack", f"v5_fusion_diff__{name or sample_clean['sample_token']}")
    fig.savefig(path, bbox_inches="tight", pad_inches=0.1); plt.close(fig)
    return path
