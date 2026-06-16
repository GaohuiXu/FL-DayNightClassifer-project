"""V2 single-modality encoder renderers (fl_v3 T2) — into the ``encoder`` viz stage.

Makes the per-modality encoder behavior inspectable (the SPEC V2 gate): per-camera
feature-norm heatmaps, the LSS expected-depth map, the camera→BEV norm, pillar
occupancy, the LiDAR-BEV norm, and the response-at-GT overlay. Mirrors the
``viz/calibration.py`` sibling-module pattern (headless ``Agg``, ``sample_token``-stable
filenames, into ``VizWriter``); does NOT rewrite ``writer.py``.

All renderers consume the detector's **intermediate features**
(``forward(batch, return_intermediates=True)``) — the same forward the loss/decode use,
so what is drawn is exactly what the model computes. GT overlays use the single
:mod:`bev_grid` convention (so a misplaced overlay would reveal a convention bug)."""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from fl_v3.models.fusion.bev_grid import BEVConfig, points_to_fine_grid

_FIG_DPI = 100


def _bev_norm(bev: torch.Tensor) -> np.ndarray:
    """``[C,ny,nx] → [ny,nx]`` L2 channel norm (numpy)."""
    return bev.detach().float().norm(dim=0).cpu().numpy()


def _gt_fine_cells(boxes: torch.Tensor, cfg: BEVConfig):
    """GT box centers → fine-grid ``(col, row)`` for overlay (shared convention)."""
    if boxes.numel() == 0:
        return np.zeros(0), np.zeros(0)
    col, row = points_to_fine_grid(boxes[:, :2].detach().cpu(), cfg)
    return col.numpy(), row.numpy()


def _imshow_bev(ax, grid: np.ndarray, title: str, cfg: BEVConfig,
                gt_cols=None, gt_rows=None, cmap="viridis"):
    im = ax.imshow(grid, origin="lower", cmap=cmap, aspect="equal")
    if gt_cols is not None and len(gt_cols) > 0:
        ax.scatter(gt_cols, gt_rows, s=22, facecolors="none", edgecolors="red", linewidths=1.0)
    ax.set_title(title, fontsize=8)
    ax.set_xlabel("col = x bin (W→x)", fontsize=6)
    ax.set_ylabel("row = y bin (H→y)", fontsize=6)
    return im


def render_camera_features(writer, intermediates: Dict[str, torch.Tensor], sample_token: str,
                           cam_order: List[str]) -> str:
    """Per-camera feature-norm heatmaps + LSS expected-depth (batch index 0)."""
    context = intermediates["_camera_context"]  # [B*N, Cc, fH, fW]
    depth = intermediates["_depth_prob"]         # [B*N, D, fH, fW]
    N = len(cam_order)
    fnorm = context[:N].float().norm(dim=1).cpu().numpy()       # [N, fH, fW]
    D = depth.shape[1]
    dgrid = torch.arange(D, device=depth.device, dtype=depth.dtype).view(1, D, 1, 1)
    exp_depth = (depth[:N] * dgrid).sum(dim=1).cpu().numpy()     # [N, fH, fW] (bin units)

    fig, axes = plt.subplots(2, N, figsize=(2.2 * N, 4.4), dpi=_FIG_DPI)
    for i in range(N):
        axes[0, i].imshow(fnorm[i], cmap="magma"); axes[0, i].set_title(cam_order[i][:10], fontsize=6)
        axes[0, i].axis("off")
        axes[1, i].imshow(exp_depth[i], cmap="viridis"); axes[1, i].axis("off")
    axes[0, 0].set_ylabel("feat ‖·‖", fontsize=7); axes[1, 0].set_ylabel("E[depth bin]", fontsize=7)
    fig.suptitle(f"V2 camera features + LSS depth | {sample_token}", fontsize=9)
    path = writer.figure_path("encoder", f"cam_features__{sample_token}")
    fig.savefig(path, bbox_inches="tight", pad_inches=0.1); plt.close(fig)
    return path


def render_bev_encoders(writer, intermediates: Dict[str, torch.Tensor], occupancy: torch.Tensor,
                        gt_boxes: torch.Tensor, sample_token: str, cfg: BEVConfig) -> str:
    """camera→BEV norm, pillar occupancy, LiDAR-BEV norm — with GT-center overlay."""
    cam_bev = _bev_norm(intermediates["_camera_bev"][0])
    lidar_bev = _bev_norm(intermediates["_lidar_bev"][0])
    occ = occupancy[0].detach().cpu().numpy()
    gc, gr = _gt_fine_cells(gt_boxes, cfg)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.4), dpi=_FIG_DPI)
    _imshow_bev(axes[0], cam_bev, "camera→BEV ‖·‖ (response@GT)", cfg, gc, gr, cmap="viridis")
    _imshow_bev(axes[1], occ, "pillar occupancy", cfg, gc, gr, cmap="cividis")
    _imshow_bev(axes[2], lidar_bev, "LiDAR-BEV ‖·‖ (response@GT)", cfg, gc, gr, cmap="magma")
    fig.suptitle(f"V2 BEV encoders | {sample_token} | GT=red○", fontsize=9)
    path = writer.figure_path("encoder", f"bev_encoders__{sample_token}")
    fig.savefig(path, bbox_inches="tight", pad_inches=0.1); plt.close(fig)
    return path


@torch.no_grad()
def render_v2(writer, model, batch, cfg: BEVConfig, max_samples: int = 3) -> List[str]:
    """Render V2 for the first ``max_samples`` samples of ``batch``; returns paths."""
    model.eval()
    out = model(batch, return_intermediates=True)
    occupancy = model.lidar_encoder.occupancy(batch["lidar_points"], len(batch["gt_boxes"]))
    paths: List[str] = []
    n = min(max_samples, len(batch["gt_boxes"]))
    for b in range(n):
        tok = batch["sample_token"][b]
        cam_order = list(batch["cam_order"][b])
        # per-sample slice of intermediates (cameras for sample b: rows b*N..(b+1)*N)
        N = len(cam_order)
        inter_b = {
            "_camera_context": out["_camera_context"][b * N:(b + 1) * N],
            "_depth_prob": out["_depth_prob"][b * N:(b + 1) * N],
            "_camera_bev": out["_camera_bev"][b:b + 1],
            "_lidar_bev": out["_lidar_bev"][b:b + 1],
        }
        paths.append(render_camera_features(writer, inter_b, tok, cam_order))
        paths.append(render_bev_encoders(writer, inter_b, occupancy[b:b + 1],
                                         batch["gt_boxes"][b], tok, cfg))
    return paths
