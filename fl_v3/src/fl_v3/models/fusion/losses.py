"""CenterPoint detection loss (fl_v3 T2) — Gaussian focal heatmap + L1 regression.

Builds the dense target on the **head grid** via the single :mod:`bev_grid` convention,
then a Gaussian-penalty focal loss on the heatmap and an L1 on the regression channels at
the matched GT-center cells.

## The corrected ``gaussian_radius`` (NOT the CenterNet ``/2`` bug)

The radius of the target Gaussian solves three IOU-overlap quadratics; each root is
``(b + sqrt(b² − 4·a·c)) / (2·a)`` with ``a1=1, a2=4, a3=4·min_overlap`` (``min_overlap
=0.1``). The **historical CenterNet bug divides by a constant ``2`` instead of ``2·a``**
for the ``a2``/``a3`` cases — which *halves* the radius, shrinks every target Gaussian, and
**stalls the overfit** (the SPEC failure mode). We use the corrected ``/(2·a)`` form and
``radius = max(0, int(min(r1, r2, r3)))``.

Target rendering is **RNG-free + atomic-free**: a precomputed 2-D Gaussian patch written
with ``torch.maximum`` overlay (overlapping objects take the max, never a summed scatter)
into a sliced view of the heatmap. The regression L1 gathers predictions at the GT-center
cells (index gather — deterministic) and matches the **T1 canonical** parameterization
``(offset_xy, z, log(l,w,h), sin/cos yaw, vx, vy)``.
"""
from __future__ import annotations

import math
from typing import Dict, List

import torch
import torch.nn as nn

from fl_v3.models.fusion.bev_grid import BEVConfig

# CenterPoint code weights (velocity down-weighted): reg(2) z(1) dim(3) rot(2) vel(2)
DEFAULT_CODE_WEIGHTS = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2)


def gaussian_radius(det_size, min_overlap: float = 0.1) -> float:
    """Corrected 3-case Gaussian radius (denominators ``2·a``; NOT the ``/2`` bug)."""
    height, width = float(det_size[0]), float(det_size[1])

    a1 = 1.0
    b1 = height + width
    c1 = width * height * (1.0 - min_overlap) / (1.0 + min_overlap)
    sq1 = math.sqrt(max(b1 * b1 - 4.0 * a1 * c1, 0.0))
    r1 = (b1 + sq1) / (2.0 * a1)

    a2 = 4.0
    b2 = 2.0 * (height + width)
    c2 = (1.0 - min_overlap) * width * height
    sq2 = math.sqrt(max(b2 * b2 - 4.0 * a2 * c2, 0.0))
    r2 = (b2 + sq2) / (2.0 * a2)

    a3 = 4.0 * min_overlap
    b3 = -2.0 * min_overlap * (height + width)
    c3 = (min_overlap - 1.0) * width * height
    sq3 = math.sqrt(max(b3 * b3 - 4.0 * a3 * c3, 0.0))
    r3 = (b3 + sq3) / (2.0 * a3)

    return min(r1, r2, r3)


def gaussian_2d(radius: int, sigma_scale: float = 6.0) -> torch.Tensor:
    """``(2r+1, 2r+1)`` Gaussian patch, peak 1 at center (the umich/CenterNet patch)."""
    diameter = 2 * radius + 1
    sigma = diameter / sigma_scale
    ax = torch.arange(-radius, radius + 1, dtype=torch.float32)
    yy, xx = torch.meshgrid(ax, ax, indexing="ij")
    g = torch.exp(-(xx * xx + yy * yy) / (2.0 * sigma * sigma))
    g[g < torch.finfo(g.dtype).eps * g.max()] = 0.0
    return g


def draw_gaussian(heatmap: torch.Tensor, col: int, row: int, radius: int) -> None:
    """In-place ``torch.maximum`` overlay of a Gaussian at ``(col,row)`` (col→W, row→H).

    Atomic-free, RNG-free: writes a sliced view of the per-class ``[H, W]`` heatmap."""
    H, W = heatmap.shape
    if radius < 0:
        return
    g = gaussian_2d(radius).to(heatmap.device, heatmap.dtype)
    left, right = min(col, radius), min(W - col, radius + 1)
    top, bottom = min(row, radius), min(H - row, radius + 1)
    if right <= -left or bottom <= -top:
        return
    masked = heatmap[row - top : row + bottom, col - left : col + right]
    patch = g[radius - top : radius + bottom, radius - left : radius + right]
    if masked.numel() > 0 and patch.numel() > 0:
        torch.maximum(masked, patch, out=masked)


class CenterPointLoss(nn.Module):
    """Gaussian focal heatmap loss + L1 regression at matched centers.

    ``forward(pred, batch)`` where ``pred`` is the head dict and ``batch`` is the collated
    detection batch (``gt_boxes`` / ``gt_labels`` per-sample lists). Returns a scalar loss
    (the loop's injected criterion). Sub-losses are stashed on ``self.last_terms`` for
    logging / the overfit curve."""

    def __init__(
        self,
        cfg: BEVConfig = BEVConfig(),
        n_classes: int = 10,
        reg_weight: float = 0.25,
        focal_alpha: float = 2.0,
        focal_gamma: float = 4.0,
        min_radius: int = 2,
        code_weights=DEFAULT_CODE_WEIGHTS,
    ):
        super().__init__()
        self.cfg = cfg
        self.n_classes = int(n_classes)
        self.reg_weight = float(reg_weight)
        self.alpha = float(focal_alpha)
        self.gamma = float(focal_gamma)
        self.min_radius = int(min_radius)
        self.register_buffer("code_weights", torch.tensor(code_weights, dtype=torch.float32), persistent=False)
        self.last_terms: Dict[str, float] = {}

    # --- target construction (RNG-free, atomic-free) ---
    def build_targets(self, batch: dict, device, dtype=torch.float32):
        cfg = self.cfg
        H, W = cfg.head_ny, cfg.head_nx
        gt_boxes_list: List[torch.Tensor] = batch["gt_boxes"]
        gt_labels_list: List[torch.Tensor] = batch["gt_labels"]
        B = len(gt_boxes_list)
        heatmap = torch.zeros((B, self.n_classes, H, W), device=device, dtype=dtype)
        # collected per-GT regression targets + locations (for the gather-based L1)
        bidx, cells, reg_t = [], [], []
        for b in range(B):
            boxes = gt_boxes_list[b].to(device=device, dtype=dtype)
            labels = gt_labels_list[b].to(device=device)
            for m in range(boxes.shape[0]):
                cx, cy, cz, dx, dy, dz, yaw = [boxes[m, k] for k in range(7)]
                fx = (cx - cfg.x_min) / cfg.head_vx
                fy = (cy - cfg.y_min) / cfg.head_vy
                col = int(torch.floor(fx).item())
                row = int(torch.floor(fy).item())
                if not (0 <= col < W and 0 <= row < H):
                    continue  # center outside the head grid (out of BEV range)
                c = int(labels[m].item())
                if not (0 <= c < self.n_classes):
                    continue
                l_cells = float((dx / cfg.head_vx).item())
                w_cells = float((dy / cfg.head_vy).item())
                radius = max(self.min_radius, int(gaussian_radius((l_cells, w_cells))))
                draw_gaussian(heatmap[b, c], col, row, radius)
                # regression target (T1 canonical): offset, z, log-dim, sin/cos, vel
                vx, vy = (boxes[m, 0] * 0, boxes[m, 0] * 0)
                if "gt_velocity" in batch:
                    vel = batch["gt_velocity"][b].to(device=device, dtype=dtype)
                    if m < vel.shape[0]:
                        vx, vy = vel[m, 0], vel[m, 1]
                eps = 1e-6
                tvec = torch.stack([
                    fx - col, fy - row,                                   # offset xy
                    cz,                                                    # height z
                    torch.log(dx.clamp_min(eps)), torch.log(dy.clamp_min(eps)), torch.log(dz.clamp_min(eps)),
                    torch.sin(yaw), torch.cos(yaw),                        # rot
                    vx, vy,                                                # velocity
                ])
                bidx.append(b)
                cells.append(row * W + col)
                reg_t.append(tvec)
        if reg_t:
            reg_target = torch.stack(reg_t)                                # [G, 10]
            bidx_t = torch.tensor(bidx, device=device, dtype=torch.int64)
            cells_t = torch.tensor(cells, device=device, dtype=torch.int64)
        else:
            reg_target = torch.zeros((0, 10), device=device, dtype=dtype)
            bidx_t = torch.zeros((0,), device=device, dtype=torch.int64)
            cells_t = torch.zeros((0,), device=device, dtype=torch.int64)
        return heatmap, bidx_t, cells_t, reg_target

    # --- focal + L1 ---
    def gaussian_focal(self, pred_logits: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        pred = pred_logits.sigmoid().clamp(1e-4, 1 - 1e-4)
        pos = gt.eq(1.0).to(pred.dtype)
        neg = gt.lt(1.0).to(pred.dtype)
        neg_w = (1.0 - gt).pow(self.gamma)
        pos_loss = torch.log(pred) * (1.0 - pred).pow(self.alpha) * pos
        neg_loss = torch.log(1.0 - pred) * pred.pow(self.alpha) * neg_w * neg
        n_pos = pos.sum().clamp_min(1.0)
        return -(pos_loss.sum() + neg_loss.sum()) / n_pos

    def forward(self, pred: Dict[str, torch.Tensor], batch: dict) -> torch.Tensor:
        device = pred["heatmap"].device
        dtype = pred["heatmap"].dtype
        heatmap_t, bidx, cells, reg_target = self.build_targets(batch, device, dtype)
        hm_loss = self.gaussian_focal(pred["heatmap"], heatmap_t)

        # regression channels concatenated in the canonical order → [B, 10, H, W]
        reg_pred = torch.cat([pred["reg"], pred["height"], pred["dim"], pred["rot"], pred["vel"]], dim=1)
        B, C, H, W = reg_pred.shape
        if reg_target.shape[0] > 0:
            flat = reg_pred.permute(0, 2, 3, 1).reshape(B * H * W, C)  # [B*H*W, 10]
            gather_idx = bidx * (H * W) + cells
            pred_at = flat[gather_idx]                                  # [G, 10]
            # code_weights is a buffer; move on-the-fly so the loop need not .to() the
            # criterion (keeps the generic loop / dummy-task contract untouched).
            l1 = (pred_at - reg_target).abs() * self.code_weights.to(pred_at.device)
            reg_loss = l1.sum() / reg_target.shape[0]
        else:
            reg_loss = reg_pred.sum() * 0.0
        total = hm_loss + self.reg_weight * reg_loss
        self.last_terms = {
            "loss": float(total.detach().item()),
            "hm_loss": float(hm_loss.detach().item()),
            "reg_loss": float(reg_loss.detach().item()),
            "n_gt": int(reg_target.shape[0]),
        }
        return total
