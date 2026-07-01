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
from functools import lru_cache
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


@lru_cache(maxsize=128)
def _gaussian_patch(radius: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Cached device-resident Gaussian patch per radius (the radii are a tiny integer set). Byte-identical
    to ``gaussian_2d(radius).to(device, dtype)`` — gaussian_2d is deterministic — but built ONCE instead of
    rebuilt-on-CPU-and-HtoD-copied per GT object (the per-object Memcpy HtoD the profile flagged). The patch
    is read-only in ``draw_gaussian`` (torch.maximum reads it), so the cached tensor is never mutated."""
    return gaussian_2d(radius).to(device, dtype)


def draw_gaussian(heatmap: torch.Tensor, col: int, row: int, radius: int) -> None:
    """In-place ``torch.maximum`` overlay of a Gaussian at ``(col,row)`` (col→W, row→H).

    Atomic-free, RNG-free: writes a sliced view of the per-class ``[H, W]`` heatmap."""
    H, W = heatmap.shape
    if radius < 0:
        return
    g = _gaussian_patch(int(radius), heatmap.device, heatmap.dtype)   # cached (was rebuilt+HtoD per object)
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
        class_weights=None,
        reg_class_weights=None,
    ):
        super().__init__()
        self.cfg = cfg
        self.n_classes = int(n_classes)
        self.reg_weight = float(reg_weight)
        self.alpha = float(focal_alpha)
        self.gamma = float(focal_gamma)
        self.min_radius = int(min_radius)
        self.register_buffer("code_weights", torch.tensor(code_weights, dtype=torch.float32), persistent=False)
        # MCR P1: optional per-class heatmap weight (rebalance the rare/stuck classes — trailer/construction
        # the convergence teardown flagged). Normalized to mean 1 so the overall heatmap-vs-reg scale is
        # preserved (only the RELATIVE class emphasis changes). None ⇒ uniform ⇒ byte-identical to before.
        if class_weights is not None:
            cw = torch.tensor([float(x) for x in class_weights], dtype=torch.float32)
            assert cw.numel() == self.n_classes, f"class_weights must have {self.n_classes} entries"
            cw = cw * (cw.numel() / cw.sum().clamp_min(1e-6))   # → mean 1
            self.register_buffer("class_weights", cw, persistent=False)
        else:
            self.class_weights = None
        # MCR P1 (the large-vehicle localization lever): optional per-class weight on the REGRESSION L1.
        # The verified gap: bus/truck/trailer lag on box size/yaw regression (orient_err 0.631 / scale_err
        # 0.275 — the worst TP metrics), but `class_weights` only touches the heatmap focal (gaussian_focal)
        # — the reg L1 (forward) has NO class term. That is exactly why the prior 1.68x trailer heatmap
        # upweight moved trailer only +0.004: it weighted RECOGNITION, never LOCALIZATION. This routes a
        # per-class weight onto the L1, mean-1 normalized so the heatmap-vs-reg scale is preserved. The reg
        # loss becomes a class-WEIGHTED mean (sum(w*l1)/sum(w)) so it stays scale-stable across batch
        # composition. None ⇒ unweighted ⇒ byte-identical to before.
        if reg_class_weights is not None:
            rw = torch.tensor([float(x) for x in reg_class_weights], dtype=torch.float32)
            assert rw.numel() == self.n_classes, f"reg_class_weights must have {self.n_classes} entries"
            rw = rw * (rw.numel() / rw.sum().clamp_min(1e-6))   # → mean 1
            self.register_buffer("reg_class_weights", rw, persistent=False)
        else:
            self.reg_class_weights = None
        self.record_terms = True
        self.last_terms: Dict[str, float] = {}

    # --- target construction (RNG-free, atomic-free) ---
    def build_targets(self, batch: dict, device, dtype=torch.float32):
        """BIT-IDENTICAL to the original per-GT loop, but the per-box `.item()` CPU↔GPU syncs (which
        serialized the GPU — hundreds per batch) are replaced by ONE batched transfer, and the
        regression target is vectorized. The arithmetic, the device (GPU), and the `torch.maximum`
        Gaussian overlay (commutative → order-independent) are unchanged, so the heatmap + reg_target
        tensors are byte-identical and the (target-side, no-grad) construction does not alter gradients.
        Verified against the OLD reference by verify_levers.py (loss + grad_all)."""
        cfg = self.cfg
        H, W = cfg.head_ny, cfg.head_nx
        gt_boxes_list: List[torch.Tensor] = batch["gt_boxes"]
        gt_labels_list: List[torch.Tensor] = batch["gt_labels"]
        B = len(gt_boxes_list)
        heatmap = torch.zeros((B, self.n_classes, H, W), device=device, dtype=dtype)

        has_vel = "gt_velocity" in batch
        boxes_all, labels_all, bsel_all, vel_all = [], [], [], []
        for b in range(B):
            boxes_b = gt_boxes_list[b].to(device=device, dtype=dtype)
            nb = boxes_b.shape[0]
            if nb == 0:
                continue
            boxes_all.append(boxes_b)
            labels_all.append(gt_labels_list[b].to(device=device))
            bsel_all.append(torch.full((nb,), b, device=device, dtype=torch.int64))
            vv = torch.zeros((nb, 2), device=device, dtype=dtype)  # matches OLD `boxes[m,0]*0` default
            if has_vel:
                vel = batch["gt_velocity"][b].to(device=device, dtype=dtype)
                k = min(nb, vel.shape[0])
                if k > 0:
                    vv[:k] = vel[:k, :2]
            vel_all.append(vv)

        if not boxes_all:
            z = lambda *s: torch.zeros(s, device=device, dtype=dtype)
            return (heatmap, torch.zeros((0,), device=device, dtype=torch.int64),
                    torch.zeros((0,), device=device, dtype=torch.int64), z(0, 10),
                    torch.zeros((0,), device=device, dtype=torch.int64))

        boxes = torch.cat(boxes_all, dim=0)          # [G,7] in (b,m) order — identical to the OLD loop
        labels = torch.cat(labels_all, dim=0)        # [G]
        bidx = torch.cat(bsel_all, dim=0)            # [G]
        vel = torch.cat(vel_all, dim=0)              # [G,2]

        cx, cy, cz, dx, dy, dz, yaw = boxes.unbind(dim=1)
        fx = (cx - cfg.x_min) / cfg.head_vx
        fy = (cy - cfg.y_min) / cfg.head_vy
        col_f = torch.floor(fx)
        row_f = torch.floor(fy)
        coli = col_f.to(torch.int64)
        rowi = row_f.to(torch.int64)
        # same validity filter as the OLD `continue`s (center in grid AND class in range), same order.
        keep = ((coli >= 0) & (coli < W) & (rowi >= 0) & (rowi < H)
                & (labels >= 0) & (labels < self.n_classes)).nonzero(as_tuple=False).squeeze(1)
        if keep.numel() == 0:
            return (heatmap, torch.zeros((0,), device=device, dtype=torch.int64),
                    torch.zeros((0,), device=device, dtype=torch.int64),
                    torch.zeros((0, 10), device=device, dtype=dtype),
                    torch.zeros((0,), device=device, dtype=torch.int64))
        sel = lambda t: t.index_select(0, keep)
        cx, cy, cz, dx, dy, dz, yaw = (sel(cx), sel(cy), sel(cz), sel(dx), sel(dy), sel(dz), sel(yaw))
        fx, fy, col_f, row_f = sel(fx), sel(fy), sel(col_f), sel(row_f)
        coli, rowi, labels_k, vel_k, bidx_k = sel(coli), sel(rowi), sel(labels), sel(vel), sel(bidx)

        # regression target (vectorized; identical per-element arithmetic + identical (b,m) row order).
        eps = 1e-6
        reg_target = torch.stack([
            fx - col_f, fy - row_f,                                       # offset xy
            cz,                                                            # height z
            torch.log(dx.clamp_min(eps)), torch.log(dy.clamp_min(eps)), torch.log(dz.clamp_min(eps)),
            torch.sin(yaw), torch.cos(yaw),                                # rot
            vel_k[:, 0], vel_k[:, 1],                                      # velocity
        ], dim=1)                                                          # [G_keep, 10]
        cells_t = rowi * W + coli
        bidx_t = bidx_k

        # heatmap: same per-GT torch.maximum overlay, but col/row/class/dims read from ONE batched
        # transfer (no per-box sync). gaussian_radius is CPU float math on the same values → same radius.
        # TWO batched D2H transfers (one float, one int) instead of six separate `.cpu()` stream-drains —
        # identical values, identical (b,m) order → byte-identical heatmap. gaussian_radius stays Python
        # float64 on the same values (vectorizing it in fp32 could shift the int() radius for borderline
        # boxes → NOT byte-identical), and draw_gaussian now reuses the cached device patch (no per-box HtoD).
        fcells = torch.stack([dx / cfg.head_vx, dy / cfg.head_vy], dim=1).cpu().tolist()   # [G,2] float
        icells = torch.stack([coli, rowi, labels_k, bidx_k], dim=1).cpu().tolist()         # [G,4] int
        for k in range(len(icells)):
            col_k, row_k, cls_k, b_k = icells[k]
            radius = max(self.min_radius, int(gaussian_radius((fcells[k][0], fcells[k][1]))))
            draw_gaussian(heatmap[b_k, cls_k], col_k, row_k, radius)

        return heatmap, bidx_t, cells_t, reg_target, labels_k

    # --- focal + L1 ---
    def gaussian_focal(self, pred_logits: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        pred = pred_logits.sigmoid().clamp(1e-4, 1 - 1e-4)
        pos = gt.eq(1.0).to(pred.dtype)
        neg = gt.lt(1.0).to(pred.dtype)
        neg_w = (1.0 - gt).pow(self.gamma)
        pos_loss = torch.log(pred) * (1.0 - pred).pow(self.alpha) * pos
        neg_loss = torch.log(1.0 - pred) * pred.pow(self.alpha) * neg_w * neg
        n_pos = pos.sum().clamp_min(1.0)
        if self.class_weights is not None:                 # per-class rebalance (mean-1 normalized)
            # move on-the-fly (like code_weights L258) so the generic loop need not .to() the criterion
            w = self.class_weights.to(pred_logits.device).view(1, -1, 1, 1)
            pos_loss = pos_loss * w
            neg_loss = neg_loss * w
        return -(pos_loss.sum() + neg_loss.sum()) / n_pos

    def forward(self, pred: Dict[str, torch.Tensor], batch: dict) -> torch.Tensor:
        device = pred["heatmap"].device
        dtype = pred["heatmap"].dtype
        heatmap_t, bidx, cells, reg_target, labels_k = self.build_targets(batch, device, dtype)
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
            l1 = (pred_at - reg_target).abs() * self.code_weights.to(pred_at.device)   # [G, 10]
            if self.reg_class_weights is not None:                # per-GT class weight on the L1 (mean-1)
                rw = self.reg_class_weights.to(pred_at.device)[labels_k]               # [G]
                reg_loss = (l1.sum(dim=1) * rw).sum() / rw.sum().clamp_min(1e-6)       # class-weighted mean
            else:
                reg_loss = l1.sum() / reg_target.shape[0]
        else:
            reg_loss = reg_pred.sum() * 0.0
        total = hm_loss + self.reg_weight * reg_loss
        if self.record_terms:
            self.last_terms = {
                "loss": float(total.detach().item()),
                "hm_loss": float(hm_loss.detach().item()),
                "reg_loss": float(reg_loss.detach().item()),
                "n_gt": int(reg_target.shape[0]),
            }
        else:
            self.last_terms = {"n_gt": int(reg_target.shape[0])}
        return total
