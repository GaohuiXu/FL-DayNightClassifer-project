"""CenterPoint detection loss (fl_v3 T2) — Gaussian focal heatmap + L1 regression.

Builds the dense target on the **head grid** via the single :mod:`bev_grid` convention,
then a Gaussian-penalty focal loss on the heatmap and an L1 on the regression channels at
the matched GT-center cells.

## Official CenterPoint / MIT BEVFusion ``gaussian_radius`` semantics

The target exactly follows the published upstream implementations, including their
constant ``/2`` denominator for all three candidate roots (not an alternative geometric
``/(2*a)`` derivation). The nuScenes configuration pins ``min_overlap=0.1`` and
``min_radius=2``; conversion is ``max(min_radius, int(min(r1, r2, r3)))``. The Gaussian
patch likewise follows upstream NumPy-float64 generation, float32 conversion, clipping,
and maximum overlay.

This changes target tensors relative to every old fl_v3 checkpoint trained with the mixed
denominator implementation. Those checkpoints are incompatible with the new targets and
must be retrained; this is not a resume-compatible loss change.

Target rendering is **RNG-free + atomic-free**: a precomputed 2-D Gaussian patch written
with ``torch.maximum`` overlay (overlapping objects take the max, never a summed scatter)
into a sliced view of the heatmap. The regression L1 gathers predictions at the GT-center
cells (index gather — deterministic) and matches the **T1 canonical** parameterization
``(offset_xy, z, log(l,w,h), sin/cos yaw, vx, vy)``.
"""
from __future__ import annotations

import math
from contextlib import contextmanager
from functools import lru_cache
from typing import Any, Dict, List, Sequence

import numpy as np
import torch
import torch.nn as nn

from fl_v3.models.fusion.bev_grid import BEVConfig

# CenterPoint code weights (velocity down-weighted): reg(2) z(1) dim(3) rot(2) vel(2)
DEFAULT_CODE_WEIGHTS = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2)
OFFICIAL_GAUSSIAN_OVERLAP = 0.1
OFFICIAL_MIN_RADIUS = 2


def gaussian_radius(det_size, min_overlap: float = OFFICIAL_GAUSSIAN_OVERLAP) -> float:
    """Official CenterPoint/BEVFusion three-case radius (constant ``/2`` roots)."""
    height, width = float(det_size[0]), float(det_size[1])

    a1 = 1.0
    b1 = height + width
    c1 = width * height * (1.0 - min_overlap) / (1.0 + min_overlap)
    sq1 = math.sqrt(b1 * b1 - 4.0 * a1 * c1)
    r1 = (b1 + sq1) / 2.0

    a2 = 4.0
    b2 = 2.0 * (height + width)
    c2 = (1.0 - min_overlap) * width * height
    sq2 = math.sqrt(b2 * b2 - 4.0 * a2 * c2)
    r2 = (b2 + sq2) / 2.0

    a3 = 4.0 * min_overlap
    b3 = -2.0 * min_overlap * (height + width)
    c3 = (min_overlap - 1.0) * width * height
    sq3 = math.sqrt(b3 * b3 - 4.0 * a3 * c3)
    r3 = (b3 + sq3) / 2.0

    return min(r1, r2, r3)


def gaussian_2d(radius: int) -> torch.Tensor:
    """Official ``(2r+1)²`` NumPy-generated patch converted to float32."""
    diameter = 2 * radius + 1
    sigma = diameter / 6.0
    center = (diameter - 1.0) / 2.0
    yy, xx = np.ogrid[-center : center + 1, -center : center + 1]
    gaussian = np.exp(-(xx * xx + yy * yy) / (2.0 * sigma * sigma))
    gaussian[gaussian < np.finfo(gaussian.dtype).eps * gaussian.max()] = 0.0
    return torch.from_numpy(gaussian).to(torch.float32)


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
        min_radius: int = OFFICIAL_MIN_RADIUS,
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
        self.gaussian_overlap = OFFICIAL_GAUSSIAN_OVERLAP
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
        self._s10_capture = False
        self._s10_focal: Dict[str, Any] = {}
        self.last_s10_terms: Dict[str, Any] = {}

    # --- target construction (RNG-free, atomic-free) ---
    def build_targets(self, batch: dict, device, dtype=torch.float32):
        """BIT-IDENTICAL to the original per-GT loop, but the per-box `.item()` CPU↔GPU syncs (which
        serialized the GPU — hundreds per batch) are replaced by ONE batched transfer, and the
        regression target is vectorized. The arithmetic, the device (GPU), and the `torch.maximum`
        Gaussian overlay (commutative → order-independent) are unchanged, so the heatmap + reg_target
        tensors are byte-identical and the (target-side, no-grad) construction does not alter gradients."""
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
            radius = max(
                self.min_radius,
                int(
                    gaussian_radius(
                        (fcells[k][0], fcells[k][1]),
                        min_overlap=self.gaussian_overlap,
                    )
                ),
            )
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
        n_pos_raw = pos.sum()
        n_pos = n_pos_raw.clamp_min(1.0)
        if self.class_weights is not None:                 # per-class rebalance (mean-1 normalized)
            # move on-the-fly (like code_weights L258) so the generic loop need not .to() the criterion
            w = self.class_weights.to(pred_logits.device).view(1, -1, 1, 1)
            pos_loss = pos_loss * w
            neg_loss = neg_loss * w
        pos_sum = pos_loss.sum()
        neg_sum = neg_loss.sum()
        numerator = -(pos_sum + neg_sum)
        result = numerator / n_pos
        if self._s10_capture:
            self._s10_focal = {
                "loss": result,
                "numerator": numerator,
                "denominator": n_pos,
                "raw_positive_count": n_pos_raw,
                "positive_numerator": -pos_sum,
                "negative_numerator": -neg_sum,
                "sample_numerators": tuple(
                    -(pos_loss[index].sum() + neg_loss[index].sum())
                    for index in range(pred_logits.shape[0])
                ),
                "sample_denominators": tuple(
                    pos[index].sum() for index in range(pred_logits.shape[0])
                ),
                "sample_unique_positive_centers": tuple(
                    pos[index].sum() for index in range(pred_logits.shape[0])
                ),
            }
        else:
            self._s10_focal = {}
        return result

    def forward(self, pred: Dict[str, torch.Tensor], batch: dict) -> torch.Tensor:
        device = pred["heatmap"].device
        dtype = pred["heatmap"].dtype
        heatmap_t, bidx, cells, reg_target, labels_k = self.build_targets(batch, device, dtype)
        hm_loss = self.gaussian_focal(pred["heatmap"], heatmap_t)

        # regression channels concatenated in the canonical order → [B, 10, H, W]
        reg_pred = torch.cat([pred["reg"], pred["height"], pred["dim"], pred["rot"], pred["vel"]], dim=1)
        B, C, H, W = reg_pred.shape
        reg_sample_numerators: tuple[torch.Tensor, ...]
        reg_sample_denominators: tuple[torch.Tensor, ...]
        if reg_target.shape[0] > 0:
            flat = reg_pred.permute(0, 2, 3, 1).reshape(B * H * W, C)  # [B*H*W, 10]
            gather_idx = bidx * (H * W) + cells
            pred_at = flat[gather_idx]                                  # [G, 10]
            # code_weights is a buffer; move on-the-fly so the loop need not .to() the
            # criterion (keeps the generic loop / dummy-task contract untouched).
            l1 = (pred_at - reg_target).abs() * self.code_weights.to(pred_at.device)   # [G, 10]
            if self.reg_class_weights is not None:                # per-GT class weight on the L1 (mean-1)
                rw = self.reg_class_weights.to(pred_at.device)[labels_k]               # [G]
                weighted_rows = l1.sum(dim=1) * rw
                reg_numerator = weighted_rows.sum()
                reg_denominator = rw.sum().clamp_min(1e-6)
                reg_loss = reg_numerator / reg_denominator       # class-weighted mean
                if self._s10_capture:
                    reg_sample_numerators = tuple(
                        weighted_rows[bidx == index].sum() for index in range(B)
                    )
                    reg_sample_denominators = tuple(
                        rw[bidx == index].sum() for index in range(B)
                    )
                else:
                    reg_sample_numerators = reg_sample_denominators = ()
            else:
                reg_numerator = l1.sum()
                reg_denominator = reg_numerator.new_tensor(float(reg_target.shape[0]))
                reg_loss = reg_numerator / reg_target.shape[0]
                if self._s10_capture:
                    row_sums = l1.sum(dim=1)
                    reg_sample_numerators = tuple(
                        row_sums[bidx == index].sum() for index in range(B)
                    )
                    reg_sample_denominators = tuple(
                        reg_numerator.new_tensor(float((bidx == index).sum().item()))
                        for index in range(B)
                    )
                else:
                    reg_sample_numerators = reg_sample_denominators = ()
        else:
            reg_loss = reg_pred.sum() * 0.0
            reg_numerator = reg_loss
            reg_denominator = reg_loss.new_tensor(1.0)
            if self._s10_capture:
                reg_sample_numerators = tuple(reg_pred[index].sum() * 0.0 for index in range(B))
                reg_sample_denominators = tuple(reg_loss.new_tensor(0.0) for _ in range(B))
            else:
                reg_sample_numerators = reg_sample_denominators = ()
        weighted_reg = self.reg_weight * reg_loss
        total = hm_loss + weighted_reg
        if self._s10_capture:
            input_gt = [int(boxes.shape[0]) for boxes in batch["gt_boxes"]]
            target_gt = [int((bidx == index).sum().item()) for index in range(B)]
            self.last_s10_terms = {
                "tensors": {
                    "hm_loss": hm_loss,
                    "reg_loss": reg_loss,
                    "weighted_reg_loss": weighted_reg,
                    "total": total,
                    "hm_numerator": self._s10_focal["numerator"],
                    "hm_denominator": self._s10_focal["denominator"],
                    "hm_raw_positive_count": self._s10_focal["raw_positive_count"],
                    "hm_positive_numerator": self._s10_focal["positive_numerator"],
                    "hm_negative_numerator": self._s10_focal["negative_numerator"],
                    "hm_sample_numerators": self._s10_focal["sample_numerators"],
                    "hm_sample_denominators": self._s10_focal["sample_denominators"],
                    "reg_numerator": reg_numerator,
                    "reg_denominator": reg_denominator,
                    "reg_sample_numerators": reg_sample_numerators,
                    "reg_sample_denominators": reg_sample_denominators,
                },
                "metadata": {
                    "batch_size": B,
                    "input_gt_per_sample": input_gt,
                    "in_range_targets_per_sample": target_gt,
                    "unique_heatmap_positive_centers_per_sample": [
                        int(value.detach().item())
                        for value in self._s10_focal["sample_unique_positive_centers"]
                    ],
                    "center_collisions_per_sample": [
                        int(target - positive)
                        for target, positive in zip(
                            target_gt,
                            [
                                int(value.detach().item())
                                for value in self._s10_focal["sample_unique_positive_centers"]
                            ],
                            strict=True,
                        )
                    ],
                    "reg_weight": self.reg_weight,
                    "class_weighted_heatmap": self.class_weights is not None,
                    "class_weighted_regression": self.reg_class_weights is not None,
                },
            }
        else:
            self.last_s10_terms = {}
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


class MultiTaskCenterPointLoss(nn.Module):
    """Apply the reviewed S02 target/loss to each reviewed S05 task head.

    Global nuScenes labels are mapped to task-local labels by class name.  The
    six task losses are summed, matching the independent official task-head
    objective while retaining S02's exact Gaussian implementation.
    """

    def __init__(
        self,
        cfg: BEVConfig = BEVConfig(),
        *,
        reg_weight: float = 0.25,
        class_weights=None,
        reg_class_weights=None,
        global_class_names: Sequence[str] | None = None,
    ) -> None:
        super().__init__()
        from fl_v3.models.fusion.head import (
            NUSCENES_CENTERHEAD_TASKS,
            NUSCENES_DETECTION_NAMES,
        )

        names = tuple(global_class_names or NUSCENES_DETECTION_NAMES)
        if len(names) != 10 or len(names) != len(set(names)):
            raise ValueError("CenterHead global_class_names must be a ten-class bijection")
        by_name = {name: index for index, name in enumerate(names)}
        if set(by_name) != set(NUSCENES_DETECTION_NAMES):
            raise ValueError("CenterHead global_class_names differs from nuScenes taxonomy")
        self.global_ids = tuple(
            tuple(by_name[name] for name in task)
            for task in NUSCENES_CENTERHEAD_TASKS
        )
        self.losses = nn.ModuleList()
        for ids in self.global_ids:
            heat_weights = None if class_weights is None else [class_weights[i] for i in ids]
            box_weights = (
                None if reg_class_weights is None else [reg_class_weights[i] for i in ids]
            )
            self.losses.append(
                CenterPointLoss(
                    cfg=cfg,
                    n_classes=len(ids),
                    reg_weight=reg_weight,
                    class_weights=heat_weights,
                    reg_class_weights=box_weights,
                )
            )
        self._record_terms = True
        self.last_terms: Dict[str, float] = {}
        self._s10_capture = False
        self.last_s10_terms: Dict[str, Any] = {}

    @property
    def record_terms(self) -> bool:
        """Control host-side diagnostic scalar recording for every task loss."""
        return self._record_terms

    @record_terms.setter
    def record_terms(self, enabled: bool) -> None:
        self._record_terms = bool(enabled)
        for criterion in self.losses:
            criterion.record_terms = self._record_terms

    @staticmethod
    def _task_batch(batch: dict, global_ids: Sequence[int]) -> dict:
        mapping = {int(global_id): local for local, global_id in enumerate(global_ids)}
        result = dict(batch)
        boxes_out, labels_out, velocity_out = [], [], []
        velocities = batch.get("gt_velocity")
        for index, (boxes, labels) in enumerate(zip(batch["gt_boxes"], batch["gt_labels"], strict=True)):
            labels_long = labels.to(torch.int64)
            keep = torch.zeros_like(labels_long, dtype=torch.bool)
            local = torch.full_like(labels_long, -1)
            for global_id, local_id in mapping.items():
                selected = labels_long == global_id
                keep |= selected
                local[selected] = local_id
            boxes_out.append(boxes[keep])
            labels_out.append(local[keep])
            if velocities is not None:
                velocity_out.append(velocities[index][keep])
        result["gt_boxes"] = boxes_out
        result["gt_labels"] = labels_out
        if velocities is not None:
            result["gt_velocity"] = velocity_out
        return result

    @contextmanager
    def capture_s10_terms(self):
        """Retain exact task/term tensors for one STOP-B observation forward."""
        if self._s10_capture:
            raise RuntimeError("nested S10 loss-term capture is forbidden")
        self._s10_capture = True
        self.last_s10_terms = {}
        for criterion in self.losses:
            criterion._s10_capture = True
            criterion._s10_focal = {}
            criterion.last_s10_terms = {}
        try:
            yield self
        finally:
            self._s10_capture = False
            for criterion in self.losses:
                criterion._s10_capture = False

    def s10_term_bundle(self) -> Dict[str, Any]:
        if not self.last_s10_terms:
            raise RuntimeError("no captured S10 loss-term bundle is available")
        return self.last_s10_terms

    def forward(self, pred, batch: dict) -> torch.Tensor:
        if isinstance(pred, dict) and "task_outputs" in pred:
            pred = pred["task_outputs"]
        if not isinstance(pred, (list, tuple)) or len(pred) != len(self.losses):
            raise ValueError(
                f"multi-task CenterHead must return {len(self.losses)} task dictionaries"
            )
        terms = []
        aggregate = {"hm_loss": 0.0, "reg_loss": 0.0, "n_gt": 0}
        for output, criterion, global_ids in zip(pred, self.losses, self.global_ids, strict=True):
            value = criterion(output, self._task_batch(batch, global_ids))
            terms.append(value)
            aggregate["n_gt"] += criterion.last_terms.get("n_gt", 0)
            if self.record_terms:
                aggregate["hm_loss"] += criterion.last_terms["hm_loss"]
                aggregate["reg_loss"] += criterion.last_terms["reg_loss"]
        total = torch.stack(terms).sum()
        if self._s10_capture:
            task_records = []
            for index, (criterion, global_ids) in enumerate(
                zip(self.losses, self.global_ids, strict=True)
            ):
                if not criterion.last_s10_terms:
                    raise RuntimeError(f"task {index} did not emit S10 term tensors")
                task_records.append({
                    "task_index": int(index),
                    "global_class_ids": [int(value) for value in global_ids],
                    **criterion.last_s10_terms,
                })
            self.last_s10_terms = {
                "aggregate_total": total,
                "tasks": task_records,
            }
        else:
            self.last_s10_terms = {}
        if self.record_terms:
            self.last_terms = {"loss": float(total.detach().item()), **aggregate}
        else:
            self.last_terms = {"n_gt": aggregate["n_gt"]}
        return total

    def diagnostic_terms(self) -> dict:
        """Return a JSON-ready snapshot without changing loss computation."""
        return {
            "aggregate": dict(self.last_terms),
            "tasks": [
                {
                    "task_index": int(index),
                    "global_class_ids": [int(value) for value in global_ids],
                    "terms": dict(criterion.last_terms),
                }
                for index, (global_ids, criterion) in enumerate(
                    zip(self.global_ids, self.losses, strict=True)
                )
            ],
        }
