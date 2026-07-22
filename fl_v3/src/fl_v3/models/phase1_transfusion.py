"""MMCV/mmdet3d-free TransFusion head for the S10 Phase-I LiDAR candidate.

The module follows MIT BEVFusion commit ``326653dc``: one dense ten-class
heatmap, 200 heatmap-selected queries, one Transformer decoder layer, per-query
heads, Hungarian assignment, Gaussian/focal/L1 losses, and the no-NMS nuScenes
decode.  Framework plumbing is adapted to the project's canonical conventions:

* feature maps are ``[B,C,H=y,W=x]`` and query positions are ``(x,y)``;
* boxes are ``(x,y,z,l,w,h,yaw)`` with geometric-center ``z`` rather than the
  mmdet3d bottom-center wrapper representation;
* labels use the frozen MIT/reference class order.

No mmdet3d, mmcv, or custom IoU runtime is required.  Assignment uses an exact
vectorized rotated-rectangle/height overlap followed by SciPy's reference
``linear_sum_assignment`` CPU step.
"""
from __future__ import annotations

import math
from contextlib import ExitStack, contextmanager, nullcontext
from dataclasses import dataclass
from typing import Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from fl_v3.models.fusion.losses import draw_gaussian, gaussian_radius


@dataclass(frozen=True)
class TransFusionGeometry:
    point_cloud_range: tuple[float, float, float, float, float, float] = (
        -54.0,
        -54.0,
        -5.0,
        54.0,
        54.0,
        3.0,
    )
    voxel_size: tuple[float, float, float] = (0.075, 0.075, 0.2)
    out_size_factor: int = 8
    bev_hw: tuple[int, int] = (180, 180)
    post_center_range: tuple[float, float, float, float, float, float] = (
        -61.2,
        -61.2,
        -10.0,
        61.2,
        61.2,
        10.0,
    )

    def __post_init__(self) -> None:
        if len(self.point_cloud_range) != 6 or len(self.voxel_size) != 3:
            raise ValueError("invalid TransFusion geometry dimensions")
        if self.out_size_factor <= 0 or min(self.bev_hw) <= 0:
            raise ValueError("invalid TransFusion output geometry")
        expected_w = int(
            round(
                (self.point_cloud_range[3] - self.point_cloud_range[0])
                / self.voxel_size[0]
                / self.out_size_factor
            )
        )
        expected_h = int(
            round(
                (self.point_cloud_range[4] - self.point_cloud_range[1])
                / self.voxel_size[1]
                / self.out_size_factor
            )
        )
        if self.bev_hw != (expected_h, expected_w):
            raise ValueError(
                f"TransFusion BEV shape {self.bev_hw} does not match geometry "
                f"{(expected_h, expected_w)}"
            )

    @property
    def cell_xy(self) -> tuple[float, float]:
        return (
            self.voxel_size[0] * self.out_size_factor,
            self.voxel_size[1] * self.out_size_factor,
        )


PHASE1_TRANSFUSION_GEOMETRY = TransFusionGeometry()
PHASE1_TRANSFUSION_CODE_WEIGHTS = (
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    0.2,
    0.2,
)


def _cross2(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    return left[..., 0] * right[..., 1] - left[..., 1] * right[..., 0]


def _rectangle_corners(boxes: torch.Tensor) -> torch.Tensor:
    """Canonical boxes ``[N,>=7]`` to CCW BEV corners ``[N,4,2]``."""
    if boxes.ndim != 2 or boxes.shape[1] < 7:
        raise ValueError(f"boxes must have shape [N,>=7], got {tuple(boxes.shape)}")
    if boxes.shape[0] == 0:
        return boxes.new_zeros((0, 4, 2))
    local = boxes.new_tensor(
        ((-0.5, -0.5), (0.5, -0.5), (0.5, 0.5), (-0.5, 0.5))
    )
    offsets = local.unsqueeze(0) * boxes[:, None, 3:5]
    cosine = torch.cos(boxes[:, 6])
    sine = torch.sin(boxes[:, 6])
    rotation = torch.stack(
        (cosine, -sine, sine, cosine), dim=-1
    ).reshape(-1, 2, 2)
    return offsets @ rotation.transpose(1, 2) + boxes[:, None, :2]


def _points_inside_rectangles(points: torch.Tensor, rectangles: torch.Tensor) -> torch.Tensor:
    """Pairwise point-in-convex-rectangle test on already broadcast tensors."""
    starts = rectangles.unsqueeze(-3)
    edges = torch.roll(rectangles, shifts=-1, dims=-2).unsqueeze(-3) - starts
    rel = points.unsqueeze(-2) - starts
    return (_cross2(edges, rel) >= -1e-6).all(dim=-1)


def pairwise_rotated_intersection_area(
    boxes1: torch.Tensor, boxes2: torch.Tensor
) -> torch.Tensor:
    """Vectorized exact convex intersection area for canonical yaw rectangles."""
    if boxes1.ndim != 2 or boxes2.ndim != 2 or boxes1.shape[1] < 7 or boxes2.shape[1] < 7:
        raise ValueError("pairwise rotated boxes must have shape [N,>=7] and [M,>=7]")
    rows, cols = boxes1.shape[0], boxes2.shape[0]
    if rows == 0 or cols == 0:
        return boxes1.new_zeros((rows, cols))
    corners1 = _rectangle_corners(boxes1)
    corners2 = _rectangle_corners(boxes2)
    c1 = corners1[:, None].expand(rows, cols, 4, 2)
    c2 = corners2[None].expand(rows, cols, 4, 2)
    inside1 = _points_inside_rectangles(c1, c2)
    inside2 = _points_inside_rectangles(c2, c1)

    p = corners1[:, None, :, None, :]
    r = (torch.roll(corners1, -1, dims=1) - corners1)[:, None, :, None, :]
    q = corners2[None, :, None, :, :]
    s = (torch.roll(corners2, -1, dims=1) - corners2)[None, :, None, :, :]
    q_minus_p = q - p
    denominator = _cross2(r, s)
    nonparallel = denominator.abs() > 1e-8
    safe_denominator = torch.where(nonparallel, denominator, torch.ones_like(denominator))
    t = _cross2(q_minus_p, s) / safe_denominator
    u = _cross2(q_minus_p, r) / safe_denominator
    intersections = p + t.unsqueeze(-1) * r
    intersection_mask = (
        nonparallel
        & (t >= -1e-6)
        & (t <= 1.0 + 1e-6)
        & (u >= -1e-6)
        & (u <= 1.0 + 1e-6)
    )

    candidates = torch.cat((c1, c2, intersections.reshape(rows, cols, 16, 2)), dim=2)
    valid = torch.cat((inside1, inside2, intersection_mask.reshape(rows, cols, 16)), dim=2)
    count = valid.sum(dim=2)
    centroid = (candidates * valid.unsqueeze(-1)).sum(dim=2) / count.clamp_min(1).unsqueeze(-1)
    angle = torch.atan2(
        candidates[..., 1] - centroid[..., None, 1],
        candidates[..., 0] - centroid[..., None, 0],
    )
    angle = torch.where(valid, angle, torch.full_like(angle, 10.0))
    order = torch.argsort(angle, dim=2, stable=True)
    ordered = torch.gather(candidates, 2, order.unsqueeze(-1).expand(-1, -1, -1, 2))
    positions = torch.arange(candidates.shape[2], device=boxes1.device).view(1, 1, -1)
    next_positions = torch.remainder(positions + 1, count.clamp_min(1).unsqueeze(-1))
    following = torch.gather(
        ordered, 2, next_positions.unsqueeze(-1).expand(rows, cols, -1, 2)
    )
    edge_cross = _cross2(ordered, following)
    area = 0.5 * (edge_cross * (positions < count.unsqueeze(-1))).sum(dim=2).abs()
    return torch.where(count >= 3, area, torch.zeros_like(area))


def pairwise_iou3d(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """Reference lidar 3-D IoU for canonical geometric-center boxes."""
    if boxes1.ndim != 2 or boxes2.ndim != 2 or boxes1.shape[1] < 7 or boxes2.shape[1] < 7:
        raise ValueError("pairwise IoU boxes must have shape [N,>=7] and [M,>=7]")
    if boxes1.shape[0] == 0 or boxes2.shape[0] == 0:
        return boxes1.new_zeros((boxes1.shape[0], boxes2.shape[0]))
    if not bool(torch.isfinite(boxes1[:, :7]).all().detach().cpu()) or not bool(
        torch.isfinite(boxes2[:, :7]).all().detach().cpu()
    ):
        raise ValueError("IoU boxes contain non-finite geometry")
    if not bool((boxes1[:, 3:6] > 0).all().detach().cpu()) or not bool(
        (boxes2[:, 3:6] > 0).all().detach().cpu()
    ):
        raise ValueError("IoU boxes must have positive dimensions")
    area = pairwise_rotated_intersection_area(boxes1, boxes2)
    bottom1 = boxes1[:, 2] - boxes1[:, 5] * 0.5
    top1 = boxes1[:, 2] + boxes1[:, 5] * 0.5
    bottom2 = boxes2[:, 2] - boxes2[:, 5] * 0.5
    top2 = boxes2[:, 2] + boxes2[:, 5] * 0.5
    overlap_h = (
        torch.minimum(top1[:, None], top2[None])
        - torch.maximum(bottom1[:, None], bottom2[None])
    ).clamp_min(0)
    intersection = area * overlap_h
    volume1 = boxes1[:, 3:6].prod(dim=1)[:, None]
    volume2 = boxes2[:, 3:6].prod(dim=1)[None]
    return intersection / (volume1 + volume2 - intersection).clamp_min(1e-8)


def focal_loss_cost(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    gamma: float = 2.0,
    alpha: float = 0.25,
    weight: float = 0.15,
    eps: float = 1e-12,
) -> torch.Tensor:
    """mmdetection ``FocalLossCost`` for ``[queries,classes]`` logits."""
    probabilities = logits.sigmoid()
    negative = -(1.0 - probabilities + eps).log() * (1.0 - alpha) * probabilities.pow(gamma)
    positive = -(probabilities + eps).log() * alpha * (1.0 - probabilities).pow(gamma)
    return (positive[:, labels] - negative[:, labels]) * weight


def encode_transfusion_boxes(
    boxes: torch.Tensor,
    geometry: TransFusionGeometry = PHASE1_TRANSFUSION_GEOMETRY,
) -> torch.Tensor:
    """Canonical ``[x,y,z,l,w,h,yaw,vx,vy]`` to the ten regression codes."""
    if boxes.ndim != 2 or boxes.shape[1] != 9:
        raise ValueError(f"TransFusion target boxes must be [N,9], got {tuple(boxes.shape)}")
    if boxes.shape[0] and (
        not bool(torch.isfinite(boxes).all().detach().cpu())
        or not bool((boxes[:, 3:6] > 0).all().detach().cpu())
    ):
        raise ValueError("TransFusion target boxes contain invalid geometry")
    cell_x, cell_y = geometry.cell_xy
    x0, y0 = geometry.point_cloud_range[:2]
    targets = boxes.new_zeros((boxes.shape[0], 10))
    targets[:, 0] = (boxes[:, 0] - x0) / cell_x
    targets[:, 1] = (boxes[:, 1] - y0) / cell_y
    targets[:, 2] = boxes[:, 2]
    targets[:, 3:6] = boxes[:, 3:6].log()
    targets[:, 6] = torch.sin(boxes[:, 6])
    targets[:, 7] = torch.cos(boxes[:, 6])
    targets[:, 8:10] = boxes[:, 7:9]
    return targets


def decode_transfusion_tensors(
    center: torch.Tensor,
    height: torch.Tensor,
    dimensions: torch.Tensor,
    rotation: torch.Tensor,
    velocity: torch.Tensor,
    geometry: TransFusionGeometry = PHASE1_TRANSFUSION_GEOMETRY,
) -> torch.Tensor:
    """Decode head tensors to canonical geometric-center ``[B,P,9]`` boxes."""
    shapes = {
        "center": tuple(center.shape),
        "height": tuple(height.shape),
        "dimensions": tuple(dimensions.shape),
        "rotation": tuple(rotation.shape),
        "velocity": tuple(velocity.shape),
    }
    if center.ndim != 3 or center.shape[1] != 2:
        raise ValueError(f"invalid TransFusion center shape inventory: {shapes}")
    batch, _, proposals = center.shape
    if (
        height.shape != (batch, 1, proposals)
        or dimensions.shape != (batch, 3, proposals)
        or rotation.shape != (batch, 2, proposals)
        or velocity.shape != (batch, 2, proposals)
    ):
        raise ValueError(f"invalid TransFusion prediction shape inventory: {shapes}")
    cell_x, cell_y = geometry.cell_xy
    x0, y0 = geometry.point_cloud_range[:2]
    metric_center = torch.stack(
        (center[:, 0] * cell_x + x0, center[:, 1] * cell_y + y0), dim=1
    )
    metric_dimensions = dimensions.exp()
    yaw = torch.atan2(rotation[:, 0:1], rotation[:, 1:2])
    return torch.cat(
        (metric_center, height, metric_dimensions, yaw, velocity), dim=1
    ).permute(0, 2, 1)


class ReferenceMultiheadAttention(nn.Module):
    """Small direct port of the pinned PyTorch-style attention arithmetic."""

    def __init__(self, embed_dim: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        if embed_dim % num_heads:
            raise ValueError("attention embedding dimension must divide evenly by heads")
        self.embed_dim = int(embed_dim)
        self.num_heads = int(num_heads)
        self.head_dim = self.embed_dim // self.num_heads
        self.dropout = float(dropout)
        self.in_proj_weight = nn.Parameter(torch.empty(3 * embed_dim, embed_dim))
        self.in_proj_bias = nn.Parameter(torch.zeros(3 * embed_dim))
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        nn.init.xavier_uniform_(self.in_proj_weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(
        self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
    ) -> torch.Tensor:
        if query.ndim != 3 or key.ndim != 3 or value.shape != key.shape:
            raise ValueError("attention expects query/key/value in [sequence,batch,channels]")
        embed = self.embed_dim
        if query is key and key is value:
            q, k, v = F.linear(query, self.in_proj_weight, self.in_proj_bias).chunk(3, dim=-1)
        elif key is value:
            q = F.linear(query, self.in_proj_weight[:embed], self.in_proj_bias[:embed])
            k, v = F.linear(
                key, self.in_proj_weight[embed:], self.in_proj_bias[embed:]
            ).chunk(2, dim=-1)
        else:
            q = F.linear(query, self.in_proj_weight[:embed], self.in_proj_bias[:embed])
            k = F.linear(
                key,
                self.in_proj_weight[embed : 2 * embed],
                self.in_proj_bias[embed : 2 * embed],
            )
            v = F.linear(value, self.in_proj_weight[2 * embed :], self.in_proj_bias[2 * embed :])
        q = q * (float(self.head_dim) ** -0.5)
        target_len, batch, _ = q.shape
        source_len = k.shape[0]
        q = q.contiguous().view(target_len, batch * self.num_heads, self.head_dim).transpose(0, 1)
        k = k.contiguous().view(source_len, batch * self.num_heads, self.head_dim).transpose(0, 1)
        v = v.contiguous().view(source_len, batch * self.num_heads, self.head_dim).transpose(0, 1)
        weights = torch.bmm(q, k.transpose(1, 2)).softmax(dim=-1)
        weights = F.dropout(weights, p=self.dropout, training=self.training)
        output = torch.bmm(weights, v)
        output = output.transpose(0, 1).contiguous().view(target_len, batch, embed)
        return self.out_proj(output)


class PositionEmbeddingLearned(nn.Module):
    def __init__(self, channels: int = 128) -> None:
        super().__init__()
        self.position_embedding_head = nn.Sequential(
            nn.Conv1d(2, channels, 1),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(channels, channels, 1),
        )

    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        return self.position_embedding_head(positions.transpose(1, 2).contiguous())


class TransformerDecoderLayer(nn.Module):
    def __init__(
        self,
        channels: int = 128,
        num_heads: int = 8,
        ffn_channels: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        # In the pinned head these two constructor arguments are evaluated before
        # the decoder itself is constructed.  Preserve that RNG order even though
        # the modules are registered after the attention/FFN parameters below.
        query_position = PositionEmbeddingLearned(channels)
        key_position = PositionEmbeddingLearned(channels)
        self.self_attn = ReferenceMultiheadAttention(channels, num_heads, dropout)
        self.cross_attn = ReferenceMultiheadAttention(channels, num_heads, dropout)
        self.linear1 = nn.Linear(channels, ffn_channels)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(ffn_channels, channels)
        self.norm1 = nn.LayerNorm(channels)
        self.norm2 = nn.LayerNorm(channels)
        self.norm3 = nn.LayerNorm(channels)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.query_position = query_position
        self.key_position = key_position
        self._operator_profile_ranges = False

    @contextmanager
    def operator_profile_ranges(self):
        if self._operator_profile_ranges:
            raise RuntimeError("TransFusion decoder profiler ranges are already active")
        self._operator_profile_ranges = True
        try:
            yield self
        finally:
            self._operator_profile_ranges = False

    def _profile_range(self, name: str):
        if not self._operator_profile_ranges:
            return nullcontext()
        return torch.profiler.record_function(f"fl_v3::lidar_decoder::{name}")

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        query_positions: torch.Tensor,
        key_positions: torch.Tensor,
    ) -> torch.Tensor:
        with self._profile_range("position_encoding"):
            query_position = self.query_position(query_positions).permute(2, 0, 1)
            key_position = self.key_position(key_positions).permute(2, 0, 1)
            query_sequence = query.permute(2, 0, 1)
            key_sequence = key.permute(2, 0, 1)
        with self._profile_range("self_attention"):
            positioned_query = query_sequence + query_position
            update = self.self_attn(positioned_query, positioned_query, positioned_query)
            query_sequence = self.norm1(query_sequence + self.dropout1(update))
        with self._profile_range("cross_attention"):
            positioned_key = key_sequence + key_position
            update = self.cross_attn(
                query_sequence + query_position,
                positioned_key,
                positioned_key,
            )
            query_sequence = self.norm2(query_sequence + self.dropout2(update))
        with self._profile_range("ffn"):
            update = self.linear2(self.dropout(F.relu(self.linear1(query_sequence))))
            query_sequence = self.norm3(query_sequence + self.dropout3(update))
        return query_sequence.permute(1, 2, 0)


class QueryPredictionHead(nn.Module):
    HEADS: Mapping[str, tuple[int, int]] = {
        "center": (2, 2),
        "height": (1, 2),
        "dim": (3, 2),
        "rot": (2, 2),
        "vel": (2, 2),
        "heatmap": (10, 2),
    }

    def __init__(self, channels: int = 128, hidden: int = 64) -> None:
        super().__init__()
        self.heads = nn.ModuleDict()
        for name, (out_channels, conv_count) in self.HEADS.items():
            if conv_count != 2:
                raise ValueError("the frozen TransFusion query heads require two convolutions")
            self.heads[name] = nn.Sequential(
                nn.Conv1d(channels, hidden, 1, bias=False),
                nn.BatchNorm1d(hidden),
                nn.ReLU(inplace=True),
                nn.Conv1d(hidden, out_channels, 1, bias=True),
            )
        nn.init.constant_(self.heads["heatmap"][-1].bias, -2.19)

    def forward(self, value: torch.Tensor) -> dict[str, torch.Tensor]:
        return {name: head(value) for name, head in self.heads.items()}


def create_bev_position_grid(
    geometry: TransFusionGeometry = PHASE1_TRANSFUSION_GEOMETRY,
) -> torch.Tensor:
    """Return ``[1,H*W,2]`` feature-cell centers in physical flatten order."""
    height, width = geometry.bev_hw
    rows, columns = torch.meshgrid(
        torch.arange(height, dtype=torch.float32),
        torch.arange(width, dtype=torch.float32),
        indexing="ij",
    )
    return torch.stack((columns + 0.5, rows + 0.5), dim=-1).reshape(1, -1, 2)


class Phase1TransFusionHead(nn.Module):
    """Frozen one-layer, 200-query nuScenes TransFusion head."""

    def __init__(
        self,
        *,
        geometry: TransFusionGeometry = PHASE1_TRANSFUSION_GEOMETRY,
        in_channels: int = 512,
        hidden_channels: int = 128,
        num_classes: int = 10,
        num_proposals: int = 200,
        num_heads: int = 8,
        ffn_channels: int = 256,
        dropout: float = 0.1,
        nms_kernel_size: int = 3,
    ) -> None:
        super().__init__()
        if (
            in_channels != 512
            or hidden_channels != 128
            or num_classes != 10
            or num_proposals != 200
            or num_heads != 8
            or ffn_channels != 256
            or dropout != 0.1
            or nms_kernel_size != 3
        ):
            raise ValueError("Phase-I TransFusion constructor drifted from the frozen graph")
        self.geometry = geometry
        self.num_classes = num_classes
        self.num_proposals = num_proposals
        self.nms_kernel_size = nms_kernel_size
        self.shared_conv = nn.Conv2d(in_channels, hidden_channels, 3, padding=1, bias=True)
        self.heatmap_head = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, num_classes, 3, padding=1, bias=True),
        )
        self.class_encoding = nn.Conv1d(num_classes, hidden_channels, 1)
        self.decoder = TransformerDecoderLayer(
            hidden_channels, num_heads, ffn_channels, dropout
        )
        self.prediction_head = QueryPredictionHead(hidden_channels)
        self.register_buffer("bev_positions", create_bev_position_grid(geometry), persistent=False)
        for parameter in self.decoder.parameters():
            if parameter.ndim > 1:
                nn.init.xavier_uniform_(parameter)
        for module in self.modules():
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
                module.momentum = 0.1
        self._operator_profile_ranges = False

    @contextmanager
    def operator_profile_ranges(self):
        if self._operator_profile_ranges:
            raise RuntimeError("TransFusion head profiler ranges are already active")
        self._operator_profile_ranges = True
        try:
            with ExitStack() as stack:
                stack.enter_context(self.decoder.operator_profile_ranges())
                yield self
        finally:
            self._operator_profile_ranges = False

    def _profile_range(self, name: str):
        if not self._operator_profile_ranges:
            return nullcontext()
        return torch.profiler.record_function(f"fl_v3::lidar_head::{name}")

    def _local_maximum(self, logits: torch.Tensor) -> torch.Tensor:
        heatmap = logits.detach().sigmoid()
        padding = self.nms_kernel_size // 2
        local_max = torch.zeros_like(heatmap)
        inner = F.max_pool2d(heatmap, self.nms_kernel_size, stride=1, padding=0)
        local_max[:, :, padding:-padding, padding:-padding] = inner
        # The reference exempts pedestrian and traffic cone from local suppression.
        local_max[:, 8] = heatmap[:, 8]
        local_max[:, 9] = heatmap[:, 9]
        return heatmap * (heatmap == local_max)

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        if features.ndim != 4 or features.shape[1] != 512:
            raise ValueError(f"TransFusion input must be [B,512,H,W], got {tuple(features.shape)}")
        if tuple(features.shape[-2:]) != self.geometry.bev_hw:
            raise ValueError(
                f"TransFusion BEV shape drift: got {tuple(features.shape[-2:])}, "
                f"expected {self.geometry.bev_hw}"
            )
        batch = features.shape[0]
        with self._profile_range("shared_conv"):
            lidar_features = self.shared_conv(features)
            flattened = lidar_features.flatten(2)
            positions = self.bev_positions.to(device=features.device).expand(batch, -1, -1)
        with self._profile_range("heatmap_head"):
            dense_heatmap = self.heatmap_head(lidar_features)
        with self._profile_range("local_maximum"):
            heatmap = self._local_maximum(dense_heatmap).flatten(2)
        with self._profile_range("proposal_full_argsort"):
            top = torch.argsort(heatmap.flatten(1), dim=-1, descending=True)[
                :, : self.num_proposals
            ]
            query_labels = torch.div(top, heatmap.shape[-1], rounding_mode="floor")
            query_indices = torch.remainder(top, heatmap.shape[-1])
        with self._profile_range("query_gather_encoding"):
            query_features = flattened.gather(
                2, query_indices[:, None].expand(-1, flattened.shape[1], -1)
            )
            one_hot = F.one_hot(query_labels, num_classes=self.num_classes).permute(0, 2, 1)
            query_features = query_features + self.class_encoding(one_hot.float())
            query_positions = positions.gather(
                1, query_indices[..., None].expand(-1, -1, 2)
            )
        with self._profile_range("decoder"):
            query_features = self.decoder(
                query_features, flattened, query_positions, positions
            )
        with self._profile_range("prediction_head"):
            output = self.prediction_head(query_features)
            output["center"] = output["center"] + query_positions.transpose(1, 2)
            output["query_heatmap_score"] = heatmap.gather(
                2, query_indices[:, None].expand(-1, self.num_classes, -1)
            )
        output["dense_heatmap"] = dense_heatmap
        output["query_labels"] = query_labels
        output["query_indices"] = query_indices
        return output

    @torch.no_grad()
    def decode(self, output: Mapping[str, torch.Tensor]) -> list[dict[str, torch.Tensor]]:
        required = {
            "heatmap", "center", "height", "dim", "rot", "vel",
            "query_heatmap_score", "query_labels",
        }
        missing = sorted(required - set(output))
        if missing:
            raise KeyError(f"TransFusion decode output is missing {missing}")
        query_scores = output["heatmap"].sigmoid()
        one_hot = F.one_hot(
            output["query_labels"], num_classes=self.num_classes
        ).permute(0, 2, 1)
        class_scores = query_scores * output["query_heatmap_score"] * one_hot
        scores, labels = class_scores.max(dim=1)
        boxes = decode_transfusion_tensors(
            output["center"], output["height"], output["dim"], output["rot"], output["vel"],
            self.geometry,
        )
        lower = boxes.new_tensor(self.geometry.post_center_range[:3])
        upper = boxes.new_tensor(self.geometry.post_center_range[3:])
        inside = ((boxes[:, :, :3] >= lower) & (boxes[:, :, :3] <= upper)).all(dim=2)
        decoded = []
        for batch_index in range(boxes.shape[0]):
            keep = inside[batch_index]
            selected = boxes[batch_index, keep]
            decoded.append(
                {
                    "boxes": selected[:, :7],
                    "scores": scores[batch_index, keep],
                    "labels": labels[batch_index, keep],
                    "velocity": selected[:, 7:9],
                }
            )
        return decoded


def _hungarian_matches(
    predicted_boxes: torch.Tensor,
    gt_boxes: torch.Tensor,
    gt_labels: torch.Tensor,
    query_logits: torch.Tensor,
    geometry: TransFusionGeometry,
    *,
    profile_range=None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    proposals, ground_truth = predicted_boxes.shape[0], gt_boxes.shape[0]
    if ground_truth == 0 or proposals == 0:
        empty = torch.empty((0,), dtype=torch.long, device=predicted_boxes.device)
        return empty, empty, predicted_boxes.new_zeros((proposals, ground_truth))
    profile = profile_range if profile_range is not None else lambda _name: nullcontext()
    with profile("hungarian_cost_gpu"):
        cls_cost = focal_loss_cost(query_logits.transpose(0, 1), gt_labels)
        x0, y0 = geometry.point_cloud_range[:2]
        extent_x = geometry.point_cloud_range[3] - x0
        extent_y = geometry.point_cloud_range[4] - y0
        start = predicted_boxes.new_tensor((x0, y0))
        extent = predicted_boxes.new_tensor((extent_x, extent_y))
        regression_cost = torch.cdist(
            (predicted_boxes[:, :2] - start) / extent,
            (gt_boxes[:, :2] - start) / extent,
            p=1,
        ) * 0.25
        iou = pairwise_iou3d(predicted_boxes, gt_boxes)
        cost = cls_cost + regression_cost - iou * 0.25
    with profile("hungarian_finite_sync"):
        if not bool(torch.isfinite(cost).all().detach().cpu()):
            raise FloatingPointError("TransFusion Hungarian cost contains non-finite values")
    try:
        from scipy.optimize import linear_sum_assignment
    except ImportError as exc:  # pragma: no cover - Arrhenius dependency gate
        raise RuntimeError("TransFusion Hungarian assignment requires scipy") from exc
    with profile("hungarian_d2h_scipy"):
        cpu_cost = cost.detach().to(device="cpu", dtype=torch.float64).numpy()
        row, column = linear_sum_assignment(cpu_cost)
    with profile("hungarian_h2d_indices"):
        row_tensor = torch.as_tensor(
            row, dtype=torch.long, device=predicted_boxes.device
        )
        column_tensor = torch.as_tensor(
            column, dtype=torch.long, device=predicted_boxes.device
        )
    return row_tensor, column_tensor, iou


class Phase1TransFusionLoss(nn.Module):
    """Reference dense/query losses and Hungarian target construction."""

    def __init__(
        self,
        geometry: TransFusionGeometry = PHASE1_TRANSFUSION_GEOMETRY,
        num_classes: int = 10,
        num_proposals: int = 200,
    ) -> None:
        super().__init__()
        if num_classes != 10 or num_proposals != 200:
            raise ValueError("Phase-I TransFusion loss requires 10 classes and 200 proposals")
        self.geometry = geometry
        self.num_classes = num_classes
        self.num_proposals = num_proposals
        self.register_buffer(
            "code_weights",
            torch.tensor(PHASE1_TRANSFUSION_CODE_WEIGHTS, dtype=torch.float32),
            persistent=False,
        )
        self.last_terms: dict[str, torch.Tensor] = {}
        self._operator_profile_ranges = False

    @contextmanager
    def operator_profile_ranges(self):
        """Enable output-neutral loss/Hungarian ranges for a bounded trace."""
        if self._operator_profile_ranges:
            raise RuntimeError("TransFusion loss profiler ranges are already active")
        self._operator_profile_ranges = True
        try:
            yield self
        finally:
            self._operator_profile_ranges = False

    def _profile_range(self, name: str):
        if not self._operator_profile_ranges:
            return nullcontext()
        return torch.profiler.record_function(f"fl_v3::lidar_loss::{name}")

    @staticmethod
    def _gaussian_focal(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        prediction = logits.sigmoid().clamp(1e-4, 1.0 - 1e-4)
        positive = target.eq(1.0).to(prediction.dtype)
        negative = target.lt(1.0).to(prediction.dtype)
        negative_weight = (1.0 - target).pow(4.0)
        positive_loss = -torch.log(prediction) * (1.0 - prediction).pow(2.0) * positive
        negative_loss = (
            -torch.log(1.0 - prediction)
            * prediction.pow(2.0)
            * negative_weight
            * negative
        )
        return (positive_loss.sum() + negative_loss.sum()) / positive.sum().clamp_min(1.0)

    @staticmethod
    def _sigmoid_focal(
        logits: torch.Tensor,
        labels: torch.Tensor,
        weights: torch.Tensor,
        num_positive: torch.Tensor,
        num_classes: int,
    ) -> torch.Tensor:
        target = logits.new_zeros(logits.shape)
        positive = labels < num_classes
        if positive.any():
            target[positive, labels[positive]] = 1.0
        probability = logits.sigmoid()
        pt = (1.0 - probability) * target + probability * (1.0 - target)
        focal_weight = (0.25 * target + 0.75 * (1.0 - target)) * pt.pow(2.0)
        loss = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        return (loss * focal_weight * weights[:, None]).sum() / num_positive.clamp_min(1.0)

    @torch.no_grad()
    def build_targets(
        self, output: Mapping[str, torch.Tensor], batch: Mapping[str, Sequence[torch.Tensor]]
    ) -> dict[str, torch.Tensor]:
        batch_size, classes, proposals = output["heatmap"].shape
        if classes != self.num_classes or proposals != self.num_proposals:
            raise ValueError("TransFusion query shape drift in target construction")
        if len(batch["gt_boxes"]) != batch_size or len(batch["gt_labels"]) != batch_size:
            raise ValueError("TransFusion GT batch size drift")
        device = output["heatmap"].device
        dtype = torch.float32
        labels = torch.full(
            (batch_size, proposals), self.num_classes, dtype=torch.long, device=device
        )
        label_weights = torch.ones((batch_size, proposals), dtype=dtype, device=device)
        bbox_targets = torch.zeros((batch_size, proposals, 10), dtype=dtype, device=device)
        bbox_weights = torch.zeros_like(bbox_targets)
        dense_target = torch.zeros(
            (batch_size, self.num_classes, *self.geometry.bev_hw), dtype=dtype, device=device
        )
        per_sample_matched_iou: list[torch.Tensor] = []
        decoded = decode_transfusion_tensors(
            output["center"].detach().float(),
            output["height"].detach().float(),
            output["dim"].detach().float(),
            output["rot"].detach().float(),
            output["vel"].detach().float(),
            self.geometry,
        )
        cell_x, cell_y = self.geometry.cell_xy
        x0, y0 = self.geometry.point_cloud_range[:2]
        height, width = self.geometry.bev_hw
        num_positive = torch.zeros((), dtype=dtype, device=device)
        for batch_index in range(batch_size):
            with self._profile_range("gt_prepare_validation"):
                gt7 = batch["gt_boxes"][batch_index].to(device=device, dtype=dtype)
                gt_labels = batch["gt_labels"][batch_index].to(
                    device=device, dtype=torch.long
                )
                if (
                    gt7.ndim != 2
                    or gt7.shape[1] != 7
                    or gt_labels.shape != (gt7.shape[0],)
                ):
                    raise ValueError("TransFusion GT tensor schema drift")
                if gt7.shape[0] and (
                    not bool(torch.isfinite(gt7).all().detach().cpu())
                    or not bool((gt7[:, 3:6] > 0).all().detach().cpu())
                    or not bool(
                        ((gt_labels >= 0) & (gt_labels < self.num_classes))
                        .all()
                        .detach()
                        .cpu()
                    )
                ):
                    raise ValueError("TransFusion GT contains invalid boxes or labels")
                velocity = torch.zeros(
                    (gt7.shape[0], 2), dtype=dtype, device=device
                )
                if "gt_velocity" in batch:
                    source_velocity = batch["gt_velocity"][batch_index].to(
                        device=device, dtype=dtype
                    )
                    if source_velocity.shape != velocity.shape:
                        raise ValueError("TransFusion GT velocity schema drift")
                    velocity.copy_(source_velocity)
                gt9 = torch.cat((gt7, velocity), dim=1)
            with self._profile_range(f"hungarian_sample_{batch_index}"):
                rows, columns, iou = _hungarian_matches(
                    decoded[batch_index],
                    gt9,
                    gt_labels,
                    output["heatmap"][batch_index].detach().float(),
                    self.geometry,
                    profile_range=self._profile_range,
                )
            if rows.numel():
                labels[batch_index, rows] = gt_labels[columns]
                bbox_targets[batch_index, rows] = encode_transfusion_boxes(
                    gt9[columns], self.geometry
                )
                bbox_weights[batch_index, rows] = 1.0
                per_sample_matched_iou.append(iou[rows, columns].mean())
                num_positive += rows.numel()
            else:
                per_sample_matched_iou.append(torch.zeros((), dtype=dtype, device=device))

            with self._profile_range("gaussian_target"):
                centers_x = (gt7[:, 0] - x0) / cell_x
                centers_y = (gt7[:, 1] - y0) / cell_y
                center_columns = centers_x.to(torch.int32)
                center_rows = centers_y.to(torch.int32)
                if gt7.shape[0] and not bool(
                    (
                        (center_columns >= 0)
                        & (center_columns < width)
                        & (center_rows >= 0)
                        & (center_rows < height)
                    ).all().detach().cpu()
                ):
                    raise ValueError(
                        "post-filter Phase-I GT center lies outside the TransFusion grid"
                    )
                cpu_geometry = (
                    torch.stack((gt7[:, 3] / cell_x, gt7[:, 4] / cell_y), dim=1)
                    .detach()
                    .cpu()
                    .tolist()
                )
                cpu_centers = (
                    torch.stack((center_columns, center_rows, gt_labels), dim=1)
                    .detach()
                    .cpu()
                    .tolist()
                )
                for gt_index, (column, row, label) in enumerate(cpu_centers):
                    radius = max(
                        2,
                        int(
                            gaussian_radius(
                                (
                                    cpu_geometry[gt_index][1],
                                    cpu_geometry[gt_index][0],
                                ),
                                min_overlap=0.1,
                            )
                        ),
                    )
                    draw_gaussian(
                        dense_target[batch_index, label], column, row, radius
                    )
        matched = torch.stack(per_sample_matched_iou).mean()
        return {
            "labels": labels,
            "label_weights": label_weights,
            "bbox_targets": bbox_targets,
            "bbox_weights": bbox_weights,
            "dense_heatmap": dense_target,
            "num_positive": num_positive,
            "matched_iou": matched,
        }

    def loss_terms(
        self, output: Mapping[str, torch.Tensor], batch: Mapping[str, Sequence[torch.Tensor]]
    ) -> dict[str, torch.Tensor]:
        required = {"heatmap", "center", "height", "dim", "rot", "vel", "dense_heatmap"}
        missing = sorted(required - set(output))
        if missing:
            raise KeyError(f"TransFusion loss output is missing {missing}")
        fp32 = {
            key: value.float() if torch.is_tensor(value) and value.is_floating_point() else value
            for key, value in output.items()
        }
        with self._profile_range("target_build"):
            targets = self.build_targets(fp32, batch)
        with self._profile_range("dense_heatmap_focal"):
            heatmap_loss = self._gaussian_focal(
                fp32["dense_heatmap"], targets["dense_heatmap"]
            )
        with self._profile_range("query_classification_focal"):
            query_logits = fp32["heatmap"].permute(0, 2, 1).reshape(
                -1, self.num_classes
            )
            classification_loss = self._sigmoid_focal(
                query_logits,
                targets["labels"].reshape(-1),
                targets["label_weights"].reshape(-1),
                targets["num_positive"],
                self.num_classes,
            )
        with self._profile_range("bbox_regression"):
            regression = torch.cat(
                (
                    fp32["center"],
                    fp32["height"],
                    fp32["dim"],
                    fp32["rot"],
                    fp32["vel"],
                ),
                dim=1,
            ).permute(0, 2, 1)
            regression_weight = (
                targets["bbox_weights"]
                * self.code_weights.to(device=regression.device)
            )
            bbox_loss = (
                (regression - targets["bbox_targets"]).abs() * regression_weight
            ).sum() / targets["num_positive"].clamp_min(1.0) * 0.25
        return {
            "loss_heatmap": heatmap_loss,
            "loss_cls": classification_loss,
            "loss_bbox": bbox_loss,
            "matched_iou": targets["matched_iou"],
        }

    def forward(
        self, output: Mapping[str, torch.Tensor], batch: Mapping[str, Sequence[torch.Tensor]]
    ) -> torch.Tensor:
        terms = self.loss_terms(output, batch)
        self.last_terms = {name: value.detach() for name, value in terms.items()}
        return terms["loss_heatmap"] + terms["loss_cls"] + terms["loss_bbox"]


__all__ = [
    "PHASE1_TRANSFUSION_CODE_WEIGHTS",
    "PHASE1_TRANSFUSION_GEOMETRY",
    "Phase1TransFusionHead",
    "Phase1TransFusionLoss",
    "QueryPredictionHead",
    "ReferenceMultiheadAttention",
    "TransFusionGeometry",
    "TransformerDecoderLayer",
    "create_bev_position_grid",
    "decode_transfusion_tensors",
    "encode_transfusion_boxes",
    "focal_loss_cost",
    "pairwise_iou3d",
    "pairwise_rotated_intersection_area",
]
