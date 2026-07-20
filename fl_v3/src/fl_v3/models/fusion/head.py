"""Framework-independent multi-task CenterHead for nuScenes.

Reference contract
------------------
The topology and task grouping follow the archived MIT BEVFusion CenterHead at
commit ``326653dc06e0938edf1aae7d01efcd158ba83de5``.  The implementation is the
owner-approved O-018 **reference-faithful no-starvation adaptation**:

* one shared ``3x3 Conv -> normalization -> ReLU`` feature transform;
* six official nuScenes task heads;
* each task has independent two-convolution ``heatmap``, ``reg``, ``height``,
  ``dim``, ``rot``, and ``vel`` branches;
* the legacy/Fusion path keeps its reviewed GroupNorm default, while the S10
  Phase-I standalone Camera explicitly requests the reference BatchNorm form.

Candidate selection and NMS intentionally live in :mod:`centerhead_decode` rather
than in this module.  O-018 removes the reference coder's *second* task-wide top-K
while retaining per-class K=500; multi-class decode is therefore not claimed to be
element-wise identical to the official implementation.  Single-class tasks retain
the official candidate semantics.

The regression field order is ``offset_xy, z, log(l,w,h), sin/cos(yaw), vx/vy``.
Decoded boxes use the project canonical gravity-center convention
``(cx,cy,cz,l,w,h,yaw)`` in ``LIDAR_TOP``.
"""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn as nn


# Canonical global label order from nuscenes-devkit 1.1.11.  Keep this local to
# the model module so importing the head does not require importing the devkit.
NUSCENES_DETECTION_NAMES: Tuple[str, ...] = (
    "car",
    "truck",
    "bus",
    "trailer",
    "construction_vehicle",
    "pedestrian",
    "motorcycle",
    "bicycle",
    "traffic_cone",
    "barrier",
)

# Official CenterPoint/BEVFusion task order.  It is deliberately *not* the same
# as NUSCENES_DETECTION_NAMES, so callers must map by name, never task offsets.
NUSCENES_CENTERHEAD_TASKS: Tuple[Tuple[str, ...], ...] = (
    ("car",),
    ("truck", "construction_vehicle"),
    ("bus", "trailer"),
    ("barrier",),
    ("motorcycle", "bicycle"),
    ("pedestrian", "traffic_cone"),
)

REG_CHANNELS: Dict[str, int] = {
    "reg": 2,
    "height": 1,
    "dim": 3,
    "rot": 2,
    "vel": 2,
}


def _group_norm(channels: int, max_groups: int = 32) -> nn.GroupNorm:
    groups = min(int(max_groups), int(channels))
    while channels % groups != 0:
        groups -= 1
    return nn.GroupNorm(groups, channels)


def _normalization(channels: int, kind: str) -> nn.Module:
    if kind == "group_norm":
        return _group_norm(channels)
    if kind == "batch_norm":
        return nn.BatchNorm2d(channels, eps=1e-5, momentum=0.1)
    raise ValueError(f"unknown CenterHead normalization {kind!r}")


def _validate_tasks(tasks: Sequence[Sequence[str]]) -> Tuple[Tuple[str, ...], ...]:
    normalized = tuple(tuple(str(name) for name in task) for task in tasks)
    if not normalized or any(not task for task in normalized):
        raise ValueError("CenterHead tasks must be a non-empty sequence of non-empty tasks")
    flat = tuple(name for task in normalized for name in task)
    if len(flat) != len(set(flat)):
        raise ValueError(f"CenterHead classes must be unique across tasks, got {flat}")
    unknown = sorted(set(flat) - set(NUSCENES_DETECTION_NAMES))
    if unknown:
        raise ValueError(f"unknown nuScenes detection classes in tasks: {unknown}")
    return normalized


class _TwoConvBranch(nn.Module):
    """Official two-convolution branch with an explicit normalization policy."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        *,
        init_bias: float | None = None,
        normalization: str = "group_norm",
    ) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 3, padding=1, bias=False),
            _normalization(hidden_channels, normalization),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, out_channels, 3, padding=1, bias=True),
        )
        if init_bias is not None:
            nn.init.constant_(self.layers[-1].bias, float(init_bias))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class SeparateTaskHead(nn.Module):
    """One task's independent heatmap and five regression field branches."""

    def __init__(
        self,
        in_channels: int,
        class_names: Sequence[str],
        *,
        head_channels: int = 64,
        init_bias: float = -2.19,
        normalization: str = "group_norm",
    ) -> None:
        super().__init__()
        self.class_names = tuple(str(name) for name in class_names)
        if not self.class_names:
            raise ValueError("a task head must contain at least one class")
        self.branches = nn.ModuleDict()
        self.branches["heatmap"] = _TwoConvBranch(
            in_channels,
            head_channels,
            len(self.class_names),
            init_bias=init_bias,
            normalization=normalization,
        )
        for name, channels in REG_CHANNELS.items():
            self.branches[name] = _TwoConvBranch(
                in_channels,
                head_channels,
                channels,
                normalization=normalization,
            )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {name: branch(x) for name, branch in self.branches.items()}


class CenterPointHead(nn.Module):
    """Official six-task nuScenes CenterHead topology.

    ``forward`` returns a list in task order.  Each item is a dictionary whose
    fields are ``heatmap/reg/height/dim/rot/vel``.  The production detector and
    loss wiring are intentionally outside S05 ownership and must be updated by
    S07 against this explicit list-of-task-dicts contract.
    """

    def __init__(
        self,
        in_channels: int,
        n_classes: int = 10,
        head_channels: int = 64,
        init_bias: float = -2.19,
        conv_layers: int = 2,
        *,
        tasks: Sequence[Sequence[str]] = NUSCENES_CENTERHEAD_TASKS,
        shared_channels: int = 64,
        normalization: str = "group_norm",
    ) -> None:
        super().__init__()
        if int(conv_layers) != 2:
            raise ValueError(
                "O-018 freezes two convolutions per task field; "
                f"got conv_layers={conv_layers}"
            )
        self.class_names = _validate_tasks(tasks)
        total_classes = sum(len(task) for task in self.class_names)
        if int(n_classes) != total_classes:
            raise ValueError(
                f"n_classes={n_classes} disagrees with task total {total_classes}"
            )
        self.n_classes = total_classes
        self.shared = nn.Sequential(
            nn.Conv2d(in_channels, shared_channels, 3, padding=1, bias=False),
            _normalization(shared_channels, normalization),
            nn.ReLU(inplace=True),
        )
        self.task_heads = nn.ModuleList(
            SeparateTaskHead(
                shared_channels,
                task,
                head_channels=head_channels,
                init_bias=init_bias,
                normalization=normalization,
            )
            for task in self.class_names
        )

    def forward(self, x: torch.Tensor) -> List[Dict[str, torch.Tensor]]:
        shared = self.shared(x)
        return [head(shared) for head in self.task_heads]


__all__ = [
    "CenterPointHead",
    "SeparateTaskHead",
    "REG_CHANNELS",
    "NUSCENES_CENTERHEAD_TASKS",
    "NUSCENES_DETECTION_NAMES",
]
