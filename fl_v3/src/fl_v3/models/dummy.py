"""Tiny deterministic models for T0 smoke tests (NOT the AD model).

The real BEVFusion-class model (Swin-T + PointPillars + ConvFuser + CenterPoint
head) is built in T2. These tiny modules exist only so T0 can exercise the
determinism harness and the task-agnostic FL skeleton without nuScenes or a
heavy model. No BatchNorm (avoids a needless nondeterminism surface in the
smoke); pure Linear + ReLU.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class TinyMLP(nn.Module):
    """A minimal MLP: ``in_dim -> hidden -> out_dim``.

    Used for both the dummy regression task (out_dim=1, MSE loss) and any tiny
    classification smoke (out_dim=num_classes, injected CE loss). The skeleton
    treats it as an opaque ``nn.Module`` — the loss is supplied by the Task, not
    by the model.
    """

    def __init__(self, in_dim: int = 4, hidden: int = 8, out_dim: int = 1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
