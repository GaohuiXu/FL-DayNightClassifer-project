"""Executed-update accounting and persistent epoch iteration for S06."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch.utils.data import Sampler


@dataclass
class TrainingState:
    """Cumulative fixed-window accounting; every evaluated sample reconciles."""

    epoch: int = 0
    optimizer_step: int = 0
    exposure_samples: int = 0
    attempted_microbatches: int = 0
    attempted_samples: int = 0
    loss_evaluated_samples: int = 0
    attempted_windows: int = 0
    successful_windows: int = 0
    invalid_windows: int = 0
    invalid_samples: int = 0
    accumulation_phase: int = 0
    pending_samples: int = 0
    nonfinite_windows: int = 0
    overflow_windows: int = 0
    discarded_windows: int = 0
    discarded_samples: int = 0

    def validate(self, *, checkpoint_boundary: bool) -> None:
        for name in self.__dataclass_fields__:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise RuntimeError(f"training state {name} must be a non-negative integer")
        if checkpoint_boundary and (self.accumulation_phase or self.pending_samples):
            raise RuntimeError("checkpoint contains an unsupported pending-gradient phase")
        if self.loss_evaluated_samples != self.attempted_samples:
            raise RuntimeError("training state loss/attempted sample accounting mismatch")
        if self.attempted_samples != (
            self.exposure_samples + self.invalid_samples + self.discarded_samples
        ):
            raise RuntimeError("training state attempted sample accounting mismatch")
        if self.attempted_windows != (
            self.successful_windows + self.invalid_windows + self.discarded_windows
        ):
            raise RuntimeError("training state attempted window accounting mismatch")
        if self.optimizer_step != self.successful_windows:
            raise RuntimeError("training state optimizer/successful-window accounting mismatch")
        if self.invalid_windows != self.nonfinite_windows + self.overflow_windows:
            raise RuntimeError("training state invalid-window cause accounting mismatch")
        if (self.discarded_windows == 0) != (self.discarded_samples == 0):
            raise RuntimeError("training state discarded window/sample accounting mismatch")

    def checkpoint_dict(self) -> dict[str, int]:
        self.validate(checkpoint_boundary=True)
        return asdict(self)

    @classmethod
    def from_checkpoint(cls, raw: dict[str, Any]) -> "TrainingState":
        expected = set(cls.__dataclass_fields__)
        if set(raw) != expected:
            raise RuntimeError(
                f"training state fields mismatch: missing={sorted(expected-set(raw))}, "
                f"unknown={sorted(set(raw)-expected)}"
            )
        if any(isinstance(v, bool) or not isinstance(v, int) for v in raw.values()):
            raise RuntimeError("training state values must be integers")
        state = cls(**raw)
        state.validate(checkpoint_boundary=True)
        return state


class PersistentEpochIterator:
    """Reuse one loader/sampler object and deterministically set each epoch."""

    def __init__(self, loader: Any, sampler: Any | None = None):
        self.loader = loader
        self.sampler = sampler if sampler is not None else getattr(loader, "sampler", None)
        if self.sampler is None or not hasattr(self.sampler, "set_epoch"):
            raise RuntimeError(
                "production training requires a sampler with deterministic set_epoch; "
                "RandomSampler generator-state resume is refused"
            )
        self.loader_identity = id(loader)

    def batches(self, epoch: int) -> Any:
        if id(self.loader) != self.loader_identity:
            raise RuntimeError("persistent loader identity drift")
        if self.sampler is not None and hasattr(self.sampler, "set_epoch"):
            self.sampler.set_epoch(int(epoch))
        return self.loader


class EpochPermutationSampler(Sampler[int]):
    """One deterministic, complete permutation per epoch, resumable by epoch number."""

    def __init__(self, data_source: Any, *, seed: int):
        self.data_source = data_source
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        if int(epoch) < 0:
            raise ValueError("epoch must be non-negative")
        self.epoch = int(epoch)

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        return iter(torch.randperm(len(self.data_source), generator=generator).tolist())

    def __len__(self) -> int:
        return len(self.data_source)


_COMMON_KEYS = frozenset({
    "gt_boxes", "gt_labels", "gt_num_lidar_pts", "gt_in_range", "gt_velocity",
    "gt_ann_tokens", "gt_names", "gt_attribute", "sample_token", "lidar2ego",
    "ego2global_lidar", "batch_size",
})
_CAMERA_KEYS = frozenset({"images", "lidar2img", "cam_intrinsics"})
_DIAGNOSTIC_CAMERA_KEYS = frozenset({"augmentation_params"})
_LIDAR_KEYS = frozenset({"lidar_points", "lidar_point_offsets"})


def project_batch_for_mode(batch: Any, mode: str) -> Any:
    """Drop disabled tensors before host-to-device transfer.

    Dataset decode must already be mode-aware in production; this is the separate
    transfer/forward guard and is useful for synthetic interface tests.
    """
    if not isinstance(batch, dict):
        return batch
    if mode == "camera_only":
        allowed = _COMMON_KEYS | _CAMERA_KEYS | _DIAGNOSTIC_CAMERA_KEYS
    elif mode == "lidar_only":
        allowed = _COMMON_KEYS | _LIDAR_KEYS
    elif mode == "fusion":
        allowed = _COMMON_KEYS | _CAMERA_KEYS | _DIAGNOSTIC_CAMERA_KEYS | _LIDAR_KEYS
    else:
        raise ValueError(f"unknown model mode {mode!r}")
    return {k: v for k, v in batch.items() if k in allowed}
