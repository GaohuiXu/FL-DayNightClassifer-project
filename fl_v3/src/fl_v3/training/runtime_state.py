"""Executed-update accounting and persistent epoch iteration for S06."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterator

import torch
from torch.utils.data import Sampler


@dataclass
class TrainingState:
    """Only successful optimizer updates advance update/exposure counters."""

    epoch: int = 0
    optimizer_step: int = 0
    exposure_samples: int = 0
    accumulation_phase: int = 0
    pending_samples: int = 0
    nonfinite_windows: int = 0
    overflow_windows: int = 0
    discarded_partial_windows: int = 0

    def checkpoint_dict(self) -> dict[str, int]:
        if self.accumulation_phase != 0 or self.pending_samples != 0:
            raise RuntimeError("checkpoint requires an optimizer-update boundary (no pending gradients)")
        return asdict(self)

    @classmethod
    def from_checkpoint(cls, raw: dict[str, Any]) -> "TrainingState":
        expected = set(cls.__dataclass_fields__)
        if set(raw) != expected:
            raise RuntimeError(
                f"training state fields mismatch: missing={sorted(expected-set(raw))}, "
                f"unknown={sorted(set(raw)-expected)}"
            )
        state = cls(**{k: int(v) for k, v in raw.items()})
        if state.accumulation_phase or state.pending_samples:
            raise RuntimeError("checkpoint contains an unsupported pending-gradient phase")
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

    def batches(self, epoch: int) -> Iterator[Any]:
        if id(self.loader) != self.loader_identity:
            raise RuntimeError("persistent loader identity drift")
        if self.sampler is not None and hasattr(self.sampler, "set_epoch"):
            self.sampler.set_epoch(int(epoch))
        return iter(self.loader)


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
_LIDAR_KEYS = frozenset({"lidar_points"})


def project_batch_for_mode(batch: Any, mode: str) -> Any:
    """Drop disabled tensors before host-to-device transfer.

    Dataset decode must already be mode-aware in production; this is the separate
    transfer/forward guard and is useful for synthetic interface tests.
    """
    if not isinstance(batch, dict):
        return batch
    if mode == "camera_only":
        allowed = _COMMON_KEYS | _CAMERA_KEYS
    elif mode == "lidar_only":
        allowed = _COMMON_KEYS | _LIDAR_KEYS
    elif mode == "fusion":
        allowed = _COMMON_KEYS | _CAMERA_KEYS | _LIDAR_KEYS
    else:
        raise ValueError(f"unknown model mode {mode!r}")
    return {k: v for k, v in batch.items() if k in allowed}
