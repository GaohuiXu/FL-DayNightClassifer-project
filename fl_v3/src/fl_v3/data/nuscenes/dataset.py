"""Deterministic nuScenes multimodal Dataset — the canonical sample schema (T1).

Returns, per keyframe: synchronized 6-camera images + ``LIDAR_TOP`` point cloud +
full calibration (intrinsics / per-sensor ego-poses / ``lidar2img``) + canonical
``LIDAR_TOP``-frame 3D-box GT + the per-box eligibility fields T4 needs. **No model,
no resize, no normalization** (those are T2): images are native uint8 1600×900.

**Determinism.** Reads the host-portable info-cache (samples ordered by
``sample_token``; box rows ``ann_token``-sorted). Image decoder is pinned to PIL
``Image.open().convert("RGB")`` (opencv decodes the same JPEG to different pixels).
Same keyframe loaded twice ⇒ bit-identical images / points / boxes. Multi-worker
loaders use ``seeded_worker_init``.

**The dict schema does NOT fit the T0 2-tuple ``(inputs, targets)`` loop / default
collate.** The custom ``collate_fn`` + ``ClientData``/``loop.py`` wiring for ragged
per-box tensors is a **T2 deliverable**; T1 does not modify ``training/loop.py`` or
``ClientData``.
"""
from __future__ import annotations

import hashlib
from typing import Dict, List, Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from fl_v3.data.nuscenes import info_cache as IC
from fl_v3.data.nuscenes import paths as P
from fl_v3.utils.runtime import seeded_worker_init

# Frozen cam order — a test asserts the schema matches this exactly.
CAM_ORDER = P.CAMERA_CHANNELS
IMAGE_HW = (900, 1600)  # native nuScenes (H, W)


def _decode_image_chw(abs_path: str) -> np.ndarray:
    """Pinned decoder: PIL RGB → uint8 ``(3,H,W)``. NOT opencv (different pixels)."""
    with Image.open(abs_path) as im:
        rgb = im.convert("RGB")  # ensure 3-channel, drop any alpha/palette
        arr = np.asarray(rgb, dtype=np.uint8)  # (H,W,3)
    return np.ascontiguousarray(arr.transpose(2, 0, 1))  # (3,H,W)


def _load_lidar(abs_path: str) -> np.ndarray:
    """Load ``.pcd.bin`` → ``(P,5)`` float32 = ``x,y,z,intensity,ring`` (LIDAR_TOP).

    The devkit ``LidarPointCloud.from_file`` keeps only the first 4 cols (drops
    ``ring``); we carry all 5 as a conscious superset (devkit-parity tests compare
    only cols 0:4).
    """
    raw = np.fromfile(abs_path, dtype=np.float32)
    return raw.reshape(-1, 5)


class NuScenesMultimodalDataset(Dataset):
    """Canonical-schema nuScenes Dataset over one split's keyframe info-cache."""

    def __init__(
        self,
        info_list: List[dict],
        dataroot: str,
        sample_tokens: Optional[List[str]] = None,
    ):
        """``info_list`` from :mod:`info_cache`. If ``sample_tokens`` is given, the
        dataset is restricted to (and ordered by) those tokens — this is how a
        per-client / per-split shard is materialized without rebuilding the cache.
        """
        self.dataroot = dataroot
        by_token = {i["sample_token"]: i for i in info_list}
        if sample_tokens is None:
            order = sorted(by_token)
        else:
            missing = [t for t in sample_tokens if t not in by_token]
            if missing:
                raise KeyError(f"{len(missing)} sample_tokens not in info_list (e.g. {missing[:3]})")
            order = list(sample_tokens)  # caller-controlled, already deterministic
        self._infos = [by_token[t] for t in order]

    def __len__(self) -> int:
        return len(self._infos)

    @property
    def sample_tokens(self) -> List[str]:
        return [i["sample_token"] for i in self._infos]

    def __getitem__(self, idx: int) -> Dict[str, object]:
        info = self._infos[idx]
        # --- images (6,3,900,1600) uint8, row i ↔ cam_order[i] ---
        imgs = np.stack([
            _decode_image_chw(P.abspath_from_relative(rel, self.dataroot))
            for rel in info["cam_rel_paths"]
        ])  # (6,3,H,W)
        # --- lidar (P,5) f32 ---
        pts = _load_lidar(P.abspath_from_relative(info["lidar_rel_path"], self.dataroot))

        M = int(info["gt_boxes"].shape[0])
        sample = {
            # identity + Q2 substrate
            "sample_token": info["sample_token"],
            "scene_token": info["scene_token"],
            "log_token": info["log_token"],
            "location": info["location"],
            "timestamp": int(info["timestamp"]),
            "cam_order": tuple(info["cam_order"]),
            # images + calibration (native resolution; NO resize/norm)
            "images": torch.from_numpy(imgs),                                  # uint8 [6,3,900,1600]
            "cam_intrinsics": torch.from_numpy(info["cam_intrinsics"].astype(np.float32)),   # [6,3,3]
            "lidar2img": torch.from_numpy(info["lidar2img"].astype(np.float32)),             # [6,4,4]
            "cam2ego": torch.from_numpy(info["cam2ego"].astype(np.float32)),                 # [6,4,4]
            "ego2global_cam": torch.from_numpy(info["ego2global_cam"].astype(np.float32)),   # [6,4,4]
            "lidar_points": torch.from_numpy(np.ascontiguousarray(pts)),                     # f32 [P,5]
            "lidar2ego": torch.from_numpy(info["lidar2ego"].astype(np.float32)),             # [4,4]
            "ego2global_lidar": torch.from_numpy(info["ego2global_lidar"].astype(np.float32)),  # [4,4]
            # GT (canonical LIDAR_TOP frame), ann_token-sorted
            "gt_boxes": torch.from_numpy(info["gt_boxes"].astype(np.float32)),       # [M,7]
            "gt_velocity": torch.from_numpy(info["gt_velocity"].astype(np.float32)), # [M,2]
            "gt_labels": torch.from_numpy(info["gt_labels"].astype(np.int64)),       # [M]
            "gt_names": list(info["gt_names"]),                                       # [M] str
            "gt_num_lidar_pts": torch.from_numpy(info["gt_num_lidar_pts"].astype(np.int64)),  # [M]
            "gt_visibility": torch.from_numpy(info["gt_visibility"].astype(np.int64)),        # [M]
            "gt_in_range": torch.from_numpy(info["gt_in_range"].astype(bool)),                # [M]
            "gt_attribute": list(info["gt_attribute"]),                               # [M] str
            "gt_instance_tokens": list(info["gt_instance_tokens"]),                   # [M] str
            "gt_ann_tokens": list(info["gt_ann_tokens"]),                            # [M] str
            "num_boxes": M,
        }
        return sample


def sample_image_sha256(sample: Dict[str, object]) -> str:
    """sha256 of the decoded image tensor bytes — for the pinned-decoder gate."""
    arr = sample["images"].numpy()
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


def make_loader(
    dataset: NuScenesMultimodalDataset,
    batch_size: int = 1,
    shuffle: bool = False,
    num_workers: int = 0,
    seed: int = 42,
    collate_fn=None,
) -> DataLoader:
    """DataLoader with the seeded worker init (determinism harness).

    ``collate_fn`` is a **T2 deliverable** (the dict schema has ragged per-box
    tensors); T1 callers pass ``batch_size=1`` (identity collate of a singleton
    list) or a trivial list-collate for raw inspection.
    """
    g = torch.Generator()
    g.manual_seed(int(seed))
    if collate_fn is None:
        collate_fn = _list_collate
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        worker_init_fn=seeded_worker_init,
        generator=g,
        drop_last=False,
        collate_fn=collate_fn,
    )


def _list_collate(batch: List[dict]) -> List[dict]:
    """Trivial T1 collate: keep samples as a list (no tensor stacking).

    The real ragged-tensor ``collate_fn`` is T2; T1 only needs to iterate samples
    for determinism / viz checks, so default-collate's same-shape requirement
    (which the dict schema violates) is sidestepped.
    """
    return batch
