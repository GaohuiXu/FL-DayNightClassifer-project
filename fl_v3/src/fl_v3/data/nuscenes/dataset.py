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
import os
from io import BytesIO
from typing import Dict, List, Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from fl_v3.data.nuscenes import paths as P
from fl_v3.data.nuscenes.zip_backend import NuScenesBlobStore, canonical_member_path
from fl_v3.utils.runtime import seeded_worker_init
from fl_v3.utils.runtime import normalize_model_mode

# Frozen cam order — a test asserts the schema matches this exactly.
CAM_ORDER = P.CAMERA_CHANNELS
IMAGE_HW = (900, 1600)  # native nuScenes (H, W)


def _decode_image_bytes_chw(payload: bytes) -> np.ndarray:
    """Pinned decoder over immutable bytes: PIL RGB → uint8 ``(3,H,W)``."""
    with Image.open(BytesIO(payload)) as im:
        rgb = im.convert("RGB")  # ensure 3-channel, drop any alpha/palette
        arr = np.asarray(rgb, dtype=np.uint8)  # (H,W,3)
    return np.ascontiguousarray(arr.transpose(2, 0, 1))  # (3,H,W)


_COMPAT_BLOB_STORES: dict[tuple[str, str], NuScenesBlobStore] = {}


def _infer_root_and_relative(abs_path: str) -> tuple[str, str]:
    """Recover ``(dataroot, member)`` from a devkit-style missing blob path.

    This keeps permission-out callers such as ``build_gt_database.py`` working:
    the devkit still returns ``<module-root>/samples/...`` even though that file
    exists only inside ZIP.  New code should pass a store explicitly.
    """
    absolute = os.path.abspath(abs_path)
    normalized = absolute.replace(os.sep, "/")
    for marker in ("/samples/", "/sweeps/"):
        if marker in normalized:
            prefix, suffix = normalized.split(marker, 1)
            return prefix or os.sep, canonical_member_path(marker.strip("/") + "/" + suffix)
    raise FileNotFoundError(
        f"cannot infer nuScenes dataroot/member from missing blob path {abs_path!r}"
    )


def _compat_blob_store(dataroot: str) -> NuScenesBlobStore:
    manifest = P.get_zip_manifest()
    key = (os.path.abspath(dataroot), manifest)
    store = _COMPAT_BLOB_STORES.get(key)
    if store is None:
        # Let the store ignore a shell-exported module manifest when ``dataroot``
        # is an extracted mini/local tree; ZIP roots still discover the env value.
        store = NuScenesBlobStore(key[0])
        _COMPAT_BLOB_STORES[key] = store
    return store


def _read_blob_bytes(
    path: str,
    *,
    blob_store: NuScenesBlobStore | None = None,
    dataroot: str | None = None,
) -> bytes:
    if blob_store is not None:
        rel = P.relative_to_dataroot(path, blob_store.dataroot) if os.path.isabs(path) else path
        return blob_store.read_bytes(rel)
    if dataroot is not None:
        rel = P.relative_to_dataroot(path, dataroot) if os.path.isabs(path) else path
        return _compat_blob_store(dataroot).read_bytes(rel)
    try:
        with open(path, "rb") as stream:
            return stream.read()
    except FileNotFoundError:
        root, rel = _infer_root_and_relative(path)
        return _compat_blob_store(root).read_bytes(rel)


def _load_lidar_bytes(payload: bytes, source: str = "<bytes>") -> np.ndarray:
    """Load ``.pcd.bin`` → ``(P,5)`` float32 = ``x,y,z,intensity,ring`` (LIDAR_TOP).

    The devkit ``LidarPointCloud.from_file`` keeps only the first 4 cols (drops
    ``ring``); we carry all 5 as a conscious superset (devkit-parity tests compare
    only cols 0:4).
    """
    if len(payload) % (5 * np.dtype(np.float32).itemsize) != 0:
        raise ValueError(
            f"LiDAR blob {source!r} has {len(payload)} bytes, not divisible by 5 float32 columns"
        )
    # ``copy`` preserves np.fromfile's writable-array semantics.  A bare
    # frombuffer(bytes) array is read-only and unsafe to expose through torch.
    return np.frombuffer(payload, dtype=np.float32).copy().reshape(-1, 5)


def _load_lidar(
    path: str,
    *,
    blob_store: NuScenesBlobStore | None = None,
    dataroot: str | None = None,
) -> np.ndarray:
    return _load_lidar_bytes(
        _read_blob_bytes(path, blob_store=blob_store, dataroot=dataroot), source=path
    )


def _load_multisweep(
    info: dict,
    dataroot: str,
    n_sweeps: int,
    blob_store: NuScenesBlobStore | None = None,
) -> np.ndarray:
    """Accumulate the keyframe + up to ``n_sweeps-1`` PREV sweeps into the keyframe LIDAR_TOP
    frame → ``(Ptot, 6)`` f32 = ``x,y,z,intensity,ring,dt`` (the keyframe has ``dt=0``).

    Each prev sweep's XYZ is ego-motion-compensated by the cache-precomputed ``sweep2keylidar``
    4×4 (devkit ``from_file_multisweep`` convention); intensity/ring are carried as-is and the
    per-point ``dt`` (seconds into the past) is appended as the timestamp channel. Deterministic
    (pure index/matmul; sweep order is the cache's fixed prev-walk order)."""
    store = blob_store or _compat_blob_store(dataroot)
    sweeps = info.get("lidar_sweeps", [])[: max(0, n_sweeps - 1)]
    rel_paths = [info["lidar_rel_path"], *(sw["rel_path"] for sw in sweeps)]
    payloads = store.read_many(rel_paths)
    key = _load_lidar_bytes(payloads[0], source=rel_paths[0])                       # (P0,5)
    clouds = [np.concatenate([key, np.zeros((key.shape[0], 1), np.float32)], axis=1)]  # (P0,6), dt=0
    for sw, payload in zip(sweeps, payloads[1:]):
        p = _load_lidar_bytes(payload, source=sw["rel_path"])                       # (Pi,5)
        m = sw["sweep2keylidar"].astype(np.float32)                                # (4,4)
        xyz1 = np.concatenate([p[:, :3], np.ones((p.shape[0], 1), np.float32)], axis=1)  # (Pi,4)
        xyz_key = (xyz1 @ m.T)[:, :3]                                               # (Pi,3) in key frame
        dt = np.full((p.shape[0], 1), np.float32(sw["dt"]), dtype=np.float32)
        clouds.append(np.concatenate([xyz_key, p[:, 3:5], dt], axis=1))            # (Pi,6)
    return np.ascontiguousarray(np.concatenate(clouds, axis=0))


class NuScenesMultimodalDataset(Dataset):
    """Canonical-schema nuScenes Dataset over one split's keyframe info-cache."""

    def __init__(
        self,
        info_list: List[dict],
        dataroot: str,
        sample_tokens: Optional[List[str]] = None,
        n_sweeps: int = 1,
        augment: Optional[dict] = None,
        gtpaste: Optional[dict] = None,
        zip_manifest: Optional[str] = None,
        model_mode: str = "fusion",
    ):
        """``info_list`` from :mod:`info_cache`. If ``sample_tokens`` is given, the
        dataset is restricted to (and ordered by) those tokens — this is how a
        per-client / per-split shard is materialized without rebuilding the cache.

        ``n_sweeps`` (MCR P1 lever): >1 accumulates the keyframe + (n_sweeps-1) prev LIDAR_TOP
        sweeps (ego-motion-compensated) and appends a per-point ``dt`` channel ⇒ ``lidar_points``
        is ``(P,6)`` instead of ``(P,5)``. Requires a multi-sweep cache (infos carry ``lidar_sweeps``).
        ``n_sweeps=1`` is the byte-identical single-keyframe path.

        ``augment`` (MCR P1, the overfitting fix): a dict of BEV/3D aug params (see
        :mod:`augment`); when set, each ``__getitem__`` applies a seeded GlobalRotScaleTrans+flip to the
        points/boxes/velocity/lidar2img. **TRAIN-ONLY** — pass it ONLY for the training dataset; eval
        leaves it ``None`` (clean). Default ``None`` ⇒ byte-identical to the un-augmented path.

        ``zip_manifest`` is normally discovered from ``NUSCENES_ZIP_MANIFEST``;
        passing it explicitly is useful for data-only tools and tests.
        """
        self.dataroot = dataroot
        self.n_sweeps = int(n_sweeps)
        self.augment = augment
        self.gtpaste = gtpaste
        self.model_mode = normalize_model_mode(model_mode)
        self.blob_store = NuScenesBlobStore(dataroot, manifest_path=zip_manifest)
        by_token = {i["sample_token"]: i for i in info_list}
        if sample_tokens is None:
            order = sorted(by_token)
        else:
            missing = [t for t in sample_tokens if t not in by_token]
            if missing:
                raise KeyError(f"{len(missing)} sample_tokens not in info_list (e.g. {missing[:3]})")
            order = list(sample_tokens)  # caller-controlled, already deterministic
        self._infos = [by_token[t] for t in order]
        if self.n_sweeps < 1:
            raise ValueError(f"n_sweeps must be >= 1, got {self.n_sweeps}")
        for index, info in enumerate(self._infos):
            declared = info.get("_cache_n_sweeps")
            if declared != self.n_sweeps:
                raise ValueError(
                    f"info record {index} ({info.get('sample_token', '<unknown>')}) "
                    f"declares _cache_n_sweeps={declared!r}, requested {self.n_sweeps}"
                )
            has_sweeps = "lidar_sweeps" in info
            if self.n_sweeps == 1 and has_sweeps:
                raise ValueError(f"single-sweep info record {index} contains lidar_sweeps")
            if self.n_sweeps > 1 and not has_sweeps:
                raise ValueError(f"multi-sweep info record {index} has no lidar_sweeps")
            if has_sweeps and len(info["lidar_sweeps"]) > self.n_sweeps - 1:
                raise ValueError(
                    f"info record {index} has {len(info['lidar_sweeps'])} previous sweeps, "
                    f"exceeding requested n_sweeps={self.n_sweeps}"
                )

    def __len__(self) -> int:
        return len(self._infos)

    @property
    def sample_tokens(self) -> List[str]:
        return [i["sample_token"] for i in self._infos]

    def __getitem__(self, idx: int) -> Dict[str, object]:
        info = self._infos[idx]
        use_camera = self.model_mode in {"camera_only", "fusion"}
        use_lidar = self.model_mode in {"lidar_only", "fusion"}
        imgs = None
        pts = None
        if use_camera:
            image_payloads = self.blob_store.read_many(info["cam_rel_paths"])
            imgs = np.stack([_decode_image_bytes_chw(payload) for payload in image_payloads])
        if use_lidar:
            if self.n_sweeps > 1:
                pts = _load_multisweep(info, self.dataroot, self.n_sweeps, self.blob_store)
            else:
                pts = _load_lidar(info["lidar_rel_path"], blob_store=self.blob_store)

        M = int(info["gt_boxes"].shape[0])
        sample = {
            # identity + Q2 substrate
            "sample_token": info["sample_token"],
            "scene_token": info["scene_token"],
            "log_token": info["log_token"],
            "location": info["location"],
            "timestamp": int(info["timestamp"]),
            "cam_order": tuple(info["cam_order"]),
            "model_mode": self.model_mode,
            # Calibration and poses remain available even when their raw payload is disabled.
            "cam_intrinsics": torch.from_numpy(info["cam_intrinsics"].astype(np.float32)),   # [6,3,3]
            "lidar2img": torch.from_numpy(info["lidar2img"].astype(np.float32)),             # [6,4,4]
            "cam2ego": torch.from_numpy(info["cam2ego"].astype(np.float32)),                 # [6,4,4]
            "ego2global_cam": torch.from_numpy(info["ego2global_cam"].astype(np.float32)),   # [6,4,4]
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
        if imgs is not None:
            sample["images"] = torch.from_numpy(imgs)
        if pts is not None:
            sample["lidar_points"] = torch.from_numpy(np.ascontiguousarray(pts))
        # TRAIN-ONLY GT-paste (rare-class object copy-paste). BEFORE the BEV aug so the scene transform T
        # transforms pasted points/boxes/velocity CONSISTENTLY with the host scene. Default None ⇒ byte-identical.
        if self.gtpaste is not None:
            from fl_v3.data.nuscenes.gt_paste import paste_sample
            sample = paste_sample(sample, self.gtpaste, self.n_sweeps)
        # TRAIN-ONLY BEV/3D augmentation (seeded via seeded_worker_init's per-worker numpy seed).
        if self.augment is not None:
            from fl_v3.data.nuscenes.augment import augment_sample
            sample = augment_sample(sample, self.augment)
        return sample

    @property
    def blob_backend(self) -> str:
        return self.blob_store.mode

    @property
    def payload_read_counts(self) -> dict:
        state = self.blob_store.debug_state()
        return {
            "reads": state["modality_read_count"],
            "bytes": state["modality_byte_count"],
        }

    def close(self) -> None:
        self.blob_store.close()


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
    sampler=None,
    drop_last: bool = False,
    multiprocessing_context=None,
) -> DataLoader:
    """DataLoader with the seeded worker init (determinism harness).

    ``collate_fn`` is a **T2 deliverable** (the dict schema has ragged per-box
    tensors); T1 callers pass ``batch_size=1`` (identity collate of a singleton
    list) or a trivial list-collate for raw inspection.

    ``sampler`` (MCR P2): when a ``DistributedSampler`` is passed (4-GPU DDP centralized training),
    ``shuffle`` is forced off (sampler and shuffle are mutually exclusive); the sampler owns the per-epoch
    shuffle (seed it with ``seed+epoch`` + ``set_epoch`` so order stays a pure function of (seed, epoch)).
    ``drop_last`` (MCR P2): drop the partial final batch — needed when the static-shape BEV stack is
    torch.compile'd (a partial batch would force one recompile/epoch)."""
    g = torch.Generator()
    g.manual_seed(int(seed))
    if collate_fn is None:
        collate_fn = _list_collate
    if sampler is not None:
        shuffle = False                       # sampler and shuffle are mutually exclusive
    # L5 (D15, value-neutral scheduling): pin_memory for faster H2D, and keep workers warm +
    # prefetch when num_workers>0. The dataset is RNG-free across workers (resize+normalize only;
    # asserted by test_two_worker_batch_equals_zero_worker), so these change timing, NOT batch content
    # — byte-identical in BOTH determinism levels (the shuffle generator g is unaffected).
    extra = {}
    if int(num_workers) > 0:
        extra.update(persistent_workers=True, prefetch_factor=4)
        if multiprocessing_context is not None:
            extra["multiprocessing_context"] = multiprocessing_context
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        worker_init_fn=seeded_worker_init,
        generator=g,
        drop_last=drop_last,
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available(),
        **extra,
    )


def _list_collate(batch: List[dict]) -> List[dict]:
    """Trivial T1 collate: keep samples as a list (no tensor stacking).

    The real ragged-tensor ``collate_fn`` is T2; T1 only needs to iterate samples
    for determinism / viz checks, so default-collate's same-shape requirement
    (which the dict schema violates) is sidestepped.
    """
    return batch
