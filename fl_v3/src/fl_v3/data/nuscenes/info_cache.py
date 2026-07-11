"""Deterministic, host-portable nuScenes info-cache (fl_v3 T1).

The atomic-free, dependency-free analog of the mmdet3d "info pkl": walk a split's
keyframe samples **sorted by ``sample_token``**, resolve sensor paths
(**DATAROOT-relative**), calibration, per-sensor ego-poses, and canonical-frame GT
(box rows **``ann_token``-sorted**) — using the **devkit as the record oracle** but
**our own** ``transforms`` for every piece of geometry (no ``mmdet3d``). Serialize
to a cache **outside DATAROOT** (under ``nuscenes-cache-dir``) and compute a
**host-portable content hash**.

**Host-portable hash.** The hash is taken over the *raw* index inputs — the
JSON-parsed f64 calibration/annotation values, DATAROOT-relative paths, integer
tokens, and the derived integer/string/bool fields — serialized at fixed
little-endian dtypes, samples sorted by ``sample_token``, boxes by ``ann_token``,
no ``set`` iteration, no build timestamps, no host-absolute paths. Those raw inputs
are byte-identical on any host (same dataset text → same parse), and the derived
geometry is a pure deterministic function of them and this (identical) source — so
the Arrhenius (ARM) rebuild reproduces the hash. (A same-machine "build twice → bit-
identical derived schema" test additionally certifies the derived matrices.)
"""
from __future__ import annotations

import hashlib
import json
import os
import pickle
import struct
from glob import glob
from typing import Dict, List, Optional

import numpy as np

from fl_v3.data.nuscenes import paths as P
from fl_v3.data.nuscenes import transforms as T
from fl_v3.data.nuscenes.class_map import (
    CLASS_RANGE,
    NAME_TO_ID,
    category_to_detection_name,
)
from fl_v3.data.nuscenes.zip_backend import canonical_member_path

CACHE_FORMAT_VERSION = "t1.v2"


def _validated_n_sweeps(value: int) -> int:
    n_sweeps = int(value)
    if n_sweeps < 1:
        raise ValueError(f"n_sweeps must be >= 1, got {value!r}")
    return n_sweeps


# ---------------------------------------------------------------------------
# Record resolution (devkit = oracle for the raw records only).
# ---------------------------------------------------------------------------
def _sensor_records(nusc, sample, channel: str):
    sd = nusc.get("sample_data", sample["data"][channel])
    cs = nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])
    ep = nusc.get("ego_pose", sd["ego_pose_token"])
    return sd, cs, ep


def _sample_data_rel_path(sample_data: dict) -> str:
    """Canonical blob member from metadata only (never probes/extracts payload).

    Official ``sample_data.filename`` is already DATAROOT-relative.  Reading it
    directly lets the info-cache build against the Arrhenius metadata+ZIP layout
    even though ``dataroot/filename`` is not an extracted file.
    """
    try:
        return canonical_member_path(sample_data["filename"])
    except KeyError as exc:
        raise KeyError(f"sample_data {sample_data.get('token', '<unknown>')!r} has no filename") from exc


def _attr_name(nusc, attribute_tokens) -> str:
    """First attribute name for an annotation, or '' (static classes may have none)."""
    if not attribute_tokens:
        return ""
    return nusc.get("attribute", attribute_tokens[0])["name"]


def build_keyframe_info(nusc, sample, dataroot: str, n_sweeps: int = 1) -> dict:
    """Build one canonical keyframe info dict from devkit records + our transforms.

    GT box rows are sorted by ``ann_token`` (stable, reproducible across devkit
    versions / re-extraction). All geometry is computed by ``transforms`` (ours).

    ``n_sweeps`` (MCR P1 capability lever): when >1, additionally collect up to
    ``n_sweeps-1`` PREV ``LIDAR_TOP`` sweeps — each with the precomputed sweep→keyframe-
    LIDAR 4×4 transform (devkit ``from_file_multisweep`` convention: sweep_sensor →
    sweep_ego → global → key_ego → key_sensor) and the relative timestamp
    ``dt=(key_ts−sweep_ts)·1e-6`` s. Points are loaded + ego-motion-compensated at
    train time (the dataset). Cache format t1.v2 binds the requested depth in the
    filename, metadata, every record, and the host-portable hash.
    """
    n_sweeps = _validated_n_sweeps(n_sweeps)
    sample_token = sample["token"]
    scene = nusc.get("scene", sample["scene_token"])
    log = nusc.get("log", scene["log_token"])

    # --- LiDAR (canonical frame anchor) ---
    sd_l, cs_l, ep_l = _sensor_records(nusc, sample, P.LIDAR_CHANNEL)
    lidar_rel = _sample_data_rel_path(sd_l)
    lidar2ego = T.transform_matrix(cs_l["translation"], cs_l["rotation"], inverse=False)
    ego2global_lidar = T.transform_matrix(ep_l["translation"], ep_l["rotation"], inverse=False)
    # global -> lidar (for box centers): inverse chain.
    global2lidar = (
        T.transform_matrix(cs_l["translation"], cs_l["rotation"], inverse=True)
        @ T.transform_matrix(ep_l["translation"], ep_l["rotation"], inverse=True)
    )
    # global -> lidar rotation (for velocity; rotation only).
    R_global2lidar = (
        T.quaternion_to_rotation_matrix(cs_l["rotation"]).T
        @ T.quaternion_to_rotation_matrix(ep_l["rotation"]).T
    )
    q_cs_l_inv = T.quaternion_inverse(cs_l["rotation"])
    q_ep_l_inv = T.quaternion_inverse(ep_l["rotation"])
    ego_origin_global = np.asarray(ep_l["translation"], dtype=np.float64)

    # --- LiDAR multi-sweep (n_sweeps>1): walk prev LIDAR_TOP sample_data, store sweep→key-LIDAR ---
    lidar_sweeps: List[dict] = []
    if n_sweeps and int(n_sweeps) > 1:
        key_ts = int(sample["timestamp"])
        sd_cur = sd_l
        while len(lidar_sweeps) < int(n_sweeps) - 1 and sd_cur["prev"]:
            sd_cur = nusc.get("sample_data", sd_cur["prev"])
            cs_s = nusc.get("calibrated_sensor", sd_cur["calibrated_sensor_token"])
            ep_s = nusc.get("ego_pose", sd_cur["ego_pose_token"])
            # sweep_lidar -> global, then global -> key_lidar (the keyframe's global2lidar above)
            lidar2global_sweep = (
                T.transform_matrix(ep_s["translation"], ep_s["rotation"], inverse=False)
                @ T.transform_matrix(cs_s["translation"], cs_s["rotation"], inverse=False)
            )
            sweep2keylidar = (global2lidar @ lidar2global_sweep).astype(np.float64)  # (4,4)
            rel_s = _sample_data_rel_path(sd_cur)
            dt = (key_ts - int(sd_cur["timestamp"])) * 1e-6  # seconds into the past (>=0)
            lidar_sweeps.append({
                "rel_path": rel_s,
                "sweep2keylidar": sweep2keylidar,
                "dt": float(dt),
                # raw inputs for the host-portable hash:
                "_raw": (rel_s, cs_s["translation"], cs_s["rotation"],
                         ep_s["translation"], ep_s["rotation"], int(sd_cur["timestamp"])),
            })

    # --- Cameras (cam_order is the frozen constant, NOT filesystem order) ---
    cam_rel_paths: List[str] = []
    cam_intrinsics = np.zeros((6, 3, 3), dtype=np.float64)
    lidar2img = np.zeros((6, 4, 4), dtype=np.float64)
    cam2ego = np.zeros((6, 4, 4), dtype=np.float64)
    ego2global_cam = np.zeros((6, 4, 4), dtype=np.float64)
    cam_raw = []  # raw records, for the host-portable hash
    for i, ch in enumerate(P.CAMERA_CHANNELS):
        sd_c, cs_c, ep_c = _sensor_records(nusc, sample, ch)
        rel = _sample_data_rel_path(sd_c)
        cam_rel_paths.append(rel)
        K = np.asarray(cs_c["camera_intrinsic"], dtype=np.float64)
        cam_intrinsics[i] = K
        lidar2cam = T.compose_lidar2cam(cs_l, ep_l, ep_c, cs_c)
        lidar2img[i] = T.build_lidar2img(K, lidar2cam)
        cam2ego[i] = T.transform_matrix(cs_c["translation"], cs_c["rotation"], inverse=False)
        ego2global_cam[i] = T.transform_matrix(ep_c["translation"], ep_c["rotation"], inverse=False)
        cam_raw.append((rel, cs_c["translation"], cs_c["rotation"], cs_c["camera_intrinsic"],
                        ep_c["translation"], ep_c["rotation"]))

    # --- GT boxes (canonical LIDAR_TOP frame), ann_token-sorted ---
    rows = []  # each: dict of per-box fields
    for ann_token in sample["anns"]:
        ann = nusc.get("sample_annotation", ann_token)
        det_name = category_to_detection_name(ann["category_name"])
        if det_name is None:
            continue  # dropped, not mis-mapped
        det_id = NAME_TO_ID[det_name]
        center_g = np.asarray(ann["translation"], dtype=np.float64)
        wlh = np.asarray(ann["size"], dtype=np.float64)  # (w, l, h)
        q_g = np.asarray(ann["rotation"], dtype=np.float64)

        # center: global -> lidar
        center_l = (global2lidar @ np.array([*center_g, 1.0]))[:3]
        # orientation: q_lidar = q_cs^-1 ⊗ q_ep^-1 ⊗ q_global ; yaw via our accessor
        q_lidar = T.quaternion_multiply(q_cs_l_inv, T.quaternion_multiply(q_ep_l_inv, q_g))
        yaw = T.yaw_from_quaternion(q_lidar)
        # extent: (dx,dy,dz) = (l,w,h) = (wlh[1], wlh[0], wlh[2])
        box7 = np.array([center_l[0], center_l[1], center_l[2],
                         wlh[1], wlh[0], wlh[2], yaw], dtype=np.float64)

        # velocity: global -> lidar (rotation only); NaN -> 0
        v_g = nusc.box_velocity(ann_token)
        if v_g is None or np.any(np.isnan(v_g)):
            v_l_xy = np.zeros(2, dtype=np.float64)
        else:
            v_l = R_global2lidar @ np.asarray(v_g, dtype=np.float64)
            v_l_xy = v_l[:2]

        # eligibility: devkit-exact ego_dist (global planar radial from ego origin)
        ego_dist = float(np.linalg.norm(center_g[:2] - ego_origin_global[:2]))
        in_range = bool(ego_dist < CLASS_RANGE[det_name])

        rows.append({
            "ann_token": ann_token,
            "instance_token": ann["instance_token"],
            "det_id": int(det_id),
            "det_name": det_name,
            "box7": box7,
            "vel_xy": v_l_xy,
            "num_lidar_pts": int(ann["num_lidar_pts"]),
            "visibility": int(ann["visibility_token"]),
            "in_range": in_range,
            "attribute": _attr_name(nusc, ann["attribute_tokens"]),
            # raw, for the host-portable hash:
            "_raw_center_g": center_g,
            "_raw_wlh": wlh,
            "_raw_q_g": q_g,
        })
    rows.sort(key=lambda r: r["ann_token"])  # stable canonical order
    M = len(rows)

    def _stack(key, dtype, shape_tail=()):
        if M == 0:
            return np.zeros((0, *shape_tail), dtype=dtype)
        return np.stack([np.asarray(r[key], dtype=dtype) for r in rows])

    info = {
        "sample_token": sample_token,
        "scene_token": sample["scene_token"],
        "log_token": scene["log_token"],
        "location": log["location"],
        "timestamp": int(sample["timestamp"]),
        # Cache/runtime contract. Every record carries the requested depth so a
        # mixed or lower-sweep cache cannot be silently served as another depth.
        "_cache_n_sweeps": n_sweeps,
        "cam_order": tuple(P.CAMERA_CHANNELS),
        "cam_rel_paths": cam_rel_paths,
        "lidar_rel_path": lidar_rel,
        "cam_intrinsics": cam_intrinsics,            # (6,3,3) f64
        "lidar2img": lidar2img,                      # (6,4,4) f64
        "cam2ego": cam2ego,                          # (6,4,4) f64
        "ego2global_cam": ego2global_cam,            # (6,4,4) f64
        "lidar2ego": lidar2ego,                      # (4,4) f64
        "ego2global_lidar": ego2global_lidar,        # (4,4) f64
        "gt_boxes": _stack("box7", np.float64, (7,)),       # (M,7)
        "gt_velocity": _stack("vel_xy", np.float64, (2,)),  # (M,2)
        "gt_labels": np.asarray([r["det_id"] for r in rows], dtype=np.int64),
        "gt_names": [r["det_name"] for r in rows],
        "gt_num_lidar_pts": np.asarray([r["num_lidar_pts"] for r in rows], dtype=np.int64),
        "gt_visibility": np.asarray([r["visibility"] for r in rows], dtype=np.int64),
        "gt_in_range": np.asarray([r["in_range"] for r in rows], dtype=bool),
        "gt_attribute": [r["attribute"] for r in rows],
        "gt_instance_tokens": [r["instance_token"] for r in rows],
        "gt_ann_tokens": [r["ann_token"] for r in rows],
        # raw box inputs for the host-portable hash (not served in the schema):
        "_raw_boxes": [
            (r["ann_token"], r["instance_token"], r["det_id"], r["det_name"],
             r["_raw_center_g"], r["_raw_wlh"], r["_raw_q_g"],
             r["num_lidar_pts"], r["visibility"], r["in_range"], r["attribute"])
            for r in rows
        ],
        "_cam_raw": cam_raw,
        "_lidar_raw": (lidar_rel, cs_l["translation"], cs_l["rotation"],
                       ep_l["translation"], ep_l["rotation"]),
    }
    # Additive: present for every sample in an n_sweeps>1 build, including scene
    # starts with an empty history.  This prevents a valid cache from being
    # mistaken for single-sweep merely because its first token starts a scene.
    # t1.v2 records the requested depth explicitly even when it is one.
    if int(n_sweeps) > 1:
        info["lidar_sweeps"] = lidar_sweeps
    return info


# ---------------------------------------------------------------------------
# Host-portable canonical hash.
# ---------------------------------------------------------------------------
def _h_str(h, s: str) -> None:
    b = s.encode("utf-8")
    h.update(struct.pack("<Q", len(b)))
    h.update(b)


def _h_f64(h, arr) -> None:
    a = np.ascontiguousarray(np.asarray(arr, dtype="<f8"))
    h.update(struct.pack("<Q", a.size))
    h.update(a.tobytes())


def _h_i64(h, v: int) -> None:
    h.update(struct.pack("<q", int(v)))


def canonical_hash(info_list: List[dict]) -> str:
    """sha256 over the raw index inputs (host-portable; samples sorted by token)."""
    h = hashlib.sha256()
    _h_str(h, CACHE_FORMAT_VERSION)
    for info in sorted(info_list, key=lambda x: x["sample_token"]):
        _h_str(h, info["sample_token"])
        _h_str(h, info["scene_token"])
        _h_str(h, info["log_token"])
        _h_str(h, info["location"])
        _h_i64(h, info["timestamp"])
        _h_i64(h, info["_cache_n_sweeps"])
        for ch in info["cam_order"]:
            _h_str(h, ch)
        for (rel, cst, csr, K, ept, epr) in info["_cam_raw"]:
            _h_str(h, rel)
            _h_f64(h, cst); _h_f64(h, csr); _h_f64(h, K)
            _h_f64(h, ept); _h_f64(h, epr)
        (lrel, lcst, lcsr, lept, lepr) = info["_lidar_raw"]
        _h_str(h, lrel)
        _h_f64(h, lcst); _h_f64(h, lcsr); _h_f64(h, lept); _h_f64(h, lepr)
        _h_i64(h, len(info["_raw_boxes"]))
        for (at, it, did, dn, cg, wlh, qg, nlp, vis, inr, attr) in info["_raw_boxes"]:
            _h_str(h, at); _h_str(h, it); _h_i64(h, did); _h_str(h, dn)
            _h_f64(h, cg); _h_f64(h, wlh); _h_f64(h, qg)
            _h_i64(h, nlp); _h_i64(h, vis); _h_i64(h, 1 if inr else 0); _h_str(h, attr)
        # Multi-sweep records follow the explicit depth already hashed above.
        if "lidar_sweeps" in info:
            _h_i64(h, len(info["lidar_sweeps"]))
            for sw in info["lidar_sweeps"]:
                (rel_s, cst, csr, ept, epr, ts) = sw["_raw"]
                _h_str(h, rel_s); _h_f64(h, cst); _h_f64(h, csr)
                _h_f64(h, ept); _h_f64(h, epr); _h_i64(h, ts)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Split enumeration + cache build / load.
# ---------------------------------------------------------------------------
def split_sample_tokens(nusc, split: str) -> List[str]:
    """Keyframe ``sample_token``s in an official split, **sorted by token**.

    ``split`` ∈ {train, val, mini_train, mini_val, test}. Uses
    ``create_splits_scenes`` (never a hardcoded scene list).
    """
    from nuscenes.utils.splits import create_splits_scenes

    splits = create_splits_scenes()
    if split not in splits:
        raise ValueError(f"unknown split {split!r}; known: {sorted(splits)}")
    scene_names = set(splits[split])
    scene_tokens = {s["token"] for s in nusc.scene if s["name"] in scene_names}
    tokens = [s["token"] for s in nusc.sample if s["scene_token"] in scene_tokens]
    return sorted(tokens)


def build_info_list(nusc, sample_tokens: List[str], dataroot: str, n_sweeps: int = 1) -> List[dict]:
    n_sweeps = _validated_n_sweeps(n_sweeps)
    out = []
    for tok in sorted(sample_tokens):  # defensive: enforce token order
        out.append(build_keyframe_info(nusc, nusc.get("sample", tok), dataroot, n_sweeps=n_sweeps))
    return out


def cache_paths(
    cache_dir: str, version: str, split: str, n_sweeps: int = 1
) -> tuple[str, str]:
    n_sweeps = _validated_n_sweeps(n_sweeps)
    base = f"nuscenes_info_{version}_{split}_{CACHE_FORMAT_VERSION}_nsweeps{n_sweeps}"
    return os.path.join(cache_dir, base + ".pkl"), os.path.join(cache_dir, base + ".meta.json")


def save_cache(info_list: List[dict], cache_dir: str, version: str, split: str,
               scale: str, dataroot: str, n_sweeps: int = 1) -> dict:
    """Pickle the info list + write a JSON sidecar with the host-portable hash."""
    n_sweeps = _validated_n_sweeps(n_sweeps)
    _validate_cache_depth(info_list, {"n_sweeps": n_sweeps}, n_sweeps)
    P.resolve_writable(cache_dir, dataroot)  # active read-only guard
    os.makedirs(cache_dir, exist_ok=True)
    pkl_path, meta_path = cache_paths(cache_dir, version, split, n_sweeps)
    P.resolve_writable(pkl_path, dataroot)
    chash = canonical_hash(info_list)
    n_boxes = int(sum(len(i["gt_ann_tokens"]) for i in info_list))
    meta = {
        "format_version": CACHE_FORMAT_VERSION,
        "version": version,
        "split": split,
        "scale": scale,  # "mini-smoke" | "trainval-scientific"
        "n_samples": len(info_list),
        "n_boxes": n_boxes,
        "n_sweeps": n_sweeps,
        "cache_hash": chash,
        "dataroot_relative": True,
    }
    with open(pkl_path, "wb") as f:
        pickle.dump({"info_list": info_list, "meta": meta}, f, protocol=4)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, sort_keys=True)
    return meta


def _validate_cache_depth(info_list: List[dict], meta: dict, expected_n_sweeps: int | None) -> int:
    try:
        declared = _validated_n_sweeps(meta["n_sweeps"])
    except KeyError as exc:
        raise ValueError("nuScenes cache metadata has no n_sweeps; rebuild with t1.v2") from exc
    if expected_n_sweeps is not None:
        expected = _validated_n_sweeps(expected_n_sweeps)
        if declared != expected:
            raise ValueError(
                f"nuScenes cache sweep-depth mismatch: cache={declared}, requested={expected}"
            )
    for index, info in enumerate(info_list):
        actual = info.get("_cache_n_sweeps")
        if actual != declared:
            raise ValueError(
                f"nuScenes cache record {index} ({info.get('sample_token', '<unknown>')}) "
                f"declares _cache_n_sweeps={actual!r}, expected {declared}"
            )
        has_sweeps = "lidar_sweeps" in info
        if declared == 1 and has_sweeps:
            raise ValueError(f"single-sweep cache record {index} unexpectedly contains lidar_sweeps")
        if declared > 1 and not has_sweeps:
            raise ValueError(f"multi-sweep cache record {index} has no lidar_sweeps")
        if has_sweeps and len(info["lidar_sweeps"]) > declared - 1:
            raise ValueError(
                f"cache record {index} has {len(info['lidar_sweeps'])} previous sweeps, "
                f"exceeding declared n_sweeps={declared}"
            )
    return declared


def _discover_cache_path(cache_dir: str, version: str, split: str) -> tuple[str, str]:
    prefix = f"nuscenes_info_{version}_{split}_{CACHE_FORMAT_VERSION}_nsweeps"
    matches = sorted(glob(os.path.join(cache_dir, prefix + "*.pkl")))
    if not matches:
        raise FileNotFoundError(
            f"no {CACHE_FORMAT_VERSION} nuScenes cache for ({version}, {split}) under {cache_dir!r}"
        )
    if len(matches) != 1:
        raise ValueError(
            f"ambiguous nuScenes cache depth for ({version}, {split}): {matches}; "
            "call load_cache(..., n_sweeps=...) or use a depth-specific cache directory"
        )
    pkl_path = matches[0]
    return pkl_path, pkl_path[:-4] + ".meta.json"


def load_cache(
    cache_dir: str,
    version: str,
    split: str,
    n_sweeps: int | None = None,
    expected_cache_hash: str | None = None,
) -> tuple[List[dict], dict]:
    if n_sweeps is None:
        pkl_path, meta_path = _discover_cache_path(cache_dir, version, split)
    else:
        pkl_path, meta_path = cache_paths(cache_dir, version, split, n_sweeps)
    with open(pkl_path, "rb") as f:
        blob = pickle.load(f)
    info_list, meta = blob["info_list"], blob["meta"]
    if meta.get("format_version") != CACHE_FORMAT_VERSION:
        raise ValueError(
            f"unsupported nuScenes cache format {meta.get('format_version')!r}; "
            f"expected {CACHE_FORMAT_VERSION!r}"
        )
    with open(meta_path, encoding="utf-8") as stream:
        sidecar = json.load(stream)
    if sidecar != meta:
        raise ValueError(f"nuScenes cache sidecar differs from pickle metadata: {meta_path!r}")
    _validate_cache_depth(info_list, meta, n_sweeps)
    actual_hash = canonical_hash(info_list)
    if meta.get("cache_hash") != actual_hash:
        raise ValueError(
            f"nuScenes cache hash mismatch: metadata={meta.get('cache_hash')!r}, actual={actual_hash}"
        )
    if expected_cache_hash is not None:
        expected = str(expected_cache_hash).strip().lower()
        if len(expected) != 64 or any(ch not in "0123456789abcdef" for ch in expected):
            raise ValueError(
                f"expected_cache_hash must be a 64-character SHA-256 hex digest, got "
                f"{expected_cache_hash!r}"
            )
        if actual_hash != expected:
            raise ValueError(
                f"nuScenes cache does not match frozen expected hash: "
                f"expected={expected}, actual={actual_hash}"
            )
    return info_list, meta


def get_or_build_cache(nusc, cache_dir: str, version: str, split: str, scale: str,
                       dataroot: str, rebuild: bool = False, n_sweeps: int = 1) -> tuple[List[dict], dict]:
    """Load the cache if present (and hash-consistent), else build + save it.

    Sweep depth is part of the filename, sidecar, pickle metadata, content hash, and
    every info record. A lower/mixed-depth cache is rejected rather than reused."""
    n_sweeps = _validated_n_sweeps(n_sweeps)
    pkl_path, _ = cache_paths(cache_dir, version, split, n_sweeps)
    if not rebuild and os.path.isfile(pkl_path):
        return load_cache(cache_dir, version, split, n_sweeps=n_sweeps)
    tokens = split_sample_tokens(nusc, split)
    info_list = build_info_list(nusc, tokens, dataroot, n_sweeps=n_sweeps)
    meta = save_cache(
        info_list, cache_dir, version, split, scale, dataroot, n_sweeps=n_sweeps
    )
    return info_list, meta
