"""S07-B mode/depth/backend counters and PID-fallback lifecycle hostiles."""
from __future__ import annotations

import copy
import json
import os
import zipfile

import numpy as np
import pytest
from PIL import Image

from fl_v3.data.nuscenes import dataset as DS
from fl_v3.data.nuscenes import paths as P
from fl_v3.data.nuscenes.zip_backend import (
    NuScenesBlobStore,
    TRAINVAL_ARCHIVE_NAMES,
    build_zip_manifest,
)


def _info(cam_paths, lidar_path, sweep_paths, depth):
    eye4 = np.eye(4, dtype=np.float64)
    record = {
        "sample_token": "sample-0", "_cache_n_sweeps": depth,
        "scene_token": "scene-0", "log_token": "log-0",
        "location": "boston-seaport", "timestamp": 1,
        "cam_order": tuple(P.CAMERA_CHANNELS), "cam_rel_paths": list(cam_paths),
        "lidar_rel_path": lidar_path,
        "cam_intrinsics": np.repeat(np.eye(3)[None], 6, axis=0),
        "lidar2img": np.repeat(eye4[None], 6, axis=0),
        "cam2ego": np.repeat(eye4[None], 6, axis=0),
        "ego2global_cam": np.repeat(eye4[None], 6, axis=0),
        "lidar2ego": eye4.copy(), "ego2global_lidar": eye4.copy(),
        "gt_boxes": np.zeros((0, 7)), "gt_velocity": np.zeros((0, 2)),
        "gt_labels": np.zeros(0, dtype=np.int64), "gt_names": [],
        "gt_num_lidar_pts": np.zeros(0, dtype=np.int64),
        "gt_visibility": np.zeros(0, dtype=np.int64),
        "gt_in_range": np.zeros(0, dtype=bool), "gt_attribute": [],
        "gt_instance_tokens": [], "gt_ann_tokens": [],
    }
    if depth > 1:
        record["lidar_sweeps"] = [
            {"rel_path": path, "sweep2keylidar": eye4.copy(), "dt": index * 0.05}
            for index, path in enumerate(sweep_paths[: depth - 1], start=1)
        ]
    return record


@pytest.fixture()
def mode_payloads(tmp_path):
    root = tmp_path / "directory"
    root.mkdir()
    image_path = root / "samples/CAM_FRONT/image.jpg"
    image_path.parent.mkdir(parents=True)
    Image.fromarray(np.zeros((3, 4, 3), dtype=np.uint8), mode="RGB").save(image_path)
    cam_paths = ["samples/CAM_FRONT/image.jpg"] * 6
    lidar = "samples/LIDAR_TOP/key.pcd.bin"
    sweeps = [f"sweeps/LIDAR_TOP/{index}.pcd.bin" for index in range(1, 10)]
    for index, rel in enumerate([lidar, *sweeps]):
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(np.asarray([[index, 1, 2, 3, 4]], dtype=np.float32).tobytes())
    return root, cam_paths, lidar, sweeps


def _zip_subset(tmp_path, directory, members):
    root = tmp_path / "zip"
    root.mkdir()
    unique = list(dict.fromkeys(members))
    for index, archive_name in enumerate(TRAINVAL_ARCHIVE_NAMES):
        with zipfile.ZipFile(root / archive_name, "w", compression=zipfile.ZIP_STORED) as archive:
            for rel in unique[index::len(TRAINVAL_ARCHIVE_NAMES)]:
                archive.writestr(rel, (directory / rel).read_bytes())
    manifest = tmp_path / "manifest.sqlite"
    build_zip_manifest(str(root), str(manifest))
    return root, manifest


@pytest.mark.parametrize("backend", ["directory", "zip"])
@pytest.mark.parametrize("depth", [1, 10])
@pytest.mark.parametrize("mode", ["camera_only", "lidar_only", "fusion"])
def test_mode_depth_backend_payload_counters(mode_payloads, tmp_path, backend, depth, mode):
    directory, cameras, lidar, sweeps = mode_payloads
    info = _info(cameras, lidar, sweeps, depth)
    enabled = []
    if mode in {"camera_only", "fusion"}:
        enabled.extend(cameras)
    if mode in {"lidar_only", "fusion"}:
        enabled.extend([lidar, *sweeps[: depth - 1]])
    if backend == "zip":
        root, manifest = _zip_subset(tmp_path, directory, enabled)
        kwargs = {"zip_manifest": str(manifest)}
    else:
        root = tmp_path / "subset"
        for rel in dict.fromkeys(enabled):
            target = root / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((directory / rel).read_bytes())
        kwargs = {}
    dataset = DS.NuScenesMultimodalDataset(
        [copy.deepcopy(info)], str(root), n_sweeps=depth, model_mode=mode, **kwargs,
    )
    sample = dataset[0]
    counts = dataset.payload_read_counts["reads"]
    assert ("images" in sample) is (mode in {"camera_only", "fusion"})
    assert ("lidar_points" in sample) is (mode in {"lidar_only", "fusion"})
    assert counts == {
        "camera": 6 if mode in {"camera_only", "fusion"} else 0,
        "lidar": depth if mode in {"lidar_only", "fusion"} else 0,
        "other": 0,
    }
    dataset.close()


def test_raw_fork_pid_fallback_resets_counters_and_location_cache(tmp_path):
    if not hasattr(os, "fork"):
        pytest.skip("raw fork unavailable")
    payload = tmp_path / "samples/CAM_FRONT/item.bin"
    payload.parent.mkdir(parents=True)
    payload.write_bytes(b"payload")
    store = NuScenesBlobStore(str(tmp_path))
    store.read_bytes("samples/CAM_FRONT/item.bin")
    store._locations["inherited"] = (0, 0, 0, 0, 0, 0)
    parent = store.debug_state()
    read_fd, write_fd = os.pipe()
    child = os.fork()
    if child == 0:
        try:
            os.close(read_fd)
            store._ensure_process()  # raw os.fork bypasses multiprocessing's registered hook.
            state = store.debug_state()
            os.write(write_fd, json.dumps(state).encode("utf-8"))
        finally:
            os.close(write_fd)
            os._exit(0)
    os.close(write_fd)
    encoded = b""
    while True:
        chunk = os.read(read_fd, 4096)
        if not chunk:
            break
        encoded += chunk
    os.close(read_fd)
    _, status = os.waitpid(child, 0)
    assert status == 0
    state = json.loads(encoded)
    assert state["owner_pid"] == state["current_pid"] == child
    assert state["read_count"] == state["byte_count"] == 0
    assert state["modality_read_count"] == {"camera": 0, "lidar": 0, "other": 0}
    assert state["location_cache_size"] == 0
    assert parent["read_count"] == 1
    assert store.debug_state()["read_count"] == 1
    store.close()


@pytest.mark.parametrize(
    "mode,augment,gtpaste,match",
    [
        ("camera_only", None, {"db_path": "unused"}, "GT-paste"),
        ("camera_only", {"img_flip": 0.0}, None, "BEV augmentation"),
        ("lidar_only", {"img_flip": 0.5}, None, "image flipping"),
    ],
)
def test_disabled_modality_augmentation_fails_before_iteration(
    tmp_path, mode, augment, gtpaste, match,
):
    with pytest.raises(ValueError, match=match):
        DS.NuScenesMultimodalDataset(
            [], str(tmp_path), model_mode=mode, augment=augment, gtpaste=gtpaste,
        )
