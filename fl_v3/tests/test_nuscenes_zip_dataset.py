"""S01 directory/ZIP byte, decoded-array, multi-sweep, and worker parity."""
from __future__ import annotations

import copy
import gc
import multiprocessing
import zipfile

import numpy as np
import pytest
import torch
from PIL import Image

from fl_v3.data.nuscenes import dataset as DS
from fl_v3.data.nuscenes import info_cache as IC
from fl_v3.data.nuscenes import paths as P
from fl_v3.data.nuscenes.zip_backend import (
    NuScenesBlobStore,
    TRAINVAL_ARCHIVE_NAMES,
    build_zip_manifest,
)


def _toy_info(cam_paths, lidar_path, sweep_paths):
    eye4 = np.eye(4, dtype=np.float64)
    sweeps = []
    for index, rel_path in enumerate(sweep_paths, start=1):
        transform = eye4.copy()
        transform[0, 3] = float(index)
        sweeps.append(
            {
                "rel_path": rel_path,
                "sweep2keylidar": transform,
                "dt": index * 0.05,
                "_raw": (rel_path, [0, 0, 0], [1, 0, 0, 0], [0, 0, 0], [1, 0, 0, 0], index),
            }
        )
    return {
        "sample_token": "sample-0",
        "_cache_n_sweeps": 10,
        "scene_token": "scene-0",
        "log_token": "log-0",
        "location": "boston-seaport",
        "timestamp": 123,
        "cam_order": tuple(P.CAMERA_CHANNELS),
        "cam_rel_paths": list(cam_paths),
        "lidar_rel_path": lidar_path,
        "lidar_sweeps": sweeps,
        "cam_intrinsics": np.repeat(np.eye(3, dtype=np.float64)[None], 6, axis=0),
        "lidar2img": np.repeat(eye4[None], 6, axis=0),
        "cam2ego": np.repeat(eye4[None], 6, axis=0),
        "ego2global_cam": np.repeat(eye4[None], 6, axis=0),
        "lidar2ego": eye4.copy(),
        "ego2global_lidar": eye4.copy(),
        "gt_boxes": np.zeros((0, 7), dtype=np.float64),
        "gt_velocity": np.zeros((0, 2), dtype=np.float64),
        "gt_labels": np.zeros((0,), dtype=np.int64),
        "gt_names": [],
        "gt_num_lidar_pts": np.zeros((0,), dtype=np.int64),
        "gt_visibility": np.zeros((0,), dtype=np.int64),
        "gt_in_range": np.zeros((0,), dtype=bool),
        "gt_attribute": [],
        "gt_instance_tokens": [],
        "gt_ann_tokens": [],
    }


class _DebugStateDataset(torch.utils.data.Dataset):
    """Test-only wrapper that exposes the worker-local blob-store lifecycle."""

    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        sample = self.dataset[index]
        sample["_zip_debug_state"] = self.dataset.blob_store.debug_state()
        return sample


@pytest.fixture()
def directory_and_zip(tmp_path):
    directory_root = tmp_path / "directory"
    cam_paths = []
    all_paths = []
    for index, channel in enumerate(P.CAMERA_CHANNELS):
        rel = f"samples/{channel}/{index}.jpg"
        path = directory_root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        image = np.zeros((4, 5, 3), dtype=np.uint8)
        image[..., 0] = index * 20
        image[..., 1] = np.arange(5, dtype=np.uint8)[None]
        Image.fromarray(image, mode="RGB").save(path, format="JPEG", quality=95)
        cam_paths.append(rel)
        all_paths.append(rel)

    lidar_path = "samples/LIDAR_TOP/key.pcd.bin"
    sweep_paths = [f"sweeps/LIDAR_TOP/sweep{index}.pcd.bin" for index in range(1, 10)]
    clouds = {
        lidar_path: np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]], dtype=np.float32),
    }
    for index, rel in enumerate(sweep_paths, start=1):
        clouds[rel] = np.array(
            [[index * 10 + 1, index * 10 + 2, index * 10 + 3, index * 10 + 4, index * 10 + 5]],
            dtype=np.float32,
        )
    for rel, cloud in clouds.items():
        path = directory_root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(cloud.tobytes())
        all_paths.append(rel)

    zip_root = tmp_path / "zip"
    zip_root.mkdir()
    buckets = {name: [] for name in TRAINVAL_ARCHIVE_NAMES}
    for index, rel in enumerate(all_paths):
        buckets[TRAINVAL_ARCHIVE_NAMES[index % len(TRAINVAL_ARCHIVE_NAMES)]].append(
            (rel, (directory_root / rel).read_bytes())
        )
    for name, members in buckets.items():
        with zipfile.ZipFile(zip_root / name, "w", compression=zipfile.ZIP_STORED) as archive:
            for rel, payload in members:
                archive.writestr(rel, payload)
    manifest = tmp_path / "manifest.sqlite"
    build_zip_manifest(str(zip_root), str(manifest))
    info = _toy_info(cam_paths, lidar_path, sweep_paths)
    return directory_root, zip_root, manifest, info, all_paths


def test_raw_bytes_match_directory_for_every_referenced_member(directory_and_zip):
    directory_root, zip_root, manifest, _info, paths = directory_and_zip
    directory = NuScenesBlobStore(str(directory_root))
    zipped = NuScenesBlobStore(str(zip_root), manifest_path=str(manifest))
    assert directory.read_many(paths) == zipped.read_many(paths)


def test_real_mini_directory_zip_bytes_and_decoded_arrays_match(
    nusc_mini, dataroot, tmp_path
):
    """Package only two real mini samples (one scene start, one full history)."""
    chosen = {}
    for token in IC.split_sample_tokens(nusc_mini, "mini_train"):
        sample = nusc_mini.get("sample", token)
        sample_data = nusc_mini.get("sample_data", sample["data"][P.LIDAR_CHANNEL])
        depth = 0
        current = sample_data
        while depth < 9 and current["prev"]:
            current = nusc_mini.get("sample_data", current["prev"])
            depth += 1
        if depth == 0 and "scene_start" not in chosen:
            chosen["scene_start"] = token
        if depth == 9 and "full_history" not in chosen:
            chosen["full_history"] = token
        if len(chosen) == 2:
            break
    assert set(chosen) == {"scene_start", "full_history"}
    tokens = [chosen["scene_start"], chosen["full_history"]]
    infos = IC.build_info_list(nusc_mini, tokens, dataroot, n_sweeps=10)
    depths = {info["sample_token"]: len(info["lidar_sweeps"]) for info in infos}
    assert depths == {chosen["scene_start"]: 0, chosen["full_history"]: 9}

    members = []
    for info in infos:
        members.extend(info["cam_rel_paths"])
        members.append(info["lidar_rel_path"])
        members.extend(sweep["rel_path"] for sweep in info["lidar_sweeps"])
    members = list(dict.fromkeys(members))
    zip_root = tmp_path / "mini_zip"
    zip_root.mkdir()
    buckets = {name: [] for name in TRAINVAL_ARCHIVE_NAMES}
    for index, rel in enumerate(members):
        with open(P.abspath_from_relative(rel, dataroot), "rb") as stream:
            payload = stream.read()
        buckets[TRAINVAL_ARCHIVE_NAMES[index % 10]].append((rel, payload))
    for archive_name, archive_members in buckets.items():
        with zipfile.ZipFile(
            zip_root / archive_name, "w", compression=zipfile.ZIP_STORED
        ) as archive:
            for rel, payload in archive_members:
                archive.writestr(rel, payload)
    manifest = tmp_path / "mini_manifest.sqlite"
    build_zip_manifest(str(zip_root), str(manifest))

    directory_store = NuScenesBlobStore(dataroot)
    zip_store = NuScenesBlobStore(str(zip_root), manifest_path=str(manifest))
    directory_payloads = directory_store.read_many(members)
    zip_payloads = zip_store.read_many(members)
    assert directory_payloads == zip_payloads

    directory_ds = DS.NuScenesMultimodalDataset(
        infos, dataroot, sample_tokens=tokens, n_sweeps=10
    )
    zip_ds = DS.NuScenesMultimodalDataset(
        infos,
        str(zip_root),
        sample_tokens=tokens,
        n_sweeps=10,
        zip_manifest=str(manifest),
    )
    for index in range(2):
        directory_sample = directory_ds[index]
        zip_sample = zip_ds[index]
        assert DS.sample_image_sha256(directory_sample) == DS.sample_image_sha256(zip_sample)
        assert torch.equal(directory_sample["images"], zip_sample["images"])
        assert torch.equal(directory_sample["lidar_points"], zip_sample["lidar_points"])
        assert torch.equal(directory_sample["gt_boxes"], zip_sample["gt_boxes"])
    directory_ds.close()
    zip_ds.close()
    directory_store.close()
    zip_store.close()


@pytest.mark.parametrize("n_sweeps", [1, 10])
def test_decoded_image_and_lidar_arrays_match_directory(directory_and_zip, n_sweeps):
    directory_root, zip_root, manifest, info, _paths = directory_and_zip
    runtime_info = copy.deepcopy(info)
    runtime_info["_cache_n_sweeps"] = n_sweeps
    if n_sweeps == 1:
        runtime_info.pop("lidar_sweeps")
    directory_ds = DS.NuScenesMultimodalDataset(
        [runtime_info], str(directory_root), n_sweeps=n_sweeps
    )
    zip_ds = DS.NuScenesMultimodalDataset(
        [runtime_info], str(zip_root), n_sweeps=n_sweeps, zip_manifest=str(manifest)
    )
    expected = directory_ds[0]
    actual = zip_ds[0]
    assert actual["cam_order"] == tuple(P.CAMERA_CHANNELS)
    assert torch.equal(actual["images"], expected["images"])
    assert torch.equal(actual["lidar_points"], expected["lidar_points"])
    assert torch.equal(actual["lidar2img"], expected["lidar2img"])
    assert actual["images"].dtype == torch.uint8
    assert actual["lidar_points"].dtype == torch.float32
    assert actual["lidar_points"].numpy().flags.writeable
    if n_sweeps == 10:
        # key (2 points) + nine one-point sweeps; dt is appended.
        assert tuple(actual["lidar_points"].shape) == (11, 6)
        expected_dt = torch.tensor([0.0, 0.0, *[index * 0.05 for index in range(1, 10)]])
        assert torch.equal(actual["lidar_points"][:, -1], expected_dt)


def test_legacy_absolute_lidar_and_multisweep_paths_use_zip_backend(
    directory_and_zip, monkeypatch
):
    directory_root, zip_root, manifest, info, _paths = directory_and_zip
    expected_key = DS._load_lidar(str(directory_root / info["lidar_rel_path"]))
    expected_sweeps = DS._load_multisweep(info, str(directory_root), 10)
    monkeypatch.setenv("NUSCENES_ZIP_MANIFEST", str(manifest))
    actual_key = DS._load_lidar(str(zip_root / info["lidar_rel_path"]))
    assert np.array_equal(actual_key, expected_key)
    actual_sweeps = DS._load_multisweep(info, str(zip_root), 10)
    assert np.array_equal(actual_sweeps, expected_sweeps)


@pytest.mark.parametrize("start_method", ["fork", "spawn"])
def test_repeated_persistent_multiworker_reads_are_deterministic(
    directory_and_zip, start_method
):
    if start_method not in multiprocessing.get_all_start_methods():
        pytest.skip(f"{start_method} unavailable")
    _directory_root, zip_root, manifest, base_info, _paths = directory_and_zip
    infos = []
    for index in range(4):
        info = copy.deepcopy(base_info)
        info["sample_token"] = f"sample-{index}"
        infos.append(info)
    dataset = DS.NuScenesMultimodalDataset(
        infos, str(zip_root), n_sweeps=10, zip_manifest=str(manifest)
    )
    # Open parent state first; forked workers must not reuse it.
    parent_sample = dataset[0]
    debug_dataset = _DebugStateDataset(dataset)
    loader = DS.make_loader(
        debug_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=2,
        seed=123,
        multiprocessing_context=start_method,
    )
    try:
        epochs = []
        lifecycle_epochs = []
        for _ in range(2):
            epoch = []
            lifecycle = {}
            for batch in loader:
                sample = batch[0]
                state = sample.pop("_zip_debug_state")
                assert state["owner_pid"] == state["current_pid"]
                lifecycle[state["owner_pid"]] = state
                epoch.append(
                    (
                        sample["sample_token"],
                        DS.sample_image_sha256(sample),
                        sample["lidar_points"].numpy().tobytes(),
                    )
                )
            epochs.append(epoch)
            lifecycle_epochs.append(lifecycle)
        assert epochs[0] == epochs[1]
        assert len(lifecycle_epochs[0]) == 2
        assert set(lifecycle_epochs[0]) == set(lifecycle_epochs[1])
        for pid in lifecycle_epochs[0]:
            first = lifecycle_epochs[0][pid]
            second = lifecycle_epochs[1][pid]
            assert second["reopen_count"] == first["reopen_count"]
            assert second["read_count"] > first["read_count"]
            assert set(second["open_archives"]) == set(first["open_archives"])
        assert torch.equal(parent_sample["images"], dataset[0]["images"])
    finally:
        del loader
        dataset.close()
        gc.collect()


def test_zip_root_without_manifest_fails_before_any_archive_scan(tmp_path):
    root = tmp_path / "zip"
    root.mkdir()
    (root / TRAINVAL_ARCHIVE_NAMES[0]).write_bytes(b"not-even-opened")
    with pytest.raises(FileNotFoundError, match="no external member manifest"):
        NuScenesBlobStore(str(root))


def _mode_zip(tmp_path, directory_root, members):
    root = tmp_path / "mode_zip"
    root.mkdir()
    buckets = {name: [] for name in TRAINVAL_ARCHIVE_NAMES}
    for index, rel in enumerate(members):
        buckets[TRAINVAL_ARCHIVE_NAMES[index % len(TRAINVAL_ARCHIVE_NAMES)]].append(rel)
    for name, paths in buckets.items():
        with zipfile.ZipFile(root / name, "w", compression=zipfile.ZIP_STORED) as archive:
            for rel in paths:
                archive.writestr(rel, (directory_root / rel).read_bytes())
    manifest = tmp_path / "mode_manifest.sqlite"
    build_zip_manifest(str(root), str(manifest))
    return root, manifest


@pytest.mark.parametrize("backend", ["directory", "zip"])
@pytest.mark.parametrize("model_mode", ["camera_only", "lidar_only"])
def test_mode_aware_io_never_reads_missing_disabled_payload(
    directory_and_zip, tmp_path, backend, model_mode
):
    directory_root, _zip_root, _manifest, info, _paths = directory_and_zip
    enabled = (
        list(info["cam_rel_paths"])
        if model_mode == "camera_only"
        else [info["lidar_rel_path"], *(s["rel_path"] for s in info["lidar_sweeps"])]
    )
    if backend == "zip":
        root, manifest = _mode_zip(tmp_path, directory_root, enabled)
        kwargs = {"zip_manifest": str(manifest)}
    else:
        root = tmp_path / "mode_directory"
        for rel in enabled:
            target = root / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((directory_root / rel).read_bytes())
        kwargs = {}

    dataset = DS.NuScenesMultimodalDataset(
        [info], str(root), n_sweeps=10, model_mode=model_mode, **kwargs
    )
    sample = dataset[0]
    counts = dataset.payload_read_counts["reads"]
    assert sample["model_mode"] == model_mode
    assert "images" in sample is (model_mode == "camera_only")
    assert "lidar_points" in sample is (model_mode == "lidar_only")
    assert counts["camera"] == (6 if model_mode == "camera_only" else 0)
    assert counts["lidar"] == (10 if model_mode == "lidar_only" else 0)
    assert sample["cam_intrinsics"].shape == (6, 3, 3)
    assert sample["lidar2ego"].shape == (4, 4)
    assert sample["gt_boxes"].shape == (0, 7)
    dataset.close()
