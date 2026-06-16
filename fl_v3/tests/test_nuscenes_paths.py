"""T1 — dataset wiring + read-only guard (paths.py)."""
from __future__ import annotations

import os

import pytest

from fl_v3.data.nuscenes import paths as P


def test_verify_mini_passes():
    rep = P.verify_dataset("v1.0-mini")
    assert rep["sentinel_ok"] and rep["n_samples"] == 404


def test_verify_trainval_detected():
    # Lightweight (parses sample.json only — no full devkit load).
    rep = P.verify_dataset("v1.0-trainval")
    assert rep["sentinel_ok"] and rep["n_samples"] > 30000


def test_stale_root_fails_sentinel():
    # The stale /mimer/NOBACKUP/Datasets/nuScenes has a `full/` layout (no top-level
    # version tables) → verify must FAIL, not pass.
    stale = "/mimer/NOBACKUP/Datasets/nuScenes"
    if not os.path.isdir(stale):
        pytest.skip("stale root not present on this host")
    with pytest.raises((FileNotFoundError, ValueError)):
        P.verify_dataset("v1.0-mini", dataroot=stale)


def test_wrong_dataroot_fails():
    with pytest.raises((FileNotFoundError, ValueError)):
        P.verify_dataset("v1.0-mini", dataroot="/tmp/definitely-not-nuscenes")


def test_write_under_dataroot_raises(tmp_path):
    # A path under DATAROOT must raise; a path elsewhere must pass through.
    bad = os.path.join(P.DATAROOT, "samples", "should_not_write.bin")
    with pytest.raises(PermissionError):
        P.resolve_writable(bad)
    ok = str(tmp_path / "cache" / "x.pkl")
    assert P.resolve_writable(ok) == os.path.realpath(os.path.abspath(ok))


def test_relative_roundtrip():
    abs_p = os.path.join(P.DATAROOT, "samples", "LIDAR_TOP", "foo.pcd.bin")
    rel = P.relative_to_dataroot(abs_p)
    assert rel == "samples/LIDAR_TOP/foo.pcd.bin"
    assert P.abspath_from_relative(rel) == abs_p


def test_cam_order_is_frozen_constant():
    assert P.CAMERA_CHANNELS == (
        "CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_FRONT_LEFT",
        "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT",
    )
