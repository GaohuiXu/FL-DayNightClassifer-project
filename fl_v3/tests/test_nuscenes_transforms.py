"""T1 — coordinate transforms vs the devkit geometry oracle (transforms.py).

These are the crown-jewel parity tests: our reimplemented quaternion/matrix
geometry must match the devkit exactly (it is the `fl_v2`-equivalent oracle).
"""
from __future__ import annotations

import numpy as np
import pytest

from fl_v3.data.nuscenes import transforms as T


def _rand_quats(n, seed=0):
    rng = np.random.default_rng(seed)
    q = rng.standard_normal((n, 4))
    return q / np.linalg.norm(q, axis=1, keepdims=True)


def test_quat_to_rotmat_matches_pyquaternion():
    from pyquaternion import Quaternion

    worst = 0.0
    for q in _rand_quats(2000, seed=1):
        worst = max(worst, np.abs(T.quaternion_to_rotation_matrix(q) - Quaternion(q).rotation_matrix).max())
    assert worst < 1e-12, worst


def test_transform_matrix_matches_devkit_both_conventions():
    from nuscenes.utils.geometry_utils import transform_matrix as dk_tm
    from pyquaternion import Quaternion

    rng = np.random.default_rng(2)
    worst_f = worst_i = 0.0
    for q in _rand_quats(500, seed=3):
        t = rng.standard_normal(3) * 10
        worst_f = max(worst_f, np.abs(T.transform_matrix(t, q, inverse=False) - dk_tm(t, Quaternion(q), inverse=False)).max())
        worst_i = max(worst_i, np.abs(T.transform_matrix(t, q, inverse=True) - dk_tm(t, Quaternion(q), inverse=True)).max())
    assert worst_f < 1e-12 and worst_i < 1e-12, (worst_f, worst_i)


def test_yaw_formula_matches_pyquaternion_random():
    """The decisive test: the MINUS-sign yaw formula must match over RANDOM
    quaternions (near-upright real boxes hide a sign error)."""
    from pyquaternion import Quaternion

    worst = 0.0
    for q in _rand_quats(5000, seed=4):
        d = abs(T.wrap_to_pi(T.yaw_from_quaternion(q) - Quaternion(q).yaw_pitch_roll[0]))
        worst = max(worst, d)
    assert worst < 1e-9, worst


def test_inverse_transform_roundtrip():
    rng = np.random.default_rng(5)
    for q in _rand_quats(200, seed=6):
        t = rng.standard_normal(3) * 5
        M = T.transform_matrix(t, q, inverse=False)
        Minv = T.transform_matrix(t, q, inverse=True)
        assert np.abs(M @ Minv - np.eye(4)).max() < 1e-10


def test_frame_roundtrip_closes(nusc_mini):
    """A point + box carried lidar→ego→global→ego_cam→camera→(inverse) returns to
    the lidar frame within tolerance (rotation orthogonality + translation)."""
    s = nusc_mini.sample[0]
    sd_l = nusc_mini.get("sample_data", s["data"]["LIDAR_TOP"])
    cs_l = nusc_mini.get("calibrated_sensor", sd_l["calibrated_sensor_token"])
    ep_l = nusc_mini.get("ego_pose", sd_l["ego_pose_token"])
    sd_c = nusc_mini.get("sample_data", s["data"]["CAM_FRONT"])
    cs_c = nusc_mini.get("calibrated_sensor", sd_c["calibrated_sensor_token"])
    ep_c = nusc_mini.get("ego_pose", sd_c["ego_pose_token"])
    lidar2cam = T.compose_lidar2cam(cs_l, ep_l, ep_c, cs_c)
    cam2lidar = np.linalg.inv(lidar2cam)
    # orthonormal rotation
    R = lidar2cam[:3, :3]
    assert np.abs(R @ R.T - np.eye(3)).max() < 1e-9
    rng = np.random.default_rng(7)
    pts = rng.standard_normal((50, 3)) * 20
    hom = np.concatenate([pts, np.ones((50, 1))], axis=1)
    back = (cam2lidar @ (lidar2cam @ hom.T)).T[:, :3]
    assert np.abs(back - pts).max() < 1e-8
    # AND a real box: carry an actual GT box's 8 corners through the full chain and
    # back (the SPEC §3 invariant says "a point AND a box ... and back").
    from fl_v3.viz.calibration import box7_to_corners
    from fl_v3.data.nuscenes import info_cache as IC
    from fl_v3.data.nuscenes.paths import DATAROOT

    info = IC.build_keyframe_info(nusc_mini, s, DATAROOT)
    assert info["gt_boxes"].shape[0] > 0
    corners = box7_to_corners(info["gt_boxes"][0])  # (8,3) lidar frame
    chom = np.concatenate([corners, np.ones((8, 1))], axis=1)
    cback = (cam2lidar @ (lidar2cam @ chom.T)).T[:, :3]
    assert np.abs(cback - corners).max() < 1e-7


def test_ego_motion_displacement_justifies_two_poses(nusc_mini):
    """Quantified justification for the two-distinct-ego-pose policy: the cam↔LiDAR
    ego displacement over the keyframe Δt is non-trivial (several px), so a single
    shared keyframe pose is NOT acceptable (conventions.md §2)."""
    max_disp = 0.0
    max_dt_us = 0
    for s in nusc_mini.sample[:20]:
        sd_l = nusc_mini.get("sample_data", s["data"]["LIDAR_TOP"])
        ep_l = nusc_mini.get("ego_pose", sd_l["ego_pose_token"])
        for cam in ("CAM_FRONT", "CAM_BACK", "CAM_FRONT_LEFT"):
            sd_c = nusc_mini.get("sample_data", s["data"][cam])
            ep_c = nusc_mini.get("ego_pose", sd_c["ego_pose_token"])
            disp = np.linalg.norm(np.array(ep_c["translation"]) - np.array(ep_l["translation"]))
            max_disp = max(max_disp, disp)
            max_dt_us = max(max_dt_us, abs(ep_c["timestamp"] - ep_l["timestamp"]))
    # Displacement is decimetre-scale (this subset ≈0.4 m; up to ~0.65 m over the full
    # mini set × 6 cams) — at f≈1267 px / ~10 m depth that is several px, and a single
    # shared pose shifts pixels by up to ~260 px (measured). The gate just needs the
    # margin to be comfortably non-trivial.
    assert max_disp > 0.05, max_disp
    assert max_dt_us > 10_000, max_dt_us
