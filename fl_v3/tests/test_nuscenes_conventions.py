"""T1 — canonical box / velocity / LiDAR conventions vs the devkit oracle.

Box parity is checked against ``NuScenes.get_sample_data(LIDAR_TOP)`` (the devkit's
authoritative lidar-frame boxes), on ≥200 real boxes, including the no-global-offset
test that catches a uniform mmdet3d-style −π/2 yaw shift.
"""
from __future__ import annotations

import numpy as np
import pytest

from fl_v3.data.nuscenes import info_cache as IC
from fl_v3.data.nuscenes import transforms as T
from fl_v3.data.nuscenes.class_map import category_to_detection_name


def _our_boxes_for_sample(nusc, sample_token, dataroot):
    info = IC.build_keyframe_info(nusc, nusc.get("sample", sample_token), dataroot)
    return {tok: (info["gt_boxes"][i], info["gt_velocity"][i])
            for i, tok in enumerate(info["gt_ann_tokens"])}


def test_box_parity_vs_get_sample_data(nusc_mini, dataroot):
    """≥200 boxes: center L2 <1e-3 m, extent permutation exact (zero tol),
    |wrap(Δyaw)| <1e-4 incl. ±π, AND no-global-offset (mean≈0 AND max≈0)."""
    from pyquaternion import Quaternion

    dyaws = []
    devkit_yaws = []
    n = 0
    worst_center = 0.0
    worst_extent = 0.0
    for s in nusc_mini.sample:
        ours = _our_boxes_for_sample(nusc_mini, s["token"], dataroot)
        _, boxes, _ = nusc_mini.get_sample_data(s["data"]["LIDAR_TOP"])
        for b in boxes:
            if category_to_detection_name(b.name) is None:
                continue  # dropped class — not in our schema
            assert b.token in ours, b.token
            box7, _ = ours[b.token]  # numpy arrays from build_keyframe_info
            # center
            worst_center = max(worst_center, float(np.linalg.norm(np.asarray(box7[:3]) - b.center)))
            # extent: devkit wlh=(w,l,h) → our (dx,dy,dz)=(l,w,h)
            ours_extent = np.asarray(box7[3:6])
            dk_extent = np.array([b.wlh[1], b.wlh[0], b.wlh[2]])
            worst_extent = max(worst_extent, float(np.abs(ours_extent - dk_extent).max()))
            # yaw
            dy = b.orientation.yaw_pitch_roll[0]
            dyaws.append(T.wrap_to_pi(float(box7[6]) - dy))
            devkit_yaws.append(dy)
            n += 1
    dyaws = np.array(dyaws)
    devkit_yaws = np.array(devkit_yaws)
    assert n >= 200, f"only {n} boxes compared"
    assert worst_center < 1e-3, worst_center
    assert worst_extent == 0.0, worst_extent  # exact permutation
    assert np.abs(dyaws).max() < 1e-4, np.abs(dyaws).max()
    # no-global-offset: a uniform −π/2 would show as mean≈−1.57 AND max≈1.57
    assert abs(dyaws.mean()) < 1e-5, dyaws.mean()
    # near-±π robustness: ISOLATE boxes whose devkit yaw is within 0.05 rad of ±π and
    # assert THOSE specifically (the wrap_to_pi branch-cut must close across ±π).
    near_pi_mask = np.abs(np.abs(devkit_yaws) - np.pi) < 0.05
    assert near_pi_mask.sum() >= 20, f"only {near_pi_mask.sum()} near-±π boxes"
    assert np.abs(dyaws[near_pi_mask]).max() < 1e-4, np.abs(dyaws[near_pi_mask]).max()


def test_yaw_equals_devkit_accessor(nusc_mini, dataroot):
    """Our schema yaw == Box.orientation.yaw_pitch_roll[0] to machine precision."""
    s = nusc_mini.sample[0]
    ours = _our_boxes_for_sample(nusc_mini, s["token"], dataroot)
    _, boxes, _ = nusc_mini.get_sample_data(s["data"]["LIDAR_TOP"])
    worst = 0.0
    for b in boxes:
        if category_to_detection_name(b.name) is None:
            continue
        box7, _ = ours[b.token]
        worst = max(worst, abs(T.wrap_to_pi(float(box7[6]) - b.orientation.yaw_pitch_roll[0])))
    assert worst < 1e-12, worst


def test_velocity_rotated_to_canonical_frame(nusc_mini, dataroot):
    """Velocity rotated global→lidar: magnitude preserved; stationary stays 0."""
    n_moving = 0
    worst_mag = 0.0
    for s in nusc_mini.sample:
        info = IC.build_keyframe_info(nusc_mini, s, dataroot)
        for i, tok in enumerate(info["gt_ann_tokens"]):
            v_g = nusc_mini.box_velocity(tok)
            v_l = info["gt_velocity"][i].numpy() if hasattr(info["gt_velocity"][i], "numpy") else info["gt_velocity"][i]
            if v_g is None or np.any(np.isnan(v_g)):
                assert np.allclose(v_l, 0.0)  # NaN → 0
                continue
            # only x,y kept; magnitude of full global vs full lidar preserved for the
            # planar part is NOT guaranteed, but the in-plane (x,y) speed should match
            # the global in-plane speed because nuScenes velocity z is ~0 and the
            # ground plane maps to the lidar x-y plane up to small pitch/roll.
            if np.linalg.norm(v_g[:2]) > 3.0:
                n_moving += 1
                worst_mag = max(worst_mag, abs(np.linalg.norm(v_l) - np.linalg.norm(v_g)))
        if n_moving > 5:
            break
    assert n_moving > 0
    # full 3D rotation preserves norm; we keep x,y but z-component of lidar velocity is
    # tiny (ground-plane motion), so |v_l_xy| ≈ |v_g|. Allow a small slack for the
    # dropped z (pitch/roll of the lidar mount).
    assert worst_mag < 0.5, worst_mag


def test_velocity_norm_preserved_full3d(nusc_mini):
    """Sanity: the global→lidar rotation alone preserves the full 3-vector norm."""
    s = nusc_mini.sample[0]
    sd_l = nusc_mini.get("sample_data", s["data"]["LIDAR_TOP"])
    cs_l = nusc_mini.get("calibrated_sensor", sd_l["calibrated_sensor_token"])
    ep_l = nusc_mini.get("ego_pose", sd_l["ego_pose_token"])
    R = (T.quaternion_to_rotation_matrix(cs_l["rotation"]).T
         @ T.quaternion_to_rotation_matrix(ep_l["rotation"]).T)
    rng = np.random.default_rng(0)
    for _ in range(100):
        v = rng.standard_normal(3) * 5
        assert abs(np.linalg.norm(R @ v) - np.linalg.norm(v)) < 1e-10


def test_lidar_parity_cols_0_to_4(nusc_mini):
    """Our 5-col LiDAR load matches devkit LidarPointCloud on cols 0:4 (col 4=ring is
    our conscious superset)."""
    from nuscenes.utils.data_classes import LidarPointCloud
    from fl_v3.data.nuscenes.dataset import _load_lidar

    s = nusc_mini.sample[0]
    path = nusc_mini.get_sample_data_path(s["data"]["LIDAR_TOP"])
    ours = _load_lidar(path)
    dk = LidarPointCloud.from_file(path).points  # (4,N)
    assert ours.shape[1] == 5
    assert dk.shape[0] == 4
    assert np.array_equal(ours[:, :4], dk.T)
