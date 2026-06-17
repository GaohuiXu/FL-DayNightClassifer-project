"""T4 — canonical→global conversion anchored to the RAW DEVKIT annotation (SPEC §0.1).

The crown-jewel parity: ``convert_box_to_global(T1-canonical GT)`` must reproduce the
**raw nuScenes global annotation** (reconstructed via the devkit ``Box``/``Quaternion``/
``box_velocity`` — the INDEPENDENT oracle), NOT a self-inverse of T1's own forward. A
shared yaw-sign / ``wlh``-order / ego-pose bug between T1's forward and this inverse would
self-certify in a round-trip; anchoring the yaw to a **devkit-lifted yaw-only box** and
the center/size to the **raw annotation** catches it (the devkit yaw uses the correct
``−x·y`` convention, so a wrong-sign forward yaw fails here).
"""
from __future__ import annotations

import numpy as np
import pytest

from fl_v3.eval.box_to_global import (
    convert_box_to_global,
    rotmat_to_quaternion,
    yaw_about_z_rotation,
)
from fl_v3.data.nuscenes import transforms as T


# ---------------------------------------------------------------------------
# rotmat_to_quaternion correctness (matches the devkit's own quaternion tool).
# ---------------------------------------------------------------------------
def _rand_rotations(n, seed):
    from pyquaternion import Quaternion

    rng = np.random.default_rng(seed)
    q = rng.standard_normal((n, 4))
    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    return [Quaternion(qi) for qi in q]


def test_rotmat_to_quaternion_matches_pyquaternion_and_roundtrips():
    """Our Shepperd rotmat→quat must (a) round-trip through our quat→rotmat to <1e-12,
    and (b) extract the SAME yaw as pyquaternion (the devkit tool), over random rotations."""
    worst_R = 0.0
    worst_yaw = 0.0
    for Q in _rand_rotations(3000, seed=11):
        R = Q.rotation_matrix
        q = rotmat_to_quaternion(R)
        R_back = T.quaternion_to_rotation_matrix(q)
        worst_R = max(worst_R, float(np.abs(R_back - R).max()))
        dyaw = abs(T.wrap_to_pi(T.yaw_from_quaternion(q) - Q.yaw_pitch_roll[0]))
        worst_yaw = max(worst_yaw, float(dyaw))
    assert worst_R < 1e-12, worst_R
    assert worst_yaw < 1e-9, worst_yaw


def test_yaw_about_z_is_pure_yaw():
    for yaw in (-np.pi + 1e-6, -1.3, 0.0, 0.7, np.pi - 1e-6):
        q = rotmat_to_quaternion(yaw_about_z_rotation(yaw))
        assert abs(T.wrap_to_pi(T.yaw_from_quaternion(q) - yaw)) < 1e-12


# ---------------------------------------------------------------------------
# The devkit-anchored round-trip (SPEC §0.1 GATE: ≥200 real mini boxes).
# ---------------------------------------------------------------------------
def _devkit_lift_yawonly_to_global(cs_l, ep_l, yaw_lidar: float):
    """Independent devkit lift of a YAW-ONLY lidar orientation → global ``Box.orientation``.

    Uses the devkit ``Box`` rotate ops + ``Quaternion`` only (NO fl_v3 transforms): start
    from ``Quaternion(axis=+z, radians=yaw_lidar)``, rotate by the raw ``calibrated_sensor``
    then ``ego_pose`` quaternions (the forward chain the devkit ``get_sample_data`` inverts).
    Returns the global ``Quaternion`` ``q_ego ⊗ q_cs ⊗ Quat(z,yaw)``.
    """
    from nuscenes.utils.data_classes import Box
    from pyquaternion import Quaternion

    b = Box([0.0, 0.0, 0.0], [1.0, 1.0, 1.0], Quaternion(axis=[0.0, 0.0, 1.0], radians=yaw_lidar))
    b.rotate(Quaternion(cs_l["rotation"]))   # lidar→ego rotation
    b.rotate(Quaternion(ep_l["rotation"]))   # ego→global rotation
    return b.orientation


def test_box_to_global_matches_raw_devkit_annotation(nusc_mini, mini_train_info):
    """≥200 real mini boxes vs the RAW devkit annotation oracle:

    (1) translation L2 <1e-3 m (vs raw ``ann['translation']``);
    (2) ``wlh`` EXACT (vs raw ``ann['size']``);
    (3) **rotation lift-equivalence <1e-9** — feeding the SAME lidar yaw to (a) our matrix
        lift ``R(lidar2global)·Rz(yaw)`` and (b) an INDEPENDENT devkit ``Box`` lift, the
        global orientations agree exactly (catches a wrong compose-order / Rz-sign / wlh-swap
        in box_to_global), incl. ±π;
    (4) **heading vs the raw annotation <0.02 rad** — the orient-err a *perfect* detection
        incurs; catches a gross sign(~π)/offset(~π/2) bug while tolerating the documented
        Euler↔devkit-heading convention gap (~0.004 rad; see findings_log — T1's box7 yaw is
        Tait-Bryan ``yaw_pitch_roll[0]`` but ``DetectionEval`` orient-err uses the
        rotated-x-axis ``quaternion_yaw``);
    (5) velocity DIRECTION match (not just norm) vs devkit ``box_velocity``.
    """
    from nuscenes.eval.common.utils import quaternion_yaw, angle_diff
    from pyquaternion import Quaternion

    n_checked = 0
    worst_trans = worst_wlh = worst_lift = worst_heading = 0.0
    near_pi_checked = vel_dir_checked = 0
    worst_vel_cos = 1.0

    for info in mini_train_info:
        l2e = info["lidar2ego"]; e2g = info["ego2global_lidar"]
        s = nusc_mini.get("sample", info["sample_token"])
        sd_l = nusc_mini.get("sample_data", s["data"]["LIDAR_TOP"])
        cs_l = nusc_mini.get("calibrated_sensor", sd_l["calibrated_sensor_token"])
        ep_l = nusc_mini.get("ego_pose", sd_l["ego_pose_token"])
        for m, ann_token in enumerate(info["gt_ann_tokens"]):
            ann = nusc_mini.get("sample_annotation", ann_token)
            box7 = info["gt_boxes"][m]
            yaw_lidar = float(box7[6])
            conv = convert_box_to_global(box7, l2e, e2g, velocity_lidar=info["gt_velocity"][m])

            # (1) translation vs the RAW devkit global annotation.
            raw_t = np.asarray(ann["translation"], dtype=np.float64)
            worst_trans = max(worst_trans, float(np.linalg.norm(conv["translation"] - raw_t)))
            # (2) size == raw wlh, EXACT.
            worst_wlh = max(worst_wlh, float(np.abs(conv["size"] - np.asarray(ann["size"], np.float64)).max()))

            # (3) rotation lift-equivalence vs an independent devkit Box lift (same yaw in).
            q_mine = conv["rotation"]
            q_devkit = _devkit_lift_yawonly_to_global(cs_l, ep_l, yaw_lidar)
            R_mine = T.quaternion_to_rotation_matrix(q_mine)
            R_devkit = q_devkit.rotation_matrix
            worst_lift = max(worst_lift, float(np.abs(R_mine - R_devkit).max()))

            # (4) heading vs the RAW annotation (the perfect-box orient-err; catches gross bugs).
            head_mine = quaternion_yaw(Quaternion(q_mine))
            head_raw = quaternion_yaw(Quaternion(np.asarray(ann["rotation"], np.float64)))
            dhead = abs(angle_diff(head_mine, head_raw, 2 * np.pi))
            worst_heading = max(worst_heading, float(dhead))
            if abs(head_raw) > 3.0:  # near ±π wraparound
                near_pi_checked += 1

            # (5) velocity DIRECTION vs devkit box_velocity, moving boxes only.
            vg = nusc_mini.box_velocity(ann_token)
            if vg is not None and np.all(np.isfinite(vg)) and np.linalg.norm(vg[:2]) > 1.0:
                a = conv["velocity"]; b = vg[:2]
                worst_vel_cos = min(worst_vel_cos,
                                    float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)))
                vel_dir_checked += 1
            n_checked += 1

    assert n_checked >= 200, f"only {n_checked} boxes checked (need ≥200)"
    assert worst_trans < 1e-3, f"translation L2 worst={worst_trans}"
    assert worst_wlh < 1e-9, f"wlh not exact, worst={worst_wlh}"
    assert worst_lift < 1e-9, f"rotation lift not devkit-equivalent, worst|ΔR|={worst_lift}"
    assert worst_heading < 0.02, f"heading worst={worst_heading} (gross sign/offset bug?)"
    assert near_pi_checked >= 1, "no near-±π box exercised — wraparound not covered"
    assert vel_dir_checked >= 5, f"too few moving boxes for velocity-direction check ({vel_dir_checked})"
    assert worst_vel_cos > 0.999, f"velocity direction mismatch, worst cos={worst_vel_cos}"
