"""Canonical ``LIDAR_TOP`` box → nuScenes **global** ``DetectionBox`` (fl_v3 T4).

The submission conversion every official-metric + ASR computation depends on. A
decoded box is ``(cx,cy,cz, dx=l, dy=w, dz=h, yaw)`` in the **T1 canonical**
``LIDAR_TOP`` frame (``conventions.md`` §3); the devkit ``DetectionEval`` scores
**global**-frame ``DetectionBox`` submissions. This module is the *inverse* of T1's
forward (info_cache) geometry — written **independently** (composed forward, never a
literal re-use of T1's inverted matrices) and **anchored to the raw devkit annotation**
(``Box``/``Quaternion``/``box_velocity``), NOT to T1's own forward (T4_SPEC §0.1).

Geometry (re-using the verified ``transforms`` primitives, only the inverse direction
is new):

  * **center**   ``t_global = (ego2global_lidar · lidar2ego) · [cx,cy,cz,1]``  — i.e.
    ``lidar2global = ego2global_lidar @ lidar2ego`` (the forward chain the devkit
    ``get_sample_data`` inverts; verified line-by-line against the devkit Box ops).
  * **size**     ``wlh = (w,l,h) = (dy,dx,dz)`` — inverts T1's ``wlh→(l,w,h)``; the
    raw nuScenes ``size`` order (zero-tol).
  * **rotation** ``R_box_global = R(lidar2global) · Rz(yaw)`` → unit quaternion
    ``(w,x,y,z)`` (the head emits a **yaw-only** box, so the global orientation is the
    yaw-only lidar box lifted to global — equivalently ``q_ego2global · q_lidar2ego ·
    Quat(+z, yaw)``).
  * **velocity** ``v_global = R(lidar2global) · (vx,vy,0)`` → keep ``(vx,vy)``. This is
    the **inverse ORDER** of T1's forward ``R(cs)ᵀ·R(ep)ᵀ`` (``R(lidar2global) =
    R(ego2global_lidar)·R(lidar2ego) = R(ep)·R(cs)``). ``v_z`` is dropped to 0 (T1
    dropped lidar ``v_z`` and the head emits 2D velocity) — worst-case AVE impact ~0.13
    m/s; if AVE materially hurts NDS, carrying full-3D velocity is a **T1** touch (flag,
    do not mutate T1).

**Float hygiene** (the devkit ``DetectionBox`` asserts hard): ``detection_score`` and
every translation/size/rotation float are coerced with **builtin ``float()``**; no
NaN/inf in translation/size/rotation; ``size>0``; ``attribute_name`` is in the class
vocab (or ``""``). ``velocity`` may be NaN per the devkit but we always emit finite.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from fl_v3.data.nuscenes import transforms as T

# Per-class attribute defaults (T4_SPEC §2 / devkit ``detection_name_to_rel_attributes``).
# Vehicles → vehicle.{moving,parked} by a speed threshold; pedestrian →
# pedestrian.{moving,standing}; cone/barrier → "" (no attribute). The speed threshold is
# config-pinned (``attr-vehicle-moving-speed`` / ``attr-pedestrian-moving-speed``).
_VEHICLE_CLASSES = ("car", "truck", "bus", "trailer", "construction_vehicle")
DEFAULT_VEHICLE_MOVING_SPEED = 0.2   # m/s (global planar speed) → vehicle.moving else parked
DEFAULT_PEDESTRIAN_MOVING_SPEED = 0.2


def lidar_to_global_matrix(lidar2ego, ego2global_lidar) -> np.ndarray:
    """4×4 ``lidar → global`` = ``ego2global_lidar @ lidar2ego`` (float64).

    The forward chain the devkit ``get_sample_data`` inverts (global→lidar). Built from
    the two keyframe matrices the schema carries (``lidar2ego`` = the LIDAR_TOP
    ``calibrated_sensor`` map, ``ego2global_lidar`` = the LIDAR_TOP ``ego_pose`` map).
    """
    l2e = np.asarray(lidar2ego, dtype=np.float64).reshape(4, 4)
    e2g = np.asarray(ego2global_lidar, dtype=np.float64).reshape(4, 4)
    return e2g @ l2e


def rotmat_to_quaternion(R) -> np.ndarray:
    """3×3 rotation matrix → unit quaternion ``(w,x,y,z)`` (deterministic Shepperd).

    Pure numpy float64, no RNG, branch on the largest diagonal term for numerical
    stability (the standard Shepperd method). Returned with ``w ≥ 0`` (canonical sign;
    ``q`` and ``-q`` are the same rotation — yaw extraction is sign-invariant anyway).
    Unit-tested == ``pyquaternion.Quaternion(matrix=R)`` (up to sign) and a
    round-trip through :func:`transforms.quaternion_to_rotation_matrix`.
    """
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    m00, m11, m22 = R[0, 0], R[1, 1], R[2, 2]
    tr = m00 + m11 + m22
    if tr > 0.0:
        s = np.sqrt(tr + 1.0) * 2.0  # s = 4w
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif m00 >= m11 and m00 >= m22:
        s = np.sqrt(1.0 + m00 - m11 - m22) * 2.0  # s = 4x
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif m11 >= m22:
        s = np.sqrt(1.0 + m11 - m00 - m22) * 2.0  # s = 4y
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + m22 - m00 - m11) * 2.0  # s = 4z
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    q = np.array([w, x, y, z], dtype=np.float64)
    q = q / np.linalg.norm(q)
    if q[0] < 0.0:
        q = -q
    return q


def yaw_about_z_rotation(yaw: float) -> np.ndarray:
    """3×3 rotation about +z by ``yaw`` (CCW from +x), float64."""
    c, s = np.cos(float(yaw)), np.sin(float(yaw))
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def convert_box_to_global(
    box7: Sequence[float],
    lidar2ego,
    ego2global_lidar,
    velocity_lidar: Optional[Sequence[float]] = None,
) -> Dict[str, object]:
    """Pure geometry: canonical box7 → global ``(translation, size=wlh, rotation, velocity, yaw)``.

    ``box7`` = ``(cx,cy,cz, dx=l, dy=w, dz=h, yaw)`` (LIDAR_TOP). ``velocity_lidar`` =
    ``(vx,vy)`` lidar-frame (the decode head's 2D velocity, or the GT lidar velocity);
    ``None`` → ``(0,0)``. Returns float64 numpy arrays (the ``DetectionBox`` float
    coercion happens in :func:`make_detection_box`). No RNG, deterministic.
    """
    b = np.asarray(box7, dtype=np.float64).reshape(7)
    cx, cy, cz, dl, dw, dh, yaw = b
    l2g = lidar_to_global_matrix(lidar2ego, ego2global_lidar)
    R_l2g = l2g[:3, :3]

    center_g = (l2g @ np.array([cx, cy, cz, 1.0], dtype=np.float64))[:3]
    size_wlh = np.array([dw, dl, dh], dtype=np.float64)  # (w,l,h) = (dy,dx,dz)
    R_box_global = R_l2g @ yaw_about_z_rotation(yaw)
    q_global = rotmat_to_quaternion(R_box_global)
    yaw_global = T.yaw_from_quaternion(q_global)

    if velocity_lidar is None:
        v_l = np.zeros(2, dtype=np.float64)
    else:
        v_l = np.asarray(velocity_lidar, dtype=np.float64).reshape(2)
    v_g = R_l2g @ np.array([v_l[0], v_l[1], 0.0], dtype=np.float64)
    velocity_g = v_g[:2]

    return {
        "translation": center_g,      # (3,) global
        "size": size_wlh,             # (3,) wlh
        "rotation": q_global,         # (4,) (w,x,y,z) unit
        "velocity": velocity_g,       # (2,) global xy
        "yaw_global": float(yaw_global),
    }


def attribute_for(
    detection_name: str,
    velocity_global_xy,
    vehicle_moving_speed: float = DEFAULT_VEHICLE_MOVING_SPEED,
    pedestrian_moving_speed: float = DEFAULT_PEDESTRIAN_MOVING_SPEED,
) -> str:
    """Per-class ``attribute_name`` from the predicted global speed (devkit vocab).

    Vehicles → ``vehicle.moving`` if planar speed ≥ ``vehicle_moving_speed`` else
    ``vehicle.parked``; pedestrian → ``pedestrian.{moving,standing}``; bicycle/motorcycle
    → ``cycle.{with,without}_rider`` is unknowable from geometry, so default
    ``cycle.without_rider``; cone/barrier → ``""``. Keys off the **detection name**.
    """
    v = np.asarray(velocity_global_xy, dtype=np.float64).reshape(-1)
    speed = float(np.linalg.norm(v[:2])) if v.size >= 2 and np.all(np.isfinite(v[:2])) else 0.0
    if detection_name in _VEHICLE_CLASSES:
        return "vehicle.moving" if speed >= vehicle_moving_speed else "vehicle.parked"
    if detection_name == "pedestrian":
        return "pedestrian.moving" if speed >= pedestrian_moving_speed else "pedestrian.standing"
    if detection_name in ("bicycle", "motorcycle"):
        return "cycle.without_rider"
    return ""  # traffic_cone, barrier


def make_detection_box(
    sample_token: str,
    detection_name: str,
    score: float,
    conv: Dict[str, object],
    *,
    attribute_name: Optional[str] = None,
    num_pts: int = -1,
):
    """Build a devkit ``DetectionBox`` from a converted box, enforcing the submission contract.

    Coerces ``detection_score`` + every translation/size/rotation float with builtin
    ``float()`` (the devkit asserts ``type(detection_score)==float`` exactly + no NaN/inf
    in trans/size/rot); asserts ``size>0``; resolves ``attribute_name`` (per-class default
    if ``None``). Raises loudly (not silently NaN) on a bad float — caught in-process here,
    never at devkit-load time.
    """
    from nuscenes.eval.detection.data_classes import DetectionBox

    t = tuple(float(v) for v in np.asarray(conv["translation"], dtype=np.float64).reshape(3))
    s = tuple(float(v) for v in np.asarray(conv["size"], dtype=np.float64).reshape(3))
    r = tuple(float(v) for v in np.asarray(conv["rotation"], dtype=np.float64).reshape(4))
    v = tuple(float(v) for v in np.asarray(conv["velocity"], dtype=np.float64).reshape(2))

    for name, vals in (("translation", t), ("size", s), ("rotation", r)):
        if not all(np.isfinite(vals)):
            raise ValueError(f"non-finite {name} in DetectionBox: {vals} (sample {sample_token})")
    if not all(x > 0.0 for x in s):
        raise ValueError(f"non-positive size {s} in DetectionBox (sample {sample_token})")
    if not all(np.isfinite(v)):
        v = (0.0, 0.0)  # the devkit permits NaN velocity, but emit finite for AVE sanity

    attr = attribute_name if attribute_name is not None else attribute_for(detection_name, v)
    return DetectionBox(
        sample_token=str(sample_token),
        translation=t,
        size=s,
        rotation=r,
        velocity=v,
        num_pts=int(num_pts),
        detection_name=str(detection_name),
        detection_score=float(score),   # builtin float (not np.float32) — devkit asserts
        attribute_name=str(attr),
    )


# Content-defined total order on a converted box, so equal-score boxes hit the devkit
# ``accumulate`` greedy matcher in a reproducible order (its tie-break is the emission
# index — see T4_SPEC §2 "deterministic box order"). Sort key is (−score, translation,
# size, yaw): purely a function of box CONTENT, so permuting the decode's emission order
# yields the identical serialized order → identical mAP/NDS.
def detection_box_sort_key(box) -> tuple:
    """Stable content-defined sort key for a ``DetectionBox`` (descending score first)."""
    return (
        -float(box.detection_score),
        float(box.translation[0]), float(box.translation[1]), float(box.translation[2]),
        float(box.size[0]), float(box.size[1]), float(box.size[2]),
        float(box.rotation[0]), float(box.rotation[1]),
        float(box.rotation[2]), float(box.rotation[3]),
        str(box.detection_name),
    )


def decoded_sample_to_boxes(
    decoded: Dict[str, "object"],
    lidar2ego,
    ego2global_lidar,
    sample_token: str,
    class_names: Sequence[str],
    *,
    score_threshold: Optional[float] = None,
    vehicle_moving_speed: float = DEFAULT_VEHICLE_MOVING_SPEED,
    pedestrian_moving_speed: float = DEFAULT_PEDESTRIAN_MOVING_SPEED,
) -> List["object"]:
    """Convert one sample's ``detector.decode`` dict → sorted list of global ``DetectionBox``.

    ``decoded`` = ``{"boxes":[K,7], "scores":[K], "labels":[K], "velocity":[K,2]}`` (the
    single decode). ``class_names`` maps label-id → detection name (``DETECTION_NAMES``
    order). ``score_threshold`` is an OPTIONAL additional floor — by default the decode's
    own ``score_threshold`` already applied, so this is ``None`` (no V4/eval re-threshold).
    Boxes returned in the content-defined deterministic order.
    """
    boxes = np.asarray(decoded["boxes"].detach().cpu().numpy() if hasattr(decoded["boxes"], "detach")
                       else decoded["boxes"], dtype=np.float64)
    scores = np.asarray(decoded["scores"].detach().cpu().numpy() if hasattr(decoded["scores"], "detach")
                        else decoded["scores"], dtype=np.float64)
    labels = np.asarray(decoded["labels"].detach().cpu().numpy() if hasattr(decoded["labels"], "detach")
                        else decoded["labels"]).astype(int)
    vel = decoded.get("velocity", None)
    if vel is not None:
        vel = np.asarray(vel.detach().cpu().numpy() if hasattr(vel, "detach") else vel, dtype=np.float64)

    out: List[object] = []
    for k in range(boxes.shape[0]):
        score = float(scores[k])
        if score_threshold is not None and score < score_threshold:
            continue
        cls = int(labels[k])
        if cls < 0 or cls >= len(class_names):
            continue
        name = class_names[cls]
        v_l = vel[k] if vel is not None else None
        conv = convert_box_to_global(boxes[k], lidar2ego, ego2global_lidar, velocity_lidar=v_l)
        out.append(make_detection_box(
            sample_token, name, score, conv,
            attribute_name=attribute_for(name, conv["velocity"], vehicle_moving_speed,
                                         pedestrian_moving_speed),
        ))
    out.sort(key=detection_box_sort_key)
    return out
