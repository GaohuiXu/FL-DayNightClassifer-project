"""Explicit homogeneous transforms + the nuScenes frame graph (fl_v3 T1).

**Reimplemented from scratch** (no ``mmdet3d``/``mmcv``) and unit-tested against the
``nuscenes-devkit`` geometry oracle (``geometry_utils.transform_matrix``,
``view_points``, ``Box``, pyquaternion ``yaw_pitch_roll``). All functions are pure
numpy on float64, **no RNG, no global state** — so they are bit-deterministic and
host-portable (the ARM rebuild produces identical matrices).

Frame graph (two distinct ego poses — the devkit's own ego-motion compensation):

    lidar → ego(t_lidar) → global → ego(t_cam) → camera → image

See ``conventions.md`` for the declared conventions these implement and the exact
parity tolerances the tests enforce.
"""
from __future__ import annotations

import numpy as np

# Quaternions are nuScenes/pyquaternion order: (w, x, y, z), Hamilton product.

__all__ = [
    "quaternion_to_rotation_matrix",
    "quaternion_multiply",
    "quaternion_inverse",
    "transform_matrix",
    "yaw_from_quaternion",
    "wrap_to_pi",
    "compose_lidar2cam",
    "build_viewpad",
    "build_lidar2img",
    "project_to_image",
]


def quaternion_to_rotation_matrix(q) -> np.ndarray:
    """Unit-quaternion ``(w,x,y,z)`` → 3×3 rotation matrix (pyquaternion convention).

    Normalizes defensively. Unit-tested == ``Quaternion(q).rotation_matrix``.
    """
    q = np.asarray(q, dtype=np.float64)
    n = np.linalg.norm(q)
    if n == 0.0:
        raise ValueError("zero-norm quaternion")
    w, x, y, z = q / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def quaternion_multiply(a, b) -> np.ndarray:
    """Hamilton product ``a ⊗ b`` of two ``(w,x,y,z)`` quaternions."""
    aw, ax, ay, az = np.asarray(a, dtype=np.float64)
    bw, bx, by, bz = np.asarray(b, dtype=np.float64)
    return np.array(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        dtype=np.float64,
    )


def quaternion_inverse(q) -> np.ndarray:
    """Inverse of a ``(w,x,y,z)`` quaternion (conjugate / squared-norm)."""
    w, x, y, z = np.asarray(q, dtype=np.float64)
    n2 = w * w + x * x + y * y + z * z
    if n2 == 0.0:
        raise ValueError("zero-norm quaternion")
    return np.array([w, -x, -y, -z], dtype=np.float64) / n2


def transform_matrix(translation, rotation_q, inverse: bool = False) -> np.ndarray:
    """4×4 homogeneous transform from a translation + ``(w,x,y,z)`` rotation.

    ``inverse=False`` builds the *sensor→parent* map ``p = R·s + t``.
    ``inverse=True`` builds *parent→sensor*: ``s = Rᵀ·(p − t)``.
    Unit-tested == devkit ``geometry_utils.transform_matrix`` (both conventions).
    """
    t = np.asarray(translation, dtype=np.float64).reshape(3)
    R = quaternion_to_rotation_matrix(rotation_q)
    M = np.eye(4, dtype=np.float64)
    if inverse:
        M[:3, :3] = R.T
        M[:3, 3] = -R.T @ t
    else:
        M[:3, :3] = R
        M[:3, 3] = t
    return M


def yaw_from_quaternion(q) -> float:
    """Yaw about +z (CCW-positive, from +x), in radians, from ``(w,x,y,z)``.

    Reimplements pyquaternion ``yaw_pitch_roll[0]`` (intrinsic z-y'-x'' Tait-Bryan):

        yaw = atan2( 2·(w·z − x·y),  1 − 2·(y² + z²) )

    **NOTE the minus** on the cross term — this is *not* the textbook
    ``+x·y`` aerospace yaw. Verified == ``Quaternion(q).yaw_pitch_roll[0]`` to 0
    over 5000 random quaternions (the near-upright real boxes hide a sign error;
    the random test does not).
    """
    w, x, y, z = np.asarray(q, dtype=np.float64)
    return float(np.arctan2(2.0 * (w * z - x * y), 1.0 - 2.0 * (y * y + z * z)))


def wrap_to_pi(angle):
    """Wrap angle(s) to ``[-π, π)``. Scalar or array."""
    a = np.asarray(angle, dtype=np.float64)
    out = (a + np.pi) % (2.0 * np.pi) - np.pi
    return float(out) if out.ndim == 0 else out


def compose_lidar2cam(cs_lidar, ep_lidar, ep_cam, cs_cam) -> np.ndarray:
    """4×4 ``lidar → camera`` via the full two-ego-pose graph.

    Each arg is a dict with ``"translation"`` (3,) and ``"rotation"`` (w,x,y,z)
    (the nuScenes ``calibrated_sensor`` / ``ego_pose`` records). A single shared
    keyframe pose is deliberately NOT supported (see ``conventions.md`` §2).
    """
    T_ego_from_lidar = transform_matrix(cs_lidar["translation"], cs_lidar["rotation"], inverse=False)
    T_global_from_egol = transform_matrix(ep_lidar["translation"], ep_lidar["rotation"], inverse=False)
    T_egoc_from_global = transform_matrix(ep_cam["translation"], ep_cam["rotation"], inverse=True)
    T_cam_from_egoc = transform_matrix(cs_cam["translation"], cs_cam["rotation"], inverse=True)
    return T_cam_from_egoc @ T_egoc_from_global @ T_global_from_egol @ T_ego_from_lidar


def build_viewpad(K) -> np.ndarray:
    """Embed a 3×3 pixel-space intrinsic ``K`` into a 4×4 projection pad."""
    K = np.asarray(K, dtype=np.float64).reshape(3, 3)
    viewpad = np.eye(4, dtype=np.float64)
    viewpad[:3, :3] = K
    return viewpad


def build_lidar2img(K, lidar2cam) -> np.ndarray:
    """4×4 ``lidar → image`` = ``viewpad(K) @ lidar2cam``.

    Projection is non-linear: ``uv = (lidar2img @ [x,y,z,1])[:2] / (...)[2]``.
    """
    return build_viewpad(K) @ np.asarray(lidar2cam, dtype=np.float64)


def project_to_image(points_xyz, lidar2img):
    """Project lidar-frame points ``(N,3)`` or ``(3,N)`` to pixels.

    Returns ``(uv (N,2), depth (N,))``. ``depth > 0`` ⇒ in front of the camera.
    Pixels for non-positive depth are returned as-is (caller masks on depth).
    """
    pts = np.asarray(points_xyz, dtype=np.float64)
    if pts.shape[0] == 3 and pts.ndim == 2 and pts.shape[1] != 3:
        pts = pts.T  # accept (3,N)
    n = pts.shape[0]
    hom = np.concatenate([pts[:, :3], np.ones((n, 1))], axis=1)  # (N,4)
    proj = hom @ np.asarray(lidar2img, dtype=np.float64).T  # (N,4)
    depth = proj[:, 2]
    safe = np.where(np.abs(depth) < 1e-9, 1e-9, depth)
    uv = proj[:, :2] / safe[:, None]
    return uv, depth
