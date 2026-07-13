"""Clean GT-database helpers for deterministic object sampling.

GT-paste ("db_sampler" / "ObjectSample") is the standard nuScenes/Waymo SOTA ingredient the platform lacks:
copy per-object LiDAR point crops from a pre-built database into training scenes so rare classes
(trailer/CV/bus/truck/bicycle — each in only ~18-21% of keyframes) appear in MANY more scenes, with
per-object orientation jitter for diversity (the variety CBGS lacked — CBGS replayed whole keyframes,
which overfit the few fixed instances). The pasted objects are LiDAR-only (no camera pixels); our LSS is
UNSUPERVISED (no depth loss) so this injects ZERO depth-label noise — the BEVFusion-standard tolerated cost.

This module contains the offline-build and paste-time geometry in pure NumPy.
All paste-time sampling uses ``numpy.random`` seeded by ``seeded_worker_init``;
the database build is a deterministic walk.
"""
from __future__ import annotations

import os
import pickle
from typing import Dict, List, Optional

import numpy as np


# ---------------------------------------------------------------------------
# build-time: crop an object's points into its BOX-LOCAL frame
# ---------------------------------------------------------------------------
def crop_object_points(points: np.ndarray, box7: np.ndarray) -> np.ndarray:
    """Return the points inside ``box7`` in BOX-LOCAL coords ``[n,4]`` = ``(x_l, y_l, z_l, intensity)``.

    ``points`` ``[P, W]`` with cols 0:3 = xyz and col 3 = intensity (W>=4; extra cols ignored).
    ``box7`` = ``(cx, cy, cz, dx, dy, dz, yaw)`` in the canonical LIDAR_TOP frame. A point is inside iff
    its box-local coords satisfy ``|x_l|<=dx/2 & |y_l|<=dy/2 & |z_l|<=dz/2``. Pure numpy, deterministic.
    """
    cx, cy, cz, dx, dy, dz, yaw = [float(v) for v in box7[:7]]
    xw = points[:, 0] - cx
    yw = points[:, 1] - cy
    zl = points[:, 2] - cz
    cyaw, syaw = np.cos(yaw), np.sin(yaw)
    xl = cyaw * xw + syaw * yw          # R(-yaw) @ (world - center)
    yl = -syaw * xw + cyaw * yw
    inside = (np.abs(xl) <= dx / 2) & (np.abs(yl) <= dy / 2) & (np.abs(zl) <= dz / 2)
    local = np.stack([xl[inside], yl[inside], zl[inside], points[inside, 3]], axis=1)
    return np.ascontiguousarray(local.astype(np.float32))


def place_object_points(local: np.ndarray, center_xyz, yaw: float) -> np.ndarray:
    """Inverse of :func:`crop_object_points`: box-local points → world ``[n,3]`` at ``center`` with ``yaw``.

    ``local`` ``[n,4]`` (cols 0:3 = box-local xyz). Returns world xyz ``[n,3]`` = ``R(yaw)·xy_l + center``
    (z = z_l + cz). Used at paste time to re-pose a stored crop (a jittered ``yaw`` gives orientation
    diversity)."""
    c, s = np.cos(float(yaw)), np.sin(float(yaw))
    xl, yl, zl = local[:, 0], local[:, 1], local[:, 2]
    xw = c * xl - s * yl + float(center_xyz[0])
    yw = s * xl + c * yl + float(center_xyz[1])
    zw = zl + float(center_xyz[2])
    return np.stack([xw, yw, zw], axis=1).astype(np.float32)


# ---------------------------------------------------------------------------
# paste-time: rotated-rectangle BEV overlap (SAT) for collision rejection
# ---------------------------------------------------------------------------
def box_corners_2d(box) -> np.ndarray:
    """4 BEV corners ``[4,2]`` of a box ``(cx, cy, dx, dy, yaw)`` (footprint only; z ignored)."""
    cx, cy, dx, dy, yaw = float(box[0]), float(box[1]), float(box[2]), float(box[3]), float(box[4])
    c, s = np.cos(yaw), np.sin(yaw)
    hx, hy = dx / 2.0, dy / 2.0
    local = np.array([[hx, hy], [hx, -hy], [-hx, -hy], [-hx, hy]], dtype=np.float64)
    R = np.array([[c, -s], [s, c]], dtype=np.float64)
    return local @ R.T + np.array([cx, cy], dtype=np.float64)


def rects_overlap(corners_a: np.ndarray, corners_b: np.ndarray) -> bool:
    """True iff two convex quads OVERLAP (no separating axis) — the SAT test, pure numpy/deterministic."""
    for poly in (corners_a, corners_b):
        for i in range(4):
            edge = poly[(i + 1) % 4] - poly[i]
            axis = np.array([-edge[1], edge[0]], dtype=np.float64)
            n = np.hypot(axis[0], axis[1])
            if n < 1e-9:
                continue
            axis /= n
            pa = corners_a @ axis
            pb = corners_b @ axis
            if pa.max() < pb.min() or pb.max() < pa.min():
                return False          # found a separating axis ⇒ disjoint
    return True                       # no separating axis ⇒ overlap


# ---------------------------------------------------------------------------
# DB load (lazy, per-worker module cache, read-only)
# ---------------------------------------------------------------------------
_DB_CACHE: Dict[str, Dict] = {}


def load_gt_database(db_path: str, classes: Optional[List[str]] = None) -> Dict[str, List[dict]]:
    """Load the per-class GT-DB (``{name: [ {points[n,4], box7[7], label, name, num_pts}, ... ]}``).

    Lazily cached per ``db_path`` (read-only — no RNG at load) so forked DataLoader workers each load once.
    Reads ``<db_path>/<name>.pkl`` for each requested class. Returns only the classes present.
    """
    key = os.path.abspath(db_path)
    if key in _DB_CACHE:
        db = _DB_CACHE[key]
    else:
        db = {}
        meta_p = os.path.join(db_path, "meta.json")
        if not os.path.isdir(db_path):
            raise FileNotFoundError(f"GT-database dir not found: {db_path}")
        for fn in sorted(os.listdir(db_path)):
            if fn.endswith(".pkl"):
                name = fn[:-4]
                with open(os.path.join(db_path, fn), "rb") as f:
                    db[name] = pickle.load(f)
        db["__meta_path__"] = meta_p
        _DB_CACHE[key] = db
    if classes is None:
        return {k: v for k, v in db.items() if not k.startswith("__")}
    return {c: db[c] for c in classes if c in db}
