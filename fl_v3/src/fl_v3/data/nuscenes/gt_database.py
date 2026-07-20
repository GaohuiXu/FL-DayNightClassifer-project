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

import hashlib
import json
import os
import pickle
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np


# ---------------------------------------------------------------------------
# build-time: crop an object's points into its BOX-LOCAL frame
# ---------------------------------------------------------------------------
def crop_object_points(
    points: np.ndarray,
    box7: np.ndarray,
    *,
    feature_columns: int = 4,
) -> np.ndarray:
    """Return points inside ``box7`` in box-local coordinates.

    ``feature_columns=4`` preserves the legacy ``xyz+intensity`` database.
    Phase I passes ``5`` so keyframe ``xyz+intensity+ring`` is retained exactly.
    ``points`` is ``[P,W]`` with xyz in columns 0:3.
    ``box7`` = ``(cx, cy, cz, dx, dy, dz, yaw)`` in the canonical LIDAR_TOP frame. A point is inside iff
    its box-local coords satisfy ``|x_l|<=dx/2 & |y_l|<=dy/2 & |z_l|<=dz/2``. Pure numpy, deterministic.
    """
    width = int(feature_columns)
    if points.ndim != 2 or width < 4 or width > points.shape[1]:
        raise ValueError(
            f"invalid crop feature width {feature_columns!r} for point shape {points.shape}"
        )
    cx, cy, cz, dx, dy, dz, yaw = [float(v) for v in box7[:7]]
    xw = points[:, 0] - cx
    yw = points[:, 1] - cy
    zl = points[:, 2] - cz
    cyaw, syaw = np.cos(yaw), np.sin(yaw)
    xl = cyaw * xw + syaw * yw          # R(-yaw) @ (world - center)
    yl = -syaw * xw + cyaw * yw
    inside = (np.abs(xl) <= dx / 2) & (np.abs(yl) <= dy / 2) & (np.abs(zl) <= dz / 2)
    local = np.concatenate(
        [
            np.stack([xl[inside], yl[inside], zl[inside]], axis=1),
            points[inside, 3:width],
        ],
        axis=1,
    )
    return np.ascontiguousarray(local.astype(np.float32))


def crop_object_points_center_relative(
    points: np.ndarray,
    box7: np.ndarray,
    *,
    feature_columns: int = 5,
) -> np.ndarray:
    """Pinned GTDB crop: points inside the box, translated by center only.

    MIT's database builder subtracts ``box[:3]`` but does not rotate the crop.
    This is deliberately separate from the legacy box-local helper above.
    """
    width = int(feature_columns)
    if points.ndim != 2 or width < 4 or width > points.shape[1]:
        raise ValueError(
            f"invalid crop feature width {feature_columns!r} for point shape {points.shape}"
        )
    cx, cy, cz, dx, dy, dz, yaw = [float(value) for value in box7[:7]]
    xw = points[:, 0] - cx
    yw = points[:, 1] - cy
    zw = points[:, 2] - cz
    cosine, sine = np.cos(yaw), np.sin(yaw)
    local_x = cosine * xw + sine * yw
    local_y = -sine * xw + cosine * yw
    inside = (
        (np.abs(local_x) <= dx / 2.0)
        & (np.abs(local_y) <= dy / 2.0)
        & (np.abs(zw) <= dz / 2.0)
    )
    cropped = np.ascontiguousarray(points[inside, :width], dtype=np.float32)
    cropped[:, :3] -= np.asarray([cx, cy, cz], dtype=np.float32)
    return cropped


def place_object_points(local: np.ndarray, center_xyz, yaw: float) -> np.ndarray:
    """Inverse of :func:`crop_object_points`: box-local points → world ``[n,3]`` at ``center`` with ``yaw``.

    ``local`` ``[n,W]`` (cols 0:3 = box-local xyz). Returns world xyz ``[n,3]`` = ``R(yaw)·xy_l + center``
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
            if pa.max() <= pb.min() or pb.max() <= pa.min():
                return False          # found a separating axis ⇒ disjoint
    return True                       # no separating axis ⇒ overlap


# ---------------------------------------------------------------------------
# DB load (lazy, per-worker module cache, read-only)
# ---------------------------------------------------------------------------
_DB_CACHE: Dict[str, Dict] = {}


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_phase1_gt_database(
    database_root: str,
    manifest_path: str,
    *,
    expected_manifest_sha256: str,
    class_names: Sequence[str],
    expected_contract: Mapping[str, str],
    expected_source: Mapping[str, Any],
    expected_semantics: Mapping[str, Any],
) -> Dict[str, List[dict]]:
    """Verify and load one immutable Phase-I GT database.

    Unlike the legacy loader, this entry point treats the canonical manifest and
    every class pickle as one atomic scientific input.  It verifies all bytes
    before unpickling and repeats each file hash afterwards to catch in-flight
    mutation.
    """
    root = Path(database_root).resolve()
    manifest_file = Path(manifest_path).resolve()
    if manifest_file != root / "manifest.json":
        raise ValueError("Phase-I GTDB manifest must be <database_root>/manifest.json")
    if not root.is_dir() or not manifest_file.is_file():
        raise FileNotFoundError(f"Phase-I GTDB is incomplete: {root}")
    encoded = manifest_file.read_bytes()
    actual_manifest_sha = hashlib.sha256(encoded).hexdigest()
    if actual_manifest_sha != str(expected_manifest_sha256):
        raise ValueError(
            "Phase-I GTDB manifest SHA-256 drift: "
            f"expected={expected_manifest_sha256}, actual={actual_manifest_sha}"
        )
    try:
        manifest = json.loads(encoded.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("Phase-I GTDB manifest is not valid UTF-8 JSON") from exc
    if _canonical_json_bytes(manifest) != encoded:
        raise ValueError("Phase-I GTDB manifest is not canonically encoded")
    if not isinstance(manifest, dict) or manifest.get("schema") != "s10.phase1.gtdb.v1":
        raise ValueError("Phase-I GTDB manifest schema drift")
    expected_keys = {
        "schema", "contract", "source", "semantics", "files", "counts", "total_objects",
    }
    if set(manifest) != expected_keys:
        raise ValueError("Phase-I GTDB manifest root fields drift")
    for key, expected in expected_contract.items():
        if manifest["contract"].get(key) != expected:
            raise ValueError(f"Phase-I GTDB contract field {key!r} drift")
    for key, expected in expected_source.items():
        if manifest["source"].get(key) != expected:
            raise ValueError(f"Phase-I GTDB source field {key!r} drift")
    for key, expected in expected_semantics.items():
        if manifest["semantics"].get(key) != expected:
            raise ValueError(f"Phase-I GTDB semantic field {key!r} drift")

    names = tuple(str(name) for name in class_names)
    files = manifest.get("files")
    counts = manifest.get("counts")
    if not names or len(names) != len(set(names)):
        raise ValueError("Phase-I GTDB class order must be non-empty and unique")
    if not isinstance(files, dict) or set(files) != set(names):
        raise ValueError("Phase-I GTDB file class registry drift")
    if not isinstance(counts, dict) or set(counts) != set(names):
        raise ValueError("Phase-I GTDB count class registry drift")
    allowed_entries = {f"{name}.pkl" for name in names} | {"manifest.json"}
    if set(os.listdir(root)) != allowed_entries:
        raise ValueError("Phase-I GTDB contains missing or undeclared files")

    database: Dict[str, List[dict]] = {}
    total = 0
    for name in names:
        record = files[name]
        if not isinstance(record, dict) or set(record) != {
            "path", "sha256", "bytes", "objects",
        }:
            raise ValueError(f"Phase-I GTDB file record {name!r} drift")
        path = Path(record["path"]).resolve()
        if path != root / f"{name}.pkl" or not path.is_file():
            raise ValueError(f"Phase-I GTDB file path {name!r} drift")
        before = {
            "bytes": path.stat().st_size,
            "sha256": _sha256_file(path),
        }
        if before["bytes"] != record["bytes"] or before["sha256"] != record["sha256"]:
            raise ValueError(f"Phase-I GTDB file identity {name!r} drift")
        with path.open("rb") as stream:
            values = pickle.load(stream)
        after = {
            "bytes": path.stat().st_size,
            "sha256": _sha256_file(path),
        }
        if after != before:
            raise ValueError(f"Phase-I GTDB file {name!r} changed while loading")
        count = int(record["objects"])
        if (
            not isinstance(values, list)
            or not values
            or len(values) != count
            or counts.get(name) != count
        ):
            raise ValueError(f"Phase-I GTDB object count {name!r} drift")
        database[name] = values
        total += count
    if manifest.get("total_objects") != total:
        raise ValueError("Phase-I GTDB total object count drift")
    return database


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
