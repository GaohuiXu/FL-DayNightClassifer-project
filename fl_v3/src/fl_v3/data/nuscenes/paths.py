"""nuScenes dataset paths + read-only guard (fl_v3 T1).

**Single source of truth** for the staged, read-only nuScenes dataset. Only this
file's ``DATAROOT`` changes on an Arrhenius (ARM) re-point — see ``docs/env.md``.

The dataset is **fully extracted, read-only** at ``DATAROOT`` (config-overridable
via the ``nuscenes-dataroot`` app key). T1 **extracts/copies nothing** and writes
**nothing** under ``DATAROOT``; the info-cache lives under ``nuscenes-cache-dir``
(below ``fl_outputs/``). ``resolve_writable`` is the active runtime guard that makes
any accidental write under ``DATAROOT`` raise.

``verify_dataset(version)`` is a lightweight preflight (no full devkit load): it
asserts the version's table dir + the six camera blob dirs + ``LIDAR_TOP`` exist
**and** a known per-version sentinel ``sample_token`` resolves in ``sample.json`` —
so a superficially-valid *wrong* root (e.g. the stale
``/mimer/NOBACKUP/Datasets/nuScenes`` with a ``full/`` layout and different tokens)
**fails**, not passes.
"""
from __future__ import annotations

import json
import os
from typing import Iterable

# ---------------------------------------------------------------------------
# The staged dataset (read-only). ONLY this constant moves on an ARM re-point.
# A *different*, wrong-layout /mimer/NOBACKUP/Datasets/nuScenes also exists — do
# NOT point at it (it has a `full/` subtree, not version table dirs at top level;
# verify_dataset catches the mistake via the sentinel + table-dir checks).
# ---------------------------------------------------------------------------
DATAROOT = "/mimer/NOBACKUP/Datasets/NuScenes_v1.0"

# The six cameras (all cameras nuScenes ships) + the single LiDAR we carry.
CAMERA_CHANNELS = (
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
    "CAM_FRONT_LEFT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
)
LIDAR_CHANNEL = "LIDAR_TOP"

# Tables every valid nuScenes version dir must contain (subset that we read).
_REQUIRED_TABLES = (
    "sample.json",
    "sample_data.json",
    "sample_annotation.json",
    "calibrated_sensor.json",
    "ego_pose.json",
    "scene.json",
    "log.json",
    "category.json",
    "instance.json",
    "visibility.json",
    "attribute.json",
)

# Per-version sentinel sample_token — a known-present token that a wrong root will
# not contain. Captured live from each version's sample.json (sample[0]).
_SENTINELS = {
    "v1.0-mini": "ca9a282c9e77460f8360f564131a8af5",
    "v1.0-trainval": "e93e98b63d3b40209056d129dc53ceee",
    "v1.0-test": "1b9a789e08bb4b7b89eacb2176c70840",
}

KNOWN_LOCATIONS = (
    "boston-seaport",
    "singapore-onenorth",
    "singapore-queenstown",
    "singapore-hollandvillage",
)


def get_dataroot(run_config: dict | None = None) -> str:
    """Resolve the dataroot: ``nuscenes-dataroot`` config override, else default."""
    if run_config:
        override = str(run_config.get("nuscenes-dataroot", "") or "").strip()
        if override:
            return os.path.abspath(override)
    return DATAROOT


def version_table_dir(version: str, dataroot: str | None = None) -> str:
    return os.path.join(dataroot or DATAROOT, version)


def samples_dir(dataroot: str | None = None) -> str:
    return os.path.join(dataroot or DATAROOT, "samples")


def sweeps_dir(dataroot: str | None = None) -> str:
    return os.path.join(dataroot or DATAROOT, "sweeps")


def abspath_from_relative(rel_path: str, dataroot: str | None = None) -> str:
    """Resolve a DATAROOT-relative path (as stored in the info-cache) to absolute.

    The cache stores DATAROOT-relative paths so its content hash is host-portable;
    this is the only place they become host-absolute (at read time).
    """
    return os.path.join(dataroot or DATAROOT, rel_path)


def relative_to_dataroot(abs_path: str, dataroot: str | None = None) -> str:
    """Make an absolute blob path DATAROOT-relative (with forward slashes).

    Forward-slash normalization keeps the cache hash identical across OSes. Raises if
    ``abs_path`` resolves *outside* ``dataroot`` — such a path would yield a
    ``../../host-structure`` relative string and silently leak host-specific layout
    into the host-portable info-cache hash. On the real, immutable, fully-extracted
    dataset the devkit always returns ``dataroot/<filename>``, so this never fires in
    practice; it is a hard guard against a future mis-pointed root.
    """
    root = dataroot or DATAROOT
    rel = os.path.relpath(abs_path, root).replace(os.sep, "/")
    if rel == ".." or rel.startswith("../"):
        raise ValueError(
            f"blob path {abs_path!r} resolves OUTSIDE dataroot {root!r} (rel={rel!r}); "
            f"this would leak host structure into the host-portable cache hash."
        )
    return rel


def sentinel_token(version: str) -> str:
    if version not in _SENTINELS:
        raise ValueError(
            f"No sentinel registered for version {version!r}; "
            f"known: {sorted(_SENTINELS)}"
        )
    return _SENTINELS[version]


def verify_dataset(version: str, dataroot: str | None = None) -> dict:
    """Lightweight read-only preflight. Raises ``FileNotFoundError``/``ValueError``
    on any failure; returns a small report dict on success.

    Checks: (1) the version table dir exists with all required tables; (2) the six
    camera blob dirs + ``LIDAR_TOP`` exist under ``samples/``; (3) the per-version
    sentinel ``sample_token`` is present in ``sample.json``. No full devkit load.
    """
    root = dataroot or DATAROOT
    table_dir = version_table_dir(version, root)
    if not os.path.isdir(table_dir):
        raise FileNotFoundError(
            f"nuScenes version dir not found: {table_dir!r}. "
            f"Is {root!r} the right (extracted) dataroot? The stale "
            f"/mimer/NOBACKUP/Datasets/nuScenes has a `full/` layout and will fail here."
        )
    missing = [t for t in _REQUIRED_TABLES if not os.path.isfile(os.path.join(table_dir, t))]
    if missing:
        raise FileNotFoundError(f"{table_dir!r} missing required tables: {missing}")

    sdir = samples_dir(root)
    for ch in (*CAMERA_CHANNELS, LIDAR_CHANNEL):
        if not os.path.isdir(os.path.join(sdir, ch)):
            raise FileNotFoundError(f"sensor blob dir missing: {os.path.join(sdir, ch)!r}")

    # Sentinel: parse sample.json (cheap) and assert the known token is present.
    sentinel = sentinel_token(version)
    with open(os.path.join(table_dir, "sample.json"), encoding="utf-8") as f:
        samples = json.load(f)
    tokens = {s["token"] for s in samples}
    if sentinel not in tokens:
        raise ValueError(
            f"sentinel sample_token {sentinel!r} not found in {table_dir}/sample.json "
            f"(n={len(tokens)}). This is almost certainly the WRONG dataroot."
        )
    return {
        "version": version,
        "dataroot": root,
        "n_samples": len(samples),
        "sentinel_token": sentinel,
        "sentinel_ok": True,
    }


def resolve_writable(path: str, dataroot: str | None = None) -> str:
    """Active read-only guard: raise if ``path`` resolves under ``DATAROOT``.

    Call before opening any path for writing (cache, viz, stats). The dataset is
    immutable; a write under it is a bug. Uses ``commonpath`` on the real,
    absolute paths so symlink/``..`` tricks cannot slip a write under DATAROOT.
    """
    root = os.path.realpath(dataroot or DATAROOT)
    target = os.path.realpath(os.path.abspath(path))
    try:
        common = os.path.commonpath([root, target])
    except ValueError:
        # Different drives (Windows) — cannot be under root.
        return target
    if common == root:
        raise PermissionError(
            f"Refusing to write under the read-only nuScenes DATAROOT: {path!r} "
            f"resolves under {root!r}. Write to nuscenes-cache-dir / fl_outputs instead."
        )
    return target


def iter_required_sensor_dirs(dataroot: str | None = None) -> Iterable[str]:
    sdir = samples_dir(dataroot)
    for ch in (*CAMERA_CHANNELS, LIDAR_CHANNEL):
        yield os.path.join(sdir, ch)
