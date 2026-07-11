"""nuScenes metadata/blob paths + read-only guard (fl_v3 T1/S01).

**Single source of truth** for the staged, read-only nuScenes dataset. The
dataroot is config/env driven on Arrhenius — see ``docs/env.md``.

The dataset may be either a fully extracted directory (mini/local) or the
Arrhenius module's metadata directory plus ten stored trainval ZIPs.  Both are
immutable.  S01 extracts/copies nothing and writes nothing under ``DATAROOT``;
the info-cache and ZIP member manifest live outside it. ``resolve_writable`` is
the active runtime guard that makes accidental writes under ``DATAROOT`` raise.

``verify_dataset(version)`` is a lightweight preflight (no full devkit load): it
asserts the version's table dir plus either extracted sensor dirs or the complete
version archive set exist **and** a known per-version sentinel ``sample_token``
resolves in ``sample.json`` —
so a superficially-valid *wrong* root (e.g. the stale
``/mimer/NOBACKUP/Datasets/nuScenes`` with a ``full/`` layout and different tokens)
**fails**, not passes.
"""
from __future__ import annotations

import json
import os
# ---------------------------------------------------------------------------
# The staged dataset (read-only). Prefer an explicit run config key; otherwise
# set NUSCENES_DATAROOT or ARRHENIUS_NUSCENES_DATAROOT in the Slurm launcher.
# ---------------------------------------------------------------------------
def _env_dataroot() -> str:
    return (
        os.environ.get("NUSCENES_DATAROOT", "").strip()
        or os.environ.get("ARRHENIUS_NUSCENES_DATAROOT", "").strip()
        or os.environ.get("NUSCENES_DATA_DIR", "").strip()
    )


# Compatibility snapshot for older callers/tests that import ``P.DATAROOT``.
# ``get_dataroot`` also checks the live environment so a module loaded after this
# module was imported is still discovered.
DATAROOT = _env_dataroot()

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
    "sensor.json",
    "map.json",
)

_VERSION_DIR_ALIASES = {
    "v1.0-trainval": ("v1.0-trainval", "trainval"),
    "v1.0-test": ("v1.0-test", "test"),
    "v1.0-mini": ("v1.0-mini",),
}

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
    """Resolve dataroot with explicit settings ahead of the dataset module.

    Precedence is run config, ``NUSCENES_DATAROOT``,
    ``ARRHENIUS_NUSCENES_DATAROOT``, then the canonical module variable
    ``NUSCENES_DATA_DIR``.
    """
    if run_config:
        override = str(run_config.get("nuscenes-dataroot", "") or "").strip()
        if override:
            return os.path.abspath(override)
    configured = _env_dataroot() or DATAROOT
    if configured:
        return os.path.abspath(configured)
    raise FileNotFoundError(
        "nuScenes dataroot is not configured. Set the run config key "
        "`nuscenes-dataroot`, or export NUSCENES_DATAROOT / "
        "ARRHENIUS_NUSCENES_DATAROOT; on Arrhenius load "
        "nuScenes-data/1.0-map-1.3-zip to provide NUSCENES_DATA_DIR."
    )


def get_zip_manifest(run_config: dict | None = None, *, required: bool = False) -> str:
    """Resolve the external read-only ZIP manifest path.

    The config key is supported for data-specific tools.  Existing production
    dataset constructors do not receive run config, so launchers should export
    ``NUSCENES_ZIP_MANIFEST`` (or its Arrhenius-prefixed alias).
    """
    configured = ""
    if run_config:
        configured = str(run_config.get("nuscenes-zip-manifest", "") or "").strip()
    configured = (
        configured
        or os.environ.get("NUSCENES_ZIP_MANIFEST", "").strip()
        or os.environ.get("ARRHENIUS_NUSCENES_ZIP_MANIFEST", "").strip()
    )
    if configured:
        return os.path.abspath(configured)
    if required:
        raise FileNotFoundError(
            "nuScenes ZIP manifest is not configured. Set NUSCENES_ZIP_MANIFEST "
            "to the S01 SQLite manifest outside the read-only dataroot."
        )
    return ""


def _root_or_raise(dataroot: str | None = None) -> str:
    if dataroot:
        return dataroot
    return get_dataroot()


def version_table_dir(version: str, dataroot: str | None = None) -> str:
    root = _root_or_raise(dataroot)
    candidates = _VERSION_DIR_ALIASES.get(version, (version,))
    for dirname in candidates:
        candidate = os.path.join(root, dirname)
        if os.path.isdir(candidate):
            return candidate
    return os.path.join(root, candidates[0])


def devkit_table_dirname(version: str, dataroot: str | None = None) -> str:
    """Directory name holding JSON tables for this root/version."""
    return os.path.basename(version_table_dir(version, dataroot))


def create_nuscenes(version: str, dataroot: str | None = None, *, verbose: bool = False, **kwargs):
    """Construct the official devkit over extracted or Arrhenius metadata.

    The module stores official ``v1.0-trainval`` tables in ``trainval/`` and
    official ``v1.0-test`` tables in ``test/``.  Override only ``table_root`` so
    ``nusc.version`` remains the official version and ``nusc.dataroot`` remains
    the module root used to form host-portable sample-data filenames.
    """
    from nuscenes import NuScenes

    root = os.path.abspath(_root_or_raise(dataroot))
    table_root = version_table_dir(version, root)
    if os.path.basename(table_root) == version:
        return NuScenes(version=version, dataroot=root, verbose=verbose, **kwargs)

    class _AliasedTableRootNuScenes(NuScenes):
        def __init__(self):
            self._fl_v3_table_root = table_root
            super().__init__(version=version, dataroot=root, verbose=verbose, **kwargs)

        @property
        def table_root(self) -> str:
            return self._fl_v3_table_root

    return _AliasedTableRootNuScenes()


def samples_dir(dataroot: str | None = None) -> str:
    return os.path.join(_root_or_raise(dataroot), "samples")


def abspath_from_relative(rel_path: str, dataroot: str | None = None) -> str:
    """Resolve a DATAROOT-relative path (as stored in the info-cache) to absolute.

    The cache stores DATAROOT-relative paths so its content hash is host-portable;
    this is the only place they become host-absolute (at read time).
    """
    from fl_v3.data.nuscenes.zip_backend import canonical_member_path

    rel = canonical_member_path(rel_path)
    return os.path.join(_root_or_raise(dataroot), *rel.split("/"))


def relative_to_dataroot(abs_path: str, dataroot: str | None = None) -> str:
    """Make an absolute blob path DATAROOT-relative (with forward slashes).

    Forward-slash normalization keeps the cache hash identical across OSes. Raises if
    ``abs_path`` resolves *outside* ``dataroot`` — such a path would yield a
    ``../../host-structure`` relative string and silently leak host-specific layout
    into the host-portable info-cache hash. On the real, immutable, fully-extracted
    dataset the devkit always returns ``dataroot/<filename>``, so this never fires in
    practice; it is a hard guard against a future mis-pointed root.
    """
    root = _root_or_raise(dataroot)
    rel = os.path.relpath(abs_path, root).replace(os.sep, "/")
    if rel == ".." or rel.startswith("../"):
        raise ValueError(
            f"blob path {abs_path!r} resolves OUTSIDE dataroot {root!r} (rel={rel!r}); "
            f"this would leak host structure into the host-portable cache hash."
        )
    from fl_v3.data.nuscenes.zip_backend import canonical_member_path

    return canonical_member_path(rel)


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

    Checks: (1) the version table dir exists with all devkit-required tables;
    (2) either extracted sensor directories or the exact version archive set is
    present and readable; (3) the per-version sentinel ``sample_token`` is in
    ``sample.json``.  This is not a member-coverage scan.
    """
    root = _root_or_raise(dataroot)
    table_dir = version_table_dir(version, root)
    if not os.path.isdir(table_dir):
        raise FileNotFoundError(
            f"nuScenes version dir not found: {table_dir!r}. "
            f"Is {root!r} the right extracted or module dataroot? The stale "
            f"/mimer/NOBACKUP/Datasets/nuScenes has a `full/` layout and will fail here."
        )
    missing = [t for t in _REQUIRED_TABLES if not os.path.isfile(os.path.join(table_dir, t))]
    if missing:
        raise FileNotFoundError(f"{table_dir!r} missing required tables: {missing}")

    sdir = samples_dir(root)
    extracted_dirs = [os.path.join(sdir, ch) for ch in (*CAMERA_CHANNELS, LIDAR_CHANNEL)]
    if all(os.path.isdir(path) for path in extracted_dirs):
        backend = "directory"
        archives: tuple[str, ...] = ()
    else:
        from fl_v3.data.nuscenes.zip_backend import TRAINVAL_ARCHIVE_NAMES

        if version == "v1.0-trainval":
            archives = TRAINVAL_ARCHIVE_NAMES
        elif version == "v1.0-test":
            archives = ("test_blobs.zip",)
        else:
            archives = ()
        missing_archives = [
            name for name in archives if not os.path.isfile(os.path.join(root, name))
        ]
        if not archives or missing_archives:
            missing_dirs = [path for path in extracted_dirs if not os.path.isdir(path)]
            raise FileNotFoundError(
                f"nuScenes blob backend unavailable for {version!r}: "
                f"missing extracted dirs={missing_dirs[:3]}"
                + (f", missing/unreadable archives={missing_archives}" if archives else "")
            )
        backend = "zip"

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
        "table_dir": table_dir,
        "blob_backend": backend,
        "archives": list(archives),
        "zip_manifest": get_zip_manifest() if backend == "zip" else "",
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
    root = os.path.realpath(_root_or_raise(dataroot))
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
