"""S07-A fail-closed cache/manifest provenance for GT-database preparation."""
from __future__ import annotations

import importlib.util
import inspect
import json
import os
import pickle
import zipfile
from pathlib import Path

import pytest

from fl_v3.data.nuscenes import info_cache as IC
from fl_v3.data.nuscenes import paths as P
from fl_v3.data.nuscenes.zip_backend import (
    TRAINVAL_ARCHIVE_NAMES,
    build_zip_manifest,
)


_SCRIPT = Path(__file__).parents[1] / "scripts" / "build_gt_database.py"
_SPEC = importlib.util.spec_from_file_location("fl_v3_build_gt_database", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
GTDB = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(GTDB)


def _write_required_tables(root: Path, version: str, dirname: str) -> None:
    table_dir = root / dirname
    table_dir.mkdir(parents=True)
    for table in P._REQUIRED_TABLES:
        payload = []
        if table == "sample.json":
            payload = [{"token": P.sentinel_token(version)}]
        (table_dir / table).write_text(json.dumps(payload), encoding="utf-8")


def _make_directory_root(root: Path) -> None:
    _write_required_tables(root, "v1.0-mini", "v1.0-mini")
    for channel in (*P.CAMERA_CHANNELS, P.LIDAR_CHANNEL):
        (root / "samples" / channel).mkdir(parents=True)


def _make_zip_root(root: Path) -> None:
    _write_required_tables(root, "v1.0-trainval", "trainval")
    for index, archive_name in enumerate(TRAINVAL_ARCHIVE_NAMES):
        with zipfile.ZipFile(
            root / archive_name, "w", compression=zipfile.ZIP_STORED
        ) as archive:
            archive.writestr(f"samples/CAM_FRONT/{index}.jpg", f"payload-{index}".encode())


def test_gt_database_cache_load_binds_depth_hash_and_sidecar(
    nusc_mini, dataroot, tmp_path
):
    cache_dir = str(tmp_path / "cache")
    tokens = IC.split_sample_tokens(nusc_mini, "mini_val")[:2]
    infos = IC.build_info_list(nusc_mini, tokens, dataroot, n_sweeps=2)
    meta = IC.save_cache(
        infos,
        cache_dir,
        "v1.0-mini",
        "mini_val",
        "mini-smoke",
        dataroot,
        n_sweeps=2,
    )

    loaded, loaded_meta = GTDB._load_info_list(
        cache_dir, "v1.0-mini", "mini_val", 2, meta["cache_hash"]
    )
    assert loaded_meta == meta
    assert len(loaded) == len(infos)

    with pytest.raises(ValueError, match="frozen expected hash"):
        GTDB._load_info_list(cache_dir, "v1.0-mini", "mini_val", 2, "0" * 64)
    with pytest.raises(FileNotFoundError, match="nsweeps10"):
        GTDB._load_info_list(
            cache_dir, "v1.0-mini", "mini_val", 10, meta["cache_hash"]
        )

    _, sidecar = IC.cache_paths(cache_dir, "v1.0-mini", "mini_val", 2)
    with open(sidecar, encoding="utf-8") as stream:
        sidecar_meta = json.load(stream)
    sidecar_meta["cache_hash"] = "f" * 64
    with open(sidecar, "w", encoding="utf-8") as stream:
        json.dump(sidecar_meta, stream)
    with pytest.raises(ValueError, match="sidecar differs"):
        GTDB._load_info_list(
            cache_dir, "v1.0-mini", "mini_val", 2, meta["cache_hash"]
        )
    os.unlink(sidecar)
    with pytest.raises(FileNotFoundError):
        GTDB._load_info_list(
            cache_dir, "v1.0-mini", "mini_val", 2, meta["cache_hash"]
        )


def test_gt_database_rejects_historical_t1v1_filename(tmp_path):
    legacy = tmp_path / "nuscenes_info_v1.0-trainval_train_t1.v1.pkl"
    with open(legacy, "wb") as stream:
        pickle.dump({"info_list": []}, stream)
    with pytest.raises(FileNotFoundError, match="t1.v2_nsweeps10"):
        GTDB._load_info_list(
            str(tmp_path), "v1.0-trainval", "train", 10, "0" * 64
        )


def test_gt_database_preserves_directory_backend_without_manifest(tmp_path):
    root = tmp_path / "directory"
    root.mkdir()
    _make_directory_root(root)
    store, provenance = GTDB._open_blob_store(str(root), "v1.0-mini", "", "", "")
    try:
        assert store.mode == "directory"
        assert provenance == {
            "blob_backend": "directory",
            "zip_manifest_path": None,
            "zip_manifest_hash": None,
            "zip_manifest_file_sha256": None,
        }
    finally:
        store.close()
    with pytest.raises(ValueError, match="must not be relabeled"):
        GTDB._open_blob_store(str(root), "v1.0-mini", "x", "0" * 64, "0" * 64)


def test_gt_database_zip_manifest_hashes_fail_closed(tmp_path):
    root = tmp_path / "zip"
    root.mkdir()
    _make_zip_root(root)
    manifest = tmp_path / "manifest.sqlite"
    report = build_zip_manifest(str(root), str(manifest))
    file_hash = GTDB._sha256_file(str(manifest))

    store, provenance = GTDB._open_blob_store(
        str(root),
        "v1.0-trainval",
        str(manifest),
        report["manifest_hash"],
        file_hash,
    )
    try:
        assert store.mode == "zip"
        assert provenance["zip_manifest_hash"] == report["manifest_hash"]
        assert provenance["zip_manifest_file_sha256"] == file_hash
        assert provenance["zip_manifest_archive_names"] == list(TRAINVAL_ARCHIVE_NAMES)
    finally:
        store.close()

    with pytest.raises(ValueError, match="frozen logical hash"):
        GTDB._open_blob_store(
            str(root), "v1.0-trainval", str(manifest), "0" * 64, file_hash
        )
    with pytest.raises(ValueError, match="frozen SHA-256"):
        GTDB._open_blob_store(
            str(root),
            "v1.0-trainval",
            str(manifest),
            report["manifest_hash"],
            "0" * 64,
        )


def test_gt_database_caller_has_no_direct_t1v1_bypass():
    source = inspect.getsource(GTDB)
    assert "_t1.v1.pkl" not in source
    assert "IC.load_cache(" in source
    assert "expected_cache_hash=" in source
