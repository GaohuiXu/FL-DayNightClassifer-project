"""S07-A fail-closed cache/manifest provenance for GT-database preparation."""
from __future__ import annotations

import importlib.util
import inspect
import json
import os
import pickle
import sys
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


def _cache_artifact_hashes(
    cache_dir: str, version: str, split: str, n_sweeps: int
) -> tuple[str, str]:
    pickle_path, sidecar_path = IC.cache_paths(
        cache_dir, version, split, n_sweeps=n_sweeps
    )
    return GTDB._sha256_file(pickle_path), GTDB._sha256_file(sidecar_path)


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

    pickle_sha, sidecar_sha = _cache_artifact_hashes(
        cache_dir, "v1.0-mini", "mini_val", 2
    )
    loaded, loaded_meta, provenance = GTDB._load_info_list(
        cache_dir,
        "v1.0-mini",
        "mini_val",
        2,
        meta["cache_hash"],
        pickle_sha,
        sidecar_sha,
    )
    assert loaded_meta == meta
    assert len(loaded) == len(infos)
    assert provenance == {
        "cache_format_version": "t1.v2",
        "cache_version": "v1.0-mini",
        "cache_split": "mini_val",
        "cache_n_sweeps": 2,
        "cache_hash": meta["cache_hash"],
        "cache_pickle_path": os.path.abspath(
            IC.cache_paths(cache_dir, "v1.0-mini", "mini_val", 2)[0]
        ),
        "cache_pickle_bytes": os.path.getsize(
            IC.cache_paths(cache_dir, "v1.0-mini", "mini_val", 2)[0]
        ),
        "cache_pickle_sha256": pickle_sha,
        "cache_sidecar_path": os.path.abspath(
            IC.cache_paths(cache_dir, "v1.0-mini", "mini_val", 2)[1]
        ),
        "cache_sidecar_bytes": os.path.getsize(
            IC.cache_paths(cache_dir, "v1.0-mini", "mini_val", 2)[1]
        ),
        "cache_sidecar_sha256": sidecar_sha,
    }

    with pytest.raises(ValueError, match="frozen expected hash"):
        GTDB._load_info_list(
            cache_dir,
            "v1.0-mini",
            "mini_val",
            2,
            "0" * 64,
            pickle_sha,
            sidecar_sha,
        )
    with pytest.raises(FileNotFoundError, match="cache pickle artifact.*nsweeps10"):
        GTDB._load_info_list(
            cache_dir,
            "v1.0-mini",
            "mini_val",
            10,
            meta["cache_hash"],
            pickle_sha,
            sidecar_sha,
        )
    with pytest.raises(ValueError, match="expected cache pickle SHA-256"):
        GTDB._load_info_list(
            cache_dir,
            "v1.0-mini",
            "mini_val",
            2,
            meta["cache_hash"],
            "not-a-digest",
            sidecar_sha,
        )

    _, sidecar = IC.cache_paths(cache_dir, "v1.0-mini", "mini_val", 2)
    with open(sidecar, encoding="utf-8") as stream:
        sidecar_meta = json.load(stream)
    sidecar_meta["cache_hash"] = "f" * 64
    with open(sidecar, "w", encoding="utf-8") as stream:
        json.dump(sidecar_meta, stream)
    mutated_sidecar_sha = GTDB._sha256_file(sidecar)
    with pytest.raises(ValueError, match="sidecar differs"):
        GTDB._load_info_list(
            cache_dir,
            "v1.0-mini",
            "mini_val",
            2,
            meta["cache_hash"],
            pickle_sha,
            mutated_sidecar_sha,
        )
    os.unlink(sidecar)
    with pytest.raises(FileNotFoundError):
        GTDB._load_info_list(
            cache_dir,
            "v1.0-mini",
            "mini_val",
            2,
            meta["cache_hash"],
            pickle_sha,
            mutated_sidecar_sha,
        )


def test_gt_database_rejects_historical_t1v1_filename(tmp_path):
    legacy = tmp_path / "nuscenes_info_v1.0-trainval_train_t1.v1.pkl"
    with open(legacy, "wb") as stream:
        pickle.dump({"info_list": []}, stream)
    with pytest.raises(FileNotFoundError, match="t1.v2_nsweeps10"):
        GTDB._load_info_list(
            str(tmp_path),
            "v1.0-trainval",
            "train",
            10,
            "0" * 64,
            "0" * 64,
            "0" * 64,
        )


@pytest.mark.parametrize("derived_field", ["gt_boxes", "sweep2keylidar"])
def test_gt_database_rejects_derived_cache_mutation_before_blob_or_crop(
    derived_field, nusc_mini, dataroot, tmp_path, monkeypatch
):
    cache_dir = str(tmp_path / "cache")
    tokens = IC.split_sample_tokens(nusc_mini, "mini_val")
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
    pickle_path, sidecar_path = IC.cache_paths(
        cache_dir, "v1.0-mini", "mini_val", 2
    )
    frozen_pickle_sha = GTDB._sha256_file(pickle_path)
    frozen_sidecar_sha = GTDB._sha256_file(sidecar_path)

    with open(pickle_path, "rb") as stream:
        blob = pickle.load(stream)
    if derived_field == "gt_boxes":
        target = next(info for info in blob["info_list"] if len(info["gt_boxes"]))
        target["gt_boxes"][0, 0] += 0.25
    else:
        target = next(info for info in blob["info_list"] if info["lidar_sweeps"])
        target["lidar_sweeps"][0]["sweep2keylidar"][0, 3] += 0.25
    with open(pickle_path, "wb") as stream:
        pickle.dump(blob, stream, protocol=4)

    # The raw-input canonical contract and sidecar remain logically consistent,
    # demonstrating why the separate frozen physical pickle hash is mandatory.
    loaded, loaded_meta = IC.load_cache(
        cache_dir,
        "v1.0-mini",
        "mini_val",
        n_sweeps=2,
        expected_cache_hash=meta["cache_hash"],
    )
    assert loaded_meta == meta
    assert IC.canonical_hash(loaded) == meta["cache_hash"]
    assert GTDB._sha256_file(sidecar_path) == frozen_sidecar_sha
    assert GTDB._sha256_file(pickle_path) != frozen_pickle_sha

    def forbidden(*_args, **_kwargs):
        raise AssertionError("blob opening / point cropping must not be reached")

    monkeypatch.setattr(GTDB, "_open_blob_store", forbidden)
    monkeypatch.setattr(GTDB, "crop_object_points", forbidden)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(_SCRIPT),
            "--cache-dir",
            cache_dir,
            "--version",
            "v1.0-mini",
            "--split",
            "mini_val",
            "--n-sweeps",
            "2",
            "--expected-cache-hash",
            meta["cache_hash"],
            "--expected-cache-file-sha256",
            frozen_pickle_sha,
            "--expected-cache-sidecar-sha256",
            frozen_sidecar_sha,
            "--dataroot",
            dataroot,
            "--out-dir",
            str(tmp_path / "gt_database"),
        ],
    )
    with pytest.raises(ValueError, match="pickle physical SHA mismatch"):
        GTDB.main()


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
    assert "expected_cache_file_sha256" in source
    assert "expected_cache_sidecar_sha256" in source
