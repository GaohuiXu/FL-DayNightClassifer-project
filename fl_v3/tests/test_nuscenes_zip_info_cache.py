"""S01 module discovery, metadata alias, and extraction-free cache tests."""
from __future__ import annotations

import json
import os
import sys
import types

import pytest

from fl_v3.data.nuscenes import info_cache as IC
from fl_v3.data.nuscenes import paths as P
from fl_v3.data.nuscenes.zip_backend import TRAINVAL_ARCHIVE_NAMES


def test_dataroot_precedence_includes_canonical_module_variable(monkeypatch, tmp_path):
    module_root = tmp_path / "module"
    arrhenius_root = tmp_path / "arrhenius"
    explicit_env_root = tmp_path / "explicit-env"
    config_root = tmp_path / "config"
    monkeypatch.setattr(P, "DATAROOT", "")
    monkeypatch.delenv("NUSCENES_DATAROOT", raising=False)
    monkeypatch.delenv("ARRHENIUS_NUSCENES_DATAROOT", raising=False)
    monkeypatch.setenv("NUSCENES_DATA_DIR", str(module_root))
    assert P.get_dataroot() == str(module_root)
    monkeypatch.setenv("ARRHENIUS_NUSCENES_DATAROOT", str(arrhenius_root))
    assert P.get_dataroot() == str(arrhenius_root)
    monkeypatch.setenv("NUSCENES_DATAROOT", str(explicit_env_root))
    assert P.get_dataroot() == str(explicit_env_root)
    assert P.get_dataroot({"nuscenes-dataroot": str(config_root)}) == str(config_root)


def test_zip_manifest_resolution(monkeypatch, tmp_path):
    monkeypatch.delenv("NUSCENES_ZIP_MANIFEST", raising=False)
    monkeypatch.delenv("ARRHENIUS_NUSCENES_ZIP_MANIFEST", raising=False)
    assert P.get_zip_manifest() == ""
    with pytest.raises(FileNotFoundError, match="ZIP manifest is not configured"):
        P.get_zip_manifest(required=True)
    manifest = tmp_path / "manifest.sqlite"
    monkeypatch.setenv("NUSCENES_ZIP_MANIFEST", str(manifest))
    assert P.get_zip_manifest() == str(manifest)
    configured = tmp_path / "configured.sqlite"
    assert P.get_zip_manifest({"nuscenes-zip-manifest": str(configured)}) == str(configured)


def _make_module_layout(root):
    table_dir = root / "trainval"
    table_dir.mkdir(parents=True)
    for table in P._REQUIRED_TABLES:
        payload = []
        if table == "sample.json":
            payload = [{"token": P.sentinel_token("v1.0-trainval")}]
        (table_dir / table).write_text(json.dumps(payload), encoding="utf-8")
    for archive in TRAINVAL_ARCHIVE_NAMES:
        (root / archive).write_bytes(b"present-for-lightweight-preflight")


def test_verify_dataset_accepts_arrhenius_metadata_alias_and_exact_archive_set(tmp_path):
    root = tmp_path / "module"
    root.mkdir()
    _make_module_layout(root)
    report = P.verify_dataset("v1.0-trainval", str(root))
    assert report["blob_backend"] == "zip"
    assert report["table_dir"] == str(root / "trainval")
    assert report["archives"] == list(TRAINVAL_ARCHIVE_NAMES)
    assert report["n_samples"] == 1


def test_verify_dataset_rejects_one_missing_trainval_archive(tmp_path):
    root = tmp_path / "module"
    root.mkdir()
    _make_module_layout(root)
    (root / TRAINVAL_ARCHIVE_NAMES[-1]).unlink()
    with pytest.raises(FileNotFoundError, match=TRAINVAL_ARCHIVE_NAMES[-1]):
        P.verify_dataset("v1.0-trainval", str(root))


def test_create_nuscenes_overrides_only_table_root(monkeypatch, tmp_path):
    root = tmp_path / "module"
    root.mkdir()
    _make_module_layout(root)

    class FakeNuScenes:
        def __init__(self, version, dataroot, verbose, **kwargs):
            self.version = version
            self.dataroot = dataroot
            self.verbose = verbose
            self.kwargs = kwargs
            self.table_root_seen_during_init = self.table_root

        @property
        def table_root(self):
            return os.path.join(self.dataroot, self.version)

    monkeypatch.setitem(sys.modules, "nuscenes", types.SimpleNamespace(NuScenes=FakeNuScenes))
    nusc = P.create_nuscenes("v1.0-trainval", str(root), verbose=False, map_resolution=0.2)
    assert nusc.version == "v1.0-trainval"
    assert nusc.dataroot == str(root)
    assert nusc.table_root == str(root / "trainval")
    assert nusc.table_root_seen_during_init == str(root / "trainval")
    assert nusc.kwargs == {"map_resolution": 0.2}


@pytest.mark.parametrize("bad", ["/absolute", "../escape", "a/../escape", "a\\b"])
def test_abspath_from_relative_rejects_noncanonical_path(tmp_path, bad):
    with pytest.raises(ValueError):
        P.abspath_from_relative(bad, str(tmp_path))


def test_info_cache_member_path_equals_legacy_devkit_path(nusc_mini, dataroot):
    sample = nusc_mini.sample[0]
    for channel in (*P.CAMERA_CHANNELS, P.LIDAR_CHANNEL):
        sample_data = nusc_mini.get("sample_data", sample["data"][channel])
        legacy = P.relative_to_dataroot(
            nusc_mini.get_sample_data_path(sample_data["token"]), dataroot
        )
        assert IC._sample_data_rel_path(sample_data) == legacy


def test_info_cache_build_never_calls_blob_path_probe(nusc_mini, dataroot, monkeypatch):
    token = IC.split_sample_tokens(nusc_mini, "mini_val")[0]

    def forbidden(*_args, **_kwargs):
        raise AssertionError("info-cache build must not probe extracted blob paths")

    monkeypatch.setattr(nusc_mini, "get_sample_data_path", forbidden)
    info = IC.build_info_list(nusc_mini, [token], dataroot, n_sweeps=3)
    assert len(info) == 1
    assert "lidar_sweeps" in info[0]  # present even when this token starts a scene
    assert len(info[0]["cam_rel_paths"]) == 6
    assert info[0]["lidar_rel_path"].startswith("samples/LIDAR_TOP/")
    assert all(sweep["rel_path"].startswith("sweeps/LIDAR_TOP/") for sweep in info[0]["lidar_sweeps"])
    single = IC.build_info_list(nusc_mini, [token], dataroot, n_sweeps=1)
    assert "lidar_sweeps" not in single[0]
