"""S01 stored-ZIP manifest, integrity, and process-lifecycle tests."""
from __future__ import annotations

import multiprocessing
import os
import pickle
import sqlite3
import struct
import zipfile
import inspect

import pytest

from fl_v3.data.nuscenes.zip_backend import (
    MissingBlobError,
    NuScenesBlobStore,
    TRAINVAL_ARCHIVE_NAMES,
    ZipManifestError,
    build_zip_manifest,
    canonical_member_path,
    manifest_archive_sentinels,
    manifest_member_counts,
    manifest_summary,
)
from fl_v3.data.nuscenes import zip_backend as ZIP_BACKEND_MODULE


def _write_zip(path, members, compression=zipfile.ZIP_STORED):
    with zipfile.ZipFile(path, "w", compression=compression) as archive:
        for name, payload in members:
            archive.writestr(name, payload)


def _build(tmp_path, members_by_archive):
    root = tmp_path / "data"
    root.mkdir()
    names = []
    for name, members in members_by_archive:
        names.append(name)
        _write_zip(root / name, members)
    manifest = tmp_path / "manifest.sqlite"
    report = build_zip_manifest(str(root), str(manifest), names)
    return root, manifest, report


def _child_read(store, path, queue):
    try:
        payload = store.read_bytes(path)
        queue.put((payload, store.debug_state(), ""))
    except BaseException as exc:  # pragma: no cover - only used to surface child failures
        queue.put((b"", {}, f"{type(exc).__name__}: {exc}"))


def test_manifest_exact_ten_archives_and_pread_roundtrip(tmp_path):
    archive_members = []
    expected = {}
    for index, name in enumerate(TRAINVAL_ARCHIVE_NAMES):
        member = f"samples/CAM_FRONT/{index:02d}.jpg"
        payload = f"payload-{index}".encode()
        expected[member] = payload
        archive_members.append((name, [(member, payload)]))
    root, manifest, report = _build(tmp_path, archive_members)

    assert report["archive_count"] == 10
    assert report["member_count"] == 10
    assert report["unique_member_count"] == 10
    assert report["duplicate_occurrence_count"] == 0
    summary = manifest_summary(str(manifest))
    assert summary["archive_names"] == TRAINVAL_ARCHIVE_NAMES
    assert summary["manifest_hash"] == report["manifest_hash"]
    assert os.stat(manifest).st_mode & 0o222 == 0

    store = NuScenesBlobStore(str(root), manifest_path=str(manifest))
    requested = [next(iter(expected)), *reversed(expected), next(iter(expected))]
    assert store.read_many(requested) == [expected[path] for path in requested]
    state = store.debug_state()
    assert state["owner_pid"] == os.getpid()
    assert state["manifest_open"] is True
    assert set(state["open_archives"]) == set(TRAINVAL_ARCHIVE_NAMES)
    store.close()
    assert store.debug_state()["open_archives"] == ()
    assert store.read_bytes(requested[0]) == expected[requested[0]]  # close -> lazy reopen


def test_manifest_coverage_counts_references_and_unique(tmp_path):
    root, manifest, _ = _build(
        tmp_path,
        [("a.zip", [("samples/LIDAR_TOP/a.bin", b"a"), ("sweeps/LIDAR_TOP/b.bin", b"b")])],
    )
    del root
    report = manifest_member_counts(
        str(manifest),
        ["samples/LIDAR_TOP/a.bin", "samples/LIDAR_TOP/a.bin", "missing.bin"],
    )
    assert report["references"] == 3
    assert report["unique_members"] == 2
    assert report["resolved_references"] == 2
    assert report["resolved_unique_members"] == 1
    assert report["missing_unique_members"] == ["missing.bin"]


def test_manifest_routes_identical_duplicate_member_across_archives(tmp_path):
    root = tmp_path / "data"
    root.mkdir()
    _write_zip(root / "a.zip", [("samples/CAM_FRONT/a.jpg", b"one")])
    _write_zip(root / "b.zip", [("samples/CAM_FRONT/a.jpg", b"one")])
    manifest = tmp_path / "manifest.sqlite"
    report = build_zip_manifest(str(root), str(manifest), ["a.zip", "b.zip"])
    assert report["member_count"] == 2
    assert report["unique_member_count"] == 1
    assert report["duplicate_occurrence_count"] == 1
    assert manifest_archive_sentinels(str(manifest)) == {
        "a.zip": "samples/CAM_FRONT/a.jpg",
        "b.zip": "samples/CAM_FRONT/a.jpg",
    }
    store = NuScenesBlobStore(str(root), manifest_path=str(manifest))
    assert store.read_bytes("samples/CAM_FRONT/a.jpg") == b"one"
    assert store.debug_state()["open_archives"] == ("a.zip",)
    assert store.read_archive_bytes("b.zip", "samples/CAM_FRONT/a.jpg") == b"one"
    assert store.debug_state()["open_archives"] == ("a.zip", "b.zip")


def test_manifest_rejects_conflicting_member_across_archives(tmp_path):
    root = tmp_path / "data"
    root.mkdir()
    _write_zip(root / "a.zip", [("samples/CAM_FRONT/a.jpg", b"one")])
    _write_zip(root / "b.zip", [("samples/CAM_FRONT/a.jpg", b"two")])
    with pytest.raises(ZipManifestError, match="conflicting cross-archive"):
        build_zip_manifest(str(root), str(tmp_path / "manifest.sqlite"), ["a.zip", "b.zip"])


def test_manifest_rejects_duplicate_member_within_one_archive(tmp_path):
    root = tmp_path / "data"
    root.mkdir()
    with pytest.warns(UserWarning, match="Duplicate name"):
        _write_zip(
            root / "a.zip",
            [("samples/CAM_FRONT/a.jpg", b"one"), ("samples/CAM_FRONT/a.jpg", b"two")],
        )
    with pytest.raises(ZipManifestError, match="duplicate ZIP member within archive"):
        build_zip_manifest(str(root), str(tmp_path / "manifest.sqlite"), ["a.zip"])


def test_manifest_requires_every_declared_archive(tmp_path):
    root = tmp_path / "data"
    root.mkdir()
    _write_zip(root / "a.zip", [("samples/CAM_FRONT/a.jpg", b"one")])
    with pytest.raises(FileNotFoundError, match="b.zip"):
        build_zip_manifest(str(root), str(tmp_path / "manifest.sqlite"), ["a.zip", "b.zip"])


@pytest.mark.parametrize(
    "member",
    ["../escape", "/absolute", "a//b", "a/./b", "a\\b", "a/../b", "samples/e\u0301.jpg"],
)
def test_manifest_rejects_noncanonical_member(tmp_path, member):
    root = tmp_path / "data"
    root.mkdir()
    _write_zip(root / "a.zip", [(member, b"bad")])
    with pytest.raises(ValueError, match="member path"):
        build_zip_manifest(str(root), str(tmp_path / "manifest.sqlite"), ["a.zip"])


def test_manifest_rejects_compressed_member(tmp_path):
    root = tmp_path / "data"
    root.mkdir()
    _write_zip(
        root / "a.zip",
        [("samples/CAM_FRONT/a.jpg", b"compress me" * 100)],
        compression=zipfile.ZIP_DEFLATED,
    )
    with pytest.raises(ZipManifestError, match="non-stored"):
        build_zip_manifest(str(root), str(tmp_path / "manifest.sqlite"), ["a.zip"])


def test_manifest_output_under_dataroot_is_forbidden(tmp_path):
    root = tmp_path / "data"
    root.mkdir()
    _write_zip(root / "a.zip", [("samples/CAM_FRONT/a.jpg", b"x")])
    with pytest.raises(PermissionError, match="read-only nuScenes DATAROOT"):
        build_zip_manifest(str(root), str(root / "manifest.sqlite"), ["a.zip"])


def test_missing_member_reports_manifest(tmp_path):
    root, manifest, _ = _build(
        tmp_path, [("a.zip", [("samples/CAM_FRONT/a.jpg", b"x")])]
    )
    store = NuScenesBlobStore(str(root), manifest_path=str(manifest))
    with pytest.raises(MissingBlobError, match="missing from ZIP manifest"):
        store.read_bytes("samples/CAM_FRONT/missing.jpg")
    assert store.contains("samples/CAM_FRONT/missing.jpg") is False


def test_crc_corruption_is_detected_on_read(tmp_path):
    root, manifest, _ = _build(
        tmp_path, [("a.zip", [("samples/CAM_FRONT/a.jpg", b"abcdef")])]
    )
    conn = sqlite3.connect(str(manifest))
    header_offset = conn.execute(
        "SELECT header_offset FROM members WHERE path='samples/CAM_FRONT/a.jpg'"
    ).fetchone()[0]
    conn.close()
    archive_path = root / "a.zip"
    with open(archive_path, "r+b") as stream:
        stream.seek(header_offset)
        header = stream.read(30)
        filename_len, extra_len = struct.unpack("<HH", header[-4:])
        data_offset = header_offset + 30 + filename_len + extra_len
        stream.seek(data_offset + 2)
        original = stream.read(1)
        stream.seek(data_offset + 2)
        stream.write(bytes([original[0] ^ 0xFF]))
    store = NuScenesBlobStore(str(root), manifest_path=str(manifest))
    with pytest.raises(ZipManifestError, match="CRC mismatch"):
        store.read_bytes("samples/CAM_FRONT/a.jpg")


def test_same_length_local_header_filename_mutation_is_rejected(tmp_path):
    member = "samples/CAM_FRONT/a.jpg"
    root, manifest, _ = _build(tmp_path, [("a.zip", [(member, b"abcdef")])])
    conn = sqlite3.connect(str(manifest))
    header_offset = conn.execute(
        "SELECT header_offset FROM members WHERE path=?", (member,)
    ).fetchone()[0]
    conn.close()
    archive_path = root / "a.zip"
    with open(archive_path, "r+b") as stream:
        stream.seek(header_offset + 30)
        local_name = stream.read(len(member.encode("utf-8")))
        assert local_name == member.encode("utf-8")
        stream.seek(header_offset + 30)
        stream.write(local_name.replace(b"a.jpg", b"b.jpg"))
    store = NuScenesBlobStore(str(root), manifest_path=str(manifest))
    with pytest.raises(ZipManifestError, match="local/central ZIP filename mismatch"):
        store.read_bytes(member)


def test_duplicate_sentinel_reads_each_exact_archive_occurrence(tmp_path):
    member = "LICENSE"
    root, manifest, _ = _build(
        tmp_path,
        [("a.zip", [(member, b"same")]), ("b.zip", [(member, b"same")])],
    )
    sentinels = manifest_archive_sentinels(str(manifest))
    assert sentinels == {"a.zip": member, "b.zip": member}
    store = NuScenesBlobStore(str(root), manifest_path=str(manifest))
    assert {
        archive: store.read_archive_bytes(archive, path)
        for archive, path in sentinels.items()
    } == {"a.zip": b"same", "b.zip": b"same"}
    assert store.debug_state()["open_archives"] == ("a.zip", "b.zip")


def test_archive_size_change_invalidates_manifest(tmp_path):
    root, manifest, _ = _build(
        tmp_path, [("a.zip", [("samples/CAM_FRONT/a.jpg", b"abcdef")])]
    )
    with open(root / "a.zip", "ab") as stream:
        stream.write(b"changed")
    store = NuScenesBlobStore(str(root), manifest_path=str(manifest))
    with pytest.raises(ZipManifestError, match="archive size changed"):
        store.read_bytes("samples/CAM_FRONT/a.jpg")


def test_transient_pread_error_reopens_once(tmp_path, monkeypatch):
    root, manifest, _ = _build(
        tmp_path, [("a.zip", [("samples/CAM_FRONT/a.jpg", b"abcdef")])]
    )
    original_pread = os.pread
    calls = {"count": 0}

    def flaky_pread(fd, size, offset):
        calls["count"] += 1
        if calls["count"] == 1:
            raise OSError("synthetic stale descriptor")
        return original_pread(fd, size, offset)

    monkeypatch.setattr(os, "pread", flaky_pread)
    store = NuScenesBlobStore(str(root), manifest_path=str(manifest))
    assert store.read_bytes("samples/CAM_FRONT/a.jpg") == b"abcdef"
    assert store.debug_state()["reopen_count"] == 1


def test_store_pickle_drops_open_handles_and_reopens(tmp_path):
    root, manifest, _ = _build(
        tmp_path, [("a.zip", [("samples/CAM_FRONT/a.jpg", b"abcdef")])]
    )
    store = NuScenesBlobStore(str(root), manifest_path=str(manifest))
    assert store.read_bytes("samples/CAM_FRONT/a.jpg") == b"abcdef"
    restored = pickle.loads(pickle.dumps(store))
    assert restored.debug_state()["manifest_open"] is False
    assert restored.read_bytes("samples/CAM_FRONT/a.jpg") == b"abcdef"
    assert restored.debug_state()["owner_pid"] == os.getpid()


@pytest.mark.parametrize("start_method", ["fork", "spawn"])
def test_parent_open_then_child_gets_process_owned_handles(tmp_path, start_method):
    if start_method not in multiprocessing.get_all_start_methods():
        pytest.skip(f"{start_method} unavailable")
    root, manifest, _ = _build(
        tmp_path, [("a.zip", [("samples/CAM_FRONT/a.jpg", b"abcdef")])]
    )
    store = NuScenesBlobStore(str(root), manifest_path=str(manifest))
    assert store.read_bytes("samples/CAM_FRONT/a.jpg") == b"abcdef"  # parent opens first
    parent_pid = store.debug_state()["owner_pid"]
    context = multiprocessing.get_context(start_method)
    queue = context.Queue()
    process = context.Process(
        target=_child_read,
        args=(store, "samples/CAM_FRONT/a.jpg", queue),
    )
    process.start()
    payload, state, error = queue.get(timeout=20)
    process.join(timeout=20)
    assert process.exitcode == 0
    assert error == ""
    assert payload == b"abcdef"
    assert state["owner_pid"] == state["current_pid"] != parent_pid
    assert state["open_archives"] == ("a.zip",)


def test_directory_store_is_read_only_and_uses_same_api(tmp_path):
    root = tmp_path / "data"
    member = root / "samples" / "CAM_FRONT" / "a.jpg"
    member.parent.mkdir(parents=True)
    member.write_bytes(b"directory")
    store = NuScenesBlobStore(str(root))
    assert store.mode == "directory"
    assert store.read_bytes("samples/CAM_FRONT/a.jpg") == b"directory"
    assert store.contains("samples/CAM_FRONT/a.jpg") is True
    with pytest.raises(MissingBlobError, match="directory member missing"):
        store.read_bytes("samples/CAM_FRONT/missing.jpg")


def test_canonical_member_path_rejects_absolute_and_traversal():
    assert canonical_member_path("samples/CAM_FRONT/a.jpg") == "samples/CAM_FRONT/a.jpg"
    for bad in ("", "/x", "../x", "a/../x", "a\\x", "a//x", "a/./x", "a\x00x"):
        with pytest.raises(ValueError):
            canonical_member_path(bad)


def test_backend_exposes_no_archive_extraction_path():
    source = inspect.getsource(ZIP_BACKEND_MODULE)
    assert ".extract(" not in source
    assert ".extractall(" not in source
