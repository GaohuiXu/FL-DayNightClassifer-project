"""Read-only nuScenes blob access for directory and stored-ZIP layouts.

The Arrhenius dataset module exposes metadata as ordinary JSON files and the
trainval sensor blobs in ten *stored* ZIP archives.  Opening those archives with
``zipfile.ZipFile`` in every DataLoader worker would repeatedly materialize large
central-directory dictionaries.  S01 therefore separates the one-time index
step from the hot read path:

* :func:`build_zip_manifest` scans the ten central directories once and writes a
  compact SQLite manifest **outside** the immutable dataset root.
* :class:`NuScenesBlobStore` opens that manifest read-only and uses recorded
  local-header offsets with ``os.pread``.  The tiny local header is parsed only
  for members actually read; archive descriptors are lazy and process-owned.

No extraction API exists in this module.  Every blob read is length- and CRC-
checked, and member paths are canonical DATAROOT-relative POSIX paths. ZIPs with
compression, encryption, duplicate names within one archive, or conflicting
cross-archive copies are rejected. Identical cross-archive copies (same path,
length, and CRC) are retained as auditable occurrences and routed to the first
archive deterministically. This supports sharded datasets with shared boundary
members while preserving a fail-closed content-equivalence check.
"""
from __future__ import annotations

import hashlib
import json
import multiprocessing.util
import os
import sqlite3
import struct
import threading
import unicodedata
import urllib.parse
import zipfile
import zlib
from collections import OrderedDict
from typing import Iterable, Sequence


MANIFEST_FORMAT_VERSION = "s01.nuscenes-zip.v2"
TRAINVAL_ARCHIVE_NAMES = tuple(f"trainval{i:02d}_blobs.zip" for i in range(1, 11))
_LOCAL_FILE_HEADER = struct.Struct("<4s5H3I2H")
_LOCAL_FILE_SIGNATURE = b"PK\x03\x04"
_LOCATION_CACHE_SIZE = 8192
_DIRECTORY_SENSOR_CHANNELS = (
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
    "CAM_FRONT_LEFT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
    "LIDAR_TOP",
)


class ZipManifestError(RuntimeError):
    """The ZIP set or its manifest violates the fail-closed S01 contract."""


class MissingBlobError(FileNotFoundError):
    """A canonical sensor member is absent from the selected backend."""


def canonical_member_path(path: str | os.PathLike[str]) -> str:
    """Return a strict DATAROOT-relative POSIX member path.

    Path cleanup is deliberately rejected rather than silently applied: two
    differently-spelled ZIP names must never collapse onto one cache key.
    """
    value = os.fspath(path)
    if not isinstance(value, str) or not value:
        raise ValueError("nuScenes member path must be a non-empty string")
    if "\x00" in value:
        raise ValueError(f"nuScenes member path contains NUL: {value!r}")
    if "\\" in value:
        raise ValueError(f"nuScenes member path must use forward slashes: {value!r}")
    if value.startswith("/"):
        raise ValueError(f"nuScenes member path must be relative: {value!r}")
    if unicodedata.normalize("NFC", value) != value:
        raise ValueError(f"nuScenes member path must use NFC Unicode normalization: {value!r}")
    parts = value.split("/")
    if any(part in ("", ".", "..") for part in parts):
        raise ValueError(f"nuScenes member path is not canonical: {value!r}")
    return value


def _archive_name(name: str | os.PathLike[str]) -> str:
    value = os.fspath(name)
    if not value or value != os.path.basename(value) or value in (".", ".."):
        raise ValueError(f"archive name must be a root-level basename: {value!r}")
    if "\x00" in value or "/" in value or "\\" in value:
        raise ValueError(f"archive name is not canonical: {value!r}")
    return value


def _assert_manifest_outside_dataroot(manifest_path: str, dataroot: str) -> None:
    root = os.path.realpath(os.path.abspath(dataroot))
    target = os.path.realpath(os.path.abspath(manifest_path))
    try:
        common = os.path.commonpath([root, target])
    except ValueError:
        return
    if common == root:
        raise PermissionError(
            f"Refusing to write ZIP manifest under read-only nuScenes DATAROOT: "
            f"{manifest_path!r} resolves under {root!r}"
        )


def _hash_string(digest, value: str) -> None:
    raw = value.encode("utf-8")
    digest.update(struct.pack("<Q", len(raw)))
    digest.update(raw)


def _create_manifest_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        PRAGMA journal_mode=OFF;
        PRAGMA synchronous=OFF;
        PRAGMA temp_store=MEMORY;
        CREATE TABLE metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        ) WITHOUT ROWID;
        CREATE TABLE archives (
            archive_id INTEGER PRIMARY KEY,
            name TEXT NOT NULL UNIQUE,
            size_bytes INTEGER NOT NULL,
            member_count INTEGER NOT NULL
        );
        CREATE TABLE members (
            path TEXT NOT NULL,
            archive_id INTEGER NOT NULL,
            header_offset INTEGER NOT NULL,
            file_size INTEGER NOT NULL,
            crc32 INTEGER NOT NULL,
            compression INTEGER NOT NULL,
            flags INTEGER NOT NULL,
            FOREIGN KEY (archive_id) REFERENCES archives(archive_id),
            PRIMARY KEY (path, archive_id)
        ) WITHOUT ROWID;
        CREATE INDEX members_archive_idx ON members(archive_id);
        """
    )


def build_zip_manifest(
    dataroot: str,
    manifest_path: str,
    archive_names: Sequence[str] = TRAINVAL_ARCHIVE_NAMES,
    *,
    force: bool = False,
) -> dict:
    """Build an atomic member-to-archive SQLite manifest.

    This function performs an exhaustive archive scan.  On the shared full
    dataset it is material compute and must only be called under an approved S01
    ``RUN_REQUEST.md``.  Small synthetic/local ZIPs are suitable for unit tests.
    """
    root = os.path.abspath(os.fspath(dataroot))
    target = os.path.abspath(os.fspath(manifest_path))
    names = tuple(_archive_name(name) for name in archive_names)
    if not names or len(set(names)) != len(names):
        raise ValueError("archive_names must be a non-empty duplicate-free sequence")
    _assert_manifest_outside_dataroot(target, root)
    if os.path.exists(target) and not force:
        raise FileExistsError(f"ZIP manifest already exists: {target!r}; pass force=True to replace")
    os.makedirs(os.path.dirname(target) or ".", exist_ok=True)
    temp = f"{target}.tmp.{os.getpid()}"
    if os.path.lexists(temp):
        os.unlink(temp)

    digest = hashlib.sha256()
    _hash_string(digest, MANIFEST_FORMAT_VERSION)
    total_members = 0
    archive_reports: list[dict] = []
    conn: sqlite3.Connection | None = None
    try:
        conn = sqlite3.connect(temp)
        _create_manifest_schema(conn)
        conn.execute("BEGIN")
        for archive_id, name in enumerate(names, start=1):
            archive_path = os.path.join(root, name)
            try:
                archive_size = int(os.stat(archive_path).st_size)
            except OSError as exc:
                raise FileNotFoundError(f"required nuScenes ZIP archive unavailable: {archive_path!r}") from exc
            _hash_string(digest, name)
            digest.update(struct.pack("<Q", archive_size))
            member_rows = []
            member_count = 0
            try:
                with zipfile.ZipFile(archive_path, "r") as archive:
                    for info in archive.infolist():
                        if info.is_dir():
                            continue
                        member = canonical_member_path(info.filename)
                        if info.flag_bits & 0x1:
                            raise ZipManifestError(
                                f"encrypted ZIP member is unsupported: {member!r} in {name!r}"
                            )
                        if info.compress_type != zipfile.ZIP_STORED:
                            raise ZipManifestError(
                                f"non-stored ZIP member is unsupported: {member!r} in {name!r} "
                                f"(compression={info.compress_type})"
                            )
                        if int(info.compress_size) != int(info.file_size):
                            raise ZipManifestError(
                                f"stored member has unequal compressed/file sizes: {member!r} in {name!r}"
                            )
                        row = (
                            member,
                            archive_id,
                            int(info.header_offset),
                            int(info.file_size),
                            int(info.CRC) & 0xFFFFFFFF,
                            int(info.compress_type),
                            int(info.flag_bits),
                        )
                        member_rows.append(row)
                        _hash_string(digest, member)
                        digest.update(
                            struct.pack(
                                "<QQIHH",
                                int(info.header_offset),
                                int(info.file_size),
                                int(info.CRC) & 0xFFFFFFFF,
                                int(info.compress_type),
                                int(info.flag_bits) & 0xFFFF,
                            )
                        )
                        member_count += 1
                        if len(member_rows) >= 4096:
                            try:
                                conn.executemany(
                                    "INSERT INTO members "
                                    "(path,archive_id,header_offset,file_size,crc32,compression,flags) "
                                    "VALUES (?,?,?,?,?,?,?)",
                                    member_rows,
                                )
                            except sqlite3.IntegrityError as exc:
                                raise ZipManifestError(
                                    f"duplicate ZIP member within archive {name!r}"
                                ) from exc
                            member_rows.clear()
                    if member_rows:
                        try:
                            conn.executemany(
                                "INSERT INTO members "
                                "(path,archive_id,header_offset,file_size,crc32,compression,flags) "
                                "VALUES (?,?,?,?,?,?,?)",
                                member_rows,
                            )
                        except sqlite3.IntegrityError as exc:
                            raise ZipManifestError(
                                f"duplicate ZIP member within archive {name!r}"
                            ) from exc
            except zipfile.BadZipFile as exc:
                raise ZipManifestError(f"invalid ZIP archive: {archive_path!r}") from exc
            conn.execute(
                "INSERT INTO archives (archive_id,name,size_bytes,member_count) VALUES (?,?,?,?)",
                (archive_id, name, archive_size, member_count),
            )
            total_members += member_count
            archive_reports.append(
                {"archive_id": archive_id, "name": name, "size_bytes": archive_size, "member_count": member_count}
            )

        # The licensed trainval shards were observed to repeat some paths across
        # archives. Keep every occurrence so archive coverage remains auditable,
        # but fail closed if two copies do not have the same central-directory
        # size and CRC. Runtime routing then picks the lowest archive_id.
        conflicting = list(
            conn.execute(
                "SELECT path,COUNT(*),MIN(file_size),MAX(file_size),MIN(crc32),MAX(crc32) "
                "FROM members GROUP BY path "
                "HAVING MIN(file_size) != MAX(file_size) OR MIN(crc32) != MAX(crc32) "
                "ORDER BY path LIMIT 10"
            )
        )
        if conflicting:
            preview = "; ".join(
                f"{path!r} occurrences={count} size={min_size}..{max_size} "
                f"crc={min_crc:08x}..{max_crc:08x}"
                for path, count, min_size, max_size, min_crc, max_crc in conflicting
            )
            raise ZipManifestError(
                "conflicting cross-archive ZIP copies (same path but different size/CRC): "
                + preview
            )
        unique_members = int(
            conn.execute("SELECT COUNT(DISTINCT path) FROM members").fetchone()[0]
        )
        duplicate_occurrences = total_members - unique_members
        manifest_hash = digest.hexdigest()
        metadata = {
            "format_version": MANIFEST_FORMAT_VERSION,
            "manifest_hash": manifest_hash,
            "archive_names": json.dumps(names, separators=(",", ":")),
            "archive_count": str(len(names)),
            "member_count": str(total_members),
            "unique_member_count": str(unique_members),
            "duplicate_occurrence_count": str(duplicate_occurrences),
        }
        conn.executemany("INSERT INTO metadata (key,value) VALUES (?,?)", sorted(metadata.items()))
        conn.commit()
        conn.close()
        conn = None
        os.chmod(temp, 0o444)
        os.replace(temp, target)
        return {
            "format_version": MANIFEST_FORMAT_VERSION,
            "manifest_path": target,
            "manifest_hash": manifest_hash,
            "archive_count": len(names),
            "member_count": total_members,
            "unique_member_count": unique_members,
            "duplicate_occurrence_count": duplicate_occurrences,
            "archives": archive_reports,
        }
    except BaseException:
        if conn is not None:
            conn.close()
        if os.path.lexists(temp):
            try:
                os.chmod(temp, 0o600)
                os.unlink(temp)
            except OSError:
                pass
        raise


def _manifest_uri(path: str) -> str:
    absolute = os.path.abspath(path)
    return "file:" + urllib.parse.quote(absolute, safe="/") + "?mode=ro&immutable=1"


def manifest_summary(manifest_path: str) -> dict:
    """Read small, immutable manifest metadata without loading member names."""
    conn = sqlite3.connect(_manifest_uri(manifest_path), uri=True)
    try:
        metadata = dict(conn.execute("SELECT key,value FROM metadata"))
        archives = [
            {
                "archive_id": int(row[0]),
                "name": str(row[1]),
                "size_bytes": int(row[2]),
                "member_count": int(row[3]),
            }
            for row in conn.execute(
                "SELECT archive_id,name,size_bytes,member_count FROM archives ORDER BY archive_id"
            )
        ]
    except sqlite3.DatabaseError as exc:
        raise ZipManifestError(f"invalid ZIP manifest schema: {manifest_path!r}") from exc
    finally:
        conn.close()
    try:
        if metadata.get("format_version") != MANIFEST_FORMAT_VERSION:
            raise ZipManifestError(
                f"unsupported ZIP manifest format {metadata.get('format_version')!r}; "
                f"expected {MANIFEST_FORMAT_VERSION!r}"
            )
        archive_names = tuple(json.loads(metadata["archive_names"]))
        if int(metadata["archive_count"]) != len(archives):
            raise ZipManifestError("manifest archive_count does not match archive rows")
        if tuple(archive["name"] for archive in archives) != archive_names:
            raise ZipManifestError("manifest archive_names do not match archive rows")
        if int(metadata["member_count"]) != sum(
            archive["member_count"] for archive in archives
        ):
            raise ZipManifestError("manifest member_count does not match archive rows")
        unique_members = int(metadata["unique_member_count"])
        duplicate_occurrences = int(metadata["duplicate_occurrence_count"])
        if unique_members < 0 or duplicate_occurrences < 0:
            raise ZipManifestError("manifest unique/duplicate member counts must be nonnegative")
        if unique_members + duplicate_occurrences != int(metadata["member_count"]):
            raise ZipManifestError("manifest unique + duplicate counts do not match member_count")
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ZipManifestError(f"invalid ZIP manifest metadata: {manifest_path!r}") from exc
    return {
        "format_version": metadata["format_version"],
        "manifest_hash": metadata["manifest_hash"],
        "archive_count": int(metadata["archive_count"]),
        "member_count": int(metadata["member_count"]),
        "unique_member_count": unique_members,
        "duplicate_occurrence_count": duplicate_occurrences,
        "archive_names": archive_names,
        "archives": archives,
    }


class NuScenesBlobStore:
    """Unified read-only byte store for extracted directories or stored ZIPs.

    ``manifest_path=None`` selects directory mode when ``samples/`` exists.  If
    the root has the ten trainval archives instead, the manifest is resolved from
    ``NUSCENES_ZIP_MANIFEST`` / ``ARRHENIUS_NUSCENES_ZIP_MANIFEST``.  No archive
    scan is ever performed implicitly.
    """

    def __init__(self, dataroot: str, manifest_path: str | None = None):
        self.dataroot = os.path.abspath(os.fspath(dataroot))
        explicit_manifest = os.fspath(manifest_path).strip() if manifest_path else ""
        configured_manifest = (
            explicit_manifest
            if explicit_manifest
            else (
                os.environ.get("NUSCENES_ZIP_MANIFEST", "").strip()
                or os.environ.get("ARRHENIUS_NUSCENES_ZIP_MANIFEST", "").strip()
            )
        )
        has_directory_blobs = all(
            os.path.isdir(os.path.join(self.dataroot, "samples", channel))
            for channel in _DIRECTORY_SENSOR_CHANNELS
        )
        if has_directory_blobs and not explicit_manifest:
            # A module manifest exported in the shell must not redirect a local
            # extracted mini root.  An explicitly passed manifest still wins for
            # focused parity tests/tools.
            self.mode = "directory"
            self.manifest_path = None
        elif configured_manifest:
            self.mode = "zip"
            self.manifest_path = os.path.abspath(configured_manifest)
        elif has_directory_blobs:
            self.mode = "directory"
            self.manifest_path = None
        elif any(os.path.lexists(os.path.join(self.dataroot, name)) for name in TRAINVAL_ARCHIVE_NAMES):
            raise FileNotFoundError(
                "ZIP-backed nuScenes root detected but no external member manifest is configured. "
                "Set NUSCENES_ZIP_MANIFEST (or ARRHENIUS_NUSCENES_ZIP_MANIFEST) to the "
                "S01 SQLite manifest built outside the read-only dataset root."
            )
        else:
            # Preserve the useful native FileNotFoundError at first directory read.
            self.mode = "directory"
            self.manifest_path = None
        self._lock = threading.RLock()
        self._pid: int | None = None
        self._conn: sqlite3.Connection | None = None
        self._archive_names: dict[int, str] = {}
        self._archive_fds: dict[int, int] = {}
        self._locations: OrderedDict[str, tuple[int, int, int, int, int, int]] = OrderedDict()
        self._read_count = 0
        self._byte_count = 0
        self._reopen_count = 0
        multiprocessing.util.register_after_fork(self, type(self)._after_fork)

    def __getstate__(self) -> dict:
        return {
            "dataroot": self.dataroot,
            "mode": self.mode,
            "manifest_path": self.manifest_path,
        }

    def __setstate__(self, state: dict) -> None:
        self.dataroot = state["dataroot"]
        self.mode = state["mode"]
        self.manifest_path = state["manifest_path"]
        self._lock = threading.RLock()
        self._pid = None
        self._conn = None
        self._archive_names = {}
        self._archive_fds = {}
        self._locations = OrderedDict()
        self._read_count = 0
        self._byte_count = 0
        self._reopen_count = 0
        multiprocessing.util.register_after_fork(self, type(self)._after_fork)

    @staticmethod
    def _after_fork(store: "NuScenesBlobStore") -> None:
        # The inherited sqlite object and descriptors belong to the parent.  Do
        # not call sqlite/close on them from the child; simply forget them.  The
        # child lazily opens independent read-only state on its first read.
        for fd in store._archive_fds.values():
            try:
                os.close(fd)
            except OSError:
                pass
        store._lock = threading.RLock()
        store._pid = None
        store._conn = None
        store._archive_names = {}
        store._archive_fds = {}
        store._locations = OrderedDict()
        store._read_count = 0
        store._byte_count = 0
        store._reopen_count += 1

    def _close_handles(self) -> None:
        if self._conn is not None:
            try:
                self._conn.close()
            except sqlite3.Error:
                pass
        self._conn = None
        for fd in self._archive_fds.values():
            try:
                os.close(fd)
            except OSError:
                pass
        self._archive_fds.clear()
        self._archive_names.clear()

    def _ensure_process(self) -> None:
        pid = os.getpid()
        if self._pid == pid:
            return
        if self._pid is not None:
            # Fallback PID guard for fork paths where the registered hook did
            # not run.  Forget the inherited sqlite object without invoking
            # sqlite after fork, but close the child's copies of raw FDs.
            for fd in self._archive_fds.values():
                try:
                    os.close(fd)
                except OSError:
                    pass
            self._conn = None
            self._reopen_count += 1
        self._pid = pid
        self._conn = None
        self._archive_names = {}
        self._archive_fds = {}

    def _ensure_manifest(self) -> sqlite3.Connection:
        self._ensure_process()
        if self.mode != "zip" or not self.manifest_path:
            raise RuntimeError("manifest access requested for directory blob store")
        if self._conn is not None:
            return self._conn
        conn: sqlite3.Connection | None = None
        try:
            conn = sqlite3.connect(
                _manifest_uri(self.manifest_path), uri=True, check_same_thread=False
            )
            metadata = dict(conn.execute("SELECT key,value FROM metadata"))
            if metadata.get("format_version") != MANIFEST_FORMAT_VERSION:
                raise ZipManifestError(
                    f"unsupported ZIP manifest format {metadata.get('format_version')!r}; "
                    f"expected {MANIFEST_FORMAT_VERSION!r}"
                )
            archive_rows = list(
                conn.execute(
                    "SELECT archive_id,name,size_bytes,member_count FROM archives ORDER BY archive_id"
                )
            )
            expected_count = int(metadata["archive_count"])
            if len(archive_rows) != expected_count:
                raise ZipManifestError(
                    f"manifest archive count mismatch: metadata={expected_count}, rows={len(archive_rows)}"
                )
            archive_names: dict[int, str] = {}
            expected_names = tuple(json.loads(metadata["archive_names"]))
            if tuple(str(row[1]) for row in archive_rows) != expected_names:
                raise ZipManifestError("manifest archive_names do not match archive rows")
            if int(metadata["member_count"]) != sum(int(row[3]) for row in archive_rows):
                raise ZipManifestError("manifest member_count does not match archive rows")
            unique_members = int(metadata["unique_member_count"])
            duplicate_occurrences = int(metadata["duplicate_occurrence_count"])
            if unique_members < 0 or duplicate_occurrences < 0:
                raise ZipManifestError("manifest unique/duplicate member counts must be nonnegative")
            if unique_members + duplicate_occurrences != int(metadata["member_count"]):
                raise ZipManifestError(
                    "manifest unique + duplicate counts do not match member_count"
                )
            for archive_id, name, size_bytes, _member_count in archive_rows:
                safe_name = _archive_name(name)
                archive_path = os.path.join(self.dataroot, safe_name)
                try:
                    actual_size = int(os.stat(archive_path).st_size)
                except OSError as exc:
                    raise FileNotFoundError(
                        f"manifest archive unavailable at current dataroot: {archive_path!r}"
                    ) from exc
                if actual_size != int(size_bytes):
                    raise ZipManifestError(
                        f"archive size changed since manifest build: {safe_name!r} "
                        f"manifest={int(size_bytes)}, actual={actual_size}"
                    )
                archive_names[int(archive_id)] = safe_name
        except BaseException as exc:
            if conn is not None:
                try:
                    conn.close()
                except sqlite3.Error:
                    pass
            if isinstance(exc, (ZipManifestError, FileNotFoundError)):
                raise
            if isinstance(exc, (sqlite3.Error, KeyError, TypeError, ValueError)):
                raise ZipManifestError(f"invalid ZIP manifest: {self.manifest_path!r}") from exc
            raise
        assert conn is not None
        self._conn = conn
        self._archive_names = archive_names
        return conn

    def _lookup_many(
        self, paths: Sequence[str]
    ) -> dict[str, tuple[int, int, int, int, int, int]]:
        result: dict[str, tuple[int, int, int, int, int, int]] = {}
        missing_lookup: list[str] = []
        for path in paths:
            location = self._locations.get(path)
            if location is None:
                missing_lookup.append(path)
            else:
                self._locations.move_to_end(path)
                result[path] = location
        if missing_lookup:
            conn = self._ensure_manifest()
            placeholders = ",".join("?" for _ in missing_lookup)
            rows = conn.execute(
                f"SELECT path,archive_id,header_offset,file_size,crc32,compression,flags "
                f"FROM members WHERE path IN ({placeholders}) ORDER BY path,archive_id",
                tuple(missing_lookup),
            )
            for path, archive_id, header_offset, file_size, crc32, compression, flags in rows:
                if str(path) in result:
                    # Identical cross-archive copy; the first/lowest archive_id
                    # is the deterministic runtime route.
                    continue
                location = (
                    int(archive_id),
                    int(header_offset),
                    int(file_size),
                    int(crc32),
                    int(compression),
                    int(flags),
                )
                result[str(path)] = location
                self._locations[str(path)] = location
                self._locations.move_to_end(str(path))
            while len(self._locations) > _LOCATION_CACHE_SIZE:
                self._locations.popitem(last=False)
        unresolved = [path for path in paths if path not in result]
        if unresolved:
            preview = ", ".join(repr(path) for path in unresolved[:3])
            raise MissingBlobError(
                f"{len(unresolved)} nuScenes member(s) missing from ZIP manifest "
                f"{self.manifest_path!r}: {preview}"
            )
        return result

    def _archive_fd(self, archive_id: int) -> int:
        fd = self._archive_fds.get(archive_id)
        if fd is not None:
            return fd
        name = self._archive_names.get(archive_id)
        if name is None:
            self._ensure_manifest()
            name = self._archive_names.get(archive_id)
        if name is None:
            raise ZipManifestError(f"member references unknown archive_id={archive_id}")
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
        fd = os.open(os.path.join(self.dataroot, name), flags)
        self._archive_fds[archive_id] = fd
        return fd

    def _drop_archive_fd(self, archive_id: int) -> None:
        fd = self._archive_fds.pop(archive_id, None)
        if fd is not None:
            try:
                os.close(fd)
            except OSError:
                pass

    @staticmethod
    def _pread_exact(fd: int, offset: int, size: int) -> bytes:
        chunks = []
        consumed = 0
        while consumed < size:
            chunk = os.pread(fd, size - consumed, offset + consumed)
            if not chunk:
                raise ZipManifestError(
                    f"short archive read at offset={offset}, expected={size}, got={consumed}"
                )
            chunks.append(chunk)
            consumed += len(chunk)
        return b"".join(chunks)

    @classmethod
    def _local_data_offset(
        cls,
        fd: int,
        path: str,
        header_offset: int,
        expected_compression: int,
        expected_flags: int,
    ) -> int:
        header = cls._pread_exact(fd, header_offset, _LOCAL_FILE_HEADER.size)
        (
            signature,
            _extract_version,
            local_flags,
            local_compression,
            _mtime,
            _mdate,
            _crc,
            _compressed_size,
            _file_size,
            filename_len,
            extra_len,
        ) = _LOCAL_FILE_HEADER.unpack(header)
        if signature != _LOCAL_FILE_SIGNATURE:
            raise ZipManifestError(
                f"bad local ZIP header signature for {path!r} at offset {header_offset}"
            )
        if local_compression != expected_compression:
            raise ZipManifestError(
                f"local/central compression mismatch for {path!r}: "
                f"local={local_compression}, manifest={expected_compression}"
            )
        if local_flags != (expected_flags & 0xFFFF):
            raise ZipManifestError(
                f"local/central flags mismatch for {path!r}: "
                f"local=0x{local_flags:04x}, manifest=0x{expected_flags & 0xFFFF:04x}"
            )
        filename_raw = cls._pread_exact(
            fd, header_offset + _LOCAL_FILE_HEADER.size, filename_len
        )
        encoding = "utf-8" if local_flags & 0x800 else "cp437"
        try:
            local_path = filename_raw.decode(encoding)
        except UnicodeDecodeError as exc:
            raise ZipManifestError(
                f"invalid {encoding} local ZIP filename for manifest member {path!r}"
            ) from exc
        if local_path != path:
            raise ZipManifestError(
                f"local/central ZIP filename mismatch: local={local_path!r}, manifest={path!r}"
            )
        return int(header_offset + _LOCAL_FILE_HEADER.size + filename_len + extra_len)

    def _read_zip_member(
        self,
        path: str,
        archive_id: int,
        header_offset: int,
        size: int,
        expected_crc: int,
        expected_compression: int,
        expected_flags: int,
    ) -> bytes:
        """Read/verify once, then reopen the archive FD and retry once.

        C3SE's official multi-worker ZIP example recommends lazy worker-local
        handles and a reopen on a bad handle.  ``pread`` avoids shared file
        offsets; this bounded retry covers a stale/transient descriptor without
        masking a persistent archive or manifest error.
        """
        last_error: BaseException | None = None
        for attempt in range(2):
            try:
                fd = self._archive_fd(archive_id)
                data_offset = self._local_data_offset(
                    fd,
                    path,
                    header_offset,
                    expected_compression,
                    expected_flags,
                )
                payload = self._pread_exact(fd, data_offset, size)
                actual_crc = zlib.crc32(payload) & 0xFFFFFFFF
                if actual_crc != expected_crc:
                    raise ZipManifestError(
                        f"CRC mismatch for {path!r} in {self._archive_names[archive_id]!r}: "
                        f"manifest={expected_crc:08x}, actual={actual_crc:08x}"
                    )
                return payload
            except (OSError, ZipManifestError) as exc:
                last_error = exc
                self._drop_archive_fd(archive_id)
                if attempt == 0:
                    self._reopen_count += 1
                    continue
                raise
        assert last_error is not None
        raise last_error

    def read_many(self, rel_paths: Sequence[str]) -> list[bytes]:
        """Read paths in caller order with one manifest lookup transaction."""
        paths = [canonical_member_path(path) for path in rel_paths]
        if not paths:
            return []
        with self._lock:
            self._ensure_process()
            if self.mode == "directory":
                payloads = []
                for path in paths:
                    absolute = os.path.join(self.dataroot, *path.split("/"))
                    try:
                        with open(absolute, "rb") as stream:
                            payloads.append(stream.read())
                    except FileNotFoundError as exc:
                        raise MissingBlobError(
                            f"nuScenes directory member missing: {path!r} under {self.dataroot!r}"
                        ) from exc
            else:
                locations = self._lookup_many(tuple(dict.fromkeys(paths)))
                by_path: dict[str, bytes] = {}
                for path in dict.fromkeys(paths):
                    (
                        archive_id,
                        header_offset,
                        size,
                        expected_crc,
                        expected_compression,
                        expected_flags,
                    ) = locations[path]
                    by_path[path] = self._read_zip_member(
                        path,
                        archive_id,
                        header_offset,
                        size,
                        expected_crc,
                        expected_compression,
                        expected_flags,
                    )
                payloads = [by_path[path] for path in paths]
            self._read_count += len(payloads)
            self._byte_count += sum(len(payload) for payload in payloads)
            return payloads

    def read_bytes(self, rel_path: str) -> bytes:
        return self.read_many([rel_path])[0]

    def read_archive_bytes(self, archive_name: str, rel_path: str) -> bytes:
        """Read one exact ``(archive, path)`` occurrence instead of routed path.

        Normal training reads route identical duplicates to the lowest archive ID.
        Integrity audits use this method so a duplicated sentinel still opens and
        verifies the specifically declared shard.
        """
        archive_name = _archive_name(archive_name)
        path = canonical_member_path(rel_path)
        with self._lock:
            conn = self._ensure_manifest()
            archive_id = next(
                (
                    archive_id
                    for archive_id, name in self._archive_names.items()
                    if name == archive_name
                ),
                None,
            )
            if archive_id is None:
                raise MissingBlobError(
                    f"archive {archive_name!r} is absent from ZIP manifest {self.manifest_path!r}"
                )
            row = conn.execute(
                "SELECT header_offset,file_size,crc32,compression,flags "
                "FROM members WHERE path=? AND archive_id=?",
                (path, archive_id),
            ).fetchone()
            if row is None:
                raise MissingBlobError(
                    f"member {path!r} is absent from archive {archive_name!r}"
                )
            payload = self._read_zip_member(
                path,
                archive_id,
                *(int(value) for value in row),
            )
            self._read_count += 1
            self._byte_count += len(payload)
            return payload

    def contains(self, rel_path: str) -> bool:
        path = canonical_member_path(rel_path)
        with self._lock:
            self._ensure_process()
            if self.mode == "directory":
                return os.path.isfile(os.path.join(self.dataroot, *path.split("/")))
            try:
                self._lookup_many([path])
                return True
            except MissingBlobError:
                return False

    def debug_state(self) -> dict:
        """Small lifecycle report used by deterministic worker tests/audits."""
        with self._lock:
            return {
                "mode": self.mode,
                "dataroot": self.dataroot,
                "manifest_path": self.manifest_path,
                "owner_pid": self._pid,
                "current_pid": os.getpid(),
                "manifest_open": self._conn is not None,
                "open_archives": tuple(
                    self._archive_names[archive_id]
                    for archive_id in sorted(self._archive_fds)
                ),
                "read_count": self._read_count,
                "byte_count": self._byte_count,
                "reopen_count": self._reopen_count,
                "location_cache_size": len(self._locations),
            }

    def close(self) -> None:
        with self._lock:
            self._close_handles()
            self._pid = None
            self._locations.clear()

    def __enter__(self) -> "NuScenesBlobStore":
        return self

    def __exit__(self, _exc_type, _exc, _tb) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


def manifest_member_counts(manifest_path: str, rel_paths: Iterable[str]) -> dict:
    """Resolve a path set without opening any archive payloads.

    The result is suitable for coverage reports.  It counts references and
    unique members separately and reports every missing canonical member.
    """
    paths = [canonical_member_path(path) for path in rel_paths]
    unique = tuple(dict.fromkeys(paths))
    conn = sqlite3.connect(_manifest_uri(manifest_path), uri=True)
    try:
        found: dict[str, str] = {}
        for start in range(0, len(unique), 900):
            chunk = unique[start : start + 900]
            placeholders = ",".join("?" for _ in chunk)
            if not chunk:
                continue
            rows = conn.execute(
                f"SELECT members.path,archives.name FROM members JOIN archives USING(archive_id) "
                f"WHERE members.path IN ({placeholders}) "
                f"ORDER BY members.path,members.archive_id",
                chunk,
            )
            for path, archive in rows:
                found.setdefault(str(path), str(archive))
    except sqlite3.DatabaseError as exc:
        raise ZipManifestError(f"invalid ZIP manifest: {manifest_path!r}") from exc
    finally:
        conn.close()
    missing = [path for path in unique if path not in found]
    per_archive: dict[str, int] = {}
    for path in paths:
        archive = found.get(path)
        if archive is not None:
            per_archive[archive] = per_archive.get(archive, 0) + 1
    return {
        "references": len(paths),
        "unique_members": len(unique),
        "resolved_references": sum(path in found for path in paths),
        "resolved_unique_members": len(found),
        "missing_unique_members": missing,
        "per_archive_references": dict(sorted(per_archive.items())),
    }


def manifest_archive_sentinels(manifest_path: str) -> dict[str, str]:
    """Return one deterministic occurrence path per non-empty archive.

    Call :meth:`NuScenesBlobStore.read_archive_bytes` with each ``(archive,path)``
    pair; normal routed reads are not sufficient when the selected path is shared.
    """
    conn = sqlite3.connect(_manifest_uri(manifest_path), uri=True)
    try:
        rows = conn.execute(
            "SELECT archives.name,MIN(members.path) "
            "FROM archives JOIN members USING(archive_id) "
            "GROUP BY archives.archive_id,archives.name ORDER BY archives.archive_id"
        )
        return {str(archive): str(path) for archive, path in rows}
    except sqlite3.DatabaseError as exc:
        raise ZipManifestError(f"invalid ZIP manifest: {manifest_path!r}") from exc
    finally:
        conn.close()
