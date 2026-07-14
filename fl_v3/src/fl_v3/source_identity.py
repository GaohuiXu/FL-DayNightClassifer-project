"""Exact, fail-closed tracked-source state for editable dependencies.

The accepted Arrhenius spconv checkout has one deliberate, non-executable build
metadata change.  A blanket ``git status == clean`` assertion cannot represent
that environment.  This module instead creates a canonical identity for regular
tracked files modified only in the worktree.  It intentionally rejects staged,
added, deleted, renamed, copied, conflicted, symlink, directory, and non-UTF-8
states; future support for any of those requires an explicit contract change.

Untracked files remain outside this identity, matching the earlier
``git status --untracked-files=no`` contract.  Installed executable artifacts and
import origins are attested separately by :mod:`fl_v3.utils.runtime`.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
import stat
import subprocess
from typing import Any, Mapping, Sequence


SOURCE_STATE_FORMAT = "git-tracked-regular-files.v1"
_STATE_KEYS = frozenset({"format", "changes", "sha256"})
_CHANGE_KEYS = frozenset({"status", "path", "sha256"})
_HEX = frozenset("0123456789abcdef")


def _canonical_bytes(value: object) -> bytes:
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


def _normalize_change(value: Any) -> dict[str, str]:
    if not isinstance(value, Mapping) or frozenset(value) != _CHANGE_KEYS:
        raise ValueError(
            "source-state change must contain exactly status/path/sha256"
        )
    status_value = value["status"]
    if status_value != " M":
        raise ValueError(
            "source-state permits only an unstaged modification (' M') of an "
            "existing regular tracked file"
        )
    path_value = value["path"]
    if not isinstance(path_value, str) or not path_value or "\x00" in path_value:
        raise ValueError("source-state path must be a non-empty UTF-8 string")
    pure_path = PurePosixPath(path_value)
    if (
        pure_path.is_absolute()
        or str(pure_path) != path_value
        or any(part in {"", ".", ".."} for part in pure_path.parts)
    ):
        raise ValueError("source-state path must be a normalized relative POSIX path")
    sha_value = value["sha256"]
    if (
        not isinstance(sha_value, str)
        or len(sha_value) != 64
        or any(character not in _HEX for character in sha_value)
    ):
        raise ValueError("source-state file sha256 must be exactly 64 lowercase hex")
    return {"status": status_value, "path": path_value, "sha256": sha_value}


def build_source_state(changes: Sequence[Mapping[str, str]]) -> dict[str, object]:
    """Build one canonical state object from explicit tracked-file records."""
    normalized = sorted(
        (_normalize_change(change) for change in changes),
        key=lambda change: change["path"],
    )
    paths = [change["path"] for change in normalized]
    if len(paths) != len(set(paths)):
        raise ValueError("source-state paths must be unique")
    payload = {"format": SOURCE_STATE_FORMAT, "changes": normalized}
    return {
        **payload,
        "sha256": hashlib.sha256(_canonical_bytes(payload)).hexdigest(),
    }


def validate_source_state(value: Any) -> dict[str, object]:
    """Validate and normalize an expected source-state object."""
    if not isinstance(value, Mapping) or frozenset(value) != _STATE_KEYS:
        raise ValueError("source state must contain exactly format/changes/sha256")
    if value["format"] != SOURCE_STATE_FORMAT:
        raise ValueError(f"source-state format must be exactly {SOURCE_STATE_FORMAT!r}")
    changes = value["changes"]
    if not isinstance(changes, (list, tuple)):
        raise ValueError("source-state changes must be an ordered array")
    normalized = build_source_state(changes)
    if value["sha256"] != normalized["sha256"]:
        raise ValueError(
            "source-state sha256 does not match its canonical format/changes payload"
        )
    if list(changes) != normalized["changes"]:
        raise ValueError("source-state changes must be unique and sorted by path")
    return normalized


def inspect_tracked_source_state(source: str | Path) -> dict[str, object]:
    """Inspect the exact supported tracked state of one Git checkout."""
    root = Path(source).resolve()
    try:
        completed = subprocess.run(
            [
                "git", "-C", str(root), "status", "--porcelain=v1", "-z",
                "--untracked-files=no",
            ],
            check=True,
            capture_output=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(f"cannot inspect tracked source state at {str(root)!r}") from exc

    records = completed.stdout.split(b"\0")
    if records and records[-1] == b"":
        records.pop()
    changes = []
    for record in records:
        if len(record) < 4 or record[2:3] != b" ":
            raise RuntimeError("unsupported or malformed Git porcelain source state")
        try:
            status_value = record[:2].decode("ascii")
            path_value = record[3:].decode("utf-8")
        except UnicodeDecodeError as exc:
            raise RuntimeError("source-state status/path is not canonical UTF-8") from exc
        if status_value != " M":
            raise RuntimeError(
                f"unsupported tracked source status {status_value!r} for {path_value!r}"
            )
        path = root.joinpath(*PurePosixPath(path_value).parts)
        try:
            mode = path.lstat().st_mode
        except OSError as exc:
            raise RuntimeError(f"cannot attest modified source file {path_value!r}") from exc
        if not stat.S_ISREG(mode):
            raise RuntimeError(
                f"modified source path {path_value!r} is not an existing regular file"
            )
        changes.append({
            "status": status_value,
            "path": path_value,
            "sha256": _sha256_file(path),
        })
    return build_source_state(changes)


def require_source_state(
    expected: Any,
    actual: Any,
    *,
    distribution: str,
) -> dict[str, object]:
    """Return normalized actual state or raise on any expected-state drift."""
    try:
        normalized_expected = validate_source_state(expected)
        normalized_actual = validate_source_state(actual)
    except ValueError as exc:
        raise RuntimeError(f"{distribution} source-state contract is invalid: {exc}") from exc
    if normalized_actual != normalized_expected:
        raise RuntimeError(
            f"{distribution} source checkout state drift: "
            f"expected_sha256={normalized_expected['sha256']!r}, "
            f"actual_sha256={normalized_actual['sha256']!r}, "
            f"expected_changes={normalized_expected['changes']!r}, "
            f"actual_changes={normalized_actual['changes']!r}"
        )
    return normalized_actual
