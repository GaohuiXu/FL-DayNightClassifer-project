"""Fail-closed editable-source state attestation without runtime dependencies."""
from __future__ import annotations

import hashlib
import subprocess

import pytest

from fl_v3.source_identity import (
    SOURCE_STATE_FORMAT,
    build_source_state,
    inspect_tracked_source_state,
    require_source_state,
    validate_source_state,
)


def _git(root, *args):
    return subprocess.run(
        ["git", "-C", str(root), *args],
        check=True,
        capture_output=True,
        text=True,
    )


def _checkout(tmp_path):
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "config", "user.email", "s08@example.invalid")
    _git(tmp_path, "config", "user.name", "S08 Test")
    (tmp_path / "pyproject.toml").write_text("requires = ['cumm']\n", encoding="utf-8")
    (tmp_path / "kernel.py").write_text("VALUE = 1\n", encoding="utf-8")
    _git(tmp_path, "add", "pyproject.toml", "kernel.py")
    _git(tmp_path, "commit", "-qm", "fixture")
    return tmp_path


def test_clean_and_exact_metadata_patch_have_distinct_canonical_states(tmp_path):
    root = _checkout(tmp_path)
    clean = inspect_tracked_source_state(root)
    assert clean == build_source_state([])
    assert clean["format"] == SOURCE_STATE_FORMAT
    assert clean["changes"] == []

    payload = "requires = []\n"
    (root / "pyproject.toml").write_text(payload, encoding="utf-8")
    patched = inspect_tracked_source_state(root)
    expected = build_source_state([{
        "status": " M",
        "path": "pyproject.toml",
        "sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
    }])
    assert patched == expected
    assert require_source_state(expected, patched, distribution="spconv") == patched
    assert patched["sha256"] != clean["sha256"]


def test_one_byte_or_additional_tracked_change_fails_exact_state(tmp_path):
    root = _checkout(tmp_path)
    (root / "pyproject.toml").write_text("requires = []\n", encoding="utf-8")
    expected = inspect_tracked_source_state(root)

    (root / "pyproject.toml").write_text("requires = [ ]\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="source checkout state drift"):
        require_source_state(
            expected, inspect_tracked_source_state(root), distribution="spconv"
        )

    (root / "kernel.py").write_text("VALUE = 2\n", encoding="utf-8")
    actual = inspect_tracked_source_state(root)
    assert [change["path"] for change in actual["changes"]] == [
        "kernel.py", "pyproject.toml",
    ]
    with pytest.raises(RuntimeError, match="source checkout state drift"):
        require_source_state(expected, actual, distribution="spconv")


def test_staged_or_non_regular_tracked_state_is_rejected(tmp_path):
    root = _checkout(tmp_path)
    (root / "pyproject.toml").write_text("requires = []\n", encoding="utf-8")
    _git(root, "add", "pyproject.toml")
    with pytest.raises(RuntimeError, match="unsupported tracked source status"):
        inspect_tracked_source_state(root)


def test_expected_state_schema_and_digest_are_fail_closed():
    clean = build_source_state([])
    validate_source_state(clean)

    wrong_digest = dict(clean, sha256="0" * 64)
    with pytest.raises(ValueError, match="does not match"):
        validate_source_state(wrong_digest)

    unknown = dict(clean, extra=True)
    with pytest.raises(ValueError, match="exactly"):
        validate_source_state(unknown)
