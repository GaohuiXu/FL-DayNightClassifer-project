from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

from fl_v3.data.nuscenes.s10_binding import (
    S10BindingError,
    build_stop_b_panel,
    canonical_json_bytes,
    load_frozen_stop_b_panel,
    load_frozen_split_role,
    token_vector_sha256,
    validate_stop_b_panel,
)


def _tokens(prefix: str, count: int):
    return [f"{prefix}-{index:04d}" for index in range(count)]


def _manifest(tmp_path):
    fit_logs = _tokens("log-fit", 10)
    select_logs = _tokens("log-select", 2)
    audit_logs = _tokens("log-audit", 2)
    by_log = {log: _tokens(f"sample-{log}", 12) for log in fit_logs}
    fit_samples = sorted(token for values in by_log.values() for token in values)
    low_logs = fit_logs
    roles = {
        "D_fit": {
            "log_tokens": fit_logs,
            "scene_tokens": _tokens("scene-fit", 10),
            "sample_tokens": fit_samples,
        },
        "D_select": {
            "log_tokens": select_logs,
            "scene_tokens": _tokens("scene-select", 2),
            "sample_tokens": _tokens("sample-select", 4),
        },
        "D_audit": {
            "log_tokens": audit_logs,
            "scene_tokens": _tokens("scene-audit", 2),
            "sample_tokens": _tokens("sample-audit", 4),
        },
        "D_low": {
            "log_tokens": low_logs,
            "scene_tokens": _tokens("scene-fit", 10),
            "sample_tokens": fit_samples,
        },
        "D_mid": {
            "log_tokens": low_logs,
            "scene_tokens": _tokens("scene-fit", 10),
            "sample_tokens": fit_samples,
        },
    }
    value = {
        "schema": "fl_v3.s10.split_manifest.v1",
        "parent_version": "v1.0-trainval",
        "parent_split": "train",
        "source_identities": {"cache": "a" * 64, "n_sweeps": 10},
        "roles": roles,
        "audit_policy": {},
    }
    path = tmp_path / "split_manifest.json"
    path.write_bytes(canonical_json_bytes(value) + b"\n")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return path, digest, by_log


def _infos(by_log):
    infos = []
    for log_index, (log, tokens) in enumerate(sorted(by_log.items())):
        for index, token in enumerate(tokens):
            # Every sample supports two tasks; the rotating pair gives all six
            # tasks abundant deterministic support for P_term.
            labels = np.asarray(((index + log_index) % 6, ((index + log_index) % 6) + 4), dtype=np.int64)
            labels %= 10
            boxes = np.zeros((2, 7), dtype=np.float32)
            boxes[:, 0] = 1.0 + index
            boxes[:, 1] = -1.0 - log_index
            infos.append({
                "sample_token": token,
                "log_token": log,
                "gt_boxes": boxes,
                "gt_labels": labels,
            })
    return infos


def test_split_binding_and_panel_are_one_shot_and_hash_bound(tmp_path):
    path, digest, by_log = _manifest(tmp_path)
    binding = load_frozen_split_role(
        path,
        expected_manifest_sha256=digest,
        role="D_low",
        expected_source_identities={"cache": "a" * 64, "n_sweeps": 10},
    )
    assert len(binding.log_tokens) == 10
    assert len(binding.sample_tokens) == 120
    assert binding.sample_tokens_sha256 == token_vector_sha256(binding.sample_tokens)

    panel = build_stop_b_panel(binding, _infos(by_log))
    report = validate_stop_b_panel(panel, binding)
    assert report == {
        "status": "PASS",
        "panel_sha256": panel["panel_sha256"],
        "P_core_samples": 48,
        "P_term_samples": 16,
        "P_broad_samples": 64,
        "P_core_batches_b4": 12,
        "P_term_batches_b4": 4,
    }
    assert set(panel["tokens"]["P_core"]).isdisjoint(panel["tokens"]["P_term"])
    assert all(value >= 4 for value in panel["task_positive_frames_in_P_term"])
    assert sum(panel["core_log_quotas"].values()) == 48


def test_split_binding_rejects_physical_role_and_source_drift(tmp_path):
    path, digest, _ = _manifest(tmp_path)
    with pytest.raises(S10BindingError, match="physical SHA-256 drift"):
        load_frozen_split_role(path, expected_manifest_sha256="0" * 64, role="D_low")
    with pytest.raises(S10BindingError, match="source identity"):
        load_frozen_split_role(
            path,
            expected_manifest_sha256=digest,
            role="D_low",
            expected_source_identities={"cache": "b" * 64},
        )

    value = json.loads(path.read_text())
    value["roles"]["D_low"]["sample_tokens"].append(
        value["roles"]["D_low"]["sample_tokens"][0]
    )
    path.write_bytes(canonical_json_bytes(value) + b"\n")
    changed = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(S10BindingError, match="duplicate"):
        load_frozen_split_role(path, expected_manifest_sha256=changed, role="D_low")


def test_panel_reload_rejects_mutation_and_out_of_role_token(tmp_path):
    path, digest, by_log = _manifest(tmp_path)
    binding = load_frozen_split_role(path, expected_manifest_sha256=digest, role="D_low")
    panel = build_stop_b_panel(binding, _infos(by_log))
    panel["tokens"]["P_term"][0] = "outside-D-low"
    with pytest.raises(S10BindingError, match="panel content SHA drift"):
        validate_stop_b_panel(panel, binding)


def test_frozen_panel_loader_binds_physical_and_content_hashes(tmp_path):
    split_path, split_digest, by_log = _manifest(tmp_path)
    binding = load_frozen_split_role(
        split_path, expected_manifest_sha256=split_digest, role="D_low"
    )
    panel = build_stop_b_panel(binding, _infos(by_log))
    panel_path = tmp_path / "panel_manifest.json"
    panel_path.write_bytes(canonical_json_bytes(panel) + b"\n")
    file_digest = hashlib.sha256(panel_path.read_bytes()).hexdigest()
    loaded, report = load_frozen_stop_b_panel(
        panel_path,
        expected_file_sha256=file_digest,
        expected_content_sha256=panel["panel_sha256"],
        binding=binding,
    )
    assert loaded == panel
    assert report["reconstructed"] is False
    assert report["panel_file_sha256"] == file_digest

    with pytest.raises(S10BindingError, match="physical SHA-256 drift"):
        load_frozen_stop_b_panel(
            panel_path,
            expected_file_sha256="0" * 64,
            expected_content_sha256=panel["panel_sha256"],
            binding=binding,
        )
    with pytest.raises(S10BindingError, match="expected content SHA-256 drift"):
        load_frozen_stop_b_panel(
            panel_path,
            expected_file_sha256=file_digest,
            expected_content_sha256="0" * 64,
            binding=binding,
        )
