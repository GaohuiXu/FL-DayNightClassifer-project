"""T4 — D10 provenance binding for the readiness verdict (T4_SPEC §0.2; Codex T4 finding).

The readiness verdict must be bound to a full-participation log-group trainval clean checkpoint;
a sampled/IID/defended/wrong-split checkpoint must be REFUSED even if its metric floors pass.
"""
from __future__ import annotations

import json

import pytest

from fl_v3.eval.provenance import (
    D10_REQUIRED,
    build_provenance,
    check_d10,
    verify_d10_provenance,
    build_s06_provenance,
    verify_s06_provenance,
)
from fl_v3.config import resolve_config
from test_s06_resolved_config import valid_config

_D10_CFG = {
    "task-type": "nuscenes_detection",
    "nuscenes-version": "v1.0-trainval",
    "nuscenes-train-split": "train",
    "nuscenes-val-split": "val",
    "nuscenes-partition-mode": "log_group",
    "fraction-train": 1.0,
    "defense-type": "none",
    "nuscenes-num-clients": 25,
    "num-server-rounds": 15,
    "seed": 20259,
    "det-camera-backbone": "swin_t",
}


def test_build_provenance_records_d10_keys():
    prov = build_provenance(_D10_CFG, "abc123")
    assert prov["FL_TRAINABLE_CHECKSUM"] == "abc123"
    assert prov["fraction-train"] == 1.0 and isinstance(prov["fraction-train"], float)
    assert prov["regime"].startswith("D10-")
    assert check_d10(prov) == []  # the canonical D10 config is compliant


def test_check_d10_flags_each_violation():
    base = build_provenance(_D10_CFG, "ck")
    # full participation is the headline: fraction<1 is INVALID
    p = dict(base); p["fraction-train"] = 0.2
    assert any("fraction-train" in m for m in check_d10(p))
    # IID instead of log-group
    p = dict(base); p["nuscenes-partition-mode"] = "iid"
    assert any("nuscenes-partition-mode" in m for m in check_d10(p))
    # a defended checkpoint
    p = dict(base); p["defense-type"] = "flame"
    assert any("defense-type" in m for m in check_d10(p))
    # mini instead of trainval
    p = dict(base); p["nuscenes-version"] = "v1.0-mini"
    assert any("nuscenes-version" in m for m in check_d10(p))
    # checksum binding
    assert any("FL_TRAINABLE_CHECKSUM" in m for m in check_d10(base, checksum="DIFFERENT"))
    assert check_d10(base, checksum="ck") == []


def _write(tmp_path, prov):
    (tmp_path / "provenance.json").write_text(json.dumps(prov))
    return str(tmp_path / "final_model.pt")  # the checkpoint file need not exist; provenance is beside it


def test_verify_passes_on_d10_checkpoint(tmp_path):
    ckpt = _write(tmp_path, build_provenance(_D10_CFG, "ck"))
    prov = verify_d10_provenance(ckpt, "ck")
    assert prov["_verified"] is True


def test_verify_refuses_missing_provenance(tmp_path):
    with pytest.raises(RuntimeError, match="provenance MISSING"):
        verify_d10_provenance(str(tmp_path / "final_model.pt"), "ck")


def test_verify_refuses_sampled_checkpoint(tmp_path):
    p = build_provenance({**_D10_CFG, "fraction-train": 0.2}, "ck")
    ckpt = _write(tmp_path, p)
    with pytest.raises(RuntimeError, match="INVALID"):
        verify_d10_provenance(ckpt, "ck")


def test_verify_refuses_iid_checkpoint(tmp_path):
    p = build_provenance({**_D10_CFG, "nuscenes-partition-mode": "iid"}, "ck")
    with pytest.raises(RuntimeError, match="INVALID"):
        verify_d10_provenance(_write(tmp_path, p), "ck")


def test_verify_refuses_checksum_mismatch(tmp_path):
    ckpt = _write(tmp_path, build_provenance(_D10_CFG, "ck"))
    with pytest.raises(RuntimeError, match="INVALID"):
        verify_d10_provenance(ckpt, "DIFFERENT_CHECKSUM")


def test_s06_provenance_binds_mode_config_checkpoint_and_data(tmp_path):
    config = resolve_config(valid_config(tmp_path))
    source = "1" * 40; checkpoint = "2" * 64
    prov = build_s06_provenance(config, checkpoint_sha256=checkpoint, source_sha=source)
    assert verify_s06_provenance(
        prov, config, checkpoint_sha256=checkpoint, source_sha=source,
    )["_verified"]
    drift = dict(prov); drift["model_mode"] = "fusion"
    with pytest.raises(RuntimeError, match="identity drift"):
        verify_s06_provenance(
            drift, config, checkpoint_sha256=checkpoint, source_sha=source,
        )
