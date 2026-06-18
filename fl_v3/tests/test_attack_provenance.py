"""T5 attack provenance (§0.C8): a poisoned checkpoint must verify as D10-base + poison_rate>0 + roster."""
from __future__ import annotations

import json
import os

import pytest

from fl_v3.eval.provenance import build_attack_provenance, check_attack, verify_attack_provenance

_D10_BASE = {
    "task-type": "nuscenes_detection", "nuscenes-version": "v1.0-trainval",
    "nuscenes-train-split": "train", "nuscenes-val-split": "val",
    "nuscenes-partition-mode": "log_group", "defense-type": "none", "fraction-train": 1.0,
}


def _cfg(**extra):
    return {**_D10_BASE, "attack-enabled": True, "attack-mode": "relocation",
            "attack-poison-rate": 0.5, "attack-rho": 0.2, "attack-delta-reloc": 6.0, **extra}


def test_attack_provenance_compliant():
    prov = build_attack_provenance(_cfg(), "chk123", roster=[2, 3, 12, 13, 19], m_r=5)
    assert check_attack(prov, "chk123") == []
    assert prov["m_r"] == 5 and prov["roster"] == [2, 3, 12, 13, 19]


def test_attack_provenance_rejects_zero_poison_rate():
    prov = build_attack_provenance(_cfg(**{"attack-poison-rate": 0.0}), "c", roster=[1], m_r=1)
    bad = check_attack(prov)
    assert any("poison-rate" in b for b in bad)


def test_attack_provenance_rejects_non_d10_base():
    prov = build_attack_provenance(_cfg(**{"fraction-train": 0.2}), "c", roster=[1, 2], m_r=2)
    assert any("fraction-train" in b for b in check_attack(prov))


def test_attack_provenance_rejects_empty_roster():
    prov = build_attack_provenance(_cfg(), "c", roster=[], m_r=0)
    bad = check_attack(prov)
    assert any("roster" in b for b in bad)


def test_verify_attack_provenance_roundtrip(tmp_path):
    ckpt = tmp_path / "final_model.pt"
    ckpt.write_text("x")
    prov = build_attack_provenance(_cfg(), "chkABC", roster=[2, 3, 12, 13, 19], m_r=5)
    json.dump(prov, open(tmp_path / "provenance.json", "w"))
    got = verify_attack_provenance(str(ckpt), "chkABC")
    assert got["_verified"] and got["m_r"] == 5
    with pytest.raises(RuntimeError):
        verify_attack_provenance(str(ckpt), "WRONGCHK")  # checksum mismatch
