"""The FL gate REUSES ``det_gate_a40.assert_a40`` — it must LOUD-fail (exit 2) off the A40.

A determinism pass on CPU / a login-node Tesla T4 is a FALSE PASS (the float-summation /
topk-tie drift only surfaces cross-architecture). ``fl_gate_a40`` imports ``assert_a40``
verbatim, so guarding it here guards the FL gate. (Mirrors the T2 det-gate guard.)
"""
from __future__ import annotations

import pathlib
import sys

import pytest

# det_gate_a40 / fl_gate_a40 live in fl_v3/scripts (not a package).
_SCRIPTS = pathlib.Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(_SCRIPTS))

import det_gate_a40  # noqa: E402


def test_assert_a40_exits_2_on_no_cuda(monkeypatch):
    monkeypatch.setattr(det_gate_a40.torch.cuda, "is_available", lambda: False)
    with pytest.raises(SystemExit) as e:
        det_gate_a40.assert_a40()
    assert e.value.code == 2


def test_assert_a40_exits_2_on_non_a40_device(monkeypatch):
    monkeypatch.setattr(det_gate_a40.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(det_gate_a40.torch.cuda, "get_device_name", lambda *a, **k: "Tesla T4")
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    with pytest.raises(SystemExit) as e:
        det_gate_a40.assert_a40()
    assert e.value.code == 2


def test_fl_gate_imports_the_same_assert_a40():
    import fl_gate_a40  # noqa: F401  (importable; reuses det_gate_a40.assert_a40)

    assert fl_gate_a40.assert_a40 is det_gate_a40.assert_a40
