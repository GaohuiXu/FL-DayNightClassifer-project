from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path


def _module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "s10_c1b0_fusion_health.py"
    spec = importlib.util.spec_from_file_location("s10_c1b0_fusion_health", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_h256_selection_is_label_blind_ordered_and_hash_bound(monkeypatch):
    module = _module()
    tokens = [f"token-{index:04d}" for index in range(6155)]
    salt = module.TOKEN_SELECTION_SALT.encode("utf-8") + b"\0"
    expected = tuple(sorted(
        tokens,
        key=lambda token: (hashlib.sha256(salt + token.encode()).digest(), token),
    )[:1024])
    encoded = json.dumps(
        list(expected), sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False,
    ).encode()
    monkeypatch.setattr(module, "EXPECTED_TOKEN_SHA256", hashlib.sha256(encoded).hexdigest())
    assert module.select_h256_tokens(reversed(tokens)) == expected


def test_h256_trajectory_uses_predeclared_first_and_last_windows():
    module = _module()
    windows = [
        {"loss": float(300 - index), "wall_seconds": 2.0 + index / 1000.0}
        for index in range(256)
    ]
    summary = module._trajectory_summary(windows)
    assert summary["loss_decreased"] is True
    assert summary["last_over_first_loss"] < 1.0
    assert summary["post_16_window_seconds_p50"] > 2.0


def test_c1b0_envelope_constants_are_exact():
    module = _module()
    assert module.HORIZON == 256
    assert module.PHYSICAL_BATCH == 4
    assert module.DIAGNOSTIC_WINDOWS == (1, 4, 16, 64, 128, 256)
    assert module.CANDIDATES == (
        ("F-A1-GN-H256", "group_norm"),
        ("F-A1-BN1D-H256", "batch_norm_1d"),
    )
