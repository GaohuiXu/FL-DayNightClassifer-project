"""Clean S06 provenance identity contracts."""
from __future__ import annotations

import pytest

from fl_v3.config import resolve_config
from fl_v3.eval.provenance import build_s06_provenance, verify_s06_provenance
from test_s06_resolved_config import valid_config


def test_s06_provenance_binds_config_data_mode_precision_checkpoint_and_source(tmp_path):
    config = resolve_config(valid_config(tmp_path))
    source = "1" * 40
    checkpoint = "2" * 64
    provenance = build_s06_provenance(
        config,
        checkpoint_sha256=checkpoint,
        source_sha=source,
    )
    assert provenance["resolved_config_sha256"] == config.sha256
    assert provenance["resolved_config"] == config.as_dict()
    assert provenance["data_identities"] == config.data_identities
    assert provenance["model_mode"] == config.model_mode
    assert provenance["precision"] == config.precision
    assert verify_s06_provenance(
        provenance,
        config,
        checkpoint_sha256=checkpoint,
        source_sha=source,
    )["_verified"]


@pytest.mark.parametrize(
    "field,replacement",
    [
        ("model_mode", "fusion"),
        ("precision", "fp16"),
        ("checkpoint_sha256", "3" * 64),
        ("source_sha", "4" * 40),
    ],
)
def test_s06_provenance_rejects_identity_drift(tmp_path, field, replacement):
    config = resolve_config(valid_config(tmp_path))
    source = "1" * 40
    checkpoint = "2" * 64
    provenance = build_s06_provenance(
        config,
        checkpoint_sha256=checkpoint,
        source_sha=source,
    )
    drifted = dict(provenance)
    drifted[field] = replacement
    with pytest.raises(RuntimeError, match="identity drift"):
        verify_s06_provenance(
            drifted,
            config,
            checkpoint_sha256=checkpoint,
            source_sha=source,
        )


def test_s06_provenance_rejects_partial_schema(tmp_path):
    config = resolve_config(valid_config(tmp_path))
    provenance = build_s06_provenance(
        config,
        checkpoint_sha256="2" * 64,
        source_sha="1" * 40,
    )
    provenance.pop("data_identities")
    with pytest.raises(RuntimeError, match="partial"):
        verify_s06_provenance(
            provenance,
            config,
            checkpoint_sha256="2" * 64,
            source_sha="1" * 40,
        )
