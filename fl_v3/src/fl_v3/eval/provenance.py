"""Clean production provenance for resolved evaluation inputs.

The record binds the complete resolved config (including split and cache
identities), data identities, model mode, precision, checkpoint bytes, and the
runtime source revision. It contains no experiment-specific security metadata.
"""
from __future__ import annotations

from fl_v3.config import ResolvedConfig


def build_s06_provenance(
    config: ResolvedConfig,
    *,
    checkpoint_sha256: str,
    source_sha: str,
) -> dict:
    """Build the production train/resume/evaluation identity record."""
    if len(checkpoint_sha256) != 64 or len(source_sha) != 40:
        raise ValueError("checkpoint SHA-256/source Git SHA identities are required")
    return {
        "schema": "s06.provenance.v1",
        "resolved_config_sha256": config.sha256,
        "resolved_config": config.as_dict(),
        "model_mode": config.model_mode,
        "precision": config.precision,
        "data_identities": config.data_identities,
        "checkpoint_sha256": checkpoint_sha256,
        "source_sha": source_sha,
    }


def verify_s06_provenance(
    provenance: dict,
    config: ResolvedConfig,
    *,
    checkpoint_sha256: str,
    source_sha: str,
) -> dict:
    """Reject partial provenance or any resolved identity drift."""
    expected = build_s06_provenance(
        config,
        checkpoint_sha256=checkpoint_sha256,
        source_sha=source_sha,
    )
    if set(provenance) != set(expected):
        raise RuntimeError("legacy/partial S06 provenance refused")
    drift = [key for key, value in expected.items() if provenance.get(key) != value]
    if drift:
        raise RuntimeError(f"S06 provenance identity drift: {drift}")
    return {**provenance, "_verified": True}
