from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from fl_v3.training.phase1p_compare import (
    Phase1PPairError,
    compare_output_dirs,
)


def _write_json(path: Path, value) -> str:
    payload = json.dumps(value, sort_keys=True).encode("utf-8") + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def _output(
    root: Path,
    *,
    candidate_id: str,
    batch: int,
    block_seconds: float,
    peak_reserved_bytes: int | None = None,
    device_total_bytes: int = 100_000,
) -> None:
    metrics = {
        "invalid_windows": 0.0,
        "discarded_windows": 0.0,
        "grad_scaler_skips": 0.0,
        "readiness_timing": {
            "measured_accepted_windows": 256,
            "measured_attempted_windows": 256,
            "throughput": {
                "exposure_samples_per_second": 512.0 / block_seconds,
            },
            "throughput_blocks": {
                "records": [
                    {
                        "accepted_windows": 16,
                        "exposure_samples": 512,
                        "wall_seconds": block_seconds,
                        "exposure_samples_per_second": 512.0 / block_seconds,
                    }
                    for _ in range(16)
                ]
            },
            "memory": {
                "peak_reserved_bytes": (
                    peak_reserved_bytes
                    if peak_reserved_bytes is not None
                    else 20_000 if batch == 4 else 30_000
                ),
                "device_total_bytes": device_total_bytes,
                "monotonic_reserved_growth_over_64mib": False,
            },
        },
    }
    measurement_sha = _write_json(root / "measurement.json", {"metrics": metrics})
    result = {
        "schema": "s10.phase1p.profiler-result.v2",
        "status": "COMPLETE_SUSTAINED",
        "mode": "sustained",
        "branch": "camera",
        "candidate_id": candidate_id,
        "source": {"git_sha": "a" * 40},
        "source_resolved_config_sha256": "b" * 64,
        "effective_runtime_config_sha256": "c" * 64,
        "profile_config_sha256": hashlib.sha256(candidate_id.encode()).hexdigest(),
        "physical_batch_size": batch,
        "accumulation_steps": 32 // batch,
        "first_optimizer_window_input_sha256": ["d" * 64] * (32 // batch),
        "startup_seconds": {"before_training_total": 10.0},
        "compile_evidence": {
            "warmup_including_compile_seconds": block_seconds,
            "unexpected_steady_state_recompile": False,
        },
        "sampler_prefix": {"presentations": 8704, "sample_tokens_sha256": "e" * 64},
        "checkpoint": {
            "gate_pass": True,
            "timing_seconds": {
                "save_including_device_transfer_and_atomic_replace": 0.5,
                "checkpoint_file_sha256": 0.1,
                "separate_model_state_sha256": 0.2,
            },
        },
        "measurement": metrics,
        "measurement_artifact_sha256": measurement_sha,
        "memory_safe_under_85_percent_reserved": True,
        "D_select_executed": False,
        "D_audit_executed": False,
        "official_validation_executed": False,
        "capability_metrics": False,
    }
    result_sha = _write_json(root / "result.json", result)
    _write_json(
        root / "run_identity.json",
        {
            "attempt": {"slurm_job_id": "123", "node_list": "n1"},
            "runtime": {"device_name": "NVIDIA GH200 120GB"},
        },
    )
    _write_json(
        root / "complete.json",
        {"status": result["status"], "result_sha256": result_sha},
    )


def test_pair_comparison_enforces_match_and_emits_b16_gate(tmp_path):
    reference = tmp_path / "reference"
    candidate = tmp_path / "candidate"
    _output(
        reference,
        candidate_id="camera_sdpa_compile_b4_accum8",
        batch=4,
        block_seconds=32.0,
    )
    _output(
        candidate,
        candidate_id="camera_sdpa_compile_b8_accum4",
        batch=8,
        block_seconds=28.0,
    )

    summary = compare_output_dirs(reference, candidate)
    assert summary["throughput"]["speed_verdict"] == "POSITIVE_SCREEN"
    assert summary["throughput"]["one_sided_95_percent_lower_bound"] > 1.0
    assert summary["candidate_screen_gate_pass"] is True
    assert summary["conditional_B16_gate"]["projected_fraction"] == 0.5
    assert summary["conditional_B16_gate"]["eligible_for_fresh_capacity_probe"] is True
    diagnostic = summary["conditional_B16_gate"]["projection_diagnostic"]
    assert diagnostic["former_gate_pass"] is True
    assert diagnostic["owner_withdrawn_as_capacity_veto"] is True
    assert summary["promotion_authorized"] is False


def test_b16_projection_above_former_70_percent_is_diagnostic_only(tmp_path):
    reference = tmp_path / "reference"
    candidate = tmp_path / "candidate"
    _output(
        reference,
        candidate_id="camera_sdpa_compile_fused_b4_accum8",
        batch=4,
        block_seconds=32.0,
        peak_reserved_bytes=18_000,
    )
    _output(
        candidate,
        candidate_id="camera_sdpa_compile_fused_b8_accum4",
        batch=8,
        block_seconds=28.0,
        peak_reserved_bytes=38_000,
    )

    summary = compare_output_dirs(reference, candidate)
    gate = summary["conditional_B16_gate"]
    assert gate["projected_B16_reserved_bytes"] == 78_000
    assert gate["projected_fraction"] == 0.78
    assert gate["projection_diagnostic"]["former_gate_pass"] is False
    assert gate["projection_diagnostic"]["projected_le_capacity_hard_gate"] is True
    assert gate["eligible_for_fresh_capacity_probe"] is True
    assert gate["sustained_B16_authorized_by_this_summary"] is False


def test_same_batch_pair_requires_exact_input_anchor(tmp_path):
    reference = tmp_path / "reference"
    candidate = tmp_path / "candidate"
    _output(reference, candidate_id="reference", batch=4, block_seconds=32.0)
    _output(candidate, candidate_id="candidate", batch=4, block_seconds=31.0)
    result_path = candidate / "result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result["first_optimizer_window_input_sha256"][0] = "f" * 64
    result_sha = _write_json(result_path, result)
    _write_json(
        candidate / "complete.json",
        {"status": result["status"], "result_sha256": result_sha},
    )

    with pytest.raises(Phase1PPairError, match="input/RNG anchor differs"):
        compare_output_dirs(reference, candidate)
