from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from fl_v3.training.phase1p_compare import (
    Phase1PPairError,
    compare_b16_batched_rotation_output_dirs,
    compare_b16_followup_output_dirs,
    compare_ip_e4_bulk_input_conversion_output_dirs,
    compare_ip_e4_vectorized_geometry_output_dirs,
    compare_ip_e5_ddp_output_dirs,
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


def _ddp_output(root: Path, *, block_seconds: float, hard_gate: bool = True) -> None:
    profile_sha = _write_json(root / "profile.json", {"schema": "profile"})
    rate = 512.0 / block_seconds
    measurement = {
        "measurement_wall_seconds": 16.0 * block_seconds,
        "exposure_samples": 8192,
        "exposure_samples_per_second": rate,
        "throughput_blocks": [
            {
                "accepted_windows": 16,
                "exposure_samples": 512,
                "wall_seconds": block_seconds,
            }
            for _ in range(16)
        ],
        "startup_seconds": {"before_training_total": 12.0},
        "compile_evidence": {"warmup_including_compile_seconds": 512.0 / rate},
        "rank_devices": [
            {"rank": rank, "name": "NVIDIA GH200 120GB", "total_memory_bytes": 100_000}
            for rank in range(2)
        ],
    }
    result = {
        "schema": "s10.phase1p.ip-e5-ddp-result.v1",
        "status": (
            "COMPLETE_DDP_ENGINEERING"
            if hard_gate
            else "COMPLETE_DDP_HARD_GATE_FAILURE"
        ),
        "source": {"git_sha": "a" * 40},
        "attempt": {
            "slurm_job_id": "123",
            "node_list": "n1",
            "gpus_on_node": "2",
        },
        "source_config_sha256": "b" * 64,
        "effective_config_sha256": "f" * 64,
        "profile_sha256": "1" * 64,
        "profile_artifact_sha256": profile_sha,
        "candidate_id": "camera_final_b16_ddp2",
        "world_size": 2,
        "local_batch": 16,
        "accumulation_steps": 1,
        "effective_global_batch": 32,
        "measurement": measurement,
        "bn_rank_diagnostics": {
            "rank0_vs_rank1": {"all_finite": True},
            "elementwise_exact": False,
        },
        "checkpoint": {
            "wall_seconds_including_rank_rng_sidecars_and_hash": 0.8,
        },
        "hard_gates": {"checks": {}, "gate_pass": hard_gate},
    }
    result_sha = _write_json(root / "result.json", result)
    _write_json(
        root / "complete.json",
        {"status": result["status"], "result_sha256": result_sha},
    )


def test_ip_e5_ddp_gate_requires_robust_speed_and_charged_payback(tmp_path):
    reference = tmp_path / "reference"
    candidate = tmp_path / "candidate"
    _output(
        reference,
        candidate_id="camera_final_b16_single_gpu_reference",
        batch=16,
        block_seconds=32.0,
    )
    _ddp_output(candidate, block_seconds=32.0 / 1.70)

    summary = compare_ip_e5_ddp_output_dirs(reference, candidate)
    assert summary["schema"] == "s10.phase1p.ip-e5-ddp-comparison.v1"
    assert summary["throughput"]["candidate_over_reference"] == pytest.approx(1.70)
    assert summary["throughput"]["one_sided_95_percent_lower_bound"] >= 1.60
    assert summary["projection_gates"]["candidate_charged_over_reference"] <= 1.25
    assert summary["qualification_gate"]["gate_pass"] is True
    assert summary["verdict"] == "POSITIVE_DDP_QUALIFICATION"
    assert summary["production_promotion_authorized"] is False

    slower = tmp_path / "slower"
    _ddp_output(slower, block_seconds=32.0 / 1.50)
    summary = compare_ip_e5_ddp_output_dirs(reference, slower)
    assert summary["qualification_gate"]["gate_pass"] is False
    assert summary["verdict"] == "DDP_NOT_QUALIFIED"


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


def test_b16_followup_near_neutral_gate_unlocks_implementation_only(tmp_path):
    reference = tmp_path / "reference"
    candidate = tmp_path / "candidate"
    _output(
        reference,
        candidate_id=(
            "camera_sdpa_compile_fused_b16_accum2_followup_reference"
        ),
        batch=16,
        block_seconds=32.0,
    )
    _output(
        candidate,
        candidate_id=(
            "camera_sdpa_compile_fused_b16_accum2_followup_batched_affine_grid"
        ),
        batch=16,
        block_seconds=32.0 / 0.99,
    )

    summary = compare_b16_followup_output_dirs(reference, candidate)
    gate = summary["conservative_followup_gate"]
    assert summary["schema"] == "s10.phase1p.b16-followup-comparison.v1"
    assert summary["envelope"] == "IP-E3"
    assert gate["verdict"] == "NEAR_NEUTRAL_SCREEN"
    assert gate["hard_gate_pass"] is True
    assert gate["conditional_batched_rotation_implementation_eligible"] is True
    assert summary["promotion_authorized"] is False


def test_b16_batched_rotation_comparison_requires_owner_decision(tmp_path):
    reference = tmp_path / "reference"
    candidate = tmp_path / "candidate"
    _output(
        reference,
        candidate_id=(
            "camera_sdpa_compile_fused_b16_accum2_followup_reference"
        ),
        batch=16,
        block_seconds=32.0,
    )
    _output(
        candidate,
        candidate_id=(
            "camera_sdpa_compile_fused_b16_accum2_followup_"
            "batched_rotation_grid_sample"
        ),
        batch=16,
        block_seconds=30.0,
    )

    summary = compare_b16_batched_rotation_output_dirs(reference, candidate)
    gate = summary["batched_rotation_gate"]
    assert summary["schema"] == (
        "s10.phase1p.b16-batched-rotation-comparison.v1"
    )
    assert summary["envelope"] == "IP-E3"
    assert gate["hard_gate_pass"] is True
    assert gate["verdict"] == "POSITIVE_SCREEN"
    assert gate["positive_screen"] is True
    assert summary["promotion_authorized"] is False


def test_ip_e4_vectorized_geometry_gate_promotes_at_102_percent(tmp_path):
    reference = tmp_path / "reference"
    candidate = tmp_path / "candidate"
    _output(
        reference,
        candidate_id="camera_b16_batched_affine_reference",
        batch=16,
        block_seconds=32.0,
    )
    _output(
        candidate,
        candidate_id="camera_b16_batched_affine_vectorized_geometry",
        batch=16,
        block_seconds=32.0 / 1.03,
    )

    summary = compare_ip_e4_vectorized_geometry_output_dirs(reference, candidate)
    gate = summary["ip_e4_vectorized_geometry_gate"]
    assert summary["schema"] == (
        "s10.phase1p.ip-e4-vectorized-geometry-comparison.v1"
    )
    assert summary["envelope"] == "IP-E4"
    assert gate["hard_gate_pass"] is True
    assert gate["one_sided_95_percent_lower_bound_threshold"] == 1.02
    assert gate["promoted_by_owner_gate"] is True
    assert gate["conditional_bulk_input_conversion_implementation_eligible"] is True
    assert summary["promotion_authorized"] is True


def test_ip_e4_bulk_conversion_gate_promotes_at_102_percent(tmp_path):
    reference = tmp_path / "reference"
    candidate = tmp_path / "candidate"
    _output(
        reference,
        candidate_id="camera_b16_batched_affine_vectorized_geometry",
        batch=16,
        block_seconds=32.0,
    )
    _output(
        candidate,
        candidate_id=(
            "camera_b16_batched_affine_vectorized_geometry_"
            "bulk_input_conversion"
        ),
        batch=16,
        block_seconds=32.0 / 1.03,
    )

    summary = compare_ip_e4_bulk_input_conversion_output_dirs(
        reference, candidate
    )
    gate = summary["ip_e4_bulk_input_conversion_gate"]
    assert summary["schema"] == "s10.phase1p.ip-e4-bulk-conversion-comparison.v1"
    assert summary["envelope"] == "IP-E4"
    assert gate["hard_gate_pass"] is True
    assert gate["one_sided_95_percent_lower_bound_threshold"] == 1.02
    assert gate["promoted_by_owner_gate"] is True
    assert summary["promotion_authorized"] is True
