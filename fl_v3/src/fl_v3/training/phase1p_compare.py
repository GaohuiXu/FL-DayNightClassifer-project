"""Fail-closed paired analysis for S10 Phase I-P IP-E2 sustained cells."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import random
from typing import Any, Mapping


RESULT_SCHEMA = "s10.phase1p.profiler-result.v2"
PAIR_SCHEMA = "s10.phase1p.paired-comparison.v2"
MEASURED_WINDOWS = 256
WARMUP_WINDOWS = 16
EFFECTIVE_BATCH = 32
TWENTY_EPOCH_PRESENTATIONS = 1_758_080
BOOTSTRAP_DRAWS = 50_000
B16_FOLLOWUP_NEAR_NEUTRAL_LOWER_BOUND = 0.98
B16_FOLLOWUP_REFERENCE_ID = (
    "camera_sdpa_compile_fused_b16_accum2_followup_reference"
)
B16_FOLLOWUP_CONSERVATIVE_ID = (
    "camera_sdpa_compile_fused_b16_accum2_followup_batched_affine_grid"
)
B16_BATCHED_ROTATION_ID = (
    "camera_sdpa_compile_fused_b16_accum2_followup_batched_rotation_grid_sample"
)
IP_E4_PROMOTION_LOWER_BOUND = 1.02
IP_E4_REFERENCE_ID = "camera_b16_batched_affine_reference"
IP_E4_VECTORIZED_GEOMETRY_ID = (
    "camera_b16_batched_affine_vectorized_geometry"
)
IP_E4_BULK_INPUT_CONVERSION_ID = (
    "camera_b16_batched_affine_vectorized_geometry_bulk_input_conversion"
)


class Phase1PPairError(RuntimeError):
    """A paired result is incomplete or violates its matched-allocation contract."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise Phase1PPairError(message)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as stream:
            value = json.load(stream)
    except (OSError, json.JSONDecodeError) as error:
        raise Phase1PPairError(f"cannot read paired artifact {path}: {error}") from error
    _require(isinstance(value, dict), f"paired artifact is not an object: {path}")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_output(path: str | Path) -> dict[str, Any]:
    root = Path(path).resolve()
    result_path = root / "result.json"
    result = _read_json(result_path)
    identity = _read_json(root / "run_identity.json")
    complete = _read_json(root / "complete.json")
    _require(result.get("schema") == RESULT_SCHEMA, f"result schema drift at {root}")
    _require(result.get("mode") == "sustained", f"paired output is not sustained: {root}")
    _require(
        complete.get("result_sha256") == _sha256_file(result_path),
        f"terminal result hash drift at {root}",
    )
    _require(
        complete.get("status") == result.get("status"),
        f"terminal/result status drift at {root}",
    )
    measurement_path = root / "measurement.json"
    _require(
        result.get("measurement_artifact_sha256") == _sha256_file(measurement_path),
        f"measurement hash drift at {root}",
    )
    return {"root": str(root), "result": result, "identity": identity}


def _timing(result: Mapping[str, Any]) -> Mapping[str, Any]:
    metrics = result.get("measurement")
    _require(isinstance(metrics, Mapping), "result measurement is absent")
    timing = metrics.get("readiness_timing")
    _require(isinstance(timing, Mapping), "readiness timing is absent")
    return timing


def _rate(result: Mapping[str, Any]) -> float:
    value = _timing(result).get("throughput", {}).get(
        "exposure_samples_per_second"
    )
    _require(
        isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0,
        "sustained exposure rate is invalid",
    )
    return float(value)


def _blocks(result: Mapping[str, Any]) -> list[dict[str, float]]:
    timing = _timing(result)
    raw = timing.get("throughput_blocks", {}).get("records")
    _require(isinstance(raw, list), "throughput block records are absent")
    _require(len(raw) == MEASURED_WINDOWS // 16, "throughput block count drift")
    blocks: list[dict[str, float]] = []
    for index, block in enumerate(raw):
        _require(isinstance(block, Mapping), f"throughput block {index} is invalid")
        windows = block.get("accepted_windows")
        samples = block.get("exposure_samples")
        seconds = block.get("wall_seconds")
        _require(windows == 16, f"throughput block {index} window count drift")
        _require(
            samples == 16 * EFFECTIVE_BATCH,
            f"throughput block {index} exposure drift",
        )
        _require(
            isinstance(seconds, (int, float))
            and not isinstance(seconds, bool)
            and seconds > 0,
            f"throughput block {index} wall time is invalid",
        )
        blocks.append({"samples": float(samples), "seconds": float(seconds)})
    return blocks


def _bootstrap_ratio_lower_bound(
    reference: list[dict[str, float]],
    candidate: list[dict[str, float]],
    *,
    seed: int,
) -> float:
    generator = random.Random(seed)
    ratios: list[float] = []
    ref_n = len(reference)
    cand_n = len(candidate)
    for _ in range(BOOTSTRAP_DRAWS):
        ref_samples = ref_seconds = cand_samples = cand_seconds = 0.0
        for _ in range(ref_n):
            block = reference[generator.randrange(ref_n)]
            ref_samples += block["samples"]
            ref_seconds += block["seconds"]
        for _ in range(cand_n):
            block = candidate[generator.randrange(cand_n)]
            cand_samples += block["samples"]
            cand_seconds += block["seconds"]
        ratios.append((cand_samples / cand_seconds) / (ref_samples / ref_seconds))
    ratios.sort()
    return float(ratios[math.floor(0.05 * (len(ratios) - 1))])


def _measurement_health(result: Mapping[str, Any]) -> dict[str, Any]:
    metrics = result["measurement"]
    timing = _timing(result)
    memory = timing["memory"]
    checks = {
        "terminal_sustained": result.get("status")
        in {"COMPLETE_SUSTAINED", "FAILED_CHECKPOINT_PARITY"},
        "measured_256_accepted_windows": (
            timing.get("measured_accepted_windows") == MEASURED_WINDOWS
            and timing.get("measured_attempted_windows") == MEASURED_WINDOWS
        ),
        "effective_b32": (
            int(result.get("physical_batch_size", 0))
            * int(result.get("accumulation_steps", 0))
            == EFFECTIVE_BATCH
        ),
        "zero_invalid_windows": float(metrics.get("invalid_windows", -1)) == 0.0,
        "zero_discarded_windows": float(metrics.get("discarded_windows", -1)) == 0.0,
        "zero_scaler_skips": float(metrics.get("grad_scaler_skips", -1)) == 0.0,
        "memory_under_85_percent": bool(
            result.get("memory_safe_under_85_percent_reserved")
        ),
        "no_monotonic_reserved_growth": not bool(
            memory.get("monotonic_reserved_growth_over_64mib")
        ),
        "no_steady_state_recompile": not bool(
            result.get("compile_evidence", {}).get(
                "unexpected_steady_state_recompile"
            )
        ),
        "D_select_forbidden": result.get("D_select_executed") is False,
        "D_audit_forbidden": result.get("D_audit_executed") is False,
        "official_validation_forbidden": result.get("official_validation_executed")
        is False,
        "capability_metrics_forbidden": result.get("capability_metrics") is False,
    }
    return {"checks": checks, "gate_pass": all(checks.values())}


def _continuation_health(result: Mapping[str, Any]) -> dict[str, Any]:
    checkpoint = result.get("checkpoint")
    checks = {
        "checkpoint_record_present": isinstance(checkpoint, Mapping),
        "checkpoint_gate_pass": bool(
            checkpoint.get("gate_pass") if isinstance(checkpoint, Mapping) else False
        ),
    }
    return {"checks": checks, "gate_pass": all(checks.values())}


def _projection(result: Mapping[str, Any], rate: float) -> dict[str, Any]:
    startup = float(result["startup_seconds"]["before_training_total"])
    warmup_wall = float(
        result.get("compile_evidence", {}).get(
            "warmup_including_compile_seconds", 0.0
        )
    )
    expected_warmup_wall = WARMUP_WINDOWS * EFFECTIVE_BATCH / rate
    cold_extra = max(0.0, warmup_wall - expected_warmup_wall)
    checkpoint = result.get("checkpoint")
    checkpoint_seconds = None
    if isinstance(checkpoint, Mapping):
        timing = checkpoint.get("timing_seconds")
        if isinstance(timing, Mapping):
            checkpoint_seconds = sum(
                float(timing[key])
                for key in (
                    "save_including_device_transfer_and_atomic_replace",
                    "checkpoint_file_sha256",
                    "separate_model_state_sha256",
                )
            )
    projected_seconds = None
    if checkpoint_seconds is not None:
        projected_seconds = (
            TWENTY_EPOCH_PRESENTATIONS / rate
            + startup
            + cold_extra
            + 20.0 * checkpoint_seconds
        )
    return {
        "presentations": TWENTY_EPOCH_PRESENTATIONS,
        "startup_seconds": startup,
        "warmup_or_compile_cold_extra_seconds": cold_extra,
        "checkpoint_stall_seconds_per_epoch": checkpoint_seconds,
        "projected_seconds": projected_seconds,
        "projected_GH200_hours": (
            None if projected_seconds is None else projected_seconds / 3600.0
        ),
    }


def compare_output_dirs(
    reference_dir: str | Path,
    candidate_dir: str | Path,
) -> dict[str, Any]:
    reference_output = _load_output(reference_dir)
    candidate_output = _load_output(candidate_dir)
    reference = reference_output["result"]
    candidate = candidate_output["result"]
    reference_identity = reference_output["identity"]
    candidate_identity = candidate_output["identity"]

    _require(reference.get("branch") == candidate.get("branch") == "camera", "pair is not Camera-only")
    _require(reference.get("source") == candidate.get("source"), "pair source identity differs")
    _require(
        reference.get("source_resolved_config_sha256")
        == candidate.get("source_resolved_config_sha256"),
        "pair source config differs",
    )
    _require(reference.get("sampler_prefix") == candidate.get("sampler_prefix"), "pair CBGS prefix differs")
    ref_attempt = reference_identity.get("attempt", {})
    cand_attempt = candidate_identity.get("attempt", {})
    job_id = ref_attempt.get("slurm_job_id")
    _require(job_id and job_id == cand_attempt.get("slurm_job_id"), "pair did not share one Slurm allocation")
    _require(
        ref_attempt.get("node_list")
        and ref_attempt.get("node_list") == cand_attempt.get("node_list"),
        "pair node identity differs",
    )
    _require(
        reference_identity.get("runtime", {}).get("device_name")
        == candidate_identity.get("runtime", {}).get("device_name"),
        "pair GPU identity differs",
    )
    same_physical_batch = (
        int(reference["physical_batch_size"]) == int(candidate["physical_batch_size"])
    )
    if same_physical_batch:
        _require(
            reference.get("first_optimizer_window_input_sha256")
            == candidate.get("first_optimizer_window_input_sha256"),
            "same-batch pair input/RNG anchor differs",
        )

    reference_rate = _rate(reference)
    candidate_rate = _rate(candidate)
    seed_material = (
        f"{reference.get('profile_config_sha256')}:"
        f"{candidate.get('profile_config_sha256')}:{job_id}"
    ).encode("utf-8")
    bootstrap_seed = int.from_bytes(hashlib.sha256(seed_material).digest()[:8], "big")
    lower_bound = _bootstrap_ratio_lower_bound(
        _blocks(reference), _blocks(candidate), seed=bootstrap_seed
    )
    ratio = candidate_rate / reference_rate
    if ratio <= 1.0:
        speed_verdict = "NEGATIVE"
    elif lower_bound > 1.0:
        speed_verdict = "POSITIVE_SCREEN"
    else:
        speed_verdict = "INCONCLUSIVE"

    reference_measurement = _measurement_health(reference)
    candidate_measurement = _measurement_health(candidate)
    reference_continuation = _continuation_health(reference)
    candidate_continuation = _continuation_health(candidate)
    candidate_screen_pass = bool(
        speed_verdict == "POSITIVE_SCREEN"
        and reference_measurement["gate_pass"]
        and candidate_measurement["gate_pass"]
        and reference_continuation["gate_pass"]
        and candidate_continuation["gate_pass"]
    )
    reference_projection = _projection(reference, reference_rate)
    candidate_projection = _projection(candidate, candidate_rate)
    ref_hours = reference_projection["projected_GH200_hours"]
    cand_hours = candidate_projection["projected_GH200_hours"]
    saved_hours = (
        None if ref_hours is None or cand_hours is None else float(ref_hours - cand_hours)
    )

    b16_gate = None
    if (
        int(reference["physical_batch_size"]) == 4
        and int(candidate["physical_batch_size"]) == 8
    ):
        reference_memory = _timing(reference)["memory"]
        candidate_memory = _timing(candidate)["memory"]
        r4 = int(reference_memory["peak_reserved_bytes"])
        r8 = int(candidate_memory["peak_reserved_bytes"])
        visible = int(candidate_memory["device_total_bytes"])
        projected = r8 + 2 * max(r8 - r4, 0)
        health_checks = {
            "B8_measurement_health": bool(candidate_measurement["gate_pass"]),
            "B8_checkpoint_continuation": bool(candidate_continuation["gate_pass"]),
            "B8_no_monotonic_growth": not bool(
                candidate_memory["monotonic_reserved_growth_over_64mib"]
            ),
        }
        b16_gate = {
            "R4_peak_reserved_bytes": r4,
            "R8_peak_reserved_bytes": r8,
            "visible_bytes": visible,
            "projected_B16_reserved_bytes": projected,
            "projected_fraction": projected / visible,
            "projection_diagnostic": {
                "former_threshold_fraction": 0.70,
                "former_gate_pass": projected <= 0.70 * visible,
                "owner_withdrawn_as_capacity_veto": True,
                "capacity_hard_gate_fraction": 0.85,
                "projected_le_capacity_hard_gate": projected <= 0.85 * visible,
                "rule": (
                    "projection is diagnostic only; a fresh OOM-tolerant B16 "
                    "capacity process decides the <=85% hard gate"
                ),
            },
            "checks": health_checks,
            "eligible_for_fresh_capacity_probe": all(health_checks.values()),
            "sustained_B16_authorized_by_this_summary": False,
        }

    return {
        "schema": PAIR_SCHEMA,
        "phase": "S10 Phase I-P",
        "envelope": "IP-E2",
        "matched_allocation": {
            "slurm_job_id": job_id,
            "node_list": ref_attempt["node_list"],
            "same_physical_batch": same_physical_batch,
            "same_batch_input_anchor_exact": same_physical_batch,
        },
        "reference": {
            "root": reference_output["root"],
            "candidate_id": reference["candidate_id"],
            "physical_batch_size": reference["physical_batch_size"],
            "accumulation_steps": reference["accumulation_steps"],
            "exposure_samples_per_second": reference_rate,
            "measurement_health": reference_measurement,
            "checkpoint_continuation": reference_continuation,
            "projection": reference_projection,
        },
        "candidate": {
            "root": candidate_output["root"],
            "candidate_id": candidate["candidate_id"],
            "physical_batch_size": candidate["physical_batch_size"],
            "accumulation_steps": candidate["accumulation_steps"],
            "exposure_samples_per_second": candidate_rate,
            "measurement_health": candidate_measurement,
            "checkpoint_continuation": candidate_continuation,
            "projection": candidate_projection,
        },
        "throughput": {
            "candidate_over_reference": ratio,
            "bootstrap_method": "independent_process-block percentile bootstrap",
            "bootstrap_blocks_per_process": MEASURED_WINDOWS // 16,
            "bootstrap_draws": BOOTSTRAP_DRAWS,
            "bootstrap_seed": bootstrap_seed,
            "one_sided_95_percent_lower_bound": lower_bound,
            "speed_verdict": speed_verdict,
        },
        "payback": {
            "saved_GH200_hours_per_20_epoch_run": saved_hours,
            "profiler_cost_GH200_hours": None,
            "break_even_runs": None,
            "note": "fill profiler cost from the terminal Slurm charged-time ledger",
        },
        "candidate_screen_gate_pass": candidate_screen_pass,
        "conditional_B16_gate": b16_gate,
        "promotion_authorized": False,
        "interpretation_limits": [
            "D_fit-only throughput and engineering-health evidence",
            "numerical-runtime and physical-batch candidates remain measurement-only",
            "no capability, mAP, NDS, generalization, or recipe-selection claim",
        ],
    }


def compare_b16_followup_output_dirs(
    reference_dir: str | Path,
    candidate_dir: str | Path,
) -> dict[str, Any]:
    """Evaluate the exact IP-E3 conservative B16 preprocessing screen.

    The candidate is deliberately allowed to unlock the *implementation* of the
    later batched-rotation ``grid_sample`` candidate when its robust lower bound
    is within two percent of parity. It cannot promote either candidate.
    """
    summary = compare_output_dirs(reference_dir, candidate_dir)
    _require(
        summary["reference"]["candidate_id"] == B16_FOLLOWUP_REFERENCE_ID,
        "IP-E3 reference candidate identity drift",
    )
    _require(
        summary["candidate"]["candidate_id"] == B16_FOLLOWUP_CONSERVATIVE_ID,
        "IP-E3 conservative candidate identity drift",
    )
    _require(
        summary["reference"]["physical_batch_size"]
        == summary["candidate"]["physical_batch_size"]
        == 16,
        "IP-E3 follow-up must compare physical B16",
    )
    lower_bound = float(
        summary["throughput"]["one_sided_95_percent_lower_bound"]
    )
    health = {
        "reference_measurement": bool(
            summary["reference"]["measurement_health"]["gate_pass"]
        ),
        "candidate_measurement": bool(
            summary["candidate"]["measurement_health"]["gate_pass"]
        ),
        "reference_checkpoint_continuation": bool(
            summary["reference"]["checkpoint_continuation"]["gate_pass"]
        ),
        "candidate_checkpoint_continuation": bool(
            summary["candidate"]["checkpoint_continuation"]["gate_pass"]
        ),
        "same_batch_input_anchor_exact": bool(
            summary["matched_allocation"]["same_batch_input_anchor_exact"]
        ),
    }
    hard_gate_pass = all(health.values())
    eligible = bool(
        hard_gate_pass
        and lower_bound >= B16_FOLLOWUP_NEAR_NEUTRAL_LOWER_BOUND
    )
    if not hard_gate_pass:
        verdict = "HARD_GATE_STOP"
    elif lower_bound > 1.0:
        verdict = "POSITIVE_SCREEN"
    elif eligible:
        verdict = "NEAR_NEUTRAL_SCREEN"
    else:
        verdict = "NEGATIVE_STOP"

    summary["schema"] = "s10.phase1p.b16-followup-comparison.v1"
    summary["envelope"] = "IP-E3"
    summary["conservative_followup_gate"] = {
        "one_sided_95_percent_lower_bound_threshold": (
            B16_FOLLOWUP_NEAR_NEUTRAL_LOWER_BOUND
        ),
        "health_checks": health,
        "hard_gate_pass": hard_gate_pass,
        "verdict": verdict,
        "conditional_batched_rotation_implementation_eligible": eligible,
        "rule": (
            "implement the separately owner-scoped batched rotation grid_sample "
            "candidate only when every hard gate passes and the conservative "
            "candidate's one-sided 95% speed-ratio lower bound is >=0.98"
        ),
    }
    summary["promotion_authorized"] = False
    summary["interpretation_limits"].append(
        "near-neutral screening may unlock implementation only; it cannot promote a runtime recipe"
    )
    return summary


def compare_b16_batched_rotation_output_dirs(
    reference_dir: str | Path,
    candidate_dir: str | Path,
) -> dict[str, Any]:
    """Seal the conditionally authorized combined B16 preprocessing screen."""
    summary = compare_output_dirs(reference_dir, candidate_dir)
    _require(
        summary["reference"]["candidate_id"] == B16_FOLLOWUP_REFERENCE_ID,
        "IP-E3 batched-rotation reference identity drift",
    )
    _require(
        summary["candidate"]["candidate_id"] == B16_BATCHED_ROTATION_ID,
        "IP-E3 batched-rotation candidate identity drift",
    )
    _require(
        summary["reference"]["physical_batch_size"]
        == summary["candidate"]["physical_batch_size"]
        == 16,
        "IP-E3 batched-rotation pair must compare physical B16",
    )
    health = {
        "reference_measurement": bool(
            summary["reference"]["measurement_health"]["gate_pass"]
        ),
        "candidate_measurement": bool(
            summary["candidate"]["measurement_health"]["gate_pass"]
        ),
        "reference_checkpoint_continuation": bool(
            summary["reference"]["checkpoint_continuation"]["gate_pass"]
        ),
        "candidate_checkpoint_continuation": bool(
            summary["candidate"]["checkpoint_continuation"]["gate_pass"]
        ),
        "same_batch_input_anchor_exact": bool(
            summary["matched_allocation"]["same_batch_input_anchor_exact"]
        ),
    }
    hard_gate_pass = all(health.values())
    speed_verdict = str(summary["throughput"]["speed_verdict"])
    if not hard_gate_pass:
        verdict = "HARD_GATE_STOP"
    else:
        verdict = speed_verdict
    summary["schema"] = "s10.phase1p.b16-batched-rotation-comparison.v1"
    summary["envelope"] = "IP-E3"
    summary["batched_rotation_gate"] = {
        "health_checks": health,
        "hard_gate_pass": hard_gate_pass,
        "verdict": verdict,
        "positive_screen": bool(
            hard_gate_pass and speed_verdict == "POSITIVE_SCREEN"
        ),
        "rule": (
            "return the one-pair throughput, uncertainty, memory and checkpoint "
            "evidence to the owner; no result promotes the candidate automatically"
        ),
    }
    summary["promotion_authorized"] = False
    summary["interpretation_limits"].append(
        "the combined batched-rotation candidate requires an explicit owner recipe decision"
    )
    return summary


def compare_ip_e4_vectorized_geometry_output_dirs(
    reference_dir: str | Path,
    candidate_dir: str | Path,
) -> dict[str, Any]:
    """Apply the owner-frozen IP-E4 output-neutral promotion gate."""
    summary = compare_output_dirs(reference_dir, candidate_dir)
    _require(
        summary["reference"]["candidate_id"] == IP_E4_REFERENCE_ID,
        "IP-E4 reference candidate identity drift",
    )
    _require(
        summary["candidate"]["candidate_id"]
        == IP_E4_VECTORIZED_GEOMETRY_ID,
        "IP-E4 vectorized-geometry candidate identity drift",
    )
    _require(
        summary["reference"]["physical_batch_size"]
        == summary["candidate"]["physical_batch_size"]
        == 16,
        "IP-E4 must compare physical B16",
    )
    lower_bound = float(
        summary["throughput"]["one_sided_95_percent_lower_bound"]
    )
    health = {
        "reference_measurement": bool(
            summary["reference"]["measurement_health"]["gate_pass"]
        ),
        "candidate_measurement": bool(
            summary["candidate"]["measurement_health"]["gate_pass"]
        ),
        "reference_checkpoint_continuation": bool(
            summary["reference"]["checkpoint_continuation"]["gate_pass"]
        ),
        "candidate_checkpoint_continuation": bool(
            summary["candidate"]["checkpoint_continuation"]["gate_pass"]
        ),
        "same_batch_input_anchor_exact": bool(
            summary["matched_allocation"]["same_batch_input_anchor_exact"]
        ),
    }
    hard_gate_pass = all(health.values())
    promoted = bool(
        hard_gate_pass and lower_bound >= IP_E4_PROMOTION_LOWER_BOUND
    )
    if not hard_gate_pass:
        verdict = "HARD_GATE_STOP"
    elif promoted:
        verdict = "PROMOTE_AND_UNLOCK_BULK_CONVERSION"
    elif lower_bound > 1.0:
        verdict = "POSITIVE_BELOW_PROMOTION_GATE"
    else:
        verdict = str(summary["throughput"]["speed_verdict"])
    summary["schema"] = "s10.phase1p.ip-e4-vectorized-geometry-comparison.v1"
    summary["envelope"] = "IP-E4"
    summary["ip_e4_vectorized_geometry_gate"] = {
        "one_sided_95_percent_lower_bound_threshold": (
            IP_E4_PROMOTION_LOWER_BOUND
        ),
        "health_checks": health,
        "hard_gate_pass": hard_gate_pass,
        "verdict": verdict,
        "promoted_by_owner_gate": promoted,
        "conditional_bulk_input_conversion_implementation_eligible": promoted,
        "rule": (
            "promote vectorized geometry and unlock the separately scoped bulk "
            "uint8-to-float32 candidate only when every hard gate passes and the "
            "one-sided 95% speed-ratio lower bound is >=1.02"
        ),
    }
    summary["promotion_authorized"] = promoted
    summary["interpretation_limits"].append(
        "IP-E4 may promote output-neutral runtime plumbing only; it makes no capability claim"
    )
    return summary


def compare_ip_e4_bulk_input_conversion_output_dirs(
    reference_dir: str | Path,
    candidate_dir: str | Path,
) -> dict[str, Any]:
    """Apply the frozen IP-E4 gate to the unlocked bulk-conversion pair."""
    summary = compare_output_dirs(reference_dir, candidate_dir)
    _require(
        summary["reference"]["candidate_id"] == IP_E4_VECTORIZED_GEOMETRY_ID,
        "IP-E4 bulk reference candidate identity drift",
    )
    _require(
        summary["candidate"]["candidate_id"]
        == IP_E4_BULK_INPUT_CONVERSION_ID,
        "IP-E4 bulk candidate identity drift",
    )
    _require(
        summary["reference"]["physical_batch_size"]
        == summary["candidate"]["physical_batch_size"]
        == 16,
        "IP-E4 bulk conversion must compare physical B16",
    )
    lower_bound = float(
        summary["throughput"]["one_sided_95_percent_lower_bound"]
    )
    health = {
        "reference_measurement": bool(
            summary["reference"]["measurement_health"]["gate_pass"]
        ),
        "candidate_measurement": bool(
            summary["candidate"]["measurement_health"]["gate_pass"]
        ),
        "reference_checkpoint_continuation": bool(
            summary["reference"]["checkpoint_continuation"]["gate_pass"]
        ),
        "candidate_checkpoint_continuation": bool(
            summary["candidate"]["checkpoint_continuation"]["gate_pass"]
        ),
        "same_batch_input_anchor_exact": bool(
            summary["matched_allocation"]["same_batch_input_anchor_exact"]
        ),
    }
    hard_gate_pass = all(health.values())
    promoted = bool(
        hard_gate_pass and lower_bound >= IP_E4_PROMOTION_LOWER_BOUND
    )
    if not hard_gate_pass:
        verdict = "HARD_GATE_STOP"
    elif promoted:
        verdict = "PROMOTE_BULK_INPUT_CONVERSION"
    elif lower_bound > 1.0:
        verdict = "POSITIVE_BELOW_PROMOTION_GATE"
    else:
        verdict = str(summary["throughput"]["speed_verdict"])
    summary["schema"] = "s10.phase1p.ip-e4-bulk-conversion-comparison.v1"
    summary["envelope"] = "IP-E4"
    summary["ip_e4_bulk_input_conversion_gate"] = {
        "one_sided_95_percent_lower_bound_threshold": (
            IP_E4_PROMOTION_LOWER_BOUND
        ),
        "health_checks": health,
        "hard_gate_pass": hard_gate_pass,
        "verdict": verdict,
        "promoted_by_owner_gate": promoted,
        "rule": (
            "promote the unlocked bulk native-image uint8-to-float32 conversion "
            "only when every hard gate passes and the one-sided 95% speed-ratio "
            "lower bound is >=1.02"
        ),
    }
    summary["promotion_authorized"] = promoted
    summary["interpretation_limits"].append(
        "IP-E4 bulk conversion is output-neutral runtime plumbing and makes no capability claim"
    )
    return summary


__all__ = [
    "BOOTSTRAP_DRAWS",
    "B16_FOLLOWUP_NEAR_NEUTRAL_LOWER_BOUND",
    "IP_E4_PROMOTION_LOWER_BOUND",
    "PAIR_SCHEMA",
    "Phase1PPairError",
    "compare_b16_batched_rotation_output_dirs",
    "compare_b16_followup_output_dirs",
    "compare_ip_e4_bulk_input_conversion_output_dirs",
    "compare_ip_e4_vectorized_geometry_output_dirs",
    "compare_output_dirs",
]
