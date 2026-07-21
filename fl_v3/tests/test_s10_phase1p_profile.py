from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import pytest
import torch

from fl_v3.config import load_resolved_config
from fl_v3.training.phase1_profile import (
    BASELINE_CANDIDATES,
    IP_E1_RUNNABLE_CANDIDATES,
    Phase1ProfileError,
    load_phase1_profile_spec,
    validate_phase1_profile_spec,
)


ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT / "configs" / "s10_phase1p_ip_e1.json"
AUG_CLEANUP_PROFILE = ROOT / "configs" / "s10_phase1p_camera_aug_cleanup.json"
STATIC_GRID_PROFILE = ROOT / "configs" / "s10_phase1p_camera_static_grid.json"
BATCHED_GRID_PROFILE = (
    ROOT / "configs" / "s10_phase1p_camera_batched_affine_grid.json"
)
CAMERA = ROOT / "configs" / "s10_phase1_camera.json"
LIDAR = ROOT / "configs" / "s10_phase1_lidar.json"


def _raw() -> dict:
    return json.loads(PROFILE.read_text(encoding="utf-8"))


def _runner_module():
    path = ROOT / "scripts" / "s10_phase1_throughput.py"
    spec = importlib.util.spec_from_file_location("s10_phase1_throughput_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_ip_e1_profile_binds_both_frozen_configs_and_every_candidate_off():
    profile = load_phase1_profile_spec(PROFILE)
    profile.assert_baseline()
    assert dict(profile.candidates) == BASELINE_CANDIDATES
    assert profile.measurement["warmup_accepted_windows"] == 16
    assert profile.measurement["sustained_accepted_windows"] == 256
    assert profile.measurement["trace_accepted_windows"] == 3
    assert profile.measurement["checkpoint_continuation_windows"] == 8
    profile.assert_branch_binding("camera", CAMERA, load_resolved_config(CAMERA))
    profile.assert_branch_binding("lidar", LIDAR, load_resolved_config(LIDAR))
    assert json.loads(profile.canonical_bytes) == profile.as_dict()


def test_ip_e1_aug_cleanup_profile_has_one_exact_camera_only_candidate():
    profile = load_phase1_profile_spec(AUG_CLEANUP_PROFILE)
    profile.assert_runnable("camera")
    expected = IP_E1_RUNNABLE_CANDIDATES[
        "camera_aug_transfer_cleanup_b4_accum8"
    ]["options"]
    assert dict(profile.candidates) == expected
    assert profile.candidates["camera_augmentation_transfer_cleanup"] is True
    assert sum(
        bool(value)
        for key, value in profile.candidates.items()
        if key not in {"physical_batch_size", "checkpoint_cadence_epochs"}
    ) == 1
    with pytest.raises(Phase1ProfileError, match="not runnable"):
        profile.assert_runnable("lidar")
    with pytest.raises(Phase1ProfileError, match="baseline candidate"):
        profile.assert_baseline()


def test_ip_e1_static_grid_profile_has_one_exact_camera_only_candidate():
    profile = load_phase1_profile_spec(STATIC_GRID_PROFILE)
    profile.assert_runnable("camera")
    expected = IP_E1_RUNNABLE_CANDIDATES[
        "camera_static_grid_cache_b4_accum8"
    ]["options"]
    assert dict(profile.candidates) == expected
    assert profile.candidates["camera_static_grid_cache"] is True
    assert sum(
        bool(value)
        for key, value in profile.candidates.items()
        if key not in {"physical_batch_size", "checkpoint_cadence_epochs"}
    ) == 1
    with pytest.raises(Phase1ProfileError, match="not runnable"):
        profile.assert_runnable("lidar")
    with pytest.raises(Phase1ProfileError, match="baseline candidate"):
        profile.assert_baseline()


def test_ip_e1_batched_grid_profile_has_one_exact_camera_only_candidate():
    profile = load_phase1_profile_spec(BATCHED_GRID_PROFILE)
    profile.assert_runnable("camera")
    expected = IP_E1_RUNNABLE_CANDIDATES[
        "camera_batched_affine_grid_b4_accum8"
    ]["options"]
    assert dict(profile.candidates) == expected
    assert profile.candidates["camera_batched_affine_grid"] is True
    assert sum(
        bool(value)
        for key, value in profile.candidates.items()
        if key not in {"physical_batch_size", "checkpoint_cadence_epochs"}
    ) == 1
    with pytest.raises(Phase1ProfileError, match="not runnable"):
        profile.assert_runnable("lidar")
    with pytest.raises(Phase1ProfileError, match="baseline candidate"):
        profile.assert_baseline()


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda value: value["measurement"].update(sustained_accepted_windows=64),
            "must remain 256",
        ),
        (
            lambda value: value["boundaries"].update(allowed_data_role="D_select"),
            "only D_fit",
        ),
        (
            lambda value: value["boundaries"].update(capability_metrics=True),
            "must remain disabled",
        ),
        (
            lambda value: value["boundaries"].update(
                output_root_prefix=value["boundaries"]["output_root_prefix"] + "drift"
            ),
            "output root prefix drift",
        ),
        (
            lambda value: value["candidates"].update(camera_sdpa="false"),
            "must be boolean",
        ),
    ],
)
def test_ip_e1_profile_rejects_measurement_or_boundary_drift(mutation, message):
    raw = _raw()
    mutation(raw)
    with pytest.raises(Phase1ProfileError, match=message):
        validate_phase1_profile_spec(raw)


def test_measurement_candidate_can_be_represented_but_not_run_as_ip_e1_baseline(tmp_path):
    raw = _raw()
    raw["candidates"]["camera_sdpa"] = True
    path = tmp_path / "candidate.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    profile = load_phase1_profile_spec(path)
    with pytest.raises(Phase1ProfileError, match="default-off"):
        profile.assert_baseline()


def test_profiler_state_comparison_is_exact_for_discrete_and_tolerant_for_float():
    runner = _runner_module()
    reference = {
        "step": torch.tensor(3, dtype=torch.int64),
        "weights": torch.tensor([1.0, 2.0]),
        "group": {"lr": 0.1},
    }
    candidate = copy.deepcopy(reference)
    exact = runner._compare_state_captures(
        runner._state_capture(reference),
        runner._state_capture(candidate),
        rtol=0.0,
        atol=0.0,
    )
    assert exact["gate_pass"] is True

    candidate["weights"][0] += 1e-4
    tolerant = runner._compare_state_captures(
        runner._state_capture(reference),
        runner._state_capture(candidate),
        rtol=2e-3,
        atol=2e-4,
    )
    assert tolerant["gate_pass"] is True
    candidate["step"] += 1
    discrete = runner._compare_state_captures(
        runner._state_capture(reference),
        runner._state_capture(candidate),
        rtol=2e-3,
        atol=2e-4,
    )
    assert discrete["gate_pass"] is False
    assert discrete["discrete_exact_failures"]


def test_checkpoint_diagnostic_batch_hash_is_value_sensitive_and_observational():
    runner = _runner_module()
    batch = {
        "sample_token": ["a", "b"],
        "lidar_points": [torch.arange(12, dtype=torch.float32).reshape(3, 4)],
        "gt_labels": [torch.tensor([1, 3], dtype=torch.int64)],
        "meta": (True, 4, 0.25, None),
    }
    before = copy.deepcopy(batch)
    reference = runner._batch_sha256(batch)
    assert runner._batch_sha256(copy.deepcopy(batch)) == reference
    assert batch.keys() == before.keys()
    assert torch.equal(batch["lidar_points"][0], before["lidar_points"][0])
    changed = copy.deepcopy(batch)
    changed["lidar_points"][0][0, 0] += 1.0
    assert runner._batch_sha256(changed) != reference


def test_profiler_cpu_resident_batch_field_skips_only_the_named_transfer():
    from fl_v3.training.loop import _unpack_batch

    batch = {
        "images": torch.ones((1, 2), dtype=torch.float32),
        "augmentation_params": torch.arange(7, dtype=torch.float64).view(1, 1, 7),
        "sample_token": ["token"],
    }
    moved, targets = _unpack_batch(
        batch,
        torch.device("meta"),
        cpu_resident_batch_fields=("augmentation_params",),
    )
    assert moved is targets
    assert moved["images"].device.type == "meta"
    assert moved["augmentation_params"] is batch["augmentation_params"]
    assert moved["augmentation_params"].device.type == "cpu"
    assert moved["sample_token"] == ["token"]


def test_profiler_entry_has_no_evaluation_constructor_or_metric_path():
    source = (ROOT / "scripts" / "s10_phase1_throughput.py").read_text(
        encoding="utf-8"
    )
    for forbidden in (
        "build_phase1_eval_data",
        "run_internal_manifest_eval",
        "decode_eval_set",
        "_evaluate_terminal",
    ):
        assert forbidden not in source
    assert '"D_select_executed": False' in source
    assert '"capability_metrics": False' in source
    assert '"measurement.json"' in source
    assert '"same_process_replay"' in source
