from __future__ import annotations

import copy
from contextlib import nullcontext
import importlib.util
import json
from pathlib import Path

import pytest
import torch

from fl_v3.config import load_resolved_config, resolve_config
from fl_v3.training.phase1_profile import (
    BASELINE_CANDIDATES,
    IP_E1_RUNNABLE_CANDIDATES,
    IP_E2_RUNNABLE_CANDIDATES,
    IP_E3_RUNNABLE_CANDIDATES,
    IP_E4_RUNNABLE_CANDIDATES,
    Phase1ProfileError,
    derive_profile_runtime_config,
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
IP_E2_PROFILES = tuple(sorted((ROOT / "configs").glob("s10_phase1p_ip_e2_*.json")))
IP_E3_PROFILES = tuple(sorted((ROOT / "configs").glob("s10_phase1p_ip_e3_*.json")))
IP_E4_PROFILES = tuple(sorted((ROOT / "configs").glob("s10_phase1p_ip_e4_*.json")))
HISTORICAL_CAMERA_FILE_SHA256 = (
    "567cb1b71535b4866193273960e531ae4b45318e56e81101e99ad186ac23ce60"
)
HISTORICAL_CAMERA_RESOLVED_SHA256 = (
    "e95e65a63a32c494296b38baf98fd913ff1ec6a168b78aabac48a8dc8f0ffe1d"
)


def _raw() -> dict:
    return json.loads(PROFILE.read_text(encoding="utf-8"))


def _historical_camera_config():
    """Reconstruct the immutable B4 source graph bound by terminal profiles."""
    raw = json.loads(CAMERA.read_text(encoding="utf-8"))
    raw["schema_version"] = "s10.phase1.v2"
    raw["contract"].pop("throughput_decision")
    raw["contract"].pop("throughput_evidence_commit")
    raw["optimizer"]["fused"] = False
    raw["training"].update(
        micro_batch_size=4,
        accumulation_steps=8,
        loss_accumulation="mean_over_eight_microbatches",
    )
    raw.pop("runtime_optimizations")
    config = resolve_config(raw)
    assert config.sha256 == HISTORICAL_CAMERA_RESOLVED_SHA256
    return config


def _pre_ip_e4_camera_config():
    """Reconstruct the immutable B16 source graph used by IP-E3/IP-E4."""
    raw = json.loads(CAMERA.read_text(encoding="utf-8"))
    raw["contract"]["throughput_decision"] = "IP-G2"
    raw["contract"]["throughput_evidence_commit"] = (
        "6ec7fb6d067259ac61ecaed89481e7e2562c3a2d"
    )
    raw["runtime_optimizations"].pop("camera_preprocess")
    config = resolve_config(raw)
    assert config.sha256 == (
        "f6040d30c23571f049bba3602081a9ec3bbfbdafc5d5ab8b76e9dd375eb76f25"
    )
    return config


def _assert_historical_camera_binding(profile) -> None:
    binding = profile.data["branch_bindings"]["camera"]
    assert binding["config_path"] == "fl_v3/configs/s10_phase1_camera.json"
    assert binding["config_file_sha256"] == HISTORICAL_CAMERA_FILE_SHA256
    assert binding["resolved_config_sha256"] == HISTORICAL_CAMERA_RESOLVED_SHA256


def _assert_pre_ip_e4_camera_binding(profile) -> None:
    binding = profile.data["branch_bindings"]["camera"]
    assert binding["config_path"] == "fl_v3/configs/s10_phase1_camera.json"
    assert binding["config_file_sha256"] == (
        "25f53fc554c348c329c7a9cf4b9a5c8d521d993908114fbf64a46f75b3db0bda"
    )
    assert binding["resolved_config_sha256"] == (
        "f6040d30c23571f049bba3602081a9ec3bbfbdafc5d5ab8b76e9dd375eb76f25"
    )


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
    _assert_historical_camera_binding(profile)
    with pytest.raises(Phase1ProfileError, match="source config file identity drift"):
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


def test_ip_e2_profiles_are_exact_camera_only_mappings():
    assert len(IP_E2_PROFILES) == len(IP_E2_RUNNABLE_CANDIDATES) == 8
    seen = set()
    for path in IP_E2_PROFILES:
        profile = load_phase1_profile_spec(path)
        candidate_id = str(profile.data["candidate_id"])
        assert profile.data["envelope"] == "IP-E2"
        assert candidate_id not in seen
        seen.add(candidate_id)
        profile.assert_runnable("camera")
        assert dict(profile.candidates) == IP_E2_RUNNABLE_CANDIDATES[candidate_id]["options"]
        with pytest.raises(Phase1ProfileError, match="not runnable"):
            profile.assert_runnable("lidar")
        _assert_historical_camera_binding(profile)
        assert profile.measurement["capacity_accepted_windows"] == 8
        assert profile.candidates["physical_batch_size"] in {4, 8, 16}
    assert seen == set(IP_E2_RUNNABLE_CANDIDATES)
    assert all("b12" not in candidate_id for candidate_id in seen)


def test_ip_e3_profiles_bind_the_promoted_b16_stack():
    assert len(IP_E3_PROFILES) == len(IP_E3_RUNNABLE_CANDIDATES) == 3
    source = _pre_ip_e4_camera_config()
    seen = set()
    for path in IP_E3_PROFILES:
        profile = load_phase1_profile_spec(path)
        candidate_id = str(profile.data["candidate_id"])
        seen.add(candidate_id)
        assert profile.data["envelope"] == "IP-E3"
        _assert_pre_ip_e4_camera_binding(profile)
        profile.assert_runnable("camera")
        assert dict(profile.candidates) == IP_E3_RUNNABLE_CANDIDATES[
            candidate_id
        ]["options"]
        with pytest.raises(Phase1ProfileError, match="not runnable"):
            profile.assert_runnable("lidar")
        runtime = derive_profile_runtime_config(source, profile)
        raw = runtime.as_dict()
        assert raw["training"]["micro_batch_size"] == 16
        assert raw["training"]["accumulation_steps"] == 2
        assert raw["training"]["effective_global_batch"] == 32
        assert raw["optimizer"]["fused"] is True
        assert raw["runtime_optimizations"]["camera_sdpa"] is True
        assert raw["runtime_optimizations"]["torch_compile"]["enabled"] is True
    assert seen == set(IP_E3_RUNNABLE_CANDIDATES)


def test_ip_e4_profiles_bind_the_final_b16_stack():
    assert len(IP_E4_PROFILES) == len(IP_E4_RUNNABLE_CANDIDATES) == 3
    source = _pre_ip_e4_camera_config()
    seen = set()
    for path in IP_E4_PROFILES:
        profile = load_phase1_profile_spec(path)
        candidate_id = str(profile.data["candidate_id"])
        seen.add(candidate_id)
        assert profile.data["envelope"] == "IP-E4"
        _assert_pre_ip_e4_camera_binding(profile)
        profile.assert_runnable("camera")
        assert dict(profile.candidates) == IP_E4_RUNNABLE_CANDIDATES[
            candidate_id
        ]["options"]
        with pytest.raises(Phase1ProfileError, match="not runnable"):
            profile.assert_runnable("lidar")
        runtime = derive_profile_runtime_config(source, profile)
        raw = runtime.as_dict()
        assert raw["training"]["micro_batch_size"] == 16
        assert raw["training"]["accumulation_steps"] == 2
        assert raw["training"]["effective_global_batch"] == 32
        assert raw["optimizer"]["fused"] is True
        assert raw["runtime_optimizations"]["camera_sdpa"] is True
        assert raw["runtime_optimizations"]["torch_compile"]["enabled"] is True
        assert profile.candidates["camera_batched_affine_grid"] is True
        assert profile.candidates["camera_vectorized_geometry"] is (
            candidate_id != "camera_b16_batched_affine_reference"
        )
        assert profile.candidates["camera_bulk_input_conversion"] is (
            candidate_id
            == (
                "camera_b16_batched_affine_vectorized_geometry_"
                "bulk_input_conversion"
            )
        )
    assert seen == set(IP_E4_RUNNABLE_CANDIDATES)


def test_ip_e2_runtime_views_preserve_effective_b32_and_source_bytes():
    source_bytes = CAMERA.read_bytes()
    source = _historical_camera_config()
    expected = {
        "camera_reference_b4_accum8": (4, 8, False),
        "camera_sdpa_compile_b8_accum4": (8, 4, False),
        "camera_sdpa_compile_fused_b8_accum4": (8, 4, True),
        "camera_sdpa_compile_b16_accum2": (16, 2, False),
        "camera_sdpa_compile_fused_b16_accum2": (16, 2, True),
    }
    for path in IP_E2_PROFILES:
        profile = load_phase1_profile_spec(path)
        if profile.data["candidate_id"] not in expected:
            continue
        runtime = derive_profile_runtime_config(source, profile)
        batch, accumulation, fused = expected[str(profile.data["candidate_id"])]
        raw = runtime.as_dict()
        assert raw["training"]["micro_batch_size"] == batch
        assert raw["training"]["accumulation_steps"] == accumulation
        assert raw["training"]["effective_global_batch"] == 32
        assert raw["optimizer"]["fused"] is fused
        assert raw["phase1p_runtime"]["profile_sha256"] == profile.sha256
        assert raw["phase1p_runtime"]["candidate_id"] == profile.data["candidate_id"]
        assert raw["phase1p_runtime"]["source_resolved_config_sha256"] == source.sha256
        assert runtime.sha256 != source.sha256
        assert runtime.source.sha256 == source.sha256
        assert runtime.data_identities == source.data_identities
    assert CAMERA.read_bytes() == source_bytes


def test_ip_e2_candidate_configuration_preserves_state_dict_names(monkeypatch):
    runner = _runner_module()
    profile = load_phase1_profile_spec(
        ROOT / "configs" / "s10_phase1p_ip_e2_camera_compile_b4.json"
    )

    class Preprocess(torch.nn.Module):
        def set_phase1p_augmentation_transfer_cleanup(self, value):
            assert value is False

        def set_phase1p_static_grid_cache(self, value):
            assert value is False

        def set_phase1p_batched_affine_grid(self, value):
            assert value is False

        def set_phase1p_batched_preprocess(self, value):
            assert value is False

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.preprocess = Preprocess()
            for name in runner._COMPILE_MODULES:
                setattr(self, name, torch.nn.Linear(2, 2))

            self._phase1_runtime_optimization_identity = {
                "camera_sdpa": False,
                "sdpa_modules_patched": 0,
                "torch_compile": False,
                "fused_adamw": False,
                "compiled_forward_modules": [],
                "compile_backend": None,
                "compile_dynamic": None,
                "compile_mode": None,
                "state_dict_name_sha256": runner._state_name_sha256(self),
            }

    monkeypatch.setattr(torch, "compile", lambda fn, **kwargs: fn)
    model = Model()
    before = tuple(model.state_dict())
    record = runner._configure_profile_candidate(model, profile, "camera")
    assert tuple(model.state_dict()) == before
    assert record["compiled_forward_modules"] == list(runner._COMPILE_MODULES)
    assert record["state_dict_name_sha256"] == runner._state_name_sha256(model)


def test_ip_e3_production_runtime_is_not_patched_twice():
    runner = _runner_module()
    profile = load_phase1_profile_spec(
        ROOT / "configs" / "s10_phase1p_ip_e3_camera_b16_batched_affine_grid.json"
    )

    class Preprocess(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.values = {}

        def set_phase1p_augmentation_transfer_cleanup(self, value):
            self.values["augmentation"] = value

        def set_phase1p_static_grid_cache(self, value):
            self.values["static"] = value

        def set_phase1p_batched_affine_grid(self, value):
            self.values["batched"] = value

        def set_phase1p_batched_preprocess(self, value):
            self.values["preprocess"] = value

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.preprocess = Preprocess()
            for name in runner._COMPILE_MODULES:
                setattr(self, name, torch.nn.Linear(2, 2))
            self._phase1_runtime_optimization_identity = {
                "camera_sdpa": True,
                "sdpa_modules_patched": 12,
                "torch_compile": True,
                "fused_adamw": True,
                "compiled_forward_modules": list(runner._COMPILE_MODULES),
                "compile_backend": "inductor",
                "compile_dynamic": False,
                "compile_mode": "default",
                "state_dict_name_sha256": runner._state_name_sha256(self),
            }

    model = Model()
    before = tuple(model.state_dict())
    record = runner._configure_profile_candidate(model, profile, "camera")
    assert tuple(model.state_dict()) == before
    assert model.preprocess.values == {
        "augmentation": False,
        "static": False,
        "batched": True,
        "preprocess": False,
    }
    assert record["runtime_application"] == "production_config"
    assert record["sdpa_modules_patched"] == 12
    assert record["compiled_forward_modules"] == list(runner._COMPILE_MODULES)


def test_ip_e3_batched_rotation_runtime_is_not_patched_twice():
    runner = _runner_module()
    profile = load_phase1_profile_spec(
        ROOT
        / "configs"
        / "s10_phase1p_ip_e3_camera_b16_batched_rotation_grid_sample.json"
    )

    class Preprocess(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.values = {}

        def set_phase1p_augmentation_transfer_cleanup(self, value):
            self.values["augmentation"] = value

        def set_phase1p_static_grid_cache(self, value):
            self.values["static"] = value

        def set_phase1p_batched_affine_grid(self, value):
            self.values["batched"] = value

        def set_phase1p_batched_preprocess(self, value):
            self.values["preprocess"] = value

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.preprocess = Preprocess()
            for name in runner._COMPILE_MODULES:
                setattr(self, name, torch.nn.Linear(2, 2))
            self._phase1_runtime_optimization_identity = {
                "camera_sdpa": True,
                "sdpa_modules_patched": 12,
                "torch_compile": True,
                "fused_adamw": True,
                "compiled_forward_modules": list(runner._COMPILE_MODULES),
                "compile_backend": "inductor",
                "compile_dynamic": False,
                "compile_mode": "default",
                "state_dict_name_sha256": runner._state_name_sha256(self),
            }

    model = Model()
    before = tuple(model.state_dict())
    record = runner._configure_profile_candidate(model, profile, "camera")
    assert tuple(model.state_dict()) == before
    assert model.preprocess.values == {
        "augmentation": False,
        "static": True,
        "batched": True,
        "preprocess": True,
    }
    assert record["runtime_application"] == "production_config"
    assert record["sdpa_modules_patched"] == 12
    assert record["compiled_forward_modules"] == list(runner._COMPILE_MODULES)


def test_ip_e4_candidate_configuration_is_fail_closed():
    runner = _runner_module()
    profile = load_phase1_profile_spec(
        ROOT
        / "configs"
        / "s10_phase1p_ip_e4_camera_b16_vectorized_geometry.json"
    )

    class Preprocess(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.values = {}

        def set_phase1p_augmentation_transfer_cleanup(self, value):
            self.values["augmentation"] = value

        def set_phase1p_static_grid_cache(self, value):
            self.values["static"] = value

        def set_phase1p_batched_affine_grid(self, value):
            self.values["batched"] = value

        def set_phase1p_batched_preprocess(self, value):
            self.values["preprocess"] = value

        def set_phase1p_vectorized_geometry(self, value):
            self.values["vectorized_geometry"] = value

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.preprocess = Preprocess()
            for name in runner._COMPILE_MODULES:
                setattr(self, name, torch.nn.Linear(2, 2))
            self._phase1_runtime_optimization_identity = {
                "camera_sdpa": True,
                "sdpa_modules_patched": 12,
                "torch_compile": True,
                "fused_adamw": True,
                "compiled_forward_modules": list(runner._COMPILE_MODULES),
                "compile_backend": "inductor",
                "compile_dynamic": False,
                "compile_mode": "default",
                "state_dict_name_sha256": runner._state_name_sha256(self),
            }

    model = Model()
    before = tuple(model.state_dict())
    record = runner._configure_profile_candidate(model, profile, "camera")
    assert tuple(model.state_dict()) == before
    assert model.preprocess.values == {
        "augmentation": False,
        "static": False,
        "batched": True,
        "preprocess": False,
        "vectorized_geometry": True,
    }
    assert record["runtime_application"] == "production_config"
    assert record["camera_batched_affine_grid"] is True
    assert record["camera_vectorized_geometry"] is True
    assert record["sdpa_modules_patched"] == 12
    assert record["compiled_forward_modules"] == list(runner._COMPILE_MODULES)


def test_ip_e4_bulk_candidate_configuration_is_fail_closed():
    runner = _runner_module()
    profile = load_phase1_profile_spec(
        ROOT
        / "configs"
        / "s10_phase1p_ip_e4_camera_b16_bulk_input_conversion.json"
    )

    class Preprocess(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.values = {}

        def set_phase1p_augmentation_transfer_cleanup(self, value):
            self.values["augmentation"] = value

        def set_phase1p_static_grid_cache(self, value):
            self.values["static"] = value

        def set_phase1p_batched_affine_grid(self, value):
            self.values["batched"] = value

        def set_phase1p_batched_preprocess(self, value):
            self.values["preprocess"] = value

        def set_phase1p_vectorized_geometry(self, value):
            self.values["vectorized_geometry"] = value

        def set_phase1p_bulk_input_conversion(self, value):
            self.values["bulk_input_conversion"] = value

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.preprocess = Preprocess()
            for name in runner._COMPILE_MODULES:
                setattr(self, name, torch.nn.Linear(2, 2))
            self._phase1_runtime_optimization_identity = {
                "camera_sdpa": True,
                "sdpa_modules_patched": 12,
                "torch_compile": True,
                "fused_adamw": True,
                "compiled_forward_modules": list(runner._COMPILE_MODULES),
                "compile_backend": "inductor",
                "compile_dynamic": False,
                "compile_mode": "default",
                "state_dict_name_sha256": runner._state_name_sha256(self),
            }

    model = Model()
    before = tuple(model.state_dict())
    record = runner._configure_profile_candidate(model, profile, "camera")
    assert tuple(model.state_dict()) == before
    assert model.preprocess.values == {
        "augmentation": False,
        "static": False,
        "batched": True,
        "preprocess": False,
        "vectorized_geometry": True,
        "bulk_input_conversion": True,
    }
    assert record["runtime_application"] == "production_config"
    assert record["camera_batched_affine_grid"] is True
    assert record["camera_vectorized_geometry"] is True
    assert record["camera_bulk_input_conversion"] is True
    assert record["sdpa_modules_patched"] == 12
    assert record["compiled_forward_modules"] == list(runner._COMPILE_MODULES)


def test_camera_trace_diagnosis_requires_preprocess_to_rank_first():
    runner = _runner_module()
    keys = (*runner._TRAIN_TRACE_RANGES, *runner._CAMERA_FORWARD_TRACE_RANGES)
    rows = [
        {
            "key": key,
            "cpu_time_total_us": (
                10.0 if key == "fl_v3::camera::preprocess" else 1.0
            ),
        }
        for key in keys
    ]
    diagnosis = runner._camera_trace_diagnosis(rows)
    assert diagnosis["missing_core_range_keys"] == []
    assert diagnosis["largest_camera_forward_range"] == "fl_v3::camera::preprocess"
    assert diagnosis["preprocess_is_largest_camera_forward_range"] is True

    for row in rows:
        if row["key"] == "fl_v3::camera::swin_backbone":
            row["cpu_time_total_us"] = 20.0
    diagnosis = runner._camera_trace_diagnosis(rows)
    assert diagnosis["largest_camera_forward_range"] == "fl_v3::camera::swin_backbone"
    assert diagnosis["preprocess_is_largest_camera_forward_range"] is False


def test_camera_trace_diagnosis_reports_preprocess_subranges():
    runner = _runner_module()
    keys = (
        *runner._TRAIN_TRACE_RANGES,
        *runner._CAMERA_FORWARD_TRACE_RANGES,
        *runner._CAMERA_PREPROCESS_TRACE_RANGES,
    )
    rows = [
        {
            "key": key,
            "cpu_time_total_us": (
                17.0
                if key == "fl_v3::camera_preprocess::convert_resize"
                else 2.0
            ),
        }
        for key in keys
    ]
    # Exported profiler tables may contain duplicate keys.  The diagnosis must
    # deterministically retain the largest inclusive aggregate.
    rows.append(
        {
            "key": "fl_v3::camera_preprocess::convert_resize",
            "cpu_time_total_us": 1.0,
        }
    )
    diagnosis = runner._camera_trace_diagnosis(rows)
    assert diagnosis["missing_core_range_keys"] == []
    assert diagnosis["missing_preprocess_subrange_keys"] == []
    assert diagnosis["largest_preprocess_subrange"] == (
        "fl_v3::camera_preprocess::convert_resize"
    )
    assert diagnosis["preprocess_subrange_cpu_time_total_us"][
        "fl_v3::camera_preprocess::convert_resize"
    ] == 17.0


def test_promoted_camera_runtime_stack_applies_exact_profiled_scope(monkeypatch):
    from fl_v3.models.fusion import swin_sdpa
    from fl_v3.training.phase1_runtime import apply_phase1_runtime_optimizations

    compiled = []

    def fake_compile(fn, **kwargs):
        compiled.append(kwargs)
        return fn

    monkeypatch.setattr(torch, "compile", fake_compile)
    monkeypatch.setattr(swin_sdpa, "apply_sdpa_to_swin", lambda module: 12)

    class Preprocess:
        def __init__(self):
            self.batched_affine_grid = False
            self.vectorized_geometry = False
            self.bulk_input_conversion = False

        def set_phase1p_batched_affine_grid(self, value):
            self.batched_affine_grid = bool(value)

        def set_phase1p_vectorized_geometry(self, value):
            assert self.batched_affine_grid
            self.vectorized_geometry = bool(value)

        def set_phase1p_bulk_input_conversion(self, value):
            assert self.vectorized_geometry
            self.bulk_input_conversion = bool(value)

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.preprocess = Preprocess()
            for name in (
                "camera_backbone",
                "camera_neck",
                "decoder_backbone",
                "decoder_neck",
                "head",
            ):
                setattr(self, name, torch.nn.Linear(2, 2))

    model = Model()
    before = tuple(model.state_dict())
    record = apply_phase1_runtime_optimizations(
        model, load_resolved_config(CAMERA)
    )
    assert tuple(model.state_dict()) == before
    assert record["sdpa_modules_patched"] == 12
    assert record["fused_adamw"] is True
    assert record["camera_preprocess"] == {
        "batched_affine_grid": True,
        "vectorized_geometry": True,
        "bulk_input_conversion": True,
    }
    assert model.preprocess.batched_affine_grid is True
    assert model.preprocess.vectorized_geometry is True
    assert model.preprocess.bulk_input_conversion is True
    assert record["compiled_forward_modules"] == [
        "camera_backbone",
        "camera_neck",
        "decoder_backbone",
        "decoder_neck",
        "head",
    ]
    assert compiled == [
        {"backend": "inductor", "dynamic": False, "mode": "default"}
    ] * 5
    with pytest.raises(RuntimeError, match="more than once"):
        apply_phase1_runtime_optimizations(model, load_resolved_config(CAMERA))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires the IP-E2 GH200")
@pytest.mark.parametrize(
    ("precision", "rtol", "atol"),
    [("fp32", 1e-4, 1e-6), ("fp16", 2e-3, 2e-4)],
)
def test_ip_e2_current_swin_sdpa_forward_backward_parity(precision, rtol, atol):
    from torchvision.models import swin_t

    from fl_v3.models.fusion.swin_sdpa import apply_sdpa_to_swin

    torch.manual_seed(20260721)
    features = swin_t(weights=None).features
    attentions = [
        module
        for module in features.modules()
        if type(module).__name__ == "ShiftedWindowAttention"
    ]
    assert len(attentions) == 12
    # Cover both the unshifted and shifted mask forms used by every Swin stage.
    for source in attentions[:2]:
        reference = copy.deepcopy(source).cuda().train()
        candidate = copy.deepcopy(source).cuda().train()
        assert apply_sdpa_to_swin(candidate) == 1
        assert tuple(reference.state_dict()) == tuple(candidate.state_dict())
        channels = int(reference.qkv.in_features)
        reference_input = torch.randn(
            1, 14, 14, channels, device="cuda", requires_grad=True
        )
        candidate_input = reference_input.detach().clone().requires_grad_(True)
        context = (
            nullcontext()
            if precision == "fp32"
            else torch.autocast(device_type="cuda", dtype=torch.float16)
        )
        previous = torch.backends.cudnn.deterministic
        torch.backends.cudnn.deterministic = precision == "fp32"
        try:
            with context:
                reference_output = reference(reference_input)
                candidate_output = candidate(candidate_input)
            reference_loss = reference_output.float().square().mean()
            candidate_loss = candidate_output.float().square().mean()
            reference_gradients = torch.autograd.grad(
                reference_loss,
                [reference_input, *reference.parameters()],
            )
            candidate_gradients = torch.autograd.grad(
                candidate_loss,
                [candidate_input, *candidate.parameters()],
            )
        finally:
            torch.backends.cudnn.deterministic = previous
        torch.testing.assert_close(
            candidate_output, reference_output, rtol=rtol, atol=atol
        )
        for observed, expected in zip(candidate_gradients, reference_gradients):
            torch.testing.assert_close(observed, expected, rtol=rtol, atol=atol)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires the IP-E2 GH200")
def test_ip_e2_fused_adamw_matches_unfused_accepted_updates():
    torch.manual_seed(20260721)
    reference_parameter = torch.nn.Parameter(torch.randn(4096, device="cuda"))
    candidate_parameter = torch.nn.Parameter(
        reference_parameter.detach().clone()
    )
    reference = torch.optim.AdamW(
        [reference_parameter], lr=2e-4, weight_decay=0.01, fused=False
    )
    candidate = torch.optim.AdamW(
        [candidate_parameter], lr=2e-4, weight_decay=0.01, fused=True
    )
    for step in range(8):
        gradient = torch.sin(reference_parameter.detach() * 0.01 + step)
        reference_parameter.grad = gradient.clone()
        candidate_parameter.grad = gradient.clone()
        reference.step()
        candidate.step()
    torch.testing.assert_close(
        candidate_parameter, reference_parameter, rtol=2e-3, atol=2e-4
    )
    for key in ("exp_avg", "exp_avg_sq"):
        torch.testing.assert_close(
            candidate.state[candidate_parameter][key],
            reference.state[reference_parameter][key],
            rtol=2e-3,
            atol=2e-4,
        )


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
