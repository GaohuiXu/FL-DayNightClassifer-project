from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import math
import subprocess
from pathlib import Path

import pytest

from fl_v3.config import ConfigError, load_resolved_config, resolve_config
from fl_v3.config.phase1 import (
    CAMERA_COMPILE_FORWARD_MODULES,
    LIDAR_COMPILE_FORWARD_MODULES,
    PHASE1_SCHEMA_V4,
    PHASE1_SCHEMA_V5,
    phase1_runtime_ready,
)
from fl_v3.data.nuscenes.phase1 import build_phase1_eval_data


ROOT = Path(__file__).resolve().parents[1]
CAMERA = ROOT / "configs" / "s10_phase1_camera.json"
LIDAR = ROOT / "configs" / "s10_phase1_lidar.json"
ENVELOPE = ROOT / "configs" / "s10_phase1_envelope_b_dual.json"
ENVELOPE_LAUNCHER = ROOT / "scripts" / "run_s10_phase1_envelope_b.sh"
H = "a" * 64
DUAL_OUTPUT_ROOT = (
    "/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/"
    "outputs/s10_phase1_envelope_b_dual_783173d6fe05"
)


def _raw(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_phase1_recipes_expand_to_complete_hash_bound_reference_graphs():
    camera = load_resolved_config(CAMERA)
    lidar = load_resolved_config(LIDAR)

    assert camera.is_phase1 and lidar.is_phase1
    assert camera.model_mode == "camera_only"
    assert lidar.model_mode == "lidar_only"
    assert camera.as_dict()["model"]["neck"]["in_channels"] == [192, 384, 768]
    assert camera.as_dict()["model"]["head"]["tasks"][5] == [
        "pedestrian",
        "traffic_cone",
    ]
    assert lidar.as_dict()["model"]["decoder"]["backbone"]["layer_nums"] == [5, 5]
    assert lidar.as_dict()["model"]["decoder"]["neck"]["output_channels"] == 512
    assert lidar.as_dict()["model"]["head"]["type"] == "TransFusionHead"
    assert camera.schema_version == PHASE1_SCHEMA_V4
    assert lidar.schema_version == PHASE1_SCHEMA_V5
    camera_pool = camera.as_dict()["model"]["view_transform"]
    assert camera_pool["pool_backend"] == "pytorch_sorted_segment_reduce"
    assert camera_pool["pool_optional_backend"] == "optimized_cuda_unpromoted"
    assert "pool_fallback" not in camera_pool
    assert camera.sha256 != lidar.sha256


def test_phase1_cbgs_and_effective_exposure_are_explicit_and_aligned():
    for path in (CAMERA, LIDAR):
        config = load_resolved_config(path).as_dict()
        assert config["sampling"]["expanded_length"] == 87930
        assert config["training"]["consumed_samples_per_epoch"] == 87904
        assert config["training"]["dropped_samples_per_epoch"] == 26
        assert config["training"]["optimizer_updates_per_epoch"] == 2747
        assert config["training"]["max_optimizer_updates"] == 20 * 2747
        assert (
            config["training"]["optimizer_updates_per_epoch"]
            * config["training"]["effective_global_batch"]
            == config["training"]["consumed_samples_per_epoch"]
        )


def test_phase1_camera_ip_e5_recipe_is_exact_b16_per_rank_ddp2_stack():
    resolved = load_resolved_config(CAMERA)
    assert hashlib.sha256(CAMERA.read_bytes()).hexdigest() == (
        "89a4d9982583dc213e110fcec9469be04e9b4ccf3cefb9a2ca97b294e7650014"
    )
    assert resolved.sha256 == (
        "63f77459fcb229155a0b1a6608d83abf3c55336d554c20f7629d57ed7122d1b3"
    )
    config = resolved.as_dict()
    assert config["contract"]["throughput_decision"] == "IP-E5"
    assert config["contract"]["throughput_evidence_commit"] == (
        "5da03ffdaa29614b0bcfc5c85ace93f70acfac6a"
    )
    assert config["training"]["micro_batch_size"] == 16
    assert config["training"]["world_size"] == 2
    assert config["training"]["accumulation_steps"] == 1
    assert config["training"]["effective_global_batch"] == 32
    assert config["training"]["loss_accumulation"] == (
        "ddp_mean_over_one_microbatch_per_rank"
    )
    assert config["optimizer"]["fused"] is True
    assert config["checkpointing"]["recovery_cadence_epochs"] == 1
    runtime = config["runtime_optimizations"]
    assert runtime["camera_sdpa"] is True
    assert runtime["camera_preprocess"] == {
        "batched_affine_grid": True,
        "vectorized_geometry": True,
        "bulk_input_conversion": True,
    }
    assert runtime["torch_compile"] == {
        "enabled": True,
        "scope": "forward_only",
        "backend": "inductor",
        "dynamic": False,
        "mode": "default",
        "modules": list(CAMERA_COMPILE_FORWARD_MODULES),
    }
    assert runtime["distributed_data_parallel"] == {
        "backend": "nccl",
        "topology": "single_node",
        "world_size": 2,
        "local_batch_size": 16,
        "effective_global_batch": 32,
        "broadcast_buffers": True,
        "find_unused_parameters": False,
        "gradient_as_bucket_view": True,
        "static_graph": True,
        "batch_norm": "ordinary_rank_local_b16",
        "worker_seed_formula": "seed_plus_epoch_times_world_size_plus_rank",
        "global_cbgs_partition": (
            "contiguous_rank_b16_halves_of_each_global_b32_window"
        ),
        "checkpoint_model_rank": 0,
        "checkpoint_rng": "per_rank_sidecars",
        "loss_reduction": "ddp_mean_over_one_microbatch_per_rank",
        "finite_control_flow": "all_rank_boolean_and",
    }
    assert config["execution"]["output_root"] == DUAL_OUTPUT_ROOT


def test_phase1_lidar_ip_l_e3_recipe_is_exact_b32_combined_stack():
    resolved = load_resolved_config(LIDAR)
    assert hashlib.sha256(LIDAR.read_bytes()).hexdigest() == (
        "017086bbd9a9534adf2808461da9cf881d9ef798ef3f3d7c58d3a07b2c7a15d9"
    )
    assert resolved.sha256 == (
        "c950d90db0833ecf5f50ddcc2f10671e4abf7a9f2b1edd640425eb52b888b1ad"
    )
    config = resolved.as_dict()
    assert config["contract"]["throughput_decision"] == "IP-L-E3"
    assert config["contract"]["throughput_evidence_commit"] == (
        "814a6a1ca12b16059ede9a52952f155ddafe1470"
    )
    assert config["training"]["micro_batch_size"] == 32
    assert config["training"]["world_size"] == 1
    assert config["training"]["accumulation_steps"] == 1
    assert config["training"]["effective_global_batch"] == 32
    assert config["training"]["loss_accumulation"] == "mean_over_one_microbatch"
    assert config["optimizer"]["fused"] is False
    assert config["checkpointing"]["recovery_cadence_epochs"] == 1
    runtime = config["runtime_optimizations"]
    assert runtime == {
        "lidar_host_batch_offsets": True,
        "hungarian_batched_d2h": True,
        "lidar_sdpa": False,
        "torch_compile": {
            "enabled": True,
            "scope": "forward_only",
            "backend": "inductor",
            "dynamic": False,
            "mode": "default",
            "modules": list(LIDAR_COMPILE_FORWARD_MODULES),
        },
        "cpu_resident_batch_fields": ["lidar_point_offsets"],
        "batch_norm": "ordinary_physical_b32",
        "worker_seed_formula": "seed_plus_epoch",
        "state_dict_names_unchanged_required": True,
    }
    assert config["execution"]["output_root"] == DUAL_OUTPUT_ROOT


def test_phase1_dual_envelope_b_manifest_binds_both_recipes_and_resources():
    spec = json.loads(ENVELOPE.read_text(encoding="utf-8"))
    assert spec["schema_version"] == "s10.phase1.envelope_b_dual.v2"
    assert spec["request_state"] == "parallel_amendment_owner_activation_required"
    assert spec["execution_topology"] == {
        "mode": "independent_camera_lidar_parallel",
        "runnable_units": [
            "camera_primary",
            "lidar_epoch04_diagnostic",
        ],
        "per_branch_max_concurrency": 1,
        "max_concurrent_jobs": 2,
        "max_concurrent_typed_gh200": 3,
        "camera_blocked_by_lidar_terminal": False,
        "lidar_primary_training_or_resume_authorized": False,
    }
    assert spec["candidate_count"] == 2
    assert spec["seed_policy"] == [0]
    assert spec["output_root"] == DUAL_OUTPUT_ROOT
    assert spec["activation"]["source_grants_compute_authority"] is False
    assert spec["review_gate"] == {
        "independent_recipe_freeze_review_required": True,
        "open_p0_p2_allowed": False,
        "status": "parallel_amendment_independent_review_closed_no_open_p0_p2",
        "reviewed_source_sha": "296ef9b947236c9aded6daf323f26d1a013bfb0c",
        "verdict": "PASS_WITH_RESIDUAL_RISK",
        "findings": {"p0": 0, "p1": 0, "p2": 0, "p3": 1},
        "residual_p3": (
            "wall_no_requeue_aggregate_and_cross_job_concurrency_rely_on_exact_sbatch_and_ledger"
        ),
    }
    for entry in spec["entries"].values():
        path = ROOT.parent / entry["path"]
        assert path.is_file()
        assert entry["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()

    paths = {"camera": CAMERA, "lidar": LIDAR}
    expected_resources = {
        "lidar": (1, 16, 98304, "10:00:00", 10.0),
        "camera": (2, 32, 196608, "09:00:00", 18.0),
    }
    projected_charge = 0.0
    for branch, path in paths.items():
        binding = spec["branches"][branch]
        resolved = load_resolved_config(path)
        assert binding["config_path"] == path.relative_to(ROOT.parent).as_posix()
        assert binding["config_file_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
        assert binding["resolved_config_sha256"] == resolved.sha256
        assert resolved.as_dict()["execution"]["output_root"] == DUAL_OUTPUT_ROOT
        assert binding["output_dir"] == (
            f"{DUAL_OUTPUT_ROOT}/{binding['candidate_id']}"
        )
        resource = binding["resource"]
        expected = expected_resources[branch]
        assert (
            resource["gpus_per_node"],
            resource["cpus_per_task"],
            resource["memory_mib"],
            resource["time_limit"],
            resource["initial_job_charge_ceiling_gh200_hours"],
        ) == expected
        projected_charge += binding["projected_training_charged_gh200_hours"]

    diagnostic = spec["diagnostics"]["lidar_epoch04"]
    assert diagnostic["checkpoint_sha256"] == (
        "d01b6219533e3a4c38fdd7be7727020accc4a8664951f5483cfeaeebba91c940"
    )
    assert diagnostic["epoch_record_sha256"] == (
        "98c7d9193145286fb9983a627d23b7e008f889f229cce911412fd5d4185b76da"
    )
    assert diagnostic["optimizer_updates"] == 0
    assert diagnostic["backward"] is False
    assert diagnostic["D_select"] == {
        "executions": 1,
        "checkpoint_epoch": 4,
        "selectable": False,
        "early_stopping": False,
        "raw_head_nonfinite_policy": "fail_closed_before_decode",
        "terminal_epoch20_execution_still_reserved": True,
    }
    assert diagnostic["resource"]["hard_ceiling_gh200_hours"] == 1.5
    assert diagnostic["ordered_cells"][-1] == (
        "epoch04_D_select_production_eval_after_localization_complete"
    )

    aggregate = spec["aggregate_resource"]
    assert aggregate["max_concurrency"] == 2
    assert aggregate["per_branch_max_concurrency"] == 1
    assert math.isclose(
        projected_charge,
        aggregate["original_projected_training_charged_gh200_hours"],
        rel_tol=0.0,
        abs_tol=1e-6,
    )
    computed = 1.15 * (
        aggregate["original_projected_training_charged_gh200_hours"]
        + aggregate["original_evaluation_preflight_recovery_reserve_gh200_hours"]
    )
    assert math.isclose(
        computed,
        aggregate["original_computed_need_gh200_hours"],
        rel_tol=0.0,
        abs_tol=1e-6,
    )
    assert math.isclose(
        aggregate["consumed_through_lidar_science_stop_gh200_hours"]
        + aggregate["remaining_authority_at_parallel_amendment_gh200_hours"],
        aggregate["hard_ceiling_gh200_hours"],
        rel_tol=0.0,
        abs_tol=1e-6,
    )
    assert math.isclose(
        aggregate["remaining_authority_at_parallel_amendment_gh200_hours"]
        - aggregate["camera_initial_job_maximum_gh200_hours"]
        - aggregate["lidar_epoch04_diagnostic_maximum_gh200_hours"],
        aggregate["remaining_if_both_parallel_jobs_hit_maximum_gh200_hours"],
        rel_tol=0.0,
        abs_tol=1e-6,
    )
    assert aggregate["hard_ceiling_gh200_hours"] == 30.0


def test_phase1_envelope_launcher_resolves_slurm_copy_from_submit_worktree(
    tmp_path: Path,
):
    copied_launcher = tmp_path / "slurm_script"
    copied_launcher.write_bytes(ENVELOPE_LAUNCHER.read_bytes())
    completed = subprocess.run(
        [
            "bash",
            str(copied_launcher),
            "--branch",
            "camera",
            "--envelope",
            "fl_v3/configs/s10_phase1_envelope_b_dual.json",
            "--config",
            "fl_v3/configs/s10_phase1_camera.json",
            "--output-dir",
            str(tmp_path / "fresh-output"),
            "--source-sha",
            "0" * 40,
        ],
        cwd=tmp_path,
        env={"SLURM_SUBMIT_DIR": str(ROOT.parent)},
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 2
    assert "source SHA" in completed.stderr
    assert "/var/lib/slurm" not in completed.stderr
    assert "No such file or directory" not in completed.stderr


def test_parallel_envelope_launcher_has_no_lidar_training_or_resume_path(tmp_path: Path):
    completed = subprocess.run(
        [
            "bash",
            str(ENVELOPE_LAUNCHER),
            "--branch",
            "lidar",
            "--envelope",
            str(ENVELOPE),
            "--config",
            str(LIDAR),
            "--output-dir",
            str(tmp_path / "lidar"),
            "--source-sha",
            "0" * 40,
            "--resume",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 2
    assert "usage:" in completed.stderr


def test_phase1_run_bridge_carries_full_resolved_recipe_and_leaf_inventory():
    resolved = load_resolved_config(CAMERA)
    run = resolved.to_run_config()
    assert run["resolved-config-sha256"] == resolved.sha256
    assert run["s10-phase1-runtime"] is True
    assert run["det-lidar-arch"] == "none"
    assert run["phase1"]["model"]["architecture"].startswith("mit_bevfusion")
    leaves = set(run["phase1-scientific-leaf-paths"])
    assert "model.backbone.depths[2]" in leaves
    assert "optimizer.parameter_group_rules[0].decay_mult" in leaves
    assert "sampling.expanded_indices_sha256" in leaves
    assert "data.roles.audit.sealed" in leaves
    assert load_resolved_config(LIDAR).to_run_config()["det-lidar-arch"] == "second_075"


@pytest.mark.parametrize(
    ("path", "mutation", "message"),
    [
        (CAMERA, lambda c: c["training"].pop("accumulation_steps"), "training keys invalid"),
        (CAMERA, lambda c: c["scheduler"]["lr"].update(target_ratio=[5.0, 0.1]), "scheduler.lr.target_ratio"),
        (CAMERA, lambda c: c["optimizer"]["parameter_group_rules"][0].update(decay_mult=1.0), "parameter_group_rules"),
        (CAMERA, lambda c: c["training"].update(micro_batch_size=8), "training"),
        (CAMERA, lambda c: c["training"].update(world_size=1), "training"),
        (CAMERA, lambda c: c["runtime_optimizations"].update(camera_sdpa=False), "camera_sdpa"),
        (
            CAMERA,
            lambda c: c["runtime_optimizations"]["camera_preprocess"].update(
                bulk_input_conversion=False
            ),
            "bulk_input_conversion",
        ),
        (CAMERA, lambda c: c["runtime_optimizations"]["torch_compile"].update(modules=["head"]), "modules"),
        (
            CAMERA,
            lambda c: c["runtime_optimizations"]["distributed_data_parallel"].update(
                batch_norm="sync_batch_norm"
            ),
            "distributed_data_parallel",
        ),
        (LIDAR, lambda c: c["data"].update(train_point_sweeps=10), "data.train_point_sweeps"),
        (LIDAR, lambda c: c["gt_paste"].update(yaw_jitter_radians=0.1), "gt_paste.yaw_jitter_radians"),
        (LIDAR, lambda c: c["sampling"].update(expanded_length=19877), "sampling"),
        (LIDAR, lambda c: c["training"].update(micro_batch_size=16), "training"),
        (
            LIDAR,
            lambda c: c["runtime_optimizations"].update(
                hungarian_batched_d2h=False
            ),
            "hungarian_batched_d2h",
        ),
        (
            LIDAR,
            lambda c: c["runtime_optimizations"]["torch_compile"].update(
                modules=["head"]
            ),
            "torch_compile",
        ),
    ],
)
def test_phase1_schema_rejects_missing_or_drifted_science(path, mutation, message):
    raw = _raw(path)
    mutation(raw)
    with pytest.raises(ConfigError, match=message):
        resolve_config(raw)


def test_phase1_envelope_b_recipes_are_materialized_and_runtime_ready():
    for path in (CAMERA, LIDAR):
        config = load_resolved_config(path).as_dict()
        assert config["contract"]["lifecycle"] == "envelope_b_ready"
        assert config["contract"]["amendment_decision"] == "O-150"
        assert config["execution"]["mode"] == "phase1_train_eval"
        assert config["execution"]["allowed_evaluation_roles"] == ["D_select"]
        assert config["evaluation"]["D_select"]["status"] == "open_once_in_envelope_b"
        assert config["evaluation"]["D_audit"]["status"] == "owner_sealed_until_P1_G2"
        phase1_runtime_ready(config)


def test_phase1_accepted_materialization_requires_complete_identities():
    raw = _raw(CAMERA)
    raw["initialization"]["mapping_report_sha256"] = None
    with pytest.raises(ConfigError, match="requires all three identities"):
        resolve_config(raw)

    raw["initialization"]["mapping_report_sha256"] = H
    accepted = resolve_config(raw)
    phase1_runtime_ready(accepted.as_dict())


def test_phase1_hash_changes_for_materialized_checkpoint_identity():
    raw = _raw(CAMERA)
    accepted = resolve_config(copy.deepcopy(raw))
    raw["initialization"]["physical_sha256"] = H
    changed = resolve_config(raw)
    assert accepted.sha256 != changed.sha256


def test_phase1_eval_constructor_keeps_D_audit_sealed_before_data_access():
    config = load_resolved_config(CAMERA)
    with pytest.raises(ValueError, match="not open"):
        build_phase1_eval_data(config, role="D_audit")


def test_phase1_data_identity_bridge_separates_cache_capacity_and_consumption():
    identities = load_resolved_config(LIDAR).data_identities
    assert identities["cache_capacity_sweeps"] == 10
    assert identities["train_point_sweeps"] == 1
    assert identities["eval_point_sweeps"] == 10
    assert identities["cbgs_expanded_indices_sha256"] == (
        "7f209a57e686645ae3cd3ab1e93d4ca7fc8e46b494eac35fbc2d69d27d102389"
    )


def test_phase1_calibrator_writes_exact_canonical_config_identity(tmp_path):
    calibrator_path = ROOT / "scripts" / "s10_phase1_calibrate.py"
    spec = importlib.util.spec_from_file_location(
        "s10_phase1_calibrate_test_module", calibrator_path
    )
    assert spec is not None and spec.loader is not None
    calibrator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(calibrator)

    resolved = load_resolved_config(CAMERA)
    output = tmp_path / "resolved.json"
    assert calibrator._write_config_once(output, resolved) == resolved.sha256
    assert output.read_bytes() == resolved.canonical_bytes


def test_phase1_calibrator_records_post_rename_artifact_paths(tmp_path):
    calibrator_path = ROOT / "scripts" / "s10_phase1_calibrate.py"
    spec = importlib.util.spec_from_file_location(
        "s10_phase1_calibrate_path_test_module", calibrator_path
    )
    assert spec is not None and spec.loader is not None
    calibrator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(calibrator)

    published = tmp_path / "job_b"
    working = Path(f"{published}.control") / "evidence"
    artifact = working / "resolved_config.qualified.json"
    assert calibrator._published_artifact_path(
        artifact,
        working_output_dir=working,
        published_output_root=published,
    ) == str(published / "evidence" / artifact.name)
    with pytest.raises(RuntimeError, match="layout drift"):
        calibrator._published_artifact_path(
            artifact,
            working_output_dir=tmp_path / "wrong",
            published_output_root=published,
        )
