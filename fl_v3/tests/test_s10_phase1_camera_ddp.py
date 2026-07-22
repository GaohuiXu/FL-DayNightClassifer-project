from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from fl_v3.config import load_resolved_config


ROOT = Path(__file__).resolve().parents[1]
CAMERA = ROOT / "configs" / "s10_phase1_camera.json"


def _runner_module():
    path = ROOT / "scripts" / "s10_phase1_camera_ddp.py"
    spec = importlib.util.spec_from_file_location("s10_phase1_camera_ddp_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_production_camera_ddp_runner_consumes_exact_promoted_contract():
    module = _runner_module()
    config = load_resolved_config(CAMERA)
    assert module._distributed_spec(config) == {
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


def test_production_camera_ddp_checkpoint_agreement_is_fail_closed():
    module = _runner_module()
    base = {
        "parameters_sha256": "p",
        "non_bn_buffers_sha256": "b",
        "optimizer_sha256": "o",
        "scheduler_sha256": "s",
        "scaler_sha256": "g",
        "training_state": {"epoch": 3},
    }
    assert module._agreement([base, dict(base)])["gate_pass"] is True
    drifted = dict(base)
    drifted["optimizer_sha256"] = "different"
    result = module._agreement([base, drifted])
    assert result["gate_pass"] is False
    assert result["checks"]["optimizer_sha256"] is False


def test_production_camera_ddp_resume_identity_rejects_recipe_drift():
    module = _runner_module()
    config = load_resolved_config(CAMERA)
    identity = {
        "schema": module.SCHEMA,
        "branch": "camera",
        "candidate_id": "phase1_camera_primary",
        "resolved_config_sha256": config.sha256,
        "runtime_dependencies_sha256": "d" * 64,
        "seed": 0,
        "distributed_recipe": module._distributed_spec(config),
    }
    module._validate_identity(identity, config, "d" * 64)
    identity["distributed_recipe"] = dict(identity["distributed_recipe"])
    identity["distributed_recipe"]["batch_norm"] = "sync_batch_norm"
    with pytest.raises(RuntimeError, match="distributed_recipe"):
        module._validate_identity(identity, config, "d" * 64)
