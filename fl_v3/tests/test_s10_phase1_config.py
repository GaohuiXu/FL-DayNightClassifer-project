from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from fl_v3.config import ConfigError, load_resolved_config, resolve_config
from fl_v3.config.phase1 import Phase1ConfigError, phase1_runtime_ready


ROOT = Path(__file__).resolve().parents[1]
CAMERA = ROOT / "configs" / "s10_phase1_camera.json"
LIDAR = ROOT / "configs" / "s10_phase1_lidar.json"
H = "a" * 64


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
        (LIDAR, lambda c: c["data"].update(train_point_sweeps=10), "data.train_point_sweeps"),
        (LIDAR, lambda c: c["gt_paste"].update(yaw_jitter_radians=0.1), "gt_paste.yaw_jitter_radians"),
        (LIDAR, lambda c: c["sampling"].update(expanded_length=19877), "sampling"),
    ],
)
def test_phase1_schema_rejects_missing_or_drifted_science(path, mutation, message):
    raw = _raw(path)
    mutation(raw)
    with pytest.raises(ConfigError, match=message):
        resolve_config(raw)


def test_phase1_pending_materialization_is_resolvable_but_not_runtime_ready():
    camera = load_resolved_config(CAMERA).as_dict()
    lidar = load_resolved_config(LIDAR).as_dict()
    with pytest.raises(Phase1ConfigError, match="Camera checkpoint"):
        phase1_runtime_ready(camera)
    with pytest.raises(Phase1ConfigError, match="GTDB"):
        phase1_runtime_ready(lidar)


def test_phase1_accepted_materialization_requires_complete_identities():
    raw = _raw(CAMERA)
    raw["initialization"].update(status="accepted", physical_sha256=H)
    with pytest.raises(ConfigError, match="requires all three identities"):
        resolve_config(raw)

    raw["initialization"].update(
        mapping_report_sha256=H,
        initialization_state_sha256=H,
    )
    raw["precision"]["grad_scaler"]["status"] = "accepted"
    raw["contract"]["lifecycle"] = "envelope_a_qualified"
    accepted = resolve_config(raw)
    phase1_runtime_ready(accepted.as_dict())


def test_phase1_hash_changes_for_materialized_checkpoint_identity():
    raw = _raw(CAMERA)
    pending = resolve_config(copy.deepcopy(raw))
    raw["initialization"].update(
        status="accepted",
        physical_sha256=H,
        mapping_report_sha256="b" * 64,
        initialization_state_sha256="c" * 64,
    )
    accepted = resolve_config(raw)
    assert pending.sha256 != accepted.sha256


def test_phase1_data_identity_bridge_separates_cache_capacity_and_consumption():
    identities = load_resolved_config(LIDAR).data_identities
    assert identities["cache_capacity_sweeps"] == 10
    assert identities["train_point_sweeps"] == 1
    assert identities["eval_point_sweeps"] == 10
    assert identities["cbgs_expanded_indices_sha256"] == (
        "7f209a57e686645ae3cd3ab1e93d4ca7fc8e46b494eac35fbc2d69d27d102389"
    )
