from __future__ import annotations

import copy
import hashlib

import pytest

from fl_v3.config import ConfigError, resolve_config, verify_physical_data_identities
from fl_v3.source_identity import build_source_state


H = "a" * 64


def valid_config(tmp_path=None):
    root = str(tmp_path or "/synthetic")
    return {
        "schema_version": "s09.v1",
        "model": {"mode": "camera_only", "camera_arch": "swin_t_stride8",
                  "camera_pretrained": False,
                  "lidar_arch": "none", "fusion_arch": "none",
                  "head_arch": "centerhead_multitask"},
        "precision": "fp32",
        "sparse_conv_precision": "not_applicable",
        "optimizer": {"name": "adamw", "learning_rate": 0.001, "weight_decay": 0.01},
        "training": {"max_optimizer_steps": 4, "micro_batch_size": 2, "world_size": 1,
                     "accumulation_steps": 2, "effective_global_batch": 4, "seed": 7,
                     "max_epochs": 3, "num_workers": 0, "ema_decay": 0.9,
                     "sampling": "uniform"},
        "data": {"dataroot": root, "version": "v1.0-mini", "train_split": "mini_train",
                 "val_split": "mini_val", "n_sweeps": 10,
                 "caches": {
                     "train": {"format": "t1.v2", "path": root + "/train.pkl",
                               "sidecar_path": root + "/train.meta.json",
                               "logical_sha256": H, "pickle_sha256": H, "sidecar_sha256": H},
                     "val": {"format": "t1.v2", "path": root + "/val.pkl",
                             "sidecar_path": root + "/val.meta.json",
                             "logical_sha256": H, "pickle_sha256": H, "sidecar_sha256": H}},
                 "zip_manifest": {"path": root + "/manifest.sqlite",
                                  "logical_sha256": H, "file_sha256": H}},
        "dependencies": {"torch": "2.11.0+cu128", "torch_build_sha256": H,
                         "torch_source_sha": "1" * 40, "spconv": None,
                         "spconv_build_sha256": None, "spconv_source_sha": None,
                         "spconv_source_state": None,
                         "cumm": None, "cumm_build_sha256": None,
                         "cumm_source_sha": None, "cumm_source_state": None},
        "execution": {"mode": "train_eval", "max_attempted_windows": 0,
                      "timing_warmup_successful_windows": 0, "loader_profile": None},
        "evaluation": {"timing": False, "checkpoint_weights": "raw"},
    }


def test_config_hash_is_order_stable_and_roundtrips(tmp_path):
    raw = valid_config(tmp_path)
    reverse = {k: raw[k] for k in reversed(raw)}
    a, b = resolve_config(raw), resolve_config(reverse)
    assert a.sha256 == b.sha256
    assert a.as_dict() == b.as_dict()
    assert a.data_identities["train_cache_format"] == "t1.v2"
    assert a.data_identities["val_cache_format"] == "t1.v2"
    run = a.to_run_config()
    assert run["resolved-config-sha256"] == a.sha256
    assert type(run["nuscenes-cache-identities"]) is dict
    assert all(type(value) is dict for value in run["nuscenes-cache-identities"].values())
    assert run["det-camera-arch"] == "swin_t_stride8"
    assert run["resolved-schema-version"] == "s09.v1"
    assert run["det-camera-activation-checkpoint"] is True
    assert run["det-lidar-arch"] == "none"
    assert run["det-second-normalization"] == "not_applicable"
    assert run["det-fusion-arch"] == "none"
    assert run["det-head-arch"] == "centerhead_multitask"
    assert run["det-sparse-conv-precision"] == "not_applicable"
    assert run["det-cbgs"] is False
    assert run["det-class-weights"] is None
    assert run["evaluation-timing"] is False
    assert run["evaluation-checkpoint-weights"] == "raw"
    assert run["execution-mode"] == "train_eval"
    assert run["readiness-max-attempted-windows"] == 0
    assert run["readiness-loader-profile"] is None
    assert run["readiness-operator-profile"] is None


def test_evaluation_timing_and_raw_ema_policy_are_hash_bound(tmp_path):
    raw = valid_config(tmp_path)
    plain = resolve_config(raw)
    raw["evaluation"]["timing"] = True
    timed = resolve_config(raw)
    assert timed.sha256 != plain.sha256
    assert timed.to_run_config()["evaluation-timing"] is True
    raw["evaluation"]["checkpoint_weights"] = "ema"
    ema = resolve_config(raw)
    assert ema.sha256 != timed.sha256
    raw["training"]["ema_decay"] = None
    with pytest.raises(ConfigError, match="requires training.ema_decay"):
        resolve_config(raw)


def readiness_config(tmp_path):
    raw = valid_config(tmp_path)
    raw["training"].update(
        accumulation_steps=1,
        effective_global_batch=2,
        num_workers=2,
    )
    raw["execution"] = {
        "mode": "readiness",
        "max_attempted_windows": 6,
        "timing_warmup_successful_windows": 1,
        "loader_profile": {
            "workers": [0, 2, 4],
            "repeats": 2,
            "determinism_batches": 2,
            "warmup_batches": 1,
            "measured_batches": 3,
        },
    }
    return raw


def readiness_v2_config(tmp_path):
    raw = readiness_config(tmp_path)
    raw["schema_version"] = "s09.v2"
    raw["model"]["camera_activation_checkpoint"] = False
    raw["execution"]["timing_warmup_successful_windows"] = 3
    raw["execution"]["operator_profile"] = {
        "wait_attempted_windows": 0,
        "warmup_attempted_windows": 1,
        "active_attempted_windows": 2,
        "record_shapes": True,
        "profile_memory": True,
        "row_limit": 100,
    }
    return raw


def s10_second_config(tmp_path, normalization="group_norm"):
    raw = readiness_v2_config(tmp_path)
    raw["schema_version"] = "s10.v1"
    raw["model"].update(
        mode="lidar_only",
        camera_arch="none",
        camera_pretrained=None,
        camera_activation_checkpoint=False,
        lidar_arch="second_075",
        fusion_arch="none",
        second_normalization=normalization,
    )
    raw["precision"] = "fp16"
    raw["training"]["grad_scaler_init_scale"] = 32
    raw["sparse_conv_precision"] = "fp32"
    raw["dependencies"].update(
        spconv="2.3.8",
        spconv_build_sha256=H,
        spconv_source_sha="2" * 40,
        spconv_source_state=build_source_state([]),
        cumm="0.7.13",
        cumm_build_sha256=H,
        cumm_source_sha="3" * 40,
        cumm_source_state=build_source_state([]),
    )
    return raw


def test_readiness_execution_contract_is_hash_bound_and_roundtrips(tmp_path):
    raw = readiness_config(tmp_path)
    resolved = resolve_config(raw)
    run = resolved.to_run_config()
    assert resolved.execution_mode == "readiness"
    assert run["execution-mode"] == "readiness"
    assert run["readiness-max-attempted-windows"] == 6
    assert run["readiness-timing-warmup-successful-windows"] == 1
    assert run["readiness-loader-profile"] == raw["execution"]["loader_profile"]

    changed = copy.deepcopy(raw)
    changed["execution"]["loader_profile"]["measured_batches"] = 4
    assert resolve_config(changed).sha256 != resolved.sha256


def test_s09_v2_checkpoint_and_operator_profile_are_explicit_and_hash_bound(tmp_path):
    raw = readiness_v2_config(tmp_path)
    resolved = resolve_config(raw)
    run = resolved.to_run_config()
    assert run["resolved-schema-version"] == "s09.v2"
    assert run["det-camera-activation-checkpoint"] is False
    assert run["readiness-operator-profile"] == raw["execution"]["operator_profile"]

    changed = copy.deepcopy(raw)
    changed["model"]["camera_activation_checkpoint"] = True
    assert resolve_config(changed).sha256 != resolved.sha256


def test_s10_second_normalization_is_explicit_hash_bound_and_roundtrips(tmp_path):
    gn = resolve_config(s10_second_config(tmp_path, "group_norm"))
    bn = resolve_config(s10_second_config(tmp_path, "batch_norm_1d"))
    assert gn.sha256 != bn.sha256
    assert gn.to_run_config()["det-second-normalization"] == "group_norm"
    assert bn.to_run_config()["det-second-normalization"] == "batch_norm_1d"
    assert bn.to_run_config()["grad-scaler-init-scale"] == 32.0


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda c: c["model"].pop("second_normalization"), "model keys invalid"),
        (lambda c: c["model"].update(second_normalization="bn"), "expected one of"),
        (lambda c: c["model"].update(lidar_arch="pillar_020"), "not_applicable"),
        (lambda c: c["training"].pop("grad_scaler_init_scale"), "training keys invalid"),
        (lambda c: c["training"].update(grad_scaler_init_scale=48), "power of two"),
    ],
)
def test_s10_second_normalization_rejects_missing_alias_or_wrong_route(
    tmp_path, mutation, message,
):
    raw = s10_second_config(tmp_path)
    mutation(raw)
    with pytest.raises(ConfigError, match=message):
        resolve_config(raw)
    changed = copy.deepcopy(raw)
    changed["execution"]["operator_profile"]["active_attempted_windows"] = 1
    assert resolve_config(changed).sha256 != resolved.sha256


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda c: c["model"].pop("camera_activation_checkpoint"), "model keys invalid"),
        (lambda c: c["model"].update(camera_activation_checkpoint=1), "must be boolean"),
        (lambda c: c["execution"].pop("operator_profile"), "execution keys invalid"),
        (lambda c: c["execution"]["operator_profile"].update(record_shapes=1),
         "record_shapes must be boolean"),
        (lambda c: c["execution"]["operator_profile"].update(row_limit=0),
         "row_limit must be an integer"),
        (lambda c: c["execution"]["operator_profile"].update(
            wait_attempted_windows=3, warmup_attempted_windows=2,
            active_attempted_windows=2,
        ), "schedule must finish"),
        (lambda c: c["execution"]["operator_profile"].update(
            wait_attempted_windows=1, warmup_attempted_windows=1,
            active_attempted_windows=2,
        ), "must be >= the complete operator-profile schedule"),
    ],
)
def test_s09_v2_rejects_checkpoint_or_profiler_drift(tmp_path, mutation, message):
    raw = readiness_v2_config(tmp_path)
    mutation(raw)
    with pytest.raises(ConfigError, match=message):
        resolve_config(raw)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda c: c["execution"].update(max_attempted_windows=3),
         "must be >= training.max_optimizer_steps"),
        (lambda c: c["execution"].update(timing_warmup_successful_windows=4),
         "warmup_successful_windows must be below"),
        (lambda c: c["training"].update(accumulation_steps=2, effective_global_batch=4),
         "requires world_size=1 and accumulation_steps=1"),
        (lambda c: c["evaluation"].update(timing=True),
         "requires evaluation.timing=false"),
        (lambda c: c["execution"]["loader_profile"].update(workers=[0, 2, 2]),
         "workers must be unique"),
        (lambda c: c["execution"]["loader_profile"].update(workers=[0, 4]),
         "training.num_workers must be one"),
        (lambda c: c["execution"]["loader_profile"].update(repeats=0),
         "must be an integer >= 1"),
    ],
)
def test_readiness_execution_contract_rejects_drift(tmp_path, mutation, message):
    raw = readiness_config(tmp_path)
    mutation(raw)
    with pytest.raises(ConfigError, match=message):
        resolve_config(raw)


def test_train_eval_rejects_readiness_only_fields(tmp_path):
    raw = valid_config(tmp_path)
    raw["execution"]["max_attempted_windows"] = 1
    with pytest.raises(ConfigError, match="requires max_attempted_windows=0"):
        resolve_config(raw)


@pytest.mark.parametrize("mutation", [
    lambda c: c.update(extra=1),
    lambda c: c["model"].update(mode="camera"),
    lambda c: c["data"]["caches"]["train"].update(format="t1.v1"),
    lambda c: c["data"]["caches"]["train"].pop("logical_sha256"),
    lambda c: c["training"].update(effective_global_batch=3),
])
def test_config_rejects_unknown_alias_legacy_missing_and_batch_drift(tmp_path, mutation):
    raw = valid_config(tmp_path); mutation(raw)
    with pytest.raises(ConfigError):
        resolve_config(raw)


def test_lidar_pins_exact_spconv(tmp_path):
    raw = valid_config(tmp_path)
    raw["model"].update(
        mode="lidar_only", camera_arch="none", camera_pretrained=None, lidar_arch="second_075"
    )
    raw["sparse_conv_precision"] = "fp32"
    with pytest.raises(ConfigError, match="spconv"):
        resolve_config(raw)
    raw["dependencies"]["spconv"] = "2.3.8"
    raw["dependencies"]["spconv_build_sha256"] = H
    raw["dependencies"]["spconv_source_sha"] = "2" * 40
    raw["dependencies"]["spconv_source_state"] = build_source_state([])
    raw["dependencies"]["cumm"] = "0.7.13"
    raw["dependencies"]["cumm_build_sha256"] = H
    raw["dependencies"]["cumm_source_sha"] = "3" * 40
    raw["dependencies"]["cumm_source_state"] = build_source_state([])
    resolved = resolve_config(raw)
    assert resolved.model_mode == "lidar_only"
    assert resolved.to_run_config()["dependency-spconv-source-state"] == build_source_state([])


def test_lidar_source_state_is_hash_bound_and_schema_checked(tmp_path):
    raw = valid_config(tmp_path)
    raw["model"].update(
        mode="lidar_only", camera_arch="none", camera_pretrained=None,
        lidar_arch="second_075",
    )
    raw["sparse_conv_precision"] = "fp32"
    clean = build_source_state([])
    raw["dependencies"].update(
        spconv="2.3.8", spconv_build_sha256=H, spconv_source_sha="2" * 40,
        spconv_source_state=clean,
        cumm="0.7.13", cumm_build_sha256=H, cumm_source_sha="3" * 40,
        cumm_source_state=clean,
    )
    baseline = resolve_config(raw)
    raw["dependencies"]["spconv_source_state"] = build_source_state([{
        "status": " M", "path": "pyproject.toml", "sha256": "4" * 64,
    }])
    patched = resolve_config(raw)
    assert patched.sha256 != baseline.sha256

    raw["dependencies"]["spconv_source_state"]["sha256"] = "5" * 64
    with pytest.raises(ConfigError, match="source_state is invalid"):
        resolve_config(raw)


def test_pillar_does_not_claim_spconv_and_cbgs_is_fusion_only(tmp_path):
    raw = valid_config(tmp_path)
    raw["model"].update(
        mode="lidar_only", camera_arch="none", camera_pretrained=None, lidar_arch="pillar_020"
    )
    assert resolve_config(raw).model_mode == "lidar_only"
    raw["training"]["sampling"] = "cbgs"
    with pytest.raises(ConfigError, match="F-CBGS"):
        resolve_config(raw)


def test_fusion_cbgs_is_hash_bound_and_disables_loss_weights(tmp_path):
    raw = valid_config(tmp_path)
    raw["model"].update(
        mode="fusion", lidar_arch="second_075", fusion_arch="conv_fuser_256"
    )
    raw["sparse_conv_precision"] = "fp32"
    raw["dependencies"].update(
        spconv="2.3.8", spconv_build_sha256=H, spconv_source_sha="2" * 40,
        spconv_source_state=build_source_state([]),
        cumm="0.7.13", cumm_build_sha256=H, cumm_source_sha="3" * 40,
        cumm_source_state=build_source_state([]),
    )
    uniform = resolve_config(raw)
    raw["training"]["sampling"] = "cbgs"
    cbgs = resolve_config(raw)
    assert cbgs.sha256 != uniform.sha256
    run = cbgs.to_run_config()
    assert run["det-cbgs"] is True
    assert run["det-cbgs-thresh"] == 0.5
    assert run["det-cbgs-max-repeat"] == 4.0
    assert run["det-class-weights"] is None
    assert run["det-reg-class-weights"] is None


def test_physical_identities_reject_drift(tmp_path):
    raw = valid_config(tmp_path)
    payloads = {
        tmp_path / "train.pkl": b"train-pickle",
        tmp_path / "train.meta.json": b"train-sidecar",
        tmp_path / "val.pkl": b"val-pickle",
        tmp_path / "val.meta.json": b"val-sidecar",
        tmp_path / "manifest.sqlite": b"manifest",
    }
    for path, payload in payloads.items():
        path.write_bytes(payload)
    for role in ("train", "val"):
        raw["data"]["caches"][role]["pickle_sha256"] = hashlib.sha256(
            payloads[tmp_path / f"{role}.pkl"]).hexdigest()
        raw["data"]["caches"][role]["sidecar_sha256"] = hashlib.sha256(
            payloads[tmp_path / f"{role}.meta.json"]).hexdigest()
    raw["data"]["zip_manifest"]["file_sha256"] = hashlib.sha256(b"manifest").hexdigest()
    cfg = resolve_config(raw); verify_physical_data_identities(cfg)
    (tmp_path / "train.pkl").write_bytes(b"drift")
    with pytest.raises(ConfigError, match="identity drift"):
        verify_physical_data_identities(cfg)
