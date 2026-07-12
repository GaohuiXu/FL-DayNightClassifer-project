from __future__ import annotations

import copy
import hashlib
import json

import pytest

from fl_v3.config import ConfigError, resolve_config, verify_physical_data_identities


H = "a" * 64


def valid_config(tmp_path=None):
    root = str(tmp_path or "/synthetic")
    return {
        "schema_version": "s06.v1",
        "model": {"mode": "camera_only", "camera_arch": "swin_t_stride8",
                  "lidar_arch": "none", "fusion_arch": "none",
                  "head_arch": "centerhead_multitask"},
        "precision": "fp32",
        "optimizer": {"name": "adamw", "learning_rate": 0.001, "weight_decay": 0.01},
        "training": {"max_optimizer_steps": 4, "micro_batch_size": 2, "world_size": 1,
                     "accumulation_steps": 2, "effective_global_batch": 4, "seed": 7,
                     "max_epochs": 3, "num_workers": 0, "ema_decay": 0.9},
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
        "dependencies": {"torch": "2.11.0+cu128", "spconv": None,
                         "spconv_source_sha": None, "cumm": None, "cumm_source_sha": None},
        "evaluation": {"timing": False},
    }


def test_config_hash_is_order_stable_and_roundtrips(tmp_path):
    raw = valid_config(tmp_path)
    reverse = {k: raw[k] for k in reversed(raw)}
    a, b = resolve_config(raw), resolve_config(reverse)
    assert a.sha256 == b.sha256
    assert a.as_dict() == b.as_dict()
    assert a.data_identities["train_cache_format"] == "t1.v2"
    assert a.data_identities["val_cache_format"] == "t1.v2"
    assert a.to_run_config()["resolved-config-sha256"] == a.sha256


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
    raw["model"].update(mode="lidar_only", camera_arch="none", lidar_arch="second_075")
    with pytest.raises(ConfigError, match="spconv"):
        resolve_config(raw)
    raw["dependencies"]["spconv"] = "2.3.8"
    raw["dependencies"]["spconv_source_sha"] = "2" * 40
    raw["dependencies"]["cumm"] = "0.7.13"
    raw["dependencies"]["cumm_source_sha"] = "3" * 40
    assert resolve_config(raw).model_mode == "lidar_only"


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
