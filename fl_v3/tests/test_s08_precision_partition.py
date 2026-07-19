from __future__ import annotations

import copy

import pytest

from fl_v3.config import ConfigError, resolve_config, validate_precision_partition
from fl_v3.source_identity import build_source_state
from fl_v3.training.tasks import _det_config_from_run
from test_s06_resolved_config import H, valid_config


@pytest.mark.parametrize(
    ("precision", "lidar_arch", "partition"),
    [
        ("fp32", "none", "not_applicable"),
        ("fp16", "none", "not_applicable"),
        ("fp32", "pillar_020", "not_applicable"),
        ("fp16", "pillar_020", "not_applicable"),
        ("fp32", "second_075", "fp32"),
        ("fp16", "second_075", "fp32"),
        ("fp16", "second_075", "fp16"),
    ],
)
def test_precision_partition_legal_matrix(precision, lidar_arch, partition):
    assert validate_precision_partition(precision, lidar_arch, partition) == partition


@pytest.mark.parametrize(
    ("precision", "lidar_arch", "partition"),
    [
        ("fp32", "none", "fp32"),
        ("fp16", "pillar_020", "fp16"),
        ("fp32", "second_075", "fp16"),
        ("fp16", "second_075", "not_applicable"),
        ("FP16", "second_075", "fp16"),
        ("fp16", "second_075", "FP16"),
    ],
)
def test_precision_partition_rejects_illegal_or_aliased_values(
    precision, lidar_arch, partition,
):
    with pytest.raises(ConfigError):
        validate_precision_partition(precision, lidar_arch, partition)


def _second_config(tmp_path, *, precision="fp16", partition="fp16"):
    raw = valid_config(tmp_path)
    raw["model"].update(
        mode="lidar_only",
        camera_arch="none",
        camera_pretrained=None,
        lidar_arch="second_075",
    )
    raw["precision"] = precision
    raw["sparse_conv_precision"] = partition
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


def test_partition_changes_resolved_identity_and_production_constructor(tmp_path):
    full = resolve_config(_second_config(tmp_path, partition="fp16"))
    island = resolve_config(_second_config(tmp_path, partition="fp32"))
    assert full.sha256 != island.sha256
    assert full.as_dict()["sparse_conv_precision"] == "fp16"
    assert island.as_dict()["sparse_conv_precision"] == "fp32"
    assert _det_config_from_run(full.to_run_config()).sparse_conv_fp16 is True
    assert _det_config_from_run(island.to_run_config()).sparse_conv_fp16 is False
    assert _det_config_from_run(island.to_run_config()).second_normalization == "group_norm"


def test_s10_batch_norm_reaches_production_detector_constructor(tmp_path):
    raw = _second_config(tmp_path, partition="fp32")
    raw["schema_version"] = "s10.v1"
    raw["model"]["camera_activation_checkpoint"] = False
    raw["model"]["second_normalization"] = "batch_norm_1d"
    config = _det_config_from_run(resolve_config(raw).to_run_config())
    assert config.second_normalization == "batch_norm_1d"


def test_strict_production_rejects_missing_partition_and_legacy_boolean(tmp_path):
    run = resolve_config(_second_config(tmp_path)).to_run_config()
    for field in ("precision", "det-sparse-conv-precision"):
        missing = copy.deepcopy(run)
        missing.pop(field)
        with pytest.raises(ValueError, match="mapping fields missing"):
            _det_config_from_run(missing)

    legacy = copy.deepcopy(run)
    legacy["det-sparse-conv-fp16"] = True
    with pytest.raises(ValueError, match="legacy det-sparse-conv-fp16"):
        _det_config_from_run(legacy)


def test_legacy_schema_and_missing_partition_are_refused(tmp_path):
    raw = valid_config(tmp_path)
    raw["schema_version"] = "s06.v1"
    with pytest.raises(ConfigError, match="exactly 's09.v1', 's09.v2', or 's10.v1'"):
        resolve_config(raw)
    raw = valid_config(tmp_path)
    raw.pop("sparse_conv_precision")
    with pytest.raises(ConfigError, match="missing"):
        resolve_config(raw)
