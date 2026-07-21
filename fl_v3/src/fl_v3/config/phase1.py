"""S10 Phase-I reference recipe schema.

This module is deliberately independent of MMDetection/MMCV.  It records the
mechanically resolved values inherited by the two owner-frozen Phase-I recipes
from MIT BEVFusion commit ``326653d`` and validates that a JSON recipe contains
every value explicitly.  There are no architecture, optimizer, scheduler, data,
or augmentation defaults on the production path.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping

from fl_v3.source_identity import validate_source_state


PHASE1_SCHEMA = "s10.phase1.v1"
PHASE1_SCHEMA_V2 = "s10.phase1.v2"
PHASE1_SCHEMA_V3 = "s10.phase1.v3"
PHASE1_SCHEMAS = frozenset({PHASE1_SCHEMA, PHASE1_SCHEMA_V2, PHASE1_SCHEMA_V3})
PHASE1_ENVELOPE_B_SCHEMAS = frozenset({PHASE1_SCHEMA_V2, PHASE1_SCHEMA_V3})
PHASE1_PLAN_SHA = "260750a76548208f62c384b0e0547744b619244c"
PHASE1_REQUEST_COMMIT = "e321aed749fd859c809199d52c30b2771dbef8b3"
PHASE1_O150_AMENDMENT_COMMIT = "2a26c63b61022e2947043a9ffd0538d537c51fb9"
PHASE1_IP_G2_EVIDENCE_COMMIT = "6ec7fb6d067259ac61ecaed89481e7e2562c3a2d"
PHASE1_IP_E4_EVIDENCE_COMMIT = "48fa78a60b3308c407fbc16b64dde188216f87e4"
MIT_BEVFUSION_COMMIT = "326653dc06e0938edf1aae7d01efcd158ba83de5"

CAMERA_COMPILE_FORWARD_MODULES = (
    "camera_backbone",
    "camera_neck",
    "decoder_backbone",
    "decoder_neck",
    "head",
)

REFERENCE_OBJECT_CLASSES = (
    "car",
    "truck",
    "construction_vehicle",
    "bus",
    "trailer",
    "barrier",
    "motorcycle",
    "bicycle",
    "pedestrian",
    "traffic_cone",
)

DEVKIT_DETECTION_NAMES = (
    "car",
    "truck",
    "bus",
    "trailer",
    "construction_vehicle",
    "pedestrian",
    "motorcycle",
    "bicycle",
    "traffic_cone",
    "barrier",
)

DEVKIT_TO_REFERENCE_LABEL = tuple(
    REFERENCE_OBJECT_CLASSES.index(name) for name in DEVKIT_DETECTION_NAMES
)
REFERENCE_TO_DEVKIT_LABEL = tuple(
    DEVKIT_DETECTION_NAMES.index(name) for name in REFERENCE_OBJECT_CLASSES
)

CAMERA_TASKS = (
    ("car",),
    ("truck", "construction_vehicle"),
    ("bus", "trailer"),
    ("barrier",),
    ("motorcycle", "bicycle"),
    ("pedestrian", "traffic_cone"),
)

REFERENCE_SAMPLE_GROUPS = {
    "car": 2,
    "truck": 3,
    "construction_vehicle": 7,
    "bus": 4,
    "trailer": 6,
    "barrier": 2,
    "motorcycle": 6,
    "bicycle": 6,
    "pedestrian": 2,
    "traffic_cone": 2,
}

FROZEN_TAXONOMY: dict[str, Any] = {
    "reference_object_classes": list(REFERENCE_OBJECT_CLASSES),
    "devkit_detection_names": list(DEVKIT_DETECTION_NAMES),
    "devkit_to_reference_label": list(DEVKIT_TO_REFERENCE_LABEL),
    "reference_to_devkit_label": list(REFERENCE_TO_DEVKIT_LABEL),
    "camera_tasks": [list(task) for task in CAMERA_TASKS],
    "mapping_policy": "name_based_bijection",
}


FROZEN_CAMERA_MODEL: dict[str, Any] = {
    "architecture": "mit_bevfusion_camera_swint_256x704",
    "input": {
        "camera_count": 6,
        "image_height": 256,
        "image_width": 704,
        "layout": "BNCHW",
    },
    "backbone": {
        "type": "SwinTransformer",
        "embed_dims": 96,
        "depths": [2, 2, 6, 2],
        "num_heads": [3, 6, 12, 24],
        "window_size": 7,
        "mlp_ratio": 4.0,
        "qkv_bias": True,
        "qk_scale": None,
        "drop_rate": 0.0,
        "attn_drop_rate": 0.0,
        "drop_path_rate": 0.2,
        "patch_norm": True,
        "out_indices": [1, 2, 3],
        "out_channels": [192, 384, 768],
        "out_strides": [8, 16, 32],
        "output_layer_norm": True,
        "activation_checkpoint": False,
        "convert_weights": True,
    },
    "neck": {
        "type": "GeneralizedLSSFPN",
        "in_channels": [192, 384, 768],
        "out_channels": 256,
        "start_level": 0,
        "num_outs": 3,
        "normalization": {"type": "BatchNorm2d", "eps": 1e-5, "momentum": 0.1},
        "activation": "ReLU",
        "upsample": {"mode": "bilinear", "align_corners": False},
    },
    "view_transform": {
        "type": "LSSTransform",
        "in_channels": 256,
        "out_channels": 80,
        "feature_size": [32, 88],
        "xbound": [-51.2, 51.2, 0.4],
        "ybound": [-51.2, 51.2, 0.4],
        "zbound": [-10.0, 10.0, 20.0],
        "dbound": [1.0, 60.0, 0.5],
        "downsample": 2,
        "pool_backend": "optimized_cuda",
        "pool_fallback": "sorted_segment_sum",
    },
    "decoder": {
        "backbone": {
            "type": "GeneralizedResNet",
            "in_channels": 80,
            "blocks": [[2, 128, 2], [2, 256, 2], [2, 512, 1]],
            "normalization": {"type": "BatchNorm2d", "eps": 1e-5, "momentum": 0.1},
            "basic_block": "two_conv_3x3",
        },
        "neck": {
            "type": "LSSFPN",
            "in_indices": [-1, 0],
            "in_channels": [512, 128],
            "out_channels": 256,
            "scale_factor": 2,
            "normalization": {"type": "BatchNorm2d", "eps": 1e-5, "momentum": 0.1},
            "upsample": {"mode": "bilinear", "align_corners": True},
        },
    },
    "head": {
        "type": "CenterHead",
        "in_channels": 256,
        "tasks": [list(task) for task in CAMERA_TASKS],
        "shared_conv_channels": 64,
        "common_heads": {
            "reg": [2, 2],
            "height": [1, 2],
            "dim": [3, 2],
            "rot": [2, 2],
            "vel": [2, 2],
        },
        "heatmap_convs": 2,
        "final_kernel": 3,
        "heatmap_init_bias": -2.19,
        "normalization": {"type": "BatchNorm2d", "eps": 1e-5, "momentum": 0.1},
        "train": {
            "point_cloud_range": [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
            "grid_size": [1024, 1024, 1],
            "voxel_size": [0.1, 0.1, 0.2],
            "out_size_factor": 8,
            "dense_reg": 1,
            "gaussian_overlap": 0.1,
            "max_objects": 500,
            "min_radius": 2,
            "code_weights": [1.0] * 8 + [0.2, 0.2],
            "normalize_dimensions": True,
            "heatmap_loss": "GaussianFocalLoss_mean",
            "bbox_loss": "L1Loss_mean_x0.25",
        },
        "test": {
            "post_center_range": [-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            "max_per_image": 500,
            "score_threshold": 0.1,
            "nms_type": ["circle", "rotate", "rotate", "circle", "rotate", "rotate"],
            "nms_scale": [[1.0], [1.0, 1.0], [1.0, 1.0], [1.0], [1.0, 1.0], [2.5, 4.0]],
            "min_radius": [4.0, 12.0, 10.0, 1.0, 0.85, 0.175],
            "rotate_nms_threshold": 0.2,
            "pre_max_size": 1000,
            "post_max_size": 83,
        },
    },
}

# Keep the v1 graph immutable for exact Envelope-A replay. O-150 changes only
# production dispatch: the qualified PyTorch segment-reduce path becomes primary
# while the CUDA extension remains an explicit, unpromoted option.
FROZEN_CAMERA_MODEL_V2: dict[str, Any] = json.loads(json.dumps(FROZEN_CAMERA_MODEL))
_camera_v2_pool = FROZEN_CAMERA_MODEL_V2["view_transform"]
_camera_v2_pool["pool_backend"] = "pytorch_sorted_segment_reduce"
_camera_v2_pool.pop("pool_fallback")
_camera_v2_pool["pool_optional_backend"] = "optimized_cuda_unpromoted"


FROZEN_LIDAR_MODEL: dict[str, Any] = {
    "architecture": "mit_bevfusion_voxelnet_0p075_transfusion",
    "input": {
        "point_cloud_range": [-54.0, -54.0, -5.0, 54.0, 54.0, 3.0],
        "voxel_size": [0.075, 0.075, 0.2],
        "load_dim": 5,
        "use_dim": 5,
        "model_features": ["x", "y", "z", "intensity", "time_lag"],
        "source_ring_preserved": True,
        "max_points_per_voxel": 10,
        "max_voxels_train": 120000,
        "max_voxels_eval": 160000,
    },
    "backbone": {
        "type": "SparseEncoder",
        "in_channels": 5,
        "sparse_shape_xyz": [1440, 1440, 41],
        "output_channels": 128,
        "order": ["conv", "norm", "act"],
        "encoder_channels": [[16, 16, 32], [32, 32, 64], [64, 64, 128], [128, 128]],
        "encoder_paddings": [[0, 0, 1], [0, 0, 1], [0, 0, [1, 1, 0]], [0, 0]],
        "block_type": "basicblock",
        "normalization": {"type": "BatchNorm1d", "eps": 0.001, "momentum": 0.01},
        "dense_output_channels": 256,
        "dense_output_hw": [180, 180],
    },
    "neck": None,
    "view_transform": None,
    "decoder": {
        "backbone": {
            "type": "SECOND",
            "in_channels": 256,
            "out_channels": [128, 256],
            "layer_nums": [5, 5],
            "layer_strides": [1, 2],
            "normalization": {"type": "BatchNorm2d", "eps": 0.001, "momentum": 0.01},
            "conv_bias": False,
        },
        "neck": {
            "type": "SECONDFPN",
            "in_channels": [128, 256],
            "out_channels": [256, 256],
            "upsample_strides": [1, 2],
            "normalization": {"type": "BatchNorm2d", "eps": 0.001, "momentum": 0.01},
            "upsample": "ConvTranspose2d",
            "conv_for_stride_one": True,
            "output_channels": 512,
        },
    },
    "head": {
        "type": "TransFusionHead",
        "in_channels": 512,
        "class_names": list(REFERENCE_OBJECT_CLASSES),
        "num_proposals": 200,
        "auxiliary": True,
        "hidden_channels": 128,
        "num_classes": 10,
        "num_decoder_layers": 1,
        "num_attention_heads": 8,
        "nms_kernel_size": 3,
        "ffn_channels": 256,
        "dropout": 0.1,
        "bn_momentum": 0.1,
        "activation": "ReLU",
        "common_heads": {
            "center": [2, 2],
            "height": [1, 2],
            "dim": [3, 2],
            "rot": [2, 2],
            "vel": [2, 2],
        },
        "train": {
            "point_cloud_range": [-54.0, -54.0, -5.0, 54.0, 54.0, 3.0],
            "grid_size": [1440, 1440, 41],
            "voxel_size": [0.075, 0.075, 0.2],
            "out_size_factor": 8,
            "gaussian_overlap": 0.1,
            "min_radius": 2,
            "positive_weight": -1.0,
            "code_weights": [1.0] * 8 + [0.2, 0.2],
            "assigner": {
                "type": "HungarianAssigner3D",
                "focal_cost": {"gamma": 2.0, "alpha": 0.25, "weight": 0.15},
                "bev_l1_cost_weight": 0.25,
                "iou3d_cost_weight": 0.25,
            },
            "heatmap_loss": "GaussianFocalLoss_mean_x1.0",
            "classification_loss": "FocalLoss_sigmoid_gamma2_alpha0.25_mean_x1.0",
            "bbox_loss": "L1Loss_mean_x0.25",
        },
        "test": {
            "post_center_range": [-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            "score_threshold": 0.0,
            "nms_type": None,
            "code_size": 10,
        },
    },
}


FROZEN_CAMERA_AUGMENTATION: dict[str, Any] = {
    "image": {
        "output_size": [256, 704],
        "train_resize": [0.38, 0.55],
        "eval_resize": 0.48,
        "bottom_crop_percent": [0.0, 0.0],
        "train_rotation_degrees": [-5.4, 5.4],
        "eval_rotation_degrees": [0.0, 0.0],
        "train_horizontal_flip_probability": 0.5,
        "eval_horizontal_flip_probability": 0.0,
        "interpolation": "bilinear",
        "normalization_mean": [0.485, 0.456, 0.406],
        "normalization_std": [0.229, 0.224, 0.225],
        "gridmask_probability": 0.0,
    },
    "scene_3d": {
        "train_scale": [0.95, 1.05],
        "train_yaw_radians": [-0.3925, 0.3925],
        "train_translation_limit": 0.0,
        "train_horizontal_flip_probability": 0.5,
        "train_vertical_flip_probability": 0.5,
        "eval_scale": [1.0, 1.0],
        "eval_yaw_radians": [0.0, 0.0],
        "eval_translation_limit": 0.0,
    },
    "point_shuffle": False,
}


FROZEN_LIDAR_AUGMENTATION: dict[str, Any] = {
    "image": None,
    "scene_3d": {
        "train_scale": [0.9, 1.1],
        "train_yaw_radians": [-0.78539816, 0.78539816],
        "train_translation_limit": 0.5,
        "train_horizontal_flip_probability": 0.5,
        "train_vertical_flip_probability": 0.5,
        "eval_scale": [1.0, 1.0],
        "eval_yaw_radians": [0.0, 0.0],
        "eval_translation_limit": 0.0,
    },
    "point_shuffle": True,
}


class Phase1ConfigError(ValueError):
    """The Phase-I recipe is partial, unknown, or contradicts the frozen plan."""


def _keys(value: Any, expected: set[str], where: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise Phase1ConfigError(f"{where} must be an object")
    got = set(value)
    if got != expected:
        raise Phase1ConfigError(
            f"{where} keys invalid: missing={sorted(expected-got)}, unknown={sorted(got-expected)}"
        )
    return value


def _sha(value: Any, where: str, *, allow_none: bool = False) -> str | None:
    if value is None and allow_none:
        return None
    if not isinstance(value, str) or len(value) != 64:
        raise Phase1ConfigError(f"{where} must be a lowercase SHA-256")
    if any(ch not in "0123456789abcdef" for ch in value):
        raise Phase1ConfigError(f"{where} must be a lowercase SHA-256")
    return value


def _path(value: Any, where: str, *, allow_none: bool = False) -> str | None:
    if value is None and allow_none:
        return None
    if not isinstance(value, str) or not value or "\x00" in value:
        raise Phase1ConfigError(f"{where} must be an explicit non-empty path")
    return value


def _finite(value: Any, where: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise Phase1ConfigError(f"{where} must be numeric")
    value = float(value)
    if not math.isfinite(value) or (value <= 0.0 if positive else value < 0.0):
        raise Phase1ConfigError(f"{where} must be finite and {'> 0' if positive else '>= 0'}")
    return value


def _positive_int(value: Any, where: str, *, allow_zero: bool = False) -> int:
    minimum = 0 if allow_zero else 1
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise Phase1ConfigError(f"{where} must be an integer >= {minimum}")
    return value


def _same(actual: Any, expected: Any, where: str) -> None:
    if actual != expected:
        raise Phase1ConfigError(
            f"{where} differs from the owner-frozen resolved reference recipe"
        )


def _validate_contract(raw: Any, schema_version: str) -> tuple[dict[str, Any], str]:
    keys = {
        "candidate_id", "branch", "plan_sha", "request_commit",
        "reference_repository", "reference_commit", "reference_license",
        "lifecycle", "scientific_candidate_count", "seed",
    }
    if schema_version in PHASE1_ENVELOPE_B_SCHEMAS:
        keys.update({"amendment_decision", "amendment_commit"})
    if schema_version == PHASE1_SCHEMA_V3:
        keys.update({"throughput_decision", "throughput_evidence_commit"})
    contract = _keys(
        raw,
        keys,
        "contract",
    )
    branch = contract["branch"]
    if branch not in {"camera", "lidar"}:
        raise Phase1ConfigError("contract.branch must be camera or lidar")
    _same(contract["candidate_id"], f"phase1_{branch}_primary", "contract.candidate_id")
    _same(contract["plan_sha"], PHASE1_PLAN_SHA, "contract.plan_sha")
    _same(contract["request_commit"], PHASE1_REQUEST_COMMIT, "contract.request_commit")
    if schema_version in PHASE1_ENVELOPE_B_SCHEMAS:
        _same(contract["amendment_decision"], "O-150", "contract.amendment_decision")
        _same(
            contract["amendment_commit"],
            PHASE1_O150_AMENDMENT_COMMIT,
            "contract.amendment_commit",
        )
    if schema_version == PHASE1_SCHEMA_V3:
        _same(branch, "camera", "contract.branch")
        decision = contract["throughput_decision"]
        evidence_commits = {
            "IP-G2": PHASE1_IP_G2_EVIDENCE_COMMIT,
            "IP-E4": PHASE1_IP_E4_EVIDENCE_COMMIT,
        }
        if not isinstance(decision, str) or decision not in evidence_commits:
            raise Phase1ConfigError("contract.throughput_decision is unknown")
        _same(
            contract["throughput_evidence_commit"],
            evidence_commits[decision],
            "contract.throughput_evidence_commit",
        )
    _same(contract["reference_repository"], "mit-han-lab/bevfusion", "contract.reference_repository")
    _same(contract["reference_commit"], MIT_BEVFUSION_COMMIT, "contract.reference_commit")
    _same(contract["reference_license"], "Apache-2.0", "contract.reference_license")
    if contract["lifecycle"] not in {
        "envelope_a_implementation", "envelope_a_qualified", "envelope_b_ready"
    }:
        raise Phase1ConfigError("contract.lifecycle is unknown")
    _same(contract["scientific_candidate_count"], 2, "contract.scientific_candidate_count")
    _same(contract["seed"], 0, "contract.seed")
    return contract, branch


def _validate_initialization(raw: Any, branch: str, lifecycle: str) -> None:
    init = _keys(
        raw,
        {
            "kind", "status", "source_url", "license", "quarantine_path",
            "final_path", "physical_sha256", "mapping_report_path",
            "mapping_report_sha256", "initialization_state_sha256", "scratch",
        },
        "initialization",
    )
    if branch == "camera":
        _same(init["kind"], "imagenet1k_swin_t", "initialization.kind")
        _same(
            init["source_url"],
            "https://github.com/SwinTransformer/storage/releases/download/v1.0.0/"
            "swin_tiny_patch4_window7_224.pth",
            "initialization.source_url",
        )
        _same(init["license"], "MIT", "initialization.license")
        _path(init["quarantine_path"], "initialization.quarantine_path")
        _path(init["final_path"], "initialization.final_path")
        _path(init["mapping_report_path"], "initialization.mapping_report_path")
        for key in ("physical_sha256", "mapping_report_sha256", "initialization_state_sha256"):
            _sha(init[key], f"initialization.{key}", allow_none=True)
        _same(init["scratch"], None, "initialization.scratch")
        if init["status"] not in {"pending_acquisition", "accepted"}:
            raise Phase1ConfigError("camera initialization.status is unknown")
        if init["status"] == "accepted" and any(
            init[key] is None
            for key in ("physical_sha256", "mapping_report_sha256", "initialization_state_sha256")
        ):
            raise Phase1ConfigError("accepted camera initialization requires all three identities")
        if lifecycle != "envelope_a_implementation" and init["status"] != "accepted":
            raise Phase1ConfigError("qualified/Envelope-B Camera recipe requires accepted initialization")
    else:
        _same(init["kind"], "scratch", "initialization.kind")
        _same(init["status"], "accepted", "initialization.status")
        for key in (
            "source_url", "license", "quarantine_path", "final_path", "physical_sha256",
            "mapping_report_path", "mapping_report_sha256", "initialization_state_sha256",
        ):
            _same(init[key], None, f"initialization.{key}")
        _same(
            init["scratch"],
            {
                "seed": 0,
                "conv2d": "kaiming_uniform_fan_in_leaky_relu_sqrt5",
                "conv_transpose2d": "kaiming_uniform_fan_in_leaky_relu_sqrt5",
                "linear": "kaiming_uniform_fan_in_leaky_relu_sqrt5",
                "spconv": "spconv_2.3.8_native_kaiming_uniform",
                "batch_norm_weight": 1.0,
                "batch_norm_bias": 0.0,
                "transfusion_transformer": "xavier_uniform_weights_dim_gt_1",
                "heatmap_bias": -2.19,
            },
            "initialization.scratch",
        )


def _validate_precision(raw: Any, branch: str, lifecycle: str) -> None:
    precision = _keys(
        raw,
        {
            "global_autocast", "sparse_island", "pool_input_dtype",
            "pool_accumulation_dtype", "loss_dtype", "decode_dtype",
            "grad_scaler", "tf32",
        },
        "precision",
    )
    _same(precision["global_autocast"], "fp16", "precision.global_autocast")
    _same(precision["loss_dtype"], "fp32", "precision.loss_dtype")
    _same(precision["decode_dtype"], "fp32", "precision.decode_dtype")
    _same(precision["tf32"], False, "precision.tf32")
    if branch == "camera":
        _same(precision["sparse_island"], "not_applicable", "precision.sparse_island")
        _same(precision["pool_input_dtype"], "fp32", "precision.pool_input_dtype")
        _same(precision["pool_accumulation_dtype"], "fp32", "precision.pool_accumulation_dtype")
    else:
        _same(
            precision["sparse_island"],
            "fp32_voxelize_vfe_spconv_dense_collapse_to_bev",
            "precision.sparse_island",
        )
        _same(precision["pool_input_dtype"], "not_applicable", "precision.pool_input_dtype")
        _same(precision["pool_accumulation_dtype"], "not_applicable", "precision.pool_accumulation_dtype")
    scaler = _keys(
        precision["grad_scaler"],
        {
            "status", "init_scale", "growth_factor", "backoff_factor",
            "growth_interval", "enabled",
        },
        "precision.grad_scaler",
    )
    if scaler["status"] not in {"pending_no_update_qualification", "accepted"}:
        raise Phase1ConfigError("precision.grad_scaler.status is unknown")
    _same(scaler["init_scale"], 8.0, "precision.grad_scaler.init_scale")
    _same(scaler["growth_factor"], 2.0, "precision.grad_scaler.growth_factor")
    _same(scaler["backoff_factor"], 0.5, "precision.grad_scaler.backoff_factor")
    _same(scaler["growth_interval"], 2000, "precision.grad_scaler.growth_interval")
    _same(scaler["enabled"], True, "precision.grad_scaler.enabled")
    if lifecycle != "envelope_a_implementation" and scaler["status"] != "accepted":
        raise Phase1ConfigError("qualified/Envelope-B recipe requires accepted GradScaler state")


def _validate_optimizer_scheduler(
    raw_optimizer: Any,
    raw_scheduler: Any,
    branch: str,
    schema_version: str,
) -> None:
    optimizer = _keys(
        raw_optimizer,
        {
            "name", "learning_rate", "weight_decay", "betas", "eps", "amsgrad",
            "fused", "parameter_group_rules", "coverage_policy",
        },
        "optimizer",
    )
    _same(optimizer["name"], "AdamW", "optimizer.name")
    _same(optimizer["learning_rate"], 2e-4 if branch == "camera" else 1e-4, "optimizer.learning_rate")
    _same(optimizer["weight_decay"], 0.01, "optimizer.weight_decay")
    _same(optimizer["betas"], [0.9, 0.999], "optimizer.betas")
    _same(optimizer["eps"], 1e-8, "optimizer.eps")
    _same(optimizer["amsgrad"], False, "optimizer.amsgrad")
    _same(
        optimizer["fused"],
        schema_version == PHASE1_SCHEMA_V3,
        "optimizer.fused",
    )
    _same(optimizer["coverage_policy"], "complete_disjoint_trainable_parameters", "optimizer.coverage_policy")
    expected_rules = (
        [
            {"name": "backbone_no_decay", "prefix": "camera_backbone.", "contains_any": ["absolute_pos_embed", "relative_position_bias_table"], "lr_mult": 0.1, "decay_mult": 0.0},
            {"name": "backbone", "prefix": "camera_backbone.", "contains_any": [], "lr_mult": 0.1, "decay_mult": 1.0},
            {"name": "other_no_decay", "prefix": "", "contains_any": ["absolute_pos_embed", "relative_position_bias_table"], "lr_mult": 1.0, "decay_mult": 0.0},
            {"name": "default", "prefix": "", "contains_any": [], "lr_mult": 1.0, "decay_mult": 1.0},
        ]
        if branch == "camera"
        else [{"name": "default", "prefix": "", "contains_any": [], "lr_mult": 1.0, "decay_mult": 1.0}]
    )
    _same(optimizer["parameter_group_rules"], expected_rules, "optimizer.parameter_group_rules")

    scheduler = _keys(
        raw_scheduler,
        {"interval", "lr", "momentum", "warmup"},
        "scheduler",
    )
    _same(scheduler["interval"], "accepted_optimizer_update", "scheduler.interval")
    lr = _keys(
        scheduler["lr"],
        {"policy", "target_ratio", "cyclic_times", "step_ratio_up", "anneal_strategy"},
        "scheduler.lr",
    )
    _same(lr["policy"], "cyclic", "scheduler.lr.policy")
    _same(lr["target_ratio"], [5.0, 5e-5] if branch == "camera" else [10.0, 1e-4], "scheduler.lr.target_ratio")
    _same(lr["cyclic_times"], 1, "scheduler.lr.cyclic_times")
    _same(lr["step_ratio_up"], 0.4, "scheduler.lr.step_ratio_up")
    _same(lr["anneal_strategy"], "cosine", "scheduler.lr.anneal_strategy")
    momentum = _keys(
        scheduler["momentum"],
        {"policy", "target_ratio", "cyclic_times", "step_ratio_up", "anneal_strategy"},
        "scheduler.momentum",
    )
    _same(momentum["policy"], "cyclic", "scheduler.momentum.policy")
    _same(momentum["target_ratio"], [0.85 / 0.95, 1.0], "scheduler.momentum.target_ratio")
    _same(momentum["cyclic_times"], 1, "scheduler.momentum.cyclic_times")
    _same(momentum["step_ratio_up"], 0.4, "scheduler.momentum.step_ratio_up")
    _same(momentum["anneal_strategy"], "cosine", "scheduler.momentum.anneal_strategy")
    expected_warmup = (
        {"type": "linear", "updates": 500, "ratio": 1.0 / 3.0}
        if branch == "camera"
        else None
    )
    _same(scheduler["warmup"], expected_warmup, "scheduler.warmup")


def _validate_training(raw: Any, branch: str, schema_version: str) -> None:
    training = _keys(
        raw,
        {
            "epochs", "micro_batch_size", "world_size", "accumulation_steps",
            "effective_global_batch", "loss_accumulation", "num_workers",
            "epoch_remainder_policy", "cbgs_samples_per_epoch", "consumed_samples_per_epoch",
            "dropped_samples_per_epoch", "optimizer_updates_per_epoch",
            "max_optimizer_updates", "gradient_clip", "ema", "activation_checkpoint",
            "telemetry_scalar_sync", "seed",
        },
        "training",
    )
    promoted_camera = branch == "camera" and schema_version == PHASE1_SCHEMA_V3
    expected = {
        "epochs": 20,
        "micro_batch_size": 16 if promoted_camera else 4,
        "world_size": 1,
        "accumulation_steps": 2 if promoted_camera else 8,
        "effective_global_batch": 32,
        "loss_accumulation": (
            "mean_over_two_microbatches"
            if promoted_camera
            else "mean_over_eight_microbatches"
        ),
        "num_workers": 8,
        "epoch_remainder_policy": "shuffle_then_drop_incomplete_effective_b32_window",
        "cbgs_samples_per_epoch": 87930,
        "consumed_samples_per_epoch": 87904,
        "dropped_samples_per_epoch": 26,
        "optimizer_updates_per_epoch": 2747,
        "max_optimizer_updates": 54940,
        "gradient_clip": {"max_norm": 35.0, "norm_type": 2.0},
        "ema": None,
        "activation_checkpoint": False,
        "telemetry_scalar_sync": False,
        "seed": 0,
    }
    _same(training, expected, "training")


def _validate_runtime_optimizations(
    raw: Any,
    branch: str,
    schema_version: str,
    throughput_decision: str,
) -> None:
    if schema_version != PHASE1_SCHEMA_V3 or branch != "camera":
        raise Phase1ConfigError(
            "runtime_optimizations is reserved for the promoted Camera recipe"
        )
    runtime_keys = {
        "camera_sdpa",
        "torch_compile",
        "state_dict_names_unchanged_required",
    }
    if throughput_decision == "IP-E4":
        runtime_keys.add("camera_preprocess")
    runtime = _keys(raw, runtime_keys, "runtime_optimizations")
    _same(runtime["camera_sdpa"], True, "runtime_optimizations.camera_sdpa")
    _same(
        runtime["state_dict_names_unchanged_required"],
        True,
        "runtime_optimizations.state_dict_names_unchanged_required",
    )
    preprocess_spec = runtime.get("camera_preprocess")
    if preprocess_spec is not None:
        preprocess_spec = _keys(
            preprocess_spec,
            {
                "batched_affine_grid",
                "vectorized_geometry",
                "bulk_input_conversion",
            },
            "runtime_optimizations.camera_preprocess",
        )
        for key in (
            "batched_affine_grid",
            "vectorized_geometry",
            "bulk_input_conversion",
        ):
            _same(
                preprocess_spec[key],
                True,
                f"runtime_optimizations.camera_preprocess.{key}",
            )
    compile_spec = _keys(
        runtime["torch_compile"],
        {"enabled", "scope", "backend", "dynamic", "mode", "modules"},
        "runtime_optimizations.torch_compile",
    )
    _same(compile_spec["enabled"], True, "runtime_optimizations.torch_compile.enabled")
    _same(
        compile_spec["scope"],
        "forward_only",
        "runtime_optimizations.torch_compile.scope",
    )
    _same(
        compile_spec["backend"],
        "inductor",
        "runtime_optimizations.torch_compile.backend",
    )
    _same(compile_spec["dynamic"], False, "runtime_optimizations.torch_compile.dynamic")
    _same(compile_spec["mode"], "default", "runtime_optimizations.torch_compile.mode")
    _same(
        compile_spec["modules"],
        list(CAMERA_COMPILE_FORWARD_MODULES),
        "runtime_optimizations.torch_compile.modules",
    )


def _validate_data(raw: Any) -> None:
    data = _keys(
        raw,
        {
            "dataroot", "version", "cache", "zip_manifest", "split_manifest",
            "roles", "cache_capacity_sweeps", "train_point_sweeps", "eval_point_sweeps",
            "raw_extraction", "camera_point_payload",
        },
        "data",
    )
    _path(data["dataroot"], "data.dataroot")
    _same(data["version"], "v1.0-trainval", "data.version")
    _same(data["cache_capacity_sweeps"], 10, "data.cache_capacity_sweeps")
    _same(data["train_point_sweeps"], 1, "data.train_point_sweeps")
    _same(data["eval_point_sweeps"], 10, "data.eval_point_sweeps")
    _same(data["raw_extraction"], False, "data.raw_extraction")
    if data["camera_point_payload"] not in {"disabled_after_parity", "enabled"}:
        raise Phase1ConfigError("data.camera_point_payload is unknown")
    cache = _keys(
        data["cache"],
        {"format", "path", "sidecar_path", "logical_sha256", "pickle_sha256", "sidecar_sha256"},
        "data.cache",
    )
    _same(cache["format"], "t1.v2", "data.cache.format")
    _path(cache["path"], "data.cache.path")
    _path(cache["sidecar_path"], "data.cache.sidecar_path")
    for key in ("logical_sha256", "pickle_sha256", "sidecar_sha256"):
        _sha(cache[key], f"data.cache.{key}")
    manifest = _keys(
        data["zip_manifest"],
        {"path", "logical_sha256", "file_sha256"},
        "data.zip_manifest",
    )
    _path(manifest["path"], "data.zip_manifest.path")
    _sha(manifest["logical_sha256"], "data.zip_manifest.logical_sha256")
    _sha(manifest["file_sha256"], "data.zip_manifest.file_sha256")
    split = _keys(data["split_manifest"], {"path", "sha256"}, "data.split_manifest")
    _path(split["path"], "data.split_manifest.path")
    _sha(split["sha256"], "data.split_manifest.sha256")
    roles = _keys(data["roles"], {"fit", "select", "audit"}, "data.roles")
    expected_roles = {
        "fit": {"name": "D_fit", "logs": 34, "scenes": 494, "samples": 19877, "log_tokens_sha256": "7ac31d5a3f44bfc741f148fc1d7ced9a3db9be7080233f43ca992aa2ab016aa9", "scene_tokens_sha256": "3b0a225aa3a5c550101cf90637df9bfd771415c795c1d3b02991fd49bc7dd747", "sample_tokens_sha256": "640985b4022cfe5db0efdd11d02591f113f1eab3cfe6f042fd05fc773cdc46c8"},
        "select": {"name": "D_select", "logs": 8, "scenes": 115, "samples": 4626, "log_tokens_sha256": "203e16c6761ea6a11ced6c4c2b00850bb935af8e57cfeeddec3c1e0d183cc574", "scene_tokens_sha256": "bb1a801949ab4f665edbebb6455ba0efdff944ab4f3e8ae0a4e8d756218a5b12", "sample_tokens_sha256": "72d7ca8465e3ff9dd5f0a2167b1c3837a490cf30559c2ca32afd42b98d5e45b2"},
        "audit": {"name": "D_audit", "logs": 8, "scenes": 91, "samples": 3627, "log_tokens_sha256": "509148071d74cb8c12a158e8904c93d5cea82545e0a60c58abf13689fcc1a9c3", "scene_tokens_sha256": "ac56708c38251e42c1f6acef3b6ff6b042e1976f36babedf0be51fd599e30457", "sample_tokens_sha256": "152e5cf51f3693badb96f41bd67fc36b75b3b587b7eb2fca2ce4dd675ddf167c", "sealed": True},
    }
    _same(roles, expected_roles, "data.roles")


def _validate_sampling(raw: Any) -> None:
    sampling = _keys(
        raw,
        {
            "type", "class_order", "rng", "eligibility", "artifact",
            "source_sample_order_sha256",
            "class_pool_sizes", "duplicated_class_memberships", "target_class_fraction",
            "segment_sizes", "expanded_length", "expanded_indices_sha256",
            "expanded_tokens_sha256", "class_segments_sha256", "epoch_shuffle",
            "twenty_epoch_order_sha256", "twenty_epoch_remainder_sha256",
        },
        "sampling",
    )
    expected = {
        "type": "official_cbgs",
        "class_order": list(REFERENCE_OBJECT_CLASSES),
        "rng": {"api": "numpy.random.RandomState", "algorithm": "MT19937", "seed": 0, "choice_replace": True},
        "eligibility": {
            "reference_use_valid_flag": True,
            "formula": "num_lidar_pts_plus_num_radar_pts_gt_0",
            "metadata_path": "/dataset/easybuild/data/nuScenes-data/1.0-map-1.3-zip/trainval/sample_annotation.json",
            "metadata_sha256": "1f4f3835cb86f4efe49ccad3ae6f5f6a2d068422696de564c5892b5b06bea2a9",
            "ordered_D_fit_annotation_validity_sha256": "cf69572d2428fabe1ea2366c9b2cbf33a300d811c71932891459e17beb3967d0",
        },
        "artifact": {
            "path": "/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s10_phase1_envelope_a_data_e321aed749fd/cbgs/official_cbgs_D_fit_seed0.json",
            "sha256": "64cc0d1d6cd82fae2787d397e610178cedd00887d98938b154fce9f8e8e115ef",
        },
        "source_sample_order_sha256": "640985b4022cfe5db0efdd11d02591f113f1eab3cfe6f042fd05fc773cdc46c8",
        "class_pool_sizes": [19358, 13698, 5040, 6428, 5326, 6459, 4180, 3878, 15849, 7722],
        "duplicated_class_memberships": 87938,
        "target_class_fraction": 0.1,
        "segment_sizes": [8793] * 10,
        "expanded_length": 87930,
        "expanded_indices_sha256": "7f209a57e686645ae3cd3ab1e93d4ca7fc8e46b494eac35fbc2d69d27d102389",
        "expanded_tokens_sha256": "2c5188f12aeb1c854036c5f881f8a59a802fe5706556437e6cd8067f3b1dbb4f",
        "class_segments_sha256": "15f9326237edee19054d4d3f649d7405f269d7a72a8f73f30ca2dcf4773d2cb5",
        "epoch_shuffle": {"api": "numpy.random.RandomState", "seed_formula": "seed_plus_zero_based_epoch", "permutation_domain": "expanded_position", "remainder_policy": "drop_last_26_to_effective_b32"},
        "twenty_epoch_order_sha256": "ea0ca69acebffe5cf9f8447f61e389419668874cb5bd601d0775b213464b0743",
        "twenty_epoch_remainder_sha256": "77624c45d883729a668ec04104694693bdbdcf94fc8eb897128e18b281f1e073",
    }
    _same(sampling, expected, "sampling")


def _validate_gt_paste(raw: Any, branch: str, lifecycle: str) -> None:
    paste = _keys(
        raw,
        {
            "enabled", "active_human_epochs", "active_zero_based_epochs", "database_status",
            "database_root", "manifest_path", "manifest_sha256", "all_classes",
            "min_points", "filter_by_difficulty", "sample_groups", "rate",
            "collision_policy", "yaw_jitter_radians", "source_point_dims",
            "source_role", "build_truncation",
        },
        "gt_paste",
    )
    if branch == "camera":
        expected = {
            "enabled": False,
            "active_human_epochs": [],
            "active_zero_based_epochs": [],
            "database_status": "not_applicable",
            "database_root": None,
            "manifest_path": None,
            "manifest_sha256": None,
            "all_classes": list(REFERENCE_OBJECT_CLASSES),
            "min_points": {name: 5 for name in REFERENCE_OBJECT_CLASSES},
            "filter_by_difficulty": [-1],
            "sample_groups": REFERENCE_SAMPLE_GROUPS,
            "rate": 0.0,
            "collision_policy": "reject_any_bev_overlap",
            "yaw_jitter_radians": 0.0,
            "source_point_dims": ["x", "y", "z", "intensity", "ring"],
            "source_role": "D_fit",
            "build_truncation": None,
        }
        _same(paste, expected, "gt_paste")
        return
    expected_fixed = {
        "enabled": True,
        "active_human_epochs": list(range(1, 16)),
        "active_zero_based_epochs": list(range(15)),
        "all_classes": list(REFERENCE_OBJECT_CLASSES),
        "min_points": {name: 5 for name in REFERENCE_OBJECT_CLASSES},
        "filter_by_difficulty": [-1],
        "sample_groups": REFERENCE_SAMPLE_GROUPS,
        "rate": 1.0,
        "collision_policy": "reject_any_bev_overlap",
        "yaw_jitter_radians": 0.0,
        "source_point_dims": ["x", "y", "z", "intensity", "ring"],
        "source_role": "D_fit",
        "build_truncation": None,
    }
    for key, value in expected_fixed.items():
        _same(paste[key], value, f"gt_paste.{key}")
    if paste["database_status"] not in {"pending_materialization", "accepted"}:
        raise Phase1ConfigError("gt_paste.database_status is unknown")
    _path(paste["database_root"], "gt_paste.database_root")
    _path(paste["manifest_path"], "gt_paste.manifest_path")
    _sha(paste["manifest_sha256"], "gt_paste.manifest_sha256", allow_none=True)
    if paste["database_status"] == "accepted" and paste["manifest_sha256"] is None:
        raise Phase1ConfigError("accepted GTDB requires manifest_sha256")
    if lifecycle != "envelope_a_implementation" and paste["database_status"] != "accepted":
        raise Phase1ConfigError("qualified/Envelope-B LiDAR recipe requires accepted GTDB")


def _validate_evaluation(raw: Any, schema_version: str) -> None:
    expected = {
        "evaluator": "fl_v3_subset_official_nuscenes_detection_eval",
        "class_order": list(DEVKIT_DETECTION_NAMES),
        "checkpoint_weights": "raw",
        "checkpoint_selection": "epoch_20_terminal_only",
        "D_select": {
            "executions": 1,
            "status": (
                "open_once_in_envelope_b"
                if schema_version in PHASE1_ENVELOPE_B_SCHEMAS
                else "sealed_until_envelope_b"
            ),
        },
        "D_audit": {"executions": 1, "status": "owner_sealed_until_P1_G2"},
        "official_validation": "forbidden_in_phase1_internal_selection",
        "metrics": ["mAP", "NDS", "per_class_AP", "TP_errors"],
        "timing": False,
    }
    _same(raw, expected, "evaluation")


def _validate_checkpointing(raw: Any) -> None:
    checkpointing = _keys(
        raw,
        {
            "schema", "recovery_cadence_epochs", "recovery_selectable",
            "terminal_epoch", "terminal_selectable", "retained_recovery_count",
            "resume_rng", "resume_sampler", "resume_accumulation_boundary",
        },
        "checkpointing",
    )
    expected = {
        "schema": "s10.phase1.checkpoint.v1",
        "recovery_cadence_epochs": 1,
        "recovery_selectable": False,
        "terminal_epoch": 20,
        "terminal_selectable": True,
        "retained_recovery_count": 1,
        "resume_rng": True,
        "resume_sampler": True,
        "resume_accumulation_boundary": "optimizer_window_only",
    }
    _same(checkpointing, expected, "checkpointing")


def _validate_dependencies(raw: Any, branch: str) -> None:
    deps = _keys(
        raw,
        {
            "torch", "torch_build_sha256", "torch_source_sha", "torchvision", "numpy",
            "scipy", "spconv", "spconv_build_sha256", "spconv_source_sha",
            "spconv_source_state", "cumm", "cumm_build_sha256", "cumm_source_sha",
            "cumm_source_state", "mmdet3d", "mmcv",
        },
        "dependencies",
    )
    _same(deps["torch"], "2.11.0+cu128", "dependencies.torch")
    _sha(deps["torch_build_sha256"], "dependencies.torch_build_sha256")
    if (
        not isinstance(deps["torch_source_sha"], str)
        or len(deps["torch_source_sha"]) != 40
        or any(ch not in "0123456789abcdef" for ch in deps["torch_source_sha"])
    ):
        raise Phase1ConfigError("dependencies.torch_source_sha must be a 40-character Git SHA")
    _same(deps["torchvision"], "0.26.0+cu128", "dependencies.torchvision")
    _same(deps["numpy"], "1.26.4", "dependencies.numpy")
    _same(deps["scipy"], "1.13.1", "dependencies.scipy")
    _same(deps["mmdet3d"], None, "dependencies.mmdet3d")
    _same(deps["mmcv"], None, "dependencies.mmcv")
    sparse_fields = (
        "spconv", "spconv_build_sha256", "spconv_source_sha", "spconv_source_state",
        "cumm", "cumm_build_sha256", "cumm_source_sha", "cumm_source_state",
    )
    if branch == "camera":
        for key in sparse_fields:
            _same(deps[key], None, f"dependencies.{key}")
    else:
        _same(deps["spconv"], "2.3.8", "dependencies.spconv")
        _same(deps["cumm"], "0.7.13", "dependencies.cumm")
        for key in ("spconv_build_sha256", "cumm_build_sha256"):
            _sha(deps[key], f"dependencies.{key}")
        for key in ("spconv_source_sha", "cumm_source_sha"):
            if (
                not isinstance(deps[key], str)
                or len(deps[key]) != 40
                or any(ch not in "0123456789abcdef" for ch in deps[key])
            ):
                raise Phase1ConfigError(f"dependencies.{key} must be a 40-character Git SHA")
        for key in ("spconv_source_state", "cumm_source_state"):
            try:
                normalized = validate_source_state(deps[key])
            except ValueError as exc:
                raise Phase1ConfigError(f"dependencies.{key} is invalid: {exc}") from exc
            _same(deps[key], normalized, f"dependencies.{key}")


def _validate_execution(raw: Any, branch: str) -> None:
    execution = _keys(
        raw,
        {
            "mode", "preflight_one_batch", "capability_metrics", "warmup_microbatches",
            "timed_microbatches", "calibration_updates", "output_root", "data_artifact_root",
            "pool_timing", "allowed_data_role", "allowed_evaluation_roles",
        },
        "execution",
    )
    if execution["mode"] not in {"envelope_a_calibration", "phase1_train_eval"}:
        raise Phase1ConfigError("execution.mode is unknown")
    _same(execution["preflight_one_batch"], True, "execution.preflight_one_batch")
    _same(execution["capability_metrics"], False if execution["mode"] == "envelope_a_calibration" else True, "execution.capability_metrics")
    _same(execution["warmup_microbatches"], 16 if execution["mode"] == "envelope_a_calibration" else 0, "execution.warmup_microbatches")
    _same(execution["timed_microbatches"], 64 if execution["mode"] == "envelope_a_calibration" else 0, "execution.timed_microbatches")
    _same(execution["calibration_updates"], 0, "execution.calibration_updates")
    _path(execution["output_root"], "execution.output_root")
    _path(execution["data_artifact_root"], "execution.data_artifact_root")
    _same(execution["allowed_data_role"], "D_fit", "execution.allowed_data_role")
    expected_eval_roles = [] if execution["mode"] == "envelope_a_calibration" else ["D_select"]
    _same(execution["allowed_evaluation_roles"], expected_eval_roles, "execution.allowed_evaluation_roles")
    if branch == "camera" and execution["mode"] == "envelope_a_calibration":
        _same(
            execution["pool_timing"],
            {
                "fixed_batches": 4,
                "operator_warmup": 32,
                "operator_samples": 128,
                "end_to_end_warmup": 16,
                "end_to_end_samples": 64,
                "event_clock": "cuda_event",
                "promotion_operator_median_ratio_max": 0.8,
                "promotion_step_median_ratio_max": 1.02,
                "promotion_peak_allocated_ratio_max": 1.05,
            },
            "execution.pool_timing",
        )
    else:
        _same(execution["pool_timing"], None, "execution.pool_timing")


def validate_phase1_config(raw: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and return a detached JSON-normalized Phase-I recipe."""
    raw_config = dict(raw)
    schema_version = str(raw_config.get("schema_version"))
    root_keys = {
        "schema_version", "contract", "taxonomy", "model", "initialization", "precision",
        "optimizer", "scheduler", "training", "data", "augmentation", "sampling",
        "gt_paste", "evaluation", "checkpointing", "dependencies", "execution",
    }
    if schema_version == PHASE1_SCHEMA_V3:
        root_keys.add("runtime_optimizations")
    root = _keys(
        raw_config,
        root_keys,
        "config",
    )
    schema_version = str(root["schema_version"])
    if schema_version not in PHASE1_SCHEMAS:
        raise Phase1ConfigError(
            f"schema_version must be one of {sorted(PHASE1_SCHEMAS)}"
        )
    contract, branch = _validate_contract(root["contract"], schema_version)
    taxonomy_selector = {"frozen_spec": "phase1_taxonomy_v1"}
    if root["taxonomy"] == taxonomy_selector:
        root["taxonomy"] = json.loads(json.dumps(FROZEN_TAXONOMY))
    _same(root["taxonomy"], FROZEN_TAXONOMY, "taxonomy")
    # The checked-in input files use one versioned symbolic selector for the two
    # large graph/augmentation records.  Resolution replaces it with the complete
    # owner-frozen leaf graph *before canonicalization and hashing*; consumers never
    # see or hash a library default or an unresolved inheritance node.
    model_version = (
        "v2"
        if schema_version in PHASE1_ENVELOPE_B_SCHEMAS and branch == "camera"
        else "v1"
    )
    model_selector = {"frozen_spec": f"phase1_{branch}_model_{model_version}"}
    frozen_model = (
        FROZEN_CAMERA_MODEL_V2
        if branch == "camera" and schema_version in PHASE1_ENVELOPE_B_SCHEMAS
        else FROZEN_CAMERA_MODEL
        if branch == "camera"
        else FROZEN_LIDAR_MODEL
    )
    if root["model"] == model_selector:
        root["model"] = json.loads(json.dumps(frozen_model))
    augmentation_selector = {"frozen_spec": f"phase1_{branch}_augmentation_v1"}
    if root["augmentation"] == augmentation_selector:
        root["augmentation"] = json.loads(
            json.dumps(
                FROZEN_CAMERA_AUGMENTATION
                if branch == "camera"
                else FROZEN_LIDAR_AUGMENTATION
            )
        )
    _same(root["model"], frozen_model, "model")
    _validate_initialization(root["initialization"], branch, contract["lifecycle"])
    _validate_precision(root["precision"], branch, contract["lifecycle"])
    _validate_optimizer_scheduler(
        root["optimizer"], root["scheduler"], branch, schema_version
    )
    _validate_training(root["training"], branch, schema_version)
    if schema_version == PHASE1_SCHEMA_V3:
        _validate_runtime_optimizations(
            root["runtime_optimizations"],
            branch,
            schema_version,
            str(contract["throughput_decision"]),
        )
    _validate_data(root["data"])
    _same(
        root["augmentation"],
        FROZEN_CAMERA_AUGMENTATION if branch == "camera" else FROZEN_LIDAR_AUGMENTATION,
        "augmentation",
    )
    _validate_sampling(root["sampling"])
    _validate_gt_paste(root["gt_paste"], branch, contract["lifecycle"])
    _validate_evaluation(root["evaluation"], schema_version)
    _validate_checkpointing(root["checkpointing"])
    _validate_dependencies(root["dependencies"], branch)
    _validate_execution(root["execution"], branch)
    if schema_version in PHASE1_ENVELOPE_B_SCHEMAS:
        _same(contract["lifecycle"], "envelope_b_ready", "contract.lifecycle")
        _same(root["execution"]["mode"], "phase1_train_eval", "execution.mode")
    # JSON round-trip rejects tuples/custom objects and provides a detached graph.
    return json.loads(json.dumps(root, allow_nan=False))


def phase1_scientific_leaf_paths(raw: Mapping[str, Any]) -> tuple[str, ...]:
    """Return stable leaf paths that constructors must mark as consumed.

    Provenance-only paths and mutable materialization identities are deliberately
    included: silently dropping a checkpoint/data identity is as serious as
    dropping a model hyperparameter.
    """
    leaves: list[str] = []

    def visit(prefix: str, value: Any) -> None:
        if isinstance(value, Mapping):
            for key in sorted(value):
                visit(f"{prefix}.{key}" if prefix else str(key), value[key])
        elif isinstance(value, list):
            if not value:
                leaves.append(prefix)
            else:
                for index, item in enumerate(value):
                    visit(f"{prefix}[{index}]", item)
        else:
            leaves.append(prefix)

    visit("", raw)
    return tuple(leaves)


def phase1_runtime_ready(raw: Mapping[str, Any]) -> None:
    """Fail closed before model/data construction when materialization is pending."""
    branch = str(raw["contract"]["branch"])
    if branch == "camera" and raw["initialization"]["status"] != "accepted":
        raise Phase1ConfigError("Camera checkpoint acquisition/mapping is not accepted")
    if branch == "lidar" and raw["gt_paste"]["database_status"] != "accepted":
        raise Phase1ConfigError("LiDAR D_fit GTDB materialization is not accepted")
    if raw["precision"]["grad_scaler"]["status"] != "accepted":
        raise Phase1ConfigError("GradScaler no-update qualification is not accepted")


def load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as stream:
        return json.load(stream)
