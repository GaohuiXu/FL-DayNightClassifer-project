"""Shared helpers for Arrhenius branch capability diagnostics.

These helpers are intentionally script-facing. They keep branch-isolation,
reference-parity assertions, and telemetry out of the production detector path.
"""
from __future__ import annotations

from collections import OrderedDict
from typing import Any, Dict, Iterable, List, Tuple


BRANCH_TOPOLOGIES = ("full_fusion", "lidar_only", "camera_only")
TRAIN_POLICIES = (
    "all_trainable",
    "camera_frozen",
    "lidar_only_trainable",
    "camera_only_trainable",
    "probe_no_backward",
)

BEVFUSION_075_REFERENCE = {
    "voxel_size": [0.075, 0.075, 0.2],
    "point_cloud_range": [-54.0, -54.0, -5.0, 54.0, 54.0, 3.0],
    "sparse_shape": [1440, 1440, 41],
    "head_grid_size": [1440, 1440, 41],
    "max_voxels": [120000, 160000],
    "fuser_lidar_channels": 256,
}

FULLSHAPE_COMMON = {
    "det-camera-backbone": "swin_t",
    "det-freeze-backbone": False,
    "det-pretrained-backbone": True,
    "det-max-pillars": 120000,
    "det-lidar-backbone": True,
    "det-lidar-backbone-stages": 4,
    "det-lidar-backbone-checkpoint": True,
    "det-activation-checkpoint": True,
    "det-swin-sdpa": True,
    "det-gt-paste": False,
    "det-aug-bev": False,
}

CAPABILITY_MATRIX_CELLS = {
    "full_fusion_pillar020_fp16_control": {
        **FULLSHAPE_COMMON,
        "branch_topology": "full_fusion",
        "train_policy": "all_trainable",
        "det-lidar-encoder": "pillar",
        "precision": "fp16",
        "det-sparse-conv-fp16": False,
        "det-bev-voxel": 0.2,
        "det-lidar-backbone-out": 128,
        "det-fusion-channels": 128,
    },
    "full_fusion_voxel020_fp16_current": {
        **FULLSHAPE_COMMON,
        "branch_topology": "full_fusion",
        "train_policy": "all_trainable",
        "det-lidar-encoder": "voxel",
        "precision": "fp16",
        "det-sparse-conv-fp16": False,
        "det-bev-voxel": 0.2,
        "det-lidar-z-voxel": 0.2,
        "det-lidar-backbone-out": 128,
        "det-fusion-channels": 128,
    },
    "full_fusion_voxel075_z020_ch256_fp16_parity_probe": {
        **FULLSHAPE_COMMON,
        "branch_topology": "full_fusion",
        "train_policy": "probe_no_backward",
        "det-lidar-encoder": "voxel",
        "precision": "fp16",
        "det-sparse-conv-fp16": False,
        "det-bev-voxel": 0.075,
        "det-lidar-z-voxel": 0.2,
        "det-lidar-sparse-z-size": 41,
        "det-pc-range": BEVFUSION_075_REFERENCE["point_cloud_range"],
        "det-max-pillars": 120000,
        "det-max-points-per-pillar": 10,
        "det-lidar-backbone-out": 256,
        "det-fusion-channels": 256,
        "bevfusion-parity-075": True,
        "bevfusion-max-voxels-reference": BEVFUSION_075_REFERENCE["max_voxels"],
        "bevfusion-head-grid-size-reference": BEVFUSION_075_REFERENCE["head_grid_size"],
    },
    "lidar_iso_pillar020_fp16_control": {
        **FULLSHAPE_COMMON,
        "branch_topology": "lidar_only",
        "train_policy": "lidar_only_trainable",
        "det-lidar-encoder": "pillar",
        "precision": "fp16",
        "det-sparse-conv-fp16": False,
        "det-bev-voxel": 0.2,
        "det-lidar-backbone-out": 128,
        "det-fusion-channels": 128,
    },
    "lidar_iso_voxel020_fp16_current_sparse": {
        **FULLSHAPE_COMMON,
        "branch_topology": "lidar_only",
        "train_policy": "lidar_only_trainable",
        "det-lidar-encoder": "voxel",
        "precision": "fp16",
        "det-sparse-conv-fp16": False,
        "det-bev-voxel": 0.2,
        "det-lidar-z-voxel": 0.2,
        "det-lidar-backbone-out": 128,
        "det-fusion-channels": 128,
    },
    "lidar_iso_voxel075_z020_ch256_fp16_parity_probe": {
        **FULLSHAPE_COMMON,
        "branch_topology": "lidar_only",
        "train_policy": "lidar_only_trainable",
        "det-lidar-encoder": "voxel",
        "precision": "fp16",
        "det-sparse-conv-fp16": False,
        "det-bev-voxel": 0.075,
        "det-lidar-z-voxel": 0.2,
        "det-lidar-sparse-z-size": 41,
        "det-pc-range": BEVFUSION_075_REFERENCE["point_cloud_range"],
        "det-max-pillars": 120000,
        "det-max-points-per-pillar": 10,
        "det-lidar-backbone-out": 256,
        "det-fusion-channels": 256,
        "bevfusion-parity-075": True,
        "bevfusion-max-voxels-reference": BEVFUSION_075_REFERENCE["max_voxels"],
        "bevfusion-head-grid-size-reference": BEVFUSION_075_REFERENCE["head_grid_size"],
    },
    "camera_iso_020_fp16_swin": {
        **FULLSHAPE_COMMON,
        "branch_topology": "camera_only",
        "train_policy": "camera_only_trainable",
        "det-lidar-encoder": "pillar",
        "precision": "fp16",
        "det-sparse-conv-fp16": False,
        "det-bev-voxel": 0.2,
        "det-lidar-backbone-out": 128,
        "det-fusion-channels": 128,
    },
    "camera_iso_020_fp16_swin_no_sdpa": {
        **FULLSHAPE_COMMON,
        "branch_topology": "camera_only",
        "train_policy": "camera_only_trainable",
        "det-lidar-encoder": "pillar",
        "precision": "fp16",
        "det-sparse-conv-fp16": False,
        "det-bev-voxel": 0.2,
        "det-lidar-backbone-out": 128,
        "det-fusion-channels": 128,
        "det-swin-sdpa": False,
    },
    "camera_iso_020_fp16_swin_no_ckpt": {
        **FULLSHAPE_COMMON,
        "branch_topology": "camera_only",
        "train_policy": "camera_only_trainable",
        "det-lidar-encoder": "pillar",
        "precision": "fp16",
        "det-sparse-conv-fp16": False,
        "det-bev-voxel": 0.2,
        "det-lidar-backbone-out": 128,
        "det-fusion-channels": 128,
        "det-activation-checkpoint": False,
    },
    "camera_iso_075_ch256_fp16_probe": {
        **FULLSHAPE_COMMON,
        "branch_topology": "camera_only",
        "train_policy": "probe_no_backward",
        "det-lidar-encoder": "pillar",
        "precision": "fp16",
        "det-sparse-conv-fp16": False,
        "det-bev-voxel": 0.075,
        "det-pc-range": BEVFUSION_075_REFERENCE["point_cloud_range"],
        "det-max-pillars": 120000,
        "det-max-points-per-pillar": 10,
        "det-lidar-backbone-out": 256,
        "det-fusion-channels": 256,
    },
    "camera_iso_075_ch256_fp16_swin": {
        **FULLSHAPE_COMMON,
        "branch_topology": "camera_only",
        "train_policy": "camera_only_trainable",
        "det-lidar-encoder": "pillar",
        "precision": "fp16",
        "det-sparse-conv-fp16": False,
        "det-bev-voxel": 0.075,
        "det-pc-range": BEVFUSION_075_REFERENCE["point_cloud_range"],
        "det-max-pillars": 120000,
        "det-max-points-per-pillar": 10,
        "det-lidar-backbone-out": 256,
        "det-fusion-channels": 256,
    },
}

CONFIG_FIELD_KEYS = (
    "det-camera-backbone",
    "det-freeze-backbone",
    "det-pretrained-backbone",
    "det-lidar-encoder",
    "det-lidar-sweeps",
    "det-bev-voxel",
    "det-lidar-z-voxel",
    "det-lidar-sparse-z-size",
    "det-pc-range",
    "det-max-pillars",
    "det-max-points-per-pillar",
    "det-lidar-backbone",
    "det-lidar-backbone-stages",
    "det-lidar-backbone-out",
    "det-activation-checkpoint",
    "det-swin-sdpa",
    "det-lidar-backbone-checkpoint",
    "det-fusion-channels",
    "det-neck-channels",
    "det-context-channels",
    "det-bev-neck-channels",
    "det-head-channels",
    "det-head-conv-layers",
    "det-image-h",
    "det-image-w",
    "det-feat-stride",
    "det-out-size-factor",
    "precision",
    "det-sparse-conv-fp16",
    "batch-size",
    "num-workers",
    "branch_topology",
    "train_policy",
)


def validate_controls(branch_topology: str, train_policy: str) -> Tuple[str, str]:
    if branch_topology not in BRANCH_TOPOLOGIES:
        raise ValueError(f"unknown branch_topology={branch_topology!r}; valid={BRANCH_TOPOLOGIES}")
    if train_policy not in TRAIN_POLICIES:
        raise ValueError(f"unknown train_policy={train_policy!r}; valid={TRAIN_POLICIES}")
    return branch_topology, train_policy


def cell_branch_topology(cfg: dict, default: str) -> str:
    return str(cfg.get("branch_topology", default))


def cell_train_policy(cfg: dict, default: str) -> str:
    return str(cfg.get("train_policy", default))


def config_fields(cfg: dict) -> Dict[str, Any]:
    return {k: cfg.get(k) for k in CONFIG_FIELD_KEYS if k in cfg}


CAMERA_PREFIXES = ("camera_backbone", "camera_neck", "view_transform")
LIDAR_PREFIXES = ("lidar_encoder", "lidar_backbone")


def _set_prefix_trainability(model, prefixes: Tuple[str, ...], trainable: bool) -> None:
    for name, param in model.named_parameters():
        if name.split(".", 1)[0] in prefixes:
            param.requires_grad_(bool(trainable))


def apply_train_policy(model, train_policy: str) -> None:
    validate_controls("full_fusion", train_policy)
    if train_policy == "all_trainable":
        return

    if train_policy in ("camera_frozen", "lidar_only_trainable"):
        _set_prefix_trainability(model, CAMERA_PREFIXES, False)
    elif train_policy == "camera_only_trainable":
        # Policy-level camera isolation should not inherit a frozen-backbone
        # bring-up config. This is a diagnostic-only override; camera audit cells
        # use Swin-T, so there are no ResNet BatchNorm running-stat surprises.
        backbone = getattr(model, "camera_backbone", None)
        if backbone is not None and hasattr(backbone, "frozen"):
            backbone.frozen = False
        _set_prefix_trainability(model, CAMERA_PREFIXES, True)
        _set_prefix_trainability(model, LIDAR_PREFIXES, False)


def trainable_parameters(model) -> List[Any]:
    return [p for p in model.parameters() if p.requires_grad]


def module_param_coverage(model) -> Dict[str, Dict[str, int]]:
    out: "OrderedDict[str, Dict[str, int]]" = OrderedDict()
    for name, param in model.named_parameters():
        top = name.split(".", 1)[0]
        rec = out.setdefault(top, {"total": 0, "trainable": 0})
        rec["total"] += 1
        rec["trainable"] += int(param.requires_grad)
    return dict(out)


def module_grad_coverage(model) -> Dict[str, Dict[str, int]]:
    out: "OrderedDict[str, Dict[str, int]]" = OrderedDict()
    for name, param in model.named_parameters():
        top = name.split(".", 1)[0]
        rec = out.setdefault(
            top,
            {"trainable": 0, "with_grad": 0, "finite_grad": 0, "nonzero_grad": 0},
        )
        if not param.requires_grad:
            continue
        rec["trainable"] += 1
        grad = param.grad
        if grad is None:
            continue
        rec["with_grad"] += 1
        try:
            import torch

            finite = bool(torch.isfinite(grad.detach()).all().item())
            nonzero = bool((grad.detach().abs().sum() > 0).item())
        except Exception:
            finite = False
            nonzero = False
        rec["finite_grad"] += int(finite)
        rec["nonzero_grad"] += int(nonzero)
    return dict(out)


def grad_diagnostics(model, max_params: int = 12) -> Dict[str, Any]:
    """Compact gradient health summary for failure manifests."""
    try:
        import torch
    except Exception as exc:
        return {"available": False, "error": f"{type(exc).__name__}: {exc}"}

    modules: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
    nonfinite_params: List[Dict[str, Any]] = []
    largest_params: List[Tuple[float, str, str]] = []
    for name, param in model.named_parameters():
        if not param.requires_grad or param.grad is None:
            continue
        top = name.split(".", 1)[0]
        rec = modules.setdefault(
            top,
            {
                "params_with_grad": 0,
                "finite_params": 0,
                "nonfinite_params": 0,
                "nonfinite_values": 0,
                "max_abs_grad": 0.0,
                "max_abs_param": "",
            },
        )
        rec["params_with_grad"] += 1
        grad = param.grad.detach().float()
        finite_mask = torch.isfinite(grad)
        finite = bool(finite_mask.all().item())
        finite_count = int(finite_mask.sum().item())
        nonfinite_count = int(grad.numel()) - finite_count
        abs_max = float(grad.abs().max().detach().cpu()) if grad.numel() else 0.0
        if abs_max > float(rec["max_abs_grad"]):
            rec["max_abs_grad"] = abs_max
            rec["max_abs_param"] = name
        largest_params.append((abs_max, name, top))
        if finite:
            rec["finite_params"] += 1
        else:
            rec["nonfinite_params"] += 1
            rec["nonfinite_values"] += nonfinite_count
            if len(nonfinite_params) < int(max_params):
                nonfinite_params.append({
                    "name": name,
                    "module": top,
                    "shape": tuple(param.shape),
                    "dtype": str(param.grad.dtype),
                    "nonfinite_values": nonfinite_count,
                    "max_abs_grad": abs_max,
                })
    largest_params.sort(key=lambda item: item[0], reverse=True)
    return {
        "available": True,
        "modules": dict(modules),
        "nonfinite_params": nonfinite_params,
        "largest_params": [
            {"name": name, "module": top, "max_abs_grad": value}
            for value, name, top in largest_params[: int(max_params)]
        ],
    }


def tensor_stats(x) -> Dict[str, Any]:
    try:
        import torch

        if x is None or not torch.is_tensor(x):
            return {"available": False}
        xf = x.detach().float()
        numel = int(xf.numel())
        nonzero = int((xf != 0).sum().cpu()) if numel else 0
        return {
            "available": True,
            "shape": tuple(x.shape),
            "dtype": str(x.dtype),
            "finite": bool(torch.isfinite(xf).all().item()),
            "norm": float(xf.norm().cpu()),
            "mean": float(xf.mean().cpu()) if numel else 0.0,
            "std": float(xf.std(unbiased=False).cpu()) if numel else 0.0,
            "variance": float(xf.var(unbiased=False).cpu()) if numel else 0.0,
            "numel": numel,
            "nonzero": nonzero,
            "nonzero_ratio": (float(nonzero) / float(numel)) if numel else 0.0,
        }
    except Exception as exc:
        return {"available": False, "error": f"{type(exc).__name__}: {exc}"}


def zero_bev(batch_size: int, channels: int, ny: int, nx: int, like):
    import torch

    kwargs = {"device": like.device, "dtype": like.dtype}
    return torch.zeros((int(batch_size), int(channels), int(ny), int(nx)), **kwargs)


def fuser_contract(model) -> Dict[str, Any]:
    fuser = getattr(model, "fusion", None)
    return {
        "camera_channels": int(getattr(fuser, "camera_channels", -1)),
        "lidar_channels": int(getattr(fuser, "lidar_channels", -1)),
        "out_channels": int(getattr(fuser, "out_channels", -1)),
    }


def camera_model_summary(model) -> Dict[str, Any]:
    cfg = getattr(model, "cfg", None)
    backbone = getattr(model, "camera_backbone", None)
    neck = getattr(model, "camera_neck", None)
    vt = getattr(model, "view_transform", None)
    bev = getattr(cfg, "bev", None)
    pretrained = bool(getattr(cfg, "pretrained_backbone", False))
    return {
        "backbone": getattr(backbone, "name", getattr(cfg, "camera_backbone", "")),
        "pretrained_backbone": pretrained,
        "camera_init_policy": "imagenet_pretrained" if pretrained else "scratch",
        "freeze_camera_backbone_config": bool(getattr(cfg, "freeze_camera_backbone", False)),
        "backbone_frozen_runtime": bool(getattr(backbone, "frozen", False)),
        "backbone_out_channels": list(getattr(backbone, "out_channels", [])),
        "backbone_strides": list(getattr(backbone, "strides", [])),
        "image_hw": list(getattr(cfg, "image_hw", ())),
        "neck_out_channels": int(getattr(neck, "out_channels", -1)),
        "neck_out_stride": int(getattr(neck, "out_stride", -1)),
        "view_transform_fhw": [int(getattr(vt, "fH", -1)), int(getattr(vt, "fW", -1))],
        "view_transform_depth_bins": int(getattr(vt, "D", -1)),
        "context_channels": int(getattr(vt, "context_channels", getattr(cfg, "context_channels", -1))),
        "camera_bev_grid": [int(getattr(bev, "ny", -1)), int(getattr(bev, "nx", -1))],
        "head_heatmap_grid": [int(getattr(bev, "head_ny", -1)), int(getattr(bev, "head_nx", -1))],
        "fuser": fuser_contract(model),
    }


def camera_batch_summary(batch: dict) -> Dict[str, Any]:
    images = batch.get("images")
    cam_order = batch.get("cam_order", [])
    sample_order = list(cam_order[0]) if cam_order and isinstance(cam_order, list) else []
    try:
        image_shape = tuple(images.shape)
        batch_size = int(images.shape[0])
        camera_count = int(images.shape[1])
    except Exception:
        image_shape = ()
        batch_size = int(batch.get("batch_size", 0) or 0)
        camera_count = 0
    return {
        "batch_size": batch_size,
        "camera_count": camera_count,
        "raw_image_shape": image_shape,
        "cam_order_first_sample": sample_order,
    }


def projection_meta(model) -> Dict[str, Any]:
    vt = getattr(model, "view_transform", None)
    return dict(getattr(vt, "last_projection_meta", {}) or {})


def model_shape_summary(cfg: dict, model) -> Dict[str, Any]:
    bev = model.cfg.bev
    enc = getattr(model, "lidar_encoder", None)
    sparse_nz = getattr(enc, "nz", None)
    sparse_shape_xyz = [bev.nx, bev.ny, int(sparse_nz)] if sparse_nz is not None else None
    voxel_z = getattr(enc, "vz", None)
    out = {
        "point_cloud_range": [bev.x_min, bev.y_min, bev.z_min, bev.x_max, bev.y_max, bev.z_max],
        "bev_voxel_xy": [bev.vx, bev.vy],
        "voxel_size_xyz": [bev.vx, bev.vy, voxel_z] if voxel_z is not None else None,
        "bev_grid": [bev.nx, bev.ny],
        "head_heatmap_grid": [bev.head_nx, bev.head_ny],
        "sparse_shape_order_xyz": sparse_shape_xyz,
        "spconv_spatial_shape_order_zyx": (
            [int(sparse_nz), bev.ny, bev.nx] if sparse_nz is not None else None
        ),
        "max_voxels_active": int(cfg.get("det-max-pillars", 0)),
        "fuser": fuser_contract(model),
        "camera": camera_model_summary(model),
    }
    if sparse_nz is not None:
        out["computed_z_bins"] = int(getattr(enc, "computed_nz", sparse_nz))
    return out


def parity_summary(cfg: dict, model) -> Dict[str, Any]:
    return {
        "required": dict(BEVFUSION_075_REFERENCE),
        "actual": model_shape_summary(cfg, model),
        "recorded_reference": {
            "max_voxels": cfg.get("bevfusion-max-voxels-reference"),
            "head_grid_size": cfg.get("bevfusion-head-grid-size-reference"),
        },
    }


def _float_list(vals: Iterable[Any]) -> List[float]:
    return [float(v) for v in vals]


def validate_bevfusion_075_parity(cfg: dict, model) -> List[str]:
    if not cfg.get("bevfusion-parity-075", False):
        return []
    actual = model_shape_summary(cfg, model)
    required = BEVFUSION_075_REFERENCE
    errors: List[str] = []
    if actual["voxel_size_xyz"] != required["voxel_size"]:
        errors.append(f"voxel_size {actual['voxel_size_xyz']} != {required['voxel_size']}")
    if _float_list(actual["point_cloud_range"]) != required["point_cloud_range"]:
        errors.append(f"point_cloud_range {actual['point_cloud_range']} != {required['point_cloud_range']}")
    if actual["sparse_shape_order_xyz"] != required["sparse_shape"]:
        errors.append(f"sparse_shape {actual['sparse_shape_order_xyz']} != {required['sparse_shape']}")
    if list(cfg.get("bevfusion-head-grid-size-reference", [])) != required["head_grid_size"]:
        errors.append(
            f"head_grid_size_reference {cfg.get('bevfusion-head-grid-size-reference')} "
            f"!= {required['head_grid_size']}"
        )
    if list(cfg.get("bevfusion-max-voxels-reference", [])) != required["max_voxels"]:
        errors.append(
            f"max_voxels_reference {cfg.get('bevfusion-max-voxels-reference')} "
            f"!= {required['max_voxels']}"
        )
    if actual["fuser"]["lidar_channels"] != required["fuser_lidar_channels"]:
        errors.append(
            f"fuser lidar_channels {actual['fuser']['lidar_channels']} "
            f"!= {required['fuser_lidar_channels']}"
        )
    return errors
