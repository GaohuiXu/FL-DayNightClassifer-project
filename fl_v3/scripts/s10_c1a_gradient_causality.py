#!/usr/bin/env python3
"""Execute the bounded S10 C1-A GN-versus-BN1d causal diagnostic.

The experiment reuses the frozen STOP-B L-S075 physical-B4 panel in uniform
FP32.  It compares the current tiny-group GroupNorm against MIT-reference
BatchNorm1d with identical convolution and affine parameters under (1) the
normal detection loss and (2) a coordinate-derived fixed upstream gradient at
the SECOND output.  No optimizer, update, evaluator, or checkpoint is created.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import statistics
import sys
import time
from typing import Any, Mapping, Sequence

sys.path.insert(0, "fl_v3/src")

import torch
import torch.nn as nn

from fl_v3.config import load_resolved_config, verify_physical_data_identities
from fl_v3.data.nuscenes.s10_binding import load_frozen_split_role, load_frozen_stop_b_panel
from fl_v3.models.fusion.second_sparse_backbone import SECONDSparseBackbone
from fl_v3.training.loop import _move_to_device
from fl_v3.training.s10_observation import (
    classify_c1a_gradient_causality,
    loss_term_snapshot,
    module_state_sha256,
    paired_c1a_reduction,
    spearman_rank_correlation,
    strict_json_value,
    validate_c1a_batch_norm_state_mapping,
    zero_model_gradients,
)
from fl_v3.training.tasks import NuScenesDetectionTask
from fl_v3.utils.runtime import (
    enforce_determinism,
    seed_everything,
    verify_runtime_dependency_identity,
)


SCHEMA = "fl_v3.s10.c1a_gradient_causality.v1"
CANDIDATES = ("group_norm", "batch_norm_1d")
PATHWAYS = ("normal_loss", "fixed_second_output_vjp")
REPEATS = 2
EXPECTED_GN_W0 = "a1a98033131d5496308f0a2694032a1473d582d3435cabd9db285f60b357ef0a"
BOUNDARIES = (
    "head.input",
    "bev_neck.input",
    "second.output",
    "second.stage4",
    "second.down3",
    "second.stage3",
    "second.down2",
    "second.stage2",
    "second.down1",
    "second.stage1",
    "second.stem",
)
VJP_BOUNDARIES = BOUNDARIES[2:]
BACKBONE_PREFIXES = (
    "stem",
    "stage1",
    "down1",
    "stage2",
    "down2",
    "stage3",
    "down3",
    "stage4",
    "conv_out",
)


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--split-manifest", required=True)
    parser.add_argument("--split-sha256", required=True)
    parser.add_argument("--panel-manifest", required=True)
    parser.add_argument("--panel-file-sha256", required=True)
    parser.add_argument("--panel-content-sha256", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--source-sha", required=True)
    parser.add_argument("--source-tree", required=True)
    return parser.parse_args()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(strict_json_value(value), sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _append_jsonl(path: Path, value: Any) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(strict_json_value(value), sort_keys=True, allow_nan=False) + "\n")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parameter_state_sha256(module: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(module.named_parameters()):
        tensor = value.detach().contiguous().cpu()
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(str(tensor.dtype).encode("ascii") + b"\0")
        digest.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode("ascii") + b"\0")
        digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _runtime_assertions() -> dict[str, Any]:
    if platform.machine() != "aarch64":
        raise RuntimeError("C1-A requires an aarch64 Arrhenius compute node")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("C1-A requires exactly one visible CUDA device")
    device = torch.device("cuda", 0)
    properties = torch.cuda.get_device_properties(device)
    if "GH200" not in properties.name.upper():
        raise RuntimeError(f"C1-A expected GH200, got {properties.name!r}")
    return {
        "machine": platform.machine(),
        "node": platform.node(),
        "device": str(device),
        "device_name": properties.name,
        "device_total_memory": int(properties.total_memory),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_job_gpus": os.environ.get("SLURM_JOB_GPUS"),
        "slurm_gpus_on_node": os.environ.get("SLURM_GPUS_ON_NODE"),
    }


def _expected_split_sources(config) -> dict[str, Any]:
    data = config.data["data"]
    return {
        "detection_config_sha256": "217f96cca4e80f790c4674ef72257a6863ee9a85b0ce185bc56488afc32c7a0b",
        "version": data["version"],
        "n_sweeps": data["n_sweeps"],
        "train_cache_logical_sha256": data["caches"]["train"]["logical_sha256"],
        "train_cache_pickle_sha256": data["caches"]["train"]["pickle_sha256"],
        "train_cache_sidecar_sha256": data["caches"]["train"]["sidecar_sha256"],
        "val_cache_logical_sha256": data["caches"]["val"]["logical_sha256"],
        "val_cache_pickle_sha256": data["caches"]["val"]["pickle_sha256"],
        "val_cache_sidecar_sha256": data["caches"]["val"]["sidecar_sha256"],
        "zip_manifest_logical_sha256": data["zip_manifest"]["logical_sha256"],
        "zip_manifest_file_sha256": data["zip_manifest"]["file_sha256"],
    }


def _assert_config(config) -> None:
    expected_model = {
        "mode": "lidar_only",
        "camera_arch": "none",
        "camera_pretrained": None,
        "camera_activation_checkpoint": False,
        "lidar_arch": "second_075",
        "fusion_arch": "none",
        "head_arch": "centerhead_multitask",
    }
    if config.precision != "fp32" or config.sparse_conv_precision != "fp32":
        raise RuntimeError("C1-A requires uniform FP32")
    if dict(config.data["model"]) != expected_model:
        raise RuntimeError(f"C1-A L-S075 graph drift: {dict(config.data['model'])}")
    training = config.data["training"]
    if (
        int(training["micro_batch_size"]) != 4
        or int(training["world_size"]) != 1
        or int(training["accumulation_steps"]) != 1
        or int(training["effective_global_batch"]) != 4
        or int(training["seed"]) != 0
        or str(training["sampling"]) != "uniform"
        or training["ema_decay"] is not None
    ):
        raise RuntimeError("C1-A B4/W0/uniform/no-EMA config drift")


class _C1BoundaryRecorder:
    """Explicit C1-A boundary recorder; normalization summaries are separate."""

    def __init__(self):
        self.boundaries: dict[str, dict[str, Any]] = {}

    def _add(self, name, tensor, sample_index, batch_size, spatial_shape):
        if name not in BOUNDARIES or name in self.boundaries:
            raise RuntimeError(f"C1-A unexpected or duplicate boundary {name}")
        if not tensor.requires_grad:
            raise RuntimeError(f"C1-A boundary {name} has no autograd")
        tensor.retain_grad()
        self.boundaries[name] = {
            "tensor": tensor,
            "sample_index": sample_index,
            "batch_size": int(batch_size),
            "spatial_shape": tuple(int(value) for value in spatial_shape),
        }

    def capture_dense_boundary(self, name, tensor):
        batch_size = int(tensor.shape[0])
        sample_index = torch.arange(batch_size, device=tensor.device, dtype=torch.int64)
        self._add(name, tensor, sample_index, batch_size, tensor.shape[2:])

    def capture_sparse_boundary(self, name, features, indices, batch_size, spatial_shape):
        if indices.ndim != 2 or indices.shape[1] != 4 or indices.shape[0] != features.shape[0]:
            raise RuntimeError(f"C1-A sparse boundary {name} feature/index drift")
        self._add(name, features, indices[:, 0].to(torch.int64), batch_size, spatial_shape)
        self.boundaries[name]["indices"] = indices

    def record_group_norm(self, name, value, output, module):
        # The paired normalizer hook below records both GN and BN1d identically.
        return None

    def validate(self):
        if set(self.boundaries) != set(BOUNDARIES):
            raise RuntimeError(
                f"C1-A boundary drift: missing={sorted(set(BOUNDARIES)-set(self.boundaries))}, "
                f"extra={sorted(set(self.boundaries)-set(BOUNDARIES))}"
            )


@contextmanager
def _capture_normalizers(backbone: nn.Module, expected_kind: str):
    summaries: dict[str, tuple[dict[str, Any], torch.Tensor]] = {}
    handles = []

    def register(name: str, module: nn.Module):
        def hook(_module, inputs, output):
            if name in summaries:
                raise RuntimeError(f"C1-A normalizer {name} executed more than once")
            value = inputs[0]
            if value.ndim != 2:
                raise RuntimeError(f"C1-A normalizer {name} expected [N,C], got {tuple(value.shape)}")
            detached = value.detach().to(torch.float32)
            out = output.detach().to(torch.float32)
            if isinstance(module, nn.GroupNorm):
                groups = int(module.num_groups)
                variance = detached.reshape(detached.shape[0], groups, detached.shape[1] // groups).var(
                    dim=2, unbiased=False
                )
                metadata = {"kind": "group_norm", "groups": groups, "units": int(variance.numel())}
            elif isinstance(module, nn.BatchNorm1d):
                variance = detached.var(dim=0, unbiased=False)
                metadata = {"kind": "batch_norm_1d", "groups": None, "units": int(variance.numel())}
            else:
                raise RuntimeError(f"C1-A unsupported normalizer type {type(module)!r}")
            inv_std = torch.rsqrt(variance + float(module.eps))
            finite = torch.isfinite(detached)
            finite_out = torch.isfinite(out)
            vector = torch.stack((
                variance.min(),
                variance.mean(),
                variance.max(),
                inv_std.mean(),
                inv_std.max(),
                (variance <= float(module.eps) * 10.0).sum().to(torch.float32),
                torch.tensor(float(variance.numel()), device=value.device),
                torch.sqrt(torch.square(detached[finite]).mean()),
                detached[finite].abs().max(),
                torch.sqrt(torch.square(out[finite_out]).mean()),
                out[finite_out].abs().max(),
            )).detach()
            metadata.update({
                "rows": int(value.shape[0]),
                "channels": int(value.shape[1]),
                "eps": float(module.eps),
            })
            summaries[name] = metadata, vector

        handles.append(module.register_forward_hook(hook))

    for name, module in backbone.named_modules():
        if isinstance(module, (nn.GroupNorm, nn.BatchNorm1d)):
            register(name, module)
    if len(handles) != 21:
        raise RuntimeError(f"C1-A expected 21 SECOND normalizers, got {len(handles)}")
    try:
        yield summaries
    finally:
        for handle in handles:
            handle.remove()
        kinds = {metadata["kind"] for metadata, _ in summaries.values()}
        if len(summaries) != 21 or kinds != {expected_kind}:
            raise RuntimeError(
                f"C1-A normalizer capture drift: count={len(summaries)}, kinds={sorted(kinds)}"
            )


def _finalize_normalizers(summaries) -> dict[str, Any]:
    names = sorted(summaries)
    matrix = torch.stack([summaries[name][1] for name in names]).cpu().tolist()
    fields = (
        "variance_min", "variance_mean", "variance_max", "inv_std_mean", "inv_std_max",
        "variance_le_10x_eps", "variance_units", "input_rms", "input_max_abs",
        "output_rms", "output_max_abs",
    )
    result = {}
    for name, values in zip(names, matrix, strict=True):
        metadata = dict(summaries[name][0])
        metadata.update(dict(zip(fields, values, strict=True)))
        result[name] = metadata
    return result


def _stats_vectors(tensors: Mapping[str, torch.Tensor]) -> dict[str, Any]:
    names = list(tensors)
    vectors = []
    for name in names:
        value = tensors[name].detach()
        finite = torch.isfinite(value)
        selected = value[finite].to(torch.float64)
        vectors.append(torch.stack((
            torch.tensor(float(value.numel()), device=value.device, dtype=torch.float64),
            finite.sum().to(torch.float64),
            torch.square(selected).sum(),
            selected.abs().max() if selected.numel() else torch.zeros((), device=value.device, dtype=torch.float64),
        )))
    matrix = torch.stack(vectors).cpu().tolist()
    result = {}
    for name, (total, finite, sum_sq, maximum) in zip(names, matrix, strict=True):
        total_i, finite_i = int(total), int(finite)
        result[name] = {
            "total_elements": total_i,
            "finite_elements": finite_i,
            "all_finite": total_i == finite_i,
            "stable_finite_l2": math.sqrt(max(0.0, sum_sq)),
            "stable_finite_rms": math.sqrt(max(0.0, sum_sq / finite_i)) if finite_i else 0.0,
            "max_abs_finite": maximum,
        }
    return result


def _boundary_gradient_stats(recorder: _C1BoundaryRecorder, pathway: str) -> dict[str, Any]:
    names = BOUNDARIES if pathway == "normal_loss" else VJP_BOUNDARIES
    tensors = {}
    for name in names:
        gradient = recorder.boundaries[name]["tensor"].grad
        if gradient is None:
            raise RuntimeError(f"C1-A {pathway} boundary {name} has no gradient")
        tensors[name] = gradient
    return _stats_vectors(tensors)


def _backbone_parameter_gradient_stats(model) -> dict[str, Any]:
    backbone = model.lidar_encoder.backbone
    entries: list[tuple[str, str, torch.Tensor]] = []
    missing = []
    for name, parameter in backbone.named_parameters():
        if parameter.grad is None:
            missing.append(name)
            continue
        prefix = name.split(".", 1)[0]
        if prefix not in BACKBONE_PREFIXES:
            raise RuntimeError(f"C1-A unexpected SECOND parameter prefix {prefix}")
        entries.append((name, prefix, parameter.grad))
    if missing:
        raise RuntimeError(f"C1-A missing SECOND parameter gradients: {missing}")
    individual = _stats_vectors({name: gradient for name, _, gradient in entries})
    result = {}
    for prefix in BACKBONE_PREFIXES:
        items = [individual[name] for name, group, _ in entries if group == prefix]
        if not items:
            raise RuntimeError(f"C1-A empty SECOND parameter group {prefix}")
        total = sum(item["total_elements"] for item in items)
        finite = sum(item["finite_elements"] for item in items)
        sum_sq = sum(item["stable_finite_l2"] ** 2 for item in items)
        result[prefix] = {
            "total_elements": total,
            "finite_elements": finite,
            "all_finite": total == finite,
            "stable_finite_l2": math.sqrt(max(0.0, sum_sq)),
            "stable_finite_rms": math.sqrt(max(0.0, sum_sq / finite)) if finite else 0.0,
            "max_abs_finite": max(item["max_abs_finite"] for item in items),
        }
    return result


def _fixed_upstream(boundary: Mapping[str, Any]) -> torch.Tensor:
    tensor = boundary["tensor"]
    indices = boundary.get("indices")
    if indices is None or tensor.ndim != 2:
        raise RuntimeError("C1-A fixed VJP requires sparse SECOND-output indices/features")
    prime = 2_147_483_647
    coords = indices.to(torch.int64)
    key = (((coords[:, 0] * 43 + coords[:, 1]) * 2003 + coords[:, 2]) * 2003 + coords[:, 3])
    row_hash = torch.remainder(torch.remainder(key, prime) * 48_271, prime)
    channel = torch.arange(tensor.shape[1], device=tensor.device, dtype=torch.int64)
    mixed = torch.remainder(row_hash[:, None] + channel[None, :] * 69_621, prime)
    sign = torch.remainder(mixed * 48_271, prime).bitwise_and(1).to(tensor.dtype)
    return sign.mul_(2.0).sub_(1.0)


def _reset_bn_running_state(model) -> None:
    for module in model.lidar_encoder.backbone.modules():
        if isinstance(module, nn.BatchNorm1d):
            module.reset_running_stats()


def _clear_criterion(criterion) -> None:
    criterion.last_s10_terms = {}
    for child in criterion.losses:
        child.last_s10_terms = {}
        child._s10_focal = {}


def _run_once(
    model,
    criterion,
    batch_cpu,
    *,
    candidate: str,
    pathway: str,
    batch_index: int,
    repeat: int,
    device: torch.device,
) -> dict[str, Any]:
    seed_everything(20_260_719 + batch_index * 10 + repeat)
    zero_model_gradients(model)
    if candidate == "batch_norm_1d":
        _reset_bn_running_state(model)
    batch = _move_to_device(batch_cpu, device)
    recorder = _C1BoundaryRecorder()
    loss_value = None
    loss_terms = None
    started = time.perf_counter()
    with _capture_normalizers(model.lidar_encoder.backbone, candidate) as normalizers:
        with model.capture_s10_observation(recorder):
            if pathway == "normal_loss":
                with criterion.capture_s10_terms():
                    output = model(batch)
                    loss = criterion(output, batch)
                loss.backward()
                loss_value = float(loss.detach().item())
                if repeat == 0:
                    loss_terms = loss_term_snapshot(criterion.s10_term_bundle())
            elif pathway == "fixed_second_output_vjp":
                output = model(batch)
                recorder.validate()
                upstream = _fixed_upstream(recorder.boundaries["second.output"])
                torch.autograd.backward(recorder.boundaries["second.output"]["tensor"], upstream)
                upstream_contract = {
                    "domain": "SECOND output active sparse coordinate+channel",
                    "values": "Rademacher {-1,+1}",
                    "coordinate_hash_prime": 2_147_483_647,
                    "coordinate_multiplier": 48_271,
                    "channel_multiplier": 69_621,
                    "shape": list(upstream.shape),
                    "rms": 1.0,
                }
            else:
                raise RuntimeError(f"C1-A unknown pathway {pathway}")
    recorder.validate()
    torch.cuda.synchronize(device)
    boundary_gradients = _boundary_gradient_stats(recorder, pathway)
    parameter_gradients = _backbone_parameter_gradient_stats(model)
    normalization = _finalize_normalizers(normalizers)
    second = boundary_gradients["second.output"]
    stem_boundary = boundary_gradients["second.stem"]
    stem_parameter = parameter_gradients["stem"]
    if second["stable_finite_rms"] <= 0.0:
        raise RuntimeError("C1-A SECOND-output gradient RMS must be positive")
    active_per_sample = torch.bincount(
        recorder.boundaries["second.stem"]["sample_index"], minlength=4
    ).cpu().tolist()
    record = {
        "schema": SCHEMA,
        "candidate": candidate,
        "pathway": pathway,
        "batch_index": batch_index,
        "repeat": repeat,
        "sample_tokens": [str(value) for value in batch_cpu["sample_token"]],
        "physical_batch": int(batch_cpu["batch_size"]),
        "loss": loss_value,
        "loss_terms": loss_terms,
        "fixed_upstream": upstream_contract if pathway == "fixed_second_output_vjp" else None,
        "active_stem_rows_per_sample": [int(value) for value in active_per_sample],
        "active_stem_rows": int(sum(active_per_sample)),
        "boundary_gradients": boundary_gradients,
        "backbone_parameter_gradients": parameter_gradients,
        "normalization": normalization,
        "classification_metrics": {
            "stem_parameter_max_abs": stem_parameter["max_abs_finite"],
            "stem_parameter_rms": stem_parameter["stable_finite_rms"],
            "boundary_amplification": stem_boundary["stable_finite_rms"] / second["stable_finite_rms"],
            "second_output_gradient_rms": second["stable_finite_rms"],
            "stem_boundary_gradient_rms": stem_boundary["stable_finite_rms"],
        },
        "all_second_gradients_finite": all(item["all_finite"] for item in parameter_gradients.values()),
        "parameter_state_check": "exact candidate-level hash before first and after final run",
        "optimizer_constructed": False,
        "optimizer_update": False,
        "evaluator_executed": False,
        "wall_seconds": time.perf_counter() - started,
    }
    zero_model_gradients(model)
    _clear_criterion(criterion)
    del batch, output, recorder
    if pathway == "normal_loss":
        del loss
    else:
        del upstream
    return record


def _install_batch_norm_1d(model) -> dict[str, Any]:
    old = model.lidar_encoder.backbone
    before_hash = _parameter_state_sha256(model)
    replacement = SECONDSparseBackbone(
        model.lidar_encoder.n_pt_feat,
        model.lidar_encoder.contract,
        normalization="batch_norm_1d",
    )
    incompatible = replacement.load_state_dict(old.state_dict(), strict=False)
    running_state = validate_c1a_batch_norm_state_mapping(
        replacement,
        missing_keys=incompatible.missing_keys,
        unexpected_keys=incompatible.unexpected_keys,
        expected_sites=21,
    )
    model.lidar_encoder.backbone = replacement
    after_hash = _parameter_state_sha256(model)
    if before_hash != after_hash:
        raise RuntimeError("C1-A BN1d replacement changed a trainable parameter")
    return {
        "source": "MIT BEVFusion SparseEncoder norm_cfg",
        "normalization": "torch.nn.BatchNorm1d",
        "eps": 1e-3,
        "momentum": 0.01,
        "affine_parameters_copied_exactly": True,
        "convolution_parameters_copied_exactly": True,
        "parameter_state_sha256": after_hash,
        "new_running_state": running_state,
    }


def _pairs(records, candidate, pathway, metric, batch_count):
    grouped = {index: [] for index in range(batch_count)}
    for record in records:
        if record["candidate"] == candidate and record["pathway"] == pathway:
            grouped[record["batch_index"]].append(
                (record["repeat"], record["classification_metrics"][metric])
            )
    result = []
    for index in range(batch_count):
        values = [value for _, value in sorted(grouped[index])]
        if len(values) != REPEATS:
            raise RuntimeError(f"C1-A incomplete repeats for {candidate}/{pathway}/batch{index}")
        result.append(values)
    return result


def _centres(records, candidate, pathway, field, batch_count):
    grouped = {index: [] for index in range(batch_count)}
    for record in records:
        if record["candidate"] == candidate and record["pathway"] == pathway:
            if field == "active_stem_rows":
                value = float(record[field])
            else:
                value = float(record["classification_metrics"][field])
            grouped[record["batch_index"]].append(value)
    result = []
    for index in range(batch_count):
        values = grouped[index]
        if len(values) != REPEATS:
            raise RuntimeError(f"C1-A centre missing repeats for {candidate}/{pathway}/batch{index}")
        result.append(math.sqrt(max(1e-30, values[0]) * max(1e-30, values[1])))
    return result


def _artifact_manifest(root: Path) -> dict[str, Any]:
    files = {}
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = str(path.relative_to(root))
        if relative == "artifact_sha256s.json":
            continue
        files[relative] = {"sha256": _sha256_file(path), "bytes": path.stat().st_size}
    return {"schema": "fl_v3.s10.c1a_artifacts.v1", "files": files}


def _execute(args, output: Path, raw: Path) -> None:
    runtime = _runtime_assertions()
    config = load_resolved_config(args.config)
    _assert_config(config)
    verify_physical_data_identities(config)
    runtime_dependencies = verify_runtime_dependency_identity(config.to_run_config())
    binding = load_frozen_split_role(
        args.split_manifest,
        expected_manifest_sha256=args.split_sha256,
        role="D_low",
        expected_source_identities=_expected_split_sources(config),
    )
    panel, panel_report = load_frozen_stop_b_panel(
        args.panel_manifest,
        expected_file_sha256=args.panel_file_sha256,
        expected_content_sha256=args.panel_content_sha256,
        binding=binding,
    )
    expected_batches = [*panel["batches_b4"]["P_core"], *panel["batches_b4"]["P_term"]]
    tokens = [token for batch in expected_batches for token in batch]
    if len(expected_batches) != 16 or len(tokens) != 64 or len(set(tokens)) != 64:
        raise RuntimeError("C1-A frozen panel must contain exactly 16 disjoint B4 batches")

    identity = {
        "schema": SCHEMA,
        "source_sha": args.source_sha,
        "source_tree": args.source_tree,
        "runtime": runtime,
        "runtime_dependencies": runtime_dependencies,
        "config_path": str(Path(args.config).resolve()),
        "config_resolved_sha256": config.sha256,
        "split_binding": binding.identity(),
        "panel": panel_report,
        "panel_batches": 16,
        "panel_samples": 64,
        "candidates": list(CANDIDATES),
        "pathways": list(PATHWAYS),
        "repeats": REPEATS,
        "expected_runs": 128,
        "precision": "fp32",
        "physical_microbatch": 4,
        "optimizer_constructed": False,
        "evaluator_executed": False,
    }
    _write_json(output / "execution_identity.json", identity)

    enforce_determinism(strict=True, precision="fp32")
    device = torch.device("cuda", 0)
    task = NuScenesDetectionTask()
    run_config = config.to_run_config()
    models = {}
    criteria = {}
    candidate_identity = {}
    for candidate in CANDIDATES:
        seed_everything(0)
        model = task.build_model(run_config)
        current_w0 = module_state_sha256(model)
        if current_w0 != EXPECTED_GN_W0:
            raise RuntimeError(f"C1-A L-S075 W0 drift: expected={EXPECTED_GN_W0}, actual={current_w0}")
        mapping = None
        if candidate == "batch_norm_1d":
            mapping = _install_batch_norm_1d(model)
        parameter_hash = _parameter_state_sha256(model)
        model.to(device)
        model.train()
        models[candidate] = model
        criteria[candidate] = task.build_criterion(run_config)
        candidate_identity[candidate] = {
            "full_state_sha256_before_candidate_transform": current_w0,
            "parameter_state_sha256": parameter_hash,
            "normalization_mapping": mapping,
            "normalization_count": sum(
                isinstance(module, (nn.GroupNorm, nn.BatchNorm1d))
                for module in model.lidar_encoder.backbone.modules()
            ),
        }
    if candidate_identity["group_norm"]["parameter_state_sha256"] != candidate_identity["batch_norm_1d"]["parameter_state_sha256"]:
        raise RuntimeError("C1-A candidates do not share exact trainable parameters")
    _write_json(output / "candidate_identity.json", candidate_identity)

    loader = task.fixed_train_subset_loader(run_config, tokens)
    records = []
    total_started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats(device)
    for batch_index, (batch_cpu, expected_tokens) in enumerate(zip(loader, expected_batches, strict=True)):
        actual_tokens = [str(value) for value in batch_cpu["sample_token"]]
        if actual_tokens != list(expected_tokens) or int(batch_cpu["batch_size"]) != 4:
            raise RuntimeError(
                f"C1-A batch {batch_index} token/B4 drift: expected={expected_tokens}, actual={actual_tokens}"
            )
        for pathway_index, pathway in enumerate(PATHWAYS):
            for repeat in range(REPEATS):
                order = CANDIDATES if (batch_index + pathway_index + repeat) % 2 == 0 else tuple(reversed(CANDIDATES))
                for candidate in order:
                    record = _run_once(
                        models[candidate],
                        criteria[candidate],
                        batch_cpu,
                        candidate=candidate,
                        pathway=pathway,
                        batch_index=batch_index,
                        repeat=repeat,
                        device=device,
                    )
                    _append_jsonl(raw / "runs.jsonl", record)
                    records.append(record)
        del batch_cpu
        gc.collect()

    final_parameter_hashes = {
        candidate: _parameter_state_sha256(model)
        for candidate, model in models.items()
    }
    parameter_state_unchanged = all(
        final_parameter_hashes[candidate]
        == candidate_identity[candidate]["parameter_state_sha256"]
        for candidate in CANDIDATES
    )

    expected_runs = len(CANDIDATES) * len(PATHWAYS) * len(expected_batches) * REPEATS
    if len(records) != expected_runs:
        raise RuntimeError(f"C1-A expected {expected_runs} runs, got {len(records)}")
    metrics = ("stem_parameter_max_abs", "stem_parameter_rms", "boundary_amplification")
    effects = {}
    for pathway in PATHWAYS:
        effects[pathway] = {
            metric: paired_c1a_reduction(
                _pairs(records, "group_norm", pathway, metric, len(expected_batches)),
                _pairs(records, "batch_norm_1d", pathway, metric, len(expected_batches)),
            )
            for metric in metrics
        }

    occupancy_correlations = {}
    for candidate in CANDIDATES:
        for pathway in PATHWAYS:
            occupancy = _centres(records, candidate, pathway, "active_stem_rows", len(expected_batches))
            amplification = _centres(records, candidate, pathway, "boundary_amplification", len(expected_batches))
            occupancy_correlations[f"{candidate}.{pathway}"] = spearman_rank_correlation(
                occupancy, amplification
            )
    gn_loss_upstream = _centres(
        records, "group_norm", "normal_loss", "second_output_gradient_rms", len(expected_batches)
    )
    gn_loss_stem = _centres(
        records, "group_norm", "normal_loss", "stem_parameter_max_abs", len(expected_batches)
    )
    classification = classify_c1a_gradient_causality(
        loss_effects=effects["normal_loss"],
        vjp_effects=effects["fixed_second_output_vjp"],
        occupancy_correlations=occupancy_correlations,
        loss_upstream_stem_correlation=spearman_rank_correlation(gn_loss_upstream, gn_loss_stem),
        current_loss_stem_max_abs_median=statistics.median(gn_loss_stem),
    )
    integrity_pass = bool(
        len(records) == 128
        and all(record["physical_batch"] == 4 for record in records)
        and all(record["all_second_gradients_finite"] for record in records)
        and parameter_state_unchanged
        and all(len(record["normalization"]) == 21 for record in records)
    )
    summary = {
        "schema": SCHEMA,
        "status": "PASS" if integrity_pass else "FAIL",
        "c1a_verdict": classification["label"],
        "classification": classification,
        "paired_effects": effects,
        "occupancy_correlations": occupancy_correlations,
        "final_parameter_state_sha256": final_parameter_hashes,
        "parameter_state_unchanged": parameter_state_unchanged,
        "integrity_pass": integrity_pass,
        "total_runs": len(records),
        "total_wall_seconds": time.perf_counter() - total_started,
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        "optimizer_constructed": False,
        "optimizer_updates": 0,
        "evaluator_executed": False,
        "allowed_interpretation": "bounded C1-A normalization/head-loss/occupancy mechanism localization or honest INCONCLUSIVE",
        "forbidden_interpretation": [
            "convergence", "model capability", "production architecture selection",
            "production recipe selection", "official-val claim", "automatic C1-B continuation",
        ],
    }
    _write_json(output / "summary.json", summary)
    _write_json(output / "artifact_sha256s.json", _artifact_manifest(output))
    if not integrity_pass:
        raise RuntimeError("C1-A integrity gate failed")


def main() -> None:
    args = _parse_args()
    output = Path(args.output_dir).resolve()
    if output.exists():
        raise RuntimeError(f"C1-A output must be fresh: {output}")
    output.mkdir(parents=True)
    raw = output / "raw"
    raw.mkdir()
    try:
        _execute(args, output, raw)
    except BaseException as exc:
        _write_json(output / "failure_summary.json", {
            "schema": "fl_v3.s10.c1a_failure.v1",
            "exception_type": type(exc).__name__,
            "exception_message": str(exc),
        })
        _write_json(output / "artifact_sha256s.json", _artifact_manifest(output))
        raise


if __name__ == "__main__":
    main()
