#!/usr/bin/env python3
"""Bounded S10 Phase-I Envelope-A engineering calibration.

This entry point deliberately has no capability-evaluation path.  It consumes
only four fixed D_fit B4 batches, performs zero optimizer updates, and emits
implementation/parity/performance evidence for the later measured Envelope-B
request.  Camera and LiDAR are separate invocations (Job A and Job B).
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import gc
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
from typing import Any, Iterable, Mapping, Sequence

sys.path.insert(0, "fl_v3/src")

import numpy as np
import torch
import torch.nn as nn

from fl_v3.config import load_resolved_config, resolve_config
from fl_v3.config.phase1 import REFERENCE_OBJECT_CLASSES, phase1_runtime_ready
from fl_v3.data.nuscenes.phase1 import build_phase1_train_data
from fl_v3.eval.box_to_global import decoded_sample_to_boxes
from fl_v3.models.ops.bev_pool import (
    bev_pool,
    bev_pool_build_identity,
    load_optimized_extension,
)
from fl_v3.models.phase1_camera import build_phase1_camera_model
from fl_v3.models.phase1_lidar import build_phase1_lidar_model
from fl_v3.models.phase1_swin import sha256_file, tensor_state_sha256
from fl_v3.training.checkpoint import load_checkpoint, save_checkpoint
from fl_v3.training.loop import _float_tensors, _move_to_device
from fl_v3.training.phase1 import Phase1CyclicScheduler, build_phase1_optimizer
from fl_v3.training.runtime_state import TrainingState
from fl_v3.utils.runtime import (
    enforce_determinism,
    precision_autocast_context,
    seed_everything,
    verify_runtime_dependency_identity,
)


SCHEMA = "s10.phase1.envelope-a-calibration.v1"
FIXED_BATCHES = 4
WARMUP_MICROBATCHES = 16
TIMED_MICROBATCHES = 64
ACCUMULATION_STEPS = 8
ALVIS_ROOT = Path(
    "/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/"
    "fl_weather_project/fl_outputs/nuscenes/experiments/cycle_04/"
    "p2_ddp/bb02d_r20/ema_ep15"
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _write_once(path: Path, payload: Any, *, mode: int = 0o400) -> str:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite immutable evidence {path}")
    temporary = path.with_name(path.name + ".partial")
    if temporary.exists():
        raise FileExistsError(f"stale evidence partial exists: {temporary}")
    encoded = _canonical_bytes(payload) + b"\n"
    with temporary.open("xb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.chmod(mode)
    os.replace(temporary, path)
    return sha256_file(path)


def _write_config_once(path: Path, config) -> str:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite resolved config {path}")
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("xb") as stream:
        # ResolvedConfig.sha256 is defined over these exact canonical bytes.
        # Do not append a presentation newline and then compare a different
        # physical byte stream with the canonical identity.
        stream.write(config.canonical_bytes)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.chmod(0o400)
    os.replace(temporary, path)
    _require(sha256_file(path) == config.sha256, "resolved-config physical hash drift")
    return config.sha256


def _distribution(values: Sequence[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "median": None, "p95": None, "minimum": None, "maximum": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "median": float(np.median(array)),
        "p95": float(np.percentile(array, 95)),
        "minimum": float(array.min()),
        "maximum": float(array.max()),
    }


def _tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().to(device="cpu").contiguous()
    digest = hashlib.sha256()
    digest.update(
        _canonical_bytes({"dtype": str(tensor.dtype), "shape": list(tensor.shape)})
    )
    digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _nested_digest(value: Any) -> str:
    digest = hashlib.sha256()

    def visit(item: Any) -> None:
        if torch.is_tensor(item):
            tensor = item.detach().to(device="cpu").contiguous()
            header = _canonical_bytes(
                {"kind": "tensor", "dtype": str(tensor.dtype), "shape": list(tensor.shape)}
            )
            digest.update(len(header).to_bytes(8, "little"))
            digest.update(header)
            digest.update(tensor.numpy().tobytes(order="C"))
        elif isinstance(item, Mapping):
            digest.update(b"mapping\0")
            for key in sorted(item):
                visit(str(key))
                visit(item[key])
        elif isinstance(item, (list, tuple)):
            digest.update(f"sequence:{type(item).__name__}:{len(item)}\0".encode())
            for child in item:
                visit(child)
        else:
            encoded = _canonical_bytes(item)
            digest.update(len(encoded).to_bytes(8, "little"))
            digest.update(encoded)

    visit(value)
    return digest.hexdigest()


def _parameter_sha256(model: nn.Module) -> str:
    return tensor_state_sha256(dict(model.named_parameters()))


def _state_sha256(model: nn.Module) -> str:
    return tensor_state_sha256(model.state_dict())


def _make_scaler(config, device: torch.device) -> torch.amp.GradScaler:
    spec = config.as_dict()["precision"]["grad_scaler"]
    return torch.amp.GradScaler(
        "cuda",
        enabled=bool(spec["enabled"] and device.type == "cuda"),
        init_scale=float(spec["init_scale"]),
        growth_factor=float(spec["growth_factor"]),
        backoff_factor=float(spec["backoff_factor"]),
        growth_interval=int(spec["growth_interval"]),
    )


def _source_identity(expected_sha: str) -> dict[str, str]:
    actual = subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()
    tree = subprocess.run(
        ["git", "rev-parse", "HEAD^{tree}"], check=True, capture_output=True, text=True
    ).stdout.strip()
    branch = subprocess.run(
        ["git", "branch", "--show-current"], check=True, capture_output=True, text=True
    ).stdout.strip()
    _require(actual == expected_sha, f"source SHA drift: {actual} != {expected_sha}")
    _require(branch == "codex/s10-phase1-branch-qualification", "source branch drift")
    return {"git_sha": actual, "git_tree": tree, "branch": branch}


def _validate_camera_initialization_result(path: Path) -> dict[str, Any]:
    result = json.loads(path.read_text(encoding="utf-8"))
    _require(
        result.get("schema") == "s10.phase1.swin-acceptance-result.v1",
        "Camera initialization result schema drift",
    )
    checkpoint = Path(result["physical_path"]).resolve()
    report_path = Path(result["mapping_report_path"]).resolve()
    _require(checkpoint.is_file() and report_path.is_file(), "accepted Camera artifacts missing")
    _require(sha256_file(checkpoint) == result["physical_sha256"], "Camera checkpoint hash drift")
    _require(sha256_file(report_path) == result["mapping_report_sha256"], "mapping report hash drift")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    _require(report["physical_sha256"] == result["physical_sha256"], "mapping physical hash drift")
    _require(
        report["initialization_state_sha256"] == result["initialization_state_sha256"],
        "mapping initialization-state drift",
    )
    _require(report["acquisition"]["count"] == 1, "Camera acquisition count drift")
    _require(
        report["acquisition"]["redirect_host"] == "release-assets.githubusercontent.com",
        "Camera acquisition redirect drift",
    )
    return result


def _derive_materialized_config(
    source_config,
    output_dir: Path,
    *,
    initialization_result: Path | None,
    gtdb_manifest: Path | None,
):
    raw = source_config.as_dict()
    branch = raw["contract"]["branch"]
    if branch == "camera":
        _require(initialization_result is not None, "Camera initialization result is required")
        accepted = _validate_camera_initialization_result(initialization_result)
        raw["initialization"].update(
            {
                "status": "accepted",
                "physical_sha256": accepted["physical_sha256"],
                "mapping_report_sha256": accepted["mapping_report_sha256"],
                "initialization_state_sha256": accepted["initialization_state_sha256"],
            }
        )
    else:
        _require(gtdb_manifest is not None and gtdb_manifest.is_file(), "sealed GTDB manifest is required")
        manifest = json.loads(gtdb_manifest.read_text(encoding="utf-8"))
        _require(manifest.get("schema") == "s10.phase1.gtdb.v1", "GTDB manifest schema drift")
        manifest_sha = sha256_file(gtdb_manifest)
        raw["gt_paste"].update(
            {"database_status": "accepted", "manifest_sha256": manifest_sha}
        )
    materialized = resolve_config(raw)
    path = output_dir / "resolved_config.materialized.json"
    _write_config_once(path, materialized)
    return materialized, path


def _derive_qualified_config(config, output_dir: Path):
    raw = config.as_dict()
    raw["contract"]["lifecycle"] = "envelope_a_qualified"
    raw["precision"]["grad_scaler"]["status"] = "accepted"
    qualified = resolve_config(raw)
    phase1_runtime_ready(qualified.as_dict())
    path = output_dir / "resolved_config.qualified.json"
    _write_config_once(path, qualified)
    return qualified, path


def _build_model(config, branch: str, device: torch.device, *, backend: str, build_dir: str | None):
    seed_everything(0)
    if branch == "camera":
        model = build_phase1_camera_model(
            config,
            pool_backend=backend,
            pool_build_directory=build_dir,
            require_accepted_initialization=True,
        )
    else:
        model = build_phase1_lidar_model(config)
    return model.to(device)


def _fixed_d_fit_batches(config) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    bundle = build_phase1_train_data(config)
    bundle.set_epoch(0)
    iterator = iter(bundle.loader)
    batches: list[dict[str, Any]] = []
    waits: list[float] = []
    try:
        for _ in range(FIXED_BATCHES):
            started = time.perf_counter()
            batch = next(iterator)
            waits.append((time.perf_counter() - started) * 1000.0)
            _require(int(batch["batch_size"]) == 4, "fixed calibration batch is not physical B4")
            batches.append(batch)
    finally:
        del iterator
        bundle.close()
    selected = []
    for batch in batches:
        branch = config.as_dict()["contract"]["branch"]
        content = {
            "sample_token": list(batch["sample_token"]),
            "gt_boxes": batch["gt_boxes"],
            "gt_labels": batch["gt_labels"],
        }
        if branch == "camera":
            content.update(
                {"images": batch["images"], "augmentation_params": batch["augmentation_params"]}
            )
        else:
            content["lidar_points"] = batch["lidar_points"]
        selected.append(
            {
                "sample_tokens": list(batch["sample_token"]),
                "content_sha256": _nested_digest(content),
            }
        )
    return batches, {
        "role": "D_fit",
        "fixed_batches": FIXED_BATCHES,
        "physical_batch": 4,
        "sample_count": FIXED_BATCHES * 4,
        "materialization_wait_ms": _distribution(waits),
        "batches": selected,
        "ordered_tokens_sha256": _canonical_sha256(
            [token for batch in batches for token in batch["sample_token"]]
        ),
    }


def _checkpoint_preflight(
    config,
    branch: str,
    device: torch.device,
    output_dir: Path,
    *,
    build_dir: str | None,
) -> dict[str, Any]:
    model = _build_model(
        config, branch, device, backend="optimized" if branch == "camera" else "fallback", build_dir=build_dir
    )
    optimizer = build_phase1_optimizer(model, config)
    scheduler = Phase1CyclicScheduler(optimizer, config)
    scaler = _make_scaler(config, device)
    state = TrainingState()
    state.validate(checkpoint_boundary=True)
    before = _state_sha256(model)
    checkpoint = output_dir / "engineering_recovery_preflight.pt"
    save_checkpoint(
        str(checkpoint),
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        grad_scaler=scaler,
        ema=None,
        state=state,
        config=config,
        checkpoint_identity=config.sha256,
    )
    checkpoint.chmod(0o400)
    loaded_state, identity = load_checkpoint(
        str(checkpoint),
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        grad_scaler=scaler,
        ema=None,
        config=config,
        map_location="cpu",
    )
    loaded_state.validate(checkpoint_boundary=True)
    after = _state_sha256(model)
    _require(identity == config.sha256, "checkpoint identity drift")
    _require(before == after, "checkpoint model state changed across exact reload")
    _require(loaded_state == state, "checkpoint zero-boundary state drift")
    record = {
        "schema": config.as_dict()["checkpointing"]["schema"],
        "path": str(checkpoint),
        "sha256": sha256_file(checkpoint),
        "checkpoint_identity": identity,
        "model_state_sha256": before,
        "optimizer_state_entries": len(optimizer.state),
        "scheduler_accepted_updates": scheduler.accepted_updates,
        "grad_scaler_scale": float(scaler.get_scale()),
        "training_state": state.checkpoint_dict(),
        "selectable": False,
    }
    del scaler, scheduler, optimizer, model
    gc.collect()
    torch.cuda.empty_cache()
    return record


def _decoded_schema_record(decoded: Mapping[str, torch.Tensor]) -> dict[str, Any]:
    required = {"boxes", "scores", "labels", "velocity"}
    _require(required.issubset(decoded), f"decode fields missing: {sorted(required - set(decoded))}")
    boxes = decoded["boxes"]
    scores = decoded["scores"]
    labels = decoded["labels"]
    velocity = decoded["velocity"]
    _require(boxes.ndim == 2 and boxes.shape[1] == 7, "decode boxes schema drift")
    count = int(boxes.shape[0])
    _require(scores.shape == (count,) and labels.shape == (count,), "decode vector schema drift")
    _require(velocity.shape == (count, 2), "decode velocity schema drift")
    _require(count <= 500, "decode exceeds official per-sample cap")
    for name, tensor in (("boxes", boxes), ("scores", scores), ("velocity", velocity)):
        _require(bool(torch.isfinite(tensor).all().detach().cpu()), f"decode {name} is nonfinite")
    if count:
        _require(bool(((labels >= 0) & (labels < 10)).all().detach().cpu()), "decode labels drift")
    return {
        "count": count,
        "boxes_shape": list(boxes.shape),
        "scores_shape": list(scores.shape),
        "labels_shape": list(labels.shape),
        "velocity_shape": list(velocity.shape),
    }


def _evaluator_schema_preflight(
    config,
    branch: str,
    device: torch.device,
    batch_cpu: dict[str, Any],
    *,
    build_dir: str | None,
) -> dict[str, Any]:
    model = _build_model(
        config, branch, device, backend="optimized" if branch == "camera" else "fallback", build_dir=build_dir
    )
    model.eval()
    moved = _move_to_device(batch_cpu, device)
    with torch.no_grad(), precision_autocast_context("fp16", device):
        output = model(moved, return_intermediates=(branch == "lidar"))
    fp32_output = _float_tensors(output)
    decoded = model.decode(fp32_output)
    _require(len(decoded) == 4, "one-B4 decode count drift")
    sample = _decoded_schema_record(decoded[0])
    boxes = decoded_sample_to_boxes(
        decoded[0],
        batch_cpu["lidar2ego"][0],
        batch_cpu["ego2global_lidar"][0],
        str(batch_cpu["sample_token"][0]),
        REFERENCE_OBJECT_CLASSES,
    )
    serialized = [box.serialize() for box in boxes]
    json.dumps(serialized, allow_nan=False)
    policy: dict[str, Any] = {}
    if branch == "lidar":
        _require(isinstance(output, dict), "LiDAR intermediate output missing")
        collapse = output["sparse_collapse_fp32"]
        _require(collapse.dtype == torch.float32, "LiDAR sparse island did not return FP32")
        group_norms = [name for name, module in model.named_modules() if isinstance(module, nn.GroupNorm)]
        batch_norm_1d = [name for name, module in model.named_modules() if isinstance(module, nn.BatchNorm1d)]
        batch_norm_2d = [name for name, module in model.named_modules() if isinstance(module, nn.BatchNorm2d)]
        _require(not group_norms and batch_norm_1d and batch_norm_2d, "LiDAR BN/no-GN policy drift")
        policy = {
            "sparse_collapse_dtype": str(collapse.dtype),
            "sparse_collapse_shape": list(collapse.shape),
            "group_norm_modules": group_norms,
            "batch_norm_1d_count": len(batch_norm_1d),
            "batch_norm_2d_count": len(batch_norm_2d),
        }
    record = {
        "data_role": "D_fit",
        "capability_metrics_executed": False,
        "D_select_executed": False,
        "D_audit_executed": False,
        "official_validation_executed": False,
        "sample_token": str(batch_cpu["sample_token"][0]),
        "decode": sample,
        "serialized_box_count": len(serialized),
        "serialized_schema_sha256": _canonical_sha256(serialized),
        "precision_policy": policy,
    }
    del decoded, fp32_output, output, moved, model
    gc.collect()
    torch.cuda.empty_cache()
    return record


def _flatten_tensors(value: Any, prefix: str = "") -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    if torch.is_tensor(value):
        out[prefix or "tensor"] = value.detach().to(device="cpu", dtype=torch.float32).contiguous()
    elif isinstance(value, Mapping):
        for key in sorted(value):
            child = f"{prefix}.{key}" if prefix else str(key)
            out.update(_flatten_tensors(value[key], child))
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            child = f"{prefix}[{index}]"
            out.update(_flatten_tensors(item, child))
    return out


def _compare_tensor_maps(
    fallback: Mapping[str, torch.Tensor],
    optimized: Mapping[str, torch.Tensor],
    *,
    rtol: float,
    atol: float,
) -> dict[str, Any]:
    _require(set(fallback) == set(optimized), "parity tensor coverage differs")
    failures: list[str] = []
    worst_absolute = {"name": "", "value": 0.0}
    worst_relative = {"name": "", "value": 0.0}
    for name in sorted(fallback):
        reference = fallback[name]
        candidate = optimized[name]
        _require(reference.shape == candidate.shape, f"parity shape differs for {name}")
        _require(bool(torch.isfinite(reference).all()), f"fallback parity tensor {name} nonfinite")
        _require(bool(torch.isfinite(candidate).all()), f"optimized parity tensor {name} nonfinite")
        if reference.numel():
            difference = (candidate - reference).abs()
            max_absolute = float(difference.max())
            relative = difference / reference.abs().clamp_min(1e-12)
            max_relative = float(relative.max())
        else:
            max_absolute = max_relative = 0.0
        if max_absolute > worst_absolute["value"]:
            worst_absolute = {"name": name, "value": max_absolute}
        if max_relative > worst_relative["value"]:
            worst_relative = {"name": name, "value": max_relative}
        if not torch.allclose(candidate, reference, rtol=rtol, atol=atol):
            failures.append(name)
    return {
        "passed": not failures,
        "rtol": rtol,
        "atol": atol,
        "tensor_count": len(fallback),
        "failure_count": len(failures),
        "failure_names": failures[:20],
        "worst_absolute": worst_absolute,
        "worst_relative": worst_relative,
    }


def _camera_parity_capture(
    config,
    device: torch.device,
    batch_cpu: dict[str, Any],
    *,
    backend: str,
    precision: str,
    build_dir: str,
) -> dict[str, Any]:
    enforce_determinism(strict=(precision == "fp32"), precision=precision)
    if precision == "fp16":
        # The production FP16 policy intentionally enables fast nondeterministic
        # cuDNN kernels.  A serial backend-parity comparison must hold unrelated
        # convolution/SDPA backward noise fixed or it attributes that noise to
        # BEV pooling.  Autocast, scaler, FP32 pool accumulation/output, inputs,
        # losses, and frozen tolerances stay unchanged; production timing below
        # restores the accepted relaxed FP16 policy.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)
    model = _build_model(config, "camera", device, backend=backend, build_dir=build_dir)
    criterion = model.build_criterion().to(device)
    criterion.record_terms = False
    optimizer = build_phase1_optimizer(model, config)
    scaler = _make_scaler(config, device) if precision == "fp16" else None
    model.train()
    seed_everything(0)
    moved = _move_to_device(batch_cpu, device)
    context = precision_autocast_context(precision, device) if precision == "fp16" else nullcontext()
    with context:
        output = model(moved, pool_backend=backend, return_intermediates=True)
    loss = criterion(_float_tensors(output["task_outputs"]), moved)
    _require(bool(torch.isfinite(loss).detach().cpu()), "Camera integrated parity loss nonfinite")
    if scaler is not None:
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
    else:
        loss.backward()
    selected = {
        "pool_output_fp32": output["pool_output_fp32"],
        "camera_bev": output["camera_bev"],
        "decoder_feature": output["decoder_feature"],
        "task_outputs": output["task_outputs"],
        "loss": loss.reshape(1),
    }
    tensors = _flatten_tensors(selected)
    gradients = {
        name: parameter.grad.detach().to(device="cpu", dtype=torch.float32).contiguous()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and parameter.grad is not None
    }
    missing = [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and parameter.grad is None
    ]
    _require(not missing, f"Camera parity leaves trainable parameters without gradients: {missing[:5]}")
    _require(
        all(bool(torch.isfinite(value).all()) for value in gradients.values()),
        "Camera integrated parity has nonfinite parameter gradients",
    )
    record = {
        "initial_parameter_sha256": _parameter_sha256(model),
        "outputs": tensors,
        "gradients": gradients,
        "parameter_coverage": sorted(gradients),
    }
    del selected, gradients, tensors, loss, output, moved, scaler, optimizer, criterion, model
    gc.collect()
    torch.cuda.empty_cache()
    return record


def _camera_integrated_parity(
    config,
    device: torch.device,
    batch_cpu: dict[str, Any],
    *,
    build_dir: str,
) -> dict[str, Any]:
    results: dict[str, Any] = {}
    for precision, tolerances in (
        ("fp32", (1e-4, 1e-6)),
        ("fp16", (2e-3, 2e-4)),
    ):
        fallback = _camera_parity_capture(
            config,
            device,
            batch_cpu,
            backend="fallback",
            precision=precision,
            build_dir=build_dir,
        )
        optimized = _camera_parity_capture(
            config,
            device,
            batch_cpu,
            backend="optimized",
            precision=precision,
            build_dir=build_dir,
        )
        fallback_repeat = _camera_parity_capture(
            config,
            device,
            batch_cpu,
            backend="fallback",
            precision=precision,
            build_dir=build_dir,
        )
        _require(
            fallback["initial_parameter_sha256"] == optimized["initial_parameter_sha256"],
            f"{precision} parity initial model identity differs",
        )
        _require(
            fallback["parameter_coverage"] == optimized["parameter_coverage"],
            f"{precision} parity parameter coverage differs",
        )
        outputs = _compare_tensor_maps(
            fallback["outputs"], optimized["outputs"], rtol=tolerances[0], atol=tolerances[1]
        )
        gradients = _compare_tensor_maps(
            fallback["gradients"], optimized["gradients"], rtol=tolerances[0], atol=tolerances[1]
        )
        repeat_outputs = _compare_tensor_maps(
            fallback["outputs"],
            fallback_repeat["outputs"],
            rtol=tolerances[0],
            atol=tolerances[1],
        )
        repeat_gradients = _compare_tensor_maps(
            fallback["gradients"],
            fallback_repeat["gradients"],
            rtol=tolerances[0],
            atol=tolerances[1],
        )
        repeat_passed = bool(repeat_outputs["passed"] and repeat_gradients["passed"])
        results[precision] = {
            "passed": bool(outputs["passed"] and gradients["passed"] and repeat_passed),
            "initial_parameter_sha256": fallback["initial_parameter_sha256"],
            "identical_parameter_coverage": True,
            "parameter_count": len(fallback["parameter_coverage"]),
            "outputs": outputs,
            "upstream_parameter_gradients": gradients,
            "fallback_repeat_control": {
                "passed": repeat_passed,
                "outputs": repeat_outputs,
                "upstream_parameter_gradients": repeat_gradients,
            },
            "comparison_policy": {
                "precision": precision,
                "autocast_unchanged": True,
                "strict_deterministic_backend_isolation": True,
                "production_relaxed_policy_restored_for_timing": True,
            },
        }
        del fallback, optimized, fallback_repeat
        gc.collect()
    results["passed"] = all(results[name]["passed"] for name in ("fp32", "fp16"))
    return results


def _camera_standalone_pool_gates(device: torch.device, build_dir: str) -> dict[str, Any]:
    load_optimized_extension(build_directory=build_dir)
    generator = torch.Generator(device="cpu").manual_seed(29)
    count, channels = 8192, 17
    values_cpu = torch.randn((count, channels), generator=generator, dtype=torch.float32)
    geometry_cpu = torch.stack(
        (
            torch.randint(0, 11, (count,), generator=generator, dtype=torch.int32),
            torch.randint(0, 7, (count,), generator=generator, dtype=torch.int32),
            torch.randint(0, 2, (count,), generator=generator, dtype=torch.int32),
            torch.randint(0, 4, (count,), generator=generator, dtype=torch.int32),
        ),
        dim=1,
    ).contiguous()
    regimes: dict[str, Any] = {}
    for precision in ("fp32", "fp16"):
        fallback_values = values_cpu.to(device).requires_grad_(True)
        optimized_values = values_cpu.to(device).requires_grad_(True)
        geometry = geometry_cpu.to(device)
        context = precision_autocast_context(precision, device) if precision == "fp16" else nullcontext()
        with context:
            fallback = bev_pool(fallback_values, geometry, 4, 2, 7, 11, backend="fallback")
            optimized = bev_pool(
                optimized_values,
                geometry,
                4,
                2,
                7,
                11,
                backend="optimized",
                build_directory=build_dir,
            )
        forward_close = bool(torch.allclose(optimized, fallback, rtol=1e-5, atol=1e-6))
        output_gradient = torch.randn(
            fallback.shape, generator=generator, dtype=torch.float32
        ).to(device)
        fallback.backward(output_gradient)
        optimized.backward(output_gradient)
        gradient_exact = bool(torch.equal(optimized_values.grad, fallback_values.grad))
        finite = bool(
            torch.isfinite(optimized).all()
            and torch.isfinite(optimized_values.grad).all()
            and torch.isfinite(fallback).all()
        )
        regimes[precision] = {
            "passed": forward_close and gradient_exact and finite,
            "forward_rtol": 1e-5,
            "forward_atol": 1e-6,
            "forward_close": forward_close,
            "feature_gradient_rtol": 0.0,
            "feature_gradient_atol": 0.0,
            "feature_gradient_exact": gradient_exact,
            "finite": finite,
            "output_dtype": str(optimized.dtype),
            "output_shape": list(optimized.shape),
            "max_absolute_error": float((optimized - fallback).detach().abs().max()),
        }
        del fallback_values, optimized_values, geometry, fallback, optimized, output_gradient

    empty_values = torch.empty((0, 3), device=device, dtype=torch.float32, requires_grad=True)
    empty_geometry = torch.empty((0, 4), device=device, dtype=torch.int32)
    empty_fallback = bev_pool(empty_values, empty_geometry, 2, 1, 2, 3, backend="fallback")
    empty_optimized = bev_pool(
        empty_values,
        empty_geometry,
        2,
        1,
        2,
        3,
        backend="optimized",
        build_directory=build_dir,
    )
    singleton_values = torch.tensor([[4.0]], device=device, dtype=torch.float32, requires_grad=True)
    singleton_geometry = torch.tensor([[1, 1, 0, 0]], device=device, dtype=torch.int32)
    singleton_fallback = bev_pool(singleton_values, singleton_geometry, 1, 1, 2, 2, backend="fallback")
    singleton_optimized = bev_pool(
        singleton_values,
        singleton_geometry,
        1,
        1,
        2,
        2,
        backend="optimized",
        build_directory=build_dir,
    )
    exact_cases = {
        "empty_shape": list(empty_optimized.shape),
        "empty_exact": bool(torch.equal(empty_optimized, empty_fallback)),
        "singleton_shape": list(singleton_optimized.shape),
        "singleton_exact": bool(torch.equal(singleton_optimized, singleton_fallback)),
    }
    exact_cases["passed"] = bool(exact_cases["empty_exact"] and exact_cases["singleton_exact"])
    return {
        "passed": exact_cases["passed"] and all(regimes[name]["passed"] for name in regimes),
        "exact_shape_membership_cases": exact_cases,
        "regimes": regimes,
    }


def _camera_operator_inputs(
    config,
    device: torch.device,
    fixed_batches: Sequence[dict[str, Any]],
    *,
    build_dir: str,
) -> tuple[list[tuple[torch.Tensor, torch.Tensor, tuple[int, int, int, int]]], dict[str, Any]]:
    enforce_determinism(strict=False, precision="fp16")
    model = _build_model(config, "camera", device, backend="optimized", build_dir=build_dir)
    model.train()
    seed_everything(0)
    payloads = []
    identities = []
    with torch.no_grad():
        for batch_cpu in fixed_batches:
            moved = _move_to_device(batch_cpu, device)
            with precision_autocast_context("fp16", device):
                preprocessed = model.preprocess(
                    moved["images"],
                    moved["lidar2img"],
                    moved["cam_intrinsics"],
                    augmentation_params=moved.get("augmentation_params"),
                )
                images = preprocessed["images"]
                batch_size, cameras = images.shape[:2]
                features = model.camera_backbone(
                    images.reshape(batch_size * cameras, *images.shape[2:])
                )
                pyramid = model.camera_neck(features)
                stride8 = pyramid[0].view(
                    batch_size, cameras, 256, *pyramid[0].shape[-2:]
                )
                values, geometry, dimensions, _ = model.view_transform.operator_inputs(
                    stride8, preprocessed["lidar2img"]
                )
            values = values.detach().contiguous()
            geometry = geometry.detach().contiguous()
            _require(values.dtype == torch.float32, "production pool values are not FP32")
            _require(geometry.dtype == torch.int32, "production pool geometry is not int32")
            _require(bool(torch.isfinite(values).all().detach().cpu()), "production pool values nonfinite")
            payloads.append((values, geometry, dimensions))
            identities.append(
                {
                    "sample_tokens": list(batch_cpu["sample_token"]),
                    "point_features": int(values.shape[0]),
                    "channels": int(values.shape[1]),
                    "dimensions": list(dimensions),
                    "values_sha256": _tensor_sha256(values),
                    "geometry_sha256": _tensor_sha256(geometry),
                }
            )
            del moved, preprocessed, images, features, pyramid, stride8
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return payloads, {"fixed_production_B4_inputs": identities}


def _operator_timing(
    payloads: Sequence[tuple[torch.Tensor, torch.Tensor, tuple[int, int, int, int]]],
    *,
    backend: str,
    build_dir: str,
) -> dict[str, Any]:
    _require(len(payloads) == 4, "operator timing requires four fixed B4 inputs")
    with torch.inference_mode():
        for index in range(32):
            values, geometry, dimensions = payloads[index % len(payloads)]
            output = bev_pool(
                values,
                geometry,
                *dimensions,
                backend=backend,
                build_directory=build_dir,
            )
        torch.cuda.synchronize()
        _require(bool(torch.isfinite(output).all().detach().cpu()), f"{backend} operator output nonfinite")
        pairs = []
        for index in range(128):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            values, geometry, dimensions = payloads[index % len(payloads)]
            start.record()
            output = bev_pool(
                values,
                geometry,
                *dimensions,
                backend=backend,
                build_directory=build_dir,
            )
            end.record()
            pairs.append((start, end))
        torch.cuda.synchronize()
    samples = [float(start.elapsed_time(end)) for start, end in pairs]
    return {
        "backend": backend,
        "event_clock": "cuda_event",
        "warmups": 32,
        "samples": 128,
        "forward_ms": _distribution(samples),
    }


def _run_no_update_calibration(
    config,
    branch: str,
    device: torch.device,
    fixed_batches: Sequence[dict[str, Any]],
    *,
    backend: str,
    build_dir: str | None,
) -> dict[str, Any]:
    enforce_determinism(strict=False, precision="fp16")
    model = _build_model(config, branch, device, backend=backend, build_dir=build_dir)
    criterion = model.build_criterion().to(device)
    if hasattr(criterion, "record_terms"):
        criterion.record_terms = False
    optimizer = build_phase1_optimizer(model, config)
    scheduler = Phase1CyclicScheduler(optimizer, config)
    scaler = _make_scaler(config, device)
    initial_parameter_sha = _parameter_sha256(model)
    initial_scale = float(scaler.get_scale())
    model.train()
    optimizer.zero_grad(set_to_none=True)
    seed_everything(0)
    torch.cuda.reset_peak_memory_stats(device)
    gpu_step_ms: list[float] = []
    h2d_ms: list[float] = []
    grad_norms: list[float] = []
    accepted_windows = 0
    timed_wall_started: float | None = None
    total = WARMUP_MICROBATCHES + TIMED_MICROBATCHES
    last_loss = 0.0
    for index in range(total):
        if index == WARMUP_MICROBATCHES:
            torch.cuda.synchronize(device)
            timed_wall_started = time.perf_counter()
        h2d_start = torch.cuda.Event(enable_timing=True)
        h2d_end = torch.cuda.Event(enable_timing=True)
        step_start = torch.cuda.Event(enable_timing=True)
        step_end = torch.cuda.Event(enable_timing=True)
        h2d_start.record()
        moved = _move_to_device(fixed_batches[index % len(fixed_batches)], device)
        h2d_end.record()
        step_start.record()
        with precision_autocast_context("fp16", device):
            output = model(moved)
        output_fp32 = _float_tensors(output)
        loss = criterion(output_fp32, moved)
        _require(bool(torch.isfinite(loss).detach().cpu()), "no-update calibration loss nonfinite")
        scaler.scale(loss / float(ACCUMULATION_STEPS)).backward()
        boundary = (index + 1) % ACCUMULATION_STEPS == 0
        if boundary:
            scaler.unscale_(optimizer)
            gradients = [
                parameter.grad
                for parameter in model.parameters()
                if parameter.requires_grad and parameter.grad is not None
            ]
            _require(gradients, "no-update calibration produced no gradients")
            _require(
                all(bool(torch.isfinite(gradient).all().detach().cpu()) for gradient in gradients),
                "no-update calibration produced nonfinite unscaled gradients",
            )
            norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=float(config.as_dict()["training"]["gradient_clip"]["max_norm"]),
                norm_type=float(config.as_dict()["training"]["gradient_clip"]["norm_type"]),
            )
            grad_norms.append(float(norm.detach().cpu()))
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            accepted_windows += 1
        step_end.record()
        step_end.synchronize()
        if index >= WARMUP_MICROBATCHES:
            gpu_step_ms.append(float(step_start.elapsed_time(step_end)))
            h2d_ms.append(float(h2d_start.elapsed_time(h2d_end)))
        last_loss = float(loss.detach().cpu())
        del moved, output, output_fp32, loss
    torch.cuda.synchronize(device)
    _require(timed_wall_started is not None, "timed calibration did not start")
    timed_wall_seconds = time.perf_counter() - timed_wall_started
    final_parameter_sha = _parameter_sha256(model)
    _require(initial_parameter_sha == final_parameter_sha, "no-update calibration changed parameters")
    _require(len(optimizer.state) == 0, "no-update calibration created optimizer parameter state")
    _require(scheduler.accepted_updates == 0, "no-update calibration advanced scheduler")
    _require(accepted_windows == total // ACCUMULATION_STEPS, "accepted-window count drift")
    record = {
        "backend": backend,
        "precision": "global_fp16_autocast",
        "physical_batch": 4,
        "accumulation_steps": 8,
        "effective_batch": 32,
        "warmup_microbatches": WARMUP_MICROBATCHES,
        "timed_microbatches": TIMED_MICROBATCHES,
        "attempted_microbatches": total,
        "accepted_windows": accepted_windows,
        "optimizer_updates": 0,
        "scheduler_updates": scheduler.accepted_updates,
        "initial_parameter_sha256": initial_parameter_sha,
        "final_parameter_sha256": final_parameter_sha,
        "parameters_unchanged": True,
        "optimizer_state_entries": len(optimizer.state),
        "grad_scaler": {
            "initial_scale": initial_scale,
            "final_scale": float(scaler.get_scale()),
            "state": scaler.state_dict(),
        },
        "unscaled_gradient_norm": _distribution(grad_norms),
        "last_loss_diagnostic_only": last_loss,
        "gpu_step_ms": _distribution(gpu_step_ms),
        "h2d_ms": _distribution(h2d_ms),
        "timed_wall_seconds": timed_wall_seconds,
        "timed_samples": TIMED_MICROBATCHES * 4,
        "wall_samples_per_second": TIMED_MICROBATCHES * 4 / timed_wall_seconds,
        "memory": {
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
            "device_total_bytes": int(torch.cuda.get_device_properties(device).total_memory),
        },
        "loader_wait_definition": "fixed batches materialized once; see data.materialization_wait_ms",
    }
    del scaler, scheduler, optimizer, criterion, model
    gc.collect()
    torch.cuda.empty_cache()
    return record


def _alvis_inventory() -> dict[str, Any]:
    paths = {
        "checkpoint": ALVIS_ROOT / "final_model.pt",
        "provenance": ALVIS_ROOT / "provenance.json",
        "readiness": ALVIS_ROOT / "readiness_diag" / "benchmark_readiness.json",
        "clean_cell": ALVIS_ROOT / "readiness_diag" / "clean_cell_report.json",
        "config": Path("fl_v3/configs/p1_bb02d.json").resolve(),
    }
    _require(all(path.is_file() for path in paths.values()), "historical Alvis inventory is incomplete")
    files = {
        name: {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for name, path in paths.items()
    }
    return {
        "role": "historical_non_aligned_capability_anchor_only",
        "reported_historical_metrics": {"mAP": 0.5656, "NDS": 0.5733},
        "files": files,
        "direct_comparison_executed": False,
        "alignment_gaps": [
            "Alvis x86/A100 versus Arrhenius GH200",
            "fusion detector versus independently qualified Camera/LiDAR branches",
            "ten-sweep LiDAR versus Phase-I keyframe-only LiDAR training",
            "multi-task CenterHead/GN graph versus reference BN/TransFusion LiDAR graph",
            "historical full train/eval ownership versus current D_fit role",
            "bf16/EMA best-epoch history versus current FP16/raw/terminal-only recipe",
        ],
        "interpretation": "inventory only; no aligned capability claim in Envelope A",
    }


def _camera_run(
    config,
    device: torch.device,
    fixed_batches: Sequence[dict[str, Any]],
    *,
    build_dir: str,
) -> dict[str, Any]:
    standalone = _camera_standalone_pool_gates(device, build_dir)
    integrated = _camera_integrated_parity(
        config, device, fixed_batches[0], build_dir=build_dir
    )
    payloads, payload_identity = _camera_operator_inputs(
        config, device, fixed_batches, build_dir=build_dir
    )
    fallback_operator = _operator_timing(payloads, backend="fallback", build_dir=build_dir)
    optimized_operator = _operator_timing(payloads, backend="optimized", build_dir=build_dir)
    operator_ratio = (
        float(optimized_operator["forward_ms"]["median"])
        / float(fallback_operator["forward_ms"]["median"])
    )
    del payloads
    gc.collect()
    torch.cuda.empty_cache()
    fallback_e2e = _run_no_update_calibration(
        config,
        "camera",
        device,
        fixed_batches,
        backend="fallback",
        build_dir=build_dir,
    )
    optimized_e2e = _run_no_update_calibration(
        config,
        "camera",
        device,
        fixed_batches,
        backend="optimized",
        build_dir=build_dir,
    )
    step_ratio = (
        float(optimized_e2e["gpu_step_ms"]["median"])
        / float(fallback_e2e["gpu_step_ms"]["median"])
    )
    memory_ratio = (
        int(optimized_e2e["memory"]["peak_allocated_bytes"])
        / int(fallback_e2e["memory"]["peak_allocated_bytes"])
    )
    same_initialization = (
        fallback_e2e["initial_parameter_sha256"] == optimized_e2e["initial_parameter_sha256"]
    )
    gates = {
        "standalone_correctness": bool(standalone["passed"]),
        "integrated_correctness": bool(integrated["passed"]),
        "identical_e2e_initialization": bool(same_initialization),
        "operator_median_ratio": operator_ratio,
        "operator_median_ratio_max": 0.80,
        "operator_performance": operator_ratio <= 0.80,
        "end_to_end_median_ratio": step_ratio,
        "end_to_end_median_ratio_max": 1.02,
        "end_to_end_performance": step_ratio <= 1.02,
        "peak_allocated_ratio": memory_ratio,
        "peak_allocated_ratio_max": 1.05,
        "peak_memory": memory_ratio <= 1.05,
    }
    gates["promotion_passed"] = bool(
        gates["standalone_correctness"]
        and gates["integrated_correctness"]
        and gates["identical_e2e_initialization"]
        and gates["operator_performance"]
        and gates["end_to_end_performance"]
        and gates["peak_memory"]
    )
    return {
        "bev_pool_build": bev_pool_build_identity(),
        "standalone_correctness": standalone,
        "integrated_B4_correctness": integrated,
        "operator_inputs": payload_identity,
        "operator_timing": {"fallback": fallback_operator, "optimized": optimized_operator},
        "end_to_end": {"fallback": fallback_e2e, "optimized": optimized_e2e},
        "promotion_gates": gates,
        "production_backend": "optimized" if gates["promotion_passed"] else "not_promoted",
    }


def _runtime_record(config, device: torch.device) -> dict[str, Any]:
    dependencies = verify_runtime_dependency_identity(config.to_run_config())
    properties = torch.cuda.get_device_properties(device)
    return {
        "dependencies": dependencies,
        "dependencies_sha256": _canonical_sha256(dependencies),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "device_name": properties.name,
        "compute_capability": [properties.major, properties.minor],
        "total_memory_bytes": int(properties.total_memory),
        "slurm": {
            "job_id": os.environ.get("SLURM_JOB_ID"),
            "partition": os.environ.get("SLURM_JOB_PARTITION"),
            "cpus_per_task": os.environ.get("SLURM_CPUS_PER_TASK"),
            "mem_per_node": os.environ.get("SLURM_MEM_PER_NODE"),
            "gpus_on_node": os.environ.get("SLURM_GPUS_ON_NODE"),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--branch", choices=("camera", "lidar"), required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--source-sha", required=True)
    parser.add_argument("--build-dir")
    parser.add_argument("--initialization-result")
    parser.add_argument("--gtdb-manifest")
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    source = _source_identity(args.source_sha)
    source_config = load_resolved_config(args.config)
    raw_source = source_config.as_dict()
    _require(raw_source["contract"]["branch"] == args.branch, "branch/config mismatch")
    _require(raw_source["execution"]["mode"] == "envelope_a_calibration", "execution mode drift")
    _require(not raw_source["execution"]["capability_metrics"], "capability metrics are forbidden")
    _require(raw_source["execution"]["allowed_evaluation_roles"] == [], "evaluation role opened")
    _require(torch.cuda.is_available() and torch.cuda.device_count() == 1, "exactly one CUDA GPU is required")
    device = torch.device("cuda:0")

    build_dir = str(Path(args.build_dir).resolve()) if args.build_dir else None
    if args.branch == "camera":
        _require(build_dir is not None, "Camera requires an exact CUDA build directory")
        _require(os.environ.get("FL_V3_BEV_POOL_BUILD_DIR") == build_dir, "CUDA build root env drift")
    config, config_path = _derive_materialized_config(
        source_config,
        output_dir,
        initialization_result=(
            Path(args.initialization_result).resolve() if args.initialization_result else None
        ),
        gtdb_manifest=(Path(args.gtdb_manifest).resolve() if args.gtdb_manifest else None),
    )
    enforce_determinism(strict=False, precision="fp16")
    seed_everything(0)
    runtime = _runtime_record(config, device)
    batches, data_record = _fixed_d_fit_batches(config)

    if args.branch == "camera":
        branch_record = _camera_run(config, device, batches, build_dir=str(build_dir))
        branch_gate = bool(branch_record["promotion_gates"]["promotion_passed"])
    else:
        calibration = _run_no_update_calibration(
            config,
            "lidar",
            device,
            batches,
            backend="not_applicable",
            build_dir=None,
        )
        branch_record = {
            "calibration": calibration,
            "alvis_reference_inventory": _alvis_inventory(),
        }
        branch_gate = True

    evaluator = _evaluator_schema_preflight(
        config, args.branch, device, batches[0], build_dir=build_dir
    )
    qualified = None
    qualified_path = None
    checkpoint = None
    if branch_gate:
        qualified, qualified_path = _derive_qualified_config(config, output_dir)
        checkpoint = _checkpoint_preflight(
            qualified, args.branch, device, output_dir, build_dir=build_dir
        )

    result = {
        "schema": SCHEMA,
        "status": "PASS" if branch_gate else "FAIL_POOL_PROMOTION_GATE",
        "branch": args.branch,
        "source": source,
        "source_config": {
            "path": str(Path(args.config).resolve()),
            "sha256": source_config.sha256,
        },
        "materialized_config": {"path": str(config_path), "sha256": config.sha256},
        "qualified_config": (
            None
            if qualified is None
            else {"path": str(qualified_path), "sha256": qualified.sha256}
        ),
        "runtime": runtime,
        "data": data_record,
        "checkpoint_preflight": checkpoint,
        "evaluator_schema_preflight": evaluator,
        "branch_evidence": branch_record,
        "scope": {
            "candidate": raw_source["contract"]["candidate_id"],
            "seed": 0,
            "optimizer_updates": 0,
            "capability_metrics_executed": False,
            "D_select_executed": False,
            "D_audit_executed": False,
            "official_validation_executed": False,
            "scientific_checkpoint": None,
            "allowed_interpretation": [
                "implementation conformance",
                "numerical parity",
                "bounded engineering resource estimate",
            ],
            "forbidden_interpretation": [
                "branch capability",
                "convergence",
                "mAP/NDS",
                "candidate selection",
            ],
        },
    }
    result_path = output_dir / "result.json"
    result_sha = _write_once(result_path, result)
    print(
        json.dumps(
            {
                "status": result["status"],
                "branch": args.branch,
                "result_path": str(result_path),
                "result_sha256": result_sha,
                "materialized_config_sha256": config.sha256,
                "qualified_config_sha256": None if qualified is None else qualified.sha256,
            },
            sort_keys=True,
            allow_nan=False,
        ),
        flush=True,
    )
    del batches
    gc.collect()
    torch.cuda.empty_cache()
    return 0 if branch_gate else 2


if __name__ == "__main__":
    raise SystemExit(main())
