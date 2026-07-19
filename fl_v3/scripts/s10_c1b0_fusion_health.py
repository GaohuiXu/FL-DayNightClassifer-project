#!/usr/bin/env python3
"""Bounded C1-B0 matched GN/BN1d fusion-update health experiment."""
from __future__ import annotations

import argparse
import copy
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
from typing import Any, Iterable

sys.path.insert(0, "fl_v3/src")
sys.path.insert(0, "fl_v3/scripts")

import torch
import torch.nn as nn

from centralized_train import _build_optimizer
from fl_v3.config import resolve_config, verify_physical_data_identities
from fl_v3.data.nuscenes.s10_binding import load_frozen_split_role, token_vector_sha256
from fl_v3.training.loop import _float_tensors, _move_to_device, train_one_epoch
from fl_v3.training.precision_diagnostics import (
    PrecisionDiagnosticsIdentity,
    PrecisionWindowDiagnostics,
)
from fl_v3.training.runtime_state import TrainingState
from fl_v3.training.s10_observation import strict_json_value
from fl_v3.utils.runtime import (
    enforce_determinism,
    make_grad_scaler,
    precision_autocast_context,
    seed_everything,
    verify_runtime_dependency_identity,
)


SCHEMA = "fl_v3.s10.c1b0_fusion_health.v1"
EXPECTED_SPLIT_SHA256 = "7e84a1d4f4a099c31a1d5194f17ba77d278fdc92f52462e72517b494dc5223a8"
TOKEN_SELECTION_SALT = "s10-c1b0-h256-v1"
EXPECTED_TOKEN_SHA256 = "62a096c0990e6d1d0932868a882b2418e731d1a816f481e741996e49c8e975f7"
HORIZON = 256
PHYSICAL_BATCH = 4
DIAGNOSTIC_WINDOWS = (1, 4, 16, 64, 128, 256)
CANDIDATES = (
    ("F-A1-GN-H256", "group_norm"),
    ("F-A1-BN1D-H256", "batch_norm_1d"),
)


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--split-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--source-sha", required=True)
    parser.add_argument("--source-tree", required=True)
    return parser.parse_args()


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(strict_json_value(value), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _parameter_state_sha256(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.named_parameters()):
        tensor = value.detach().contiguous().cpu()
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(str(tensor.dtype).encode("ascii") + b"\0")
        digest.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode("ascii") + b"\0")
        digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _runtime_identity() -> dict[str, Any]:
    if platform.machine() != "aarch64":
        raise RuntimeError("C1-B0 requires an aarch64 Arrhenius compute node")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("C1-B0 requires exactly one visible CUDA device")
    properties = torch.cuda.get_device_properties(0)
    if "GH200" not in properties.name.upper():
        raise RuntimeError(f"C1-B0 expected GH200, got {properties.name!r}")
    return {
        "machine": platform.machine(),
        "node": platform.node(),
        "device_name": properties.name,
        "device_total_memory": int(properties.total_memory),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_job_gpus": os.environ.get("SLURM_JOB_GPUS"),
        "slurm_gpus_on_node": os.environ.get("SLURM_GPUS_ON_NODE"),
    }


def _state_schema(model: nn.Module) -> dict[str, Any]:
    state = model.state_dict()
    keys = sorted(state)
    bn_buffers = sorted(
        key for key in keys
        if key.startswith("lidar_encoder.backbone.")
        and key.rsplit(".", 1)[-1] in {"running_mean", "running_var", "num_batches_tracked"}
    )
    return {
        "keys_sha256": _canonical_sha256(keys),
        "key_count": len(keys),
        "second_bn_running_buffer_count": len(bn_buffers),
        "second_bn_running_buffer_keys_sha256": _canonical_sha256(bn_buffers),
    }


def select_h256_tokens(tokens: Iterable[str]) -> tuple[str, ...]:
    """Select an immutable pseudo-random D_low panel without inspecting labels."""
    declared = tuple(str(token) for token in tokens)
    if len(declared) != 6155 or len(declared) != len(set(declared)):
        raise RuntimeError("C1-B0 requires the accepted 6155-token unique D_low role")
    salt = TOKEN_SELECTION_SALT.encode("utf-8") + b"\0"
    ordered = sorted(
        declared,
        key=lambda token: (hashlib.sha256(salt + token.encode("utf-8")).digest(), token),
    )[: HORIZON * PHYSICAL_BATCH]
    result = tuple(ordered)
    if token_vector_sha256(result) != EXPECTED_TOKEN_SHA256:
        raise RuntimeError("C1-B0 H256 ordered token identity drift")
    return result


def _expected_sources(config) -> dict[str, Any]:
    identities = config.data_identities
    return {
        "version": "v1.0-trainval",
        "n_sweeps": 10,
        "train_cache_logical_sha256": identities["train_cache_logical_sha256"],
        "train_cache_pickle_sha256": identities["train_cache_pickle_sha256"],
        "train_cache_sidecar_sha256": identities["train_cache_sidecar_sha256"],
        "val_cache_logical_sha256": identities["val_cache_logical_sha256"],
        "val_cache_pickle_sha256": identities["val_cache_pickle_sha256"],
        "val_cache_sidecar_sha256": identities["val_cache_sidecar_sha256"],
        "zip_manifest_logical_sha256": identities["zip_manifest_logical_sha256"],
        "zip_manifest_file_sha256": identities["zip_manifest_file_sha256"],
        "detection_config_sha256": "217f96cca4e80f790c4674ef72257a6863ee9a85b0ce185bc56488afc32c7a0b",
    }


def _resolved_candidate(base: dict[str, Any], normalization: str):
    raw = copy.deepcopy(base)
    raw["model"]["second_normalization"] = normalization
    return resolve_config(raw)


def _assert_config(config, normalization: str) -> None:
    model = config.data["model"]
    training = config.data["training"]
    if str(config.data["schema_version"]) != "s10.v1":
        raise RuntimeError("C1-B0 requires the explicit s10.v1 schema")
    if dict(model) != {
        "mode": "fusion",
        "camera_arch": "swin_t_stride8",
        "camera_pretrained": True,
        "camera_activation_checkpoint": False,
        "lidar_arch": "second_075",
        "second_normalization": normalization,
        "fusion_arch": "conv_fuser_256",
        "head_arch": "centerhead_multitask",
    }:
        raise RuntimeError("C1-B0 graph/initialization/normalization config drift")
    if config.precision != "fp16" or config.sparse_conv_precision != "fp32":
        raise RuntimeError("C1-B0 requires global FP16 plus SECOND FP32 island")
    if (
        int(training["max_optimizer_steps"]) != HORIZON
        or int(training["micro_batch_size"]) != PHYSICAL_BATCH
        or int(training["world_size"]) != 1
        or int(training["accumulation_steps"]) != 1
        or int(training["effective_global_batch"]) != PHYSICAL_BATCH
        or int(training["seed"]) != 0
        or float(training["grad_scaler_init_scale"]) != 32.0
        or training["ema_decay"] is not None
        or str(training["sampling"]) != "uniform"
    ):
        raise RuntimeError("C1-B0 B4/seed/exposure/no-EMA contract drift")
    optimizer = config.data["optimizer"]
    if (
        str(optimizer["name"]) != "adamw"
        or float(optimizer["learning_rate"]) != 1.0e-4
        or float(optimizer["weight_decay"]) != 0.01
    ):
        raise RuntimeError("C1-B0 optimizer contract drift")


def _close_loader(loader: object) -> None:
    iterator = getattr(loader, "_iterator", None)
    shutdown = getattr(iterator, "_shutdown_workers", None)
    if callable(shutdown):
        shutdown()
    dataset = getattr(loader, "dataset", None)
    base = getattr(dataset, "dataset", dataset)
    close = getattr(base, "close", None)
    if callable(close):
        close()


class _OneBatch:
    def __init__(self, batch: dict[str, Any]):
        self.batch = batch
        self.batch_size = PHYSICAL_BATCH

    def __len__(self):
        return 1

    def __iter__(self):
        yield self.batch


def _batch_tokens(batch: dict[str, Any]) -> list[str]:
    tokens = batch.get("sample_token")
    if not isinstance(tokens, (list, tuple)) or len(tokens) != PHYSICAL_BATCH:
        raise RuntimeError("C1-B0 batch does not carry exactly four sample tokens")
    result = [str(token) for token in tokens]
    if any(not token for token in result):
        raise RuntimeError("C1-B0 batch carries an empty sample token")
    return result


def _qualify_scale(model, criterion, batch_cpu, device: torch.device) -> dict[str, Any]:
    """No-update scale-32 qualification; the caller discards this model."""
    model.train()
    model.zero_grad(set_to_none=True)
    batch = _move_to_device(batch_cpu, device)
    scaler = make_grad_scaler(device, "fp16", init_scale=32.0)
    with precision_autocast_context("fp16", device):
        output = model(batch)
    loss = criterion(_float_tensors(output), batch)
    scaler.scale(loss).backward()
    gradients = [
        parameter.grad for parameter in model.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    if not gradients:
        raise RuntimeError("C1-B0 scale qualification produced no gradients")
    all_finite = all(bool(torch.isfinite(gradient).all().item()) for gradient in gradients)
    missing = [
        name for name, parameter in model.named_parameters()
        if parameter.requires_grad and parameter.grad is None
    ]
    result = {
        "scale": float(scaler.get_scale()),
        "loss": float(loss.detach().item()),
        "loss_finite": math.isfinite(float(loss.detach().item())),
        "parameters_with_gradient": len(gradients),
        "missing_gradient_count": len(missing),
        "missing_gradients": missing,
        "scaled_gradients_all_finite": all_finite,
        "optimizer_constructed": False,
        "optimizer_update": False,
    }
    model.zero_grad(set_to_none=True)
    del batch, output, loss, scaler
    if not result["loss_finite"] or not all_finite or missing:
        raise RuntimeError(f"C1-B0 common scale-32 qualification failed: {result}")
    return result


def _bn_running_state(model: nn.Module) -> dict[str, Any]:
    modules = [
        (name, module) for name, module in model.lidar_encoder.backbone.named_modules()
        if isinstance(module, nn.BatchNorm1d)
    ]
    if not modules:
        return {"applicable": False, "site_count": 0, "state_sha256": None, "sites": {}}
    digest = hashlib.sha256()
    sites = {}
    for name, module in modules:
        mean = module.running_mean.detach().to(torch.float64).cpu()
        variance = module.running_var.detach().to(torch.float64).cpu()
        count = int(module.num_batches_tracked.detach().cpu().item())
        for label, tensor in (("running_mean", mean), ("running_var", variance)):
            digest.update(name.encode("utf-8") + b"\0" + label.encode("ascii") + b"\0")
            digest.update(tensor.numpy().tobytes(order="C"))
        digest.update(str(count).encode("ascii") + b"\0")
        sites[name] = {
            "features": int(mean.numel()),
            "num_batches_tracked": count,
            "running_mean_l2": float(torch.linalg.vector_norm(mean).item()),
            "running_mean_max_abs": float(mean.abs().max().item()),
            "running_var_min": float(variance.min().item()),
            "running_var_mean": float(variance.mean().item()),
            "running_var_max": float(variance.max().item()),
            "all_finite": bool(torch.isfinite(mean).all() and torch.isfinite(variance).all()),
        }
    return {
        "applicable": True,
        "site_count": len(modules),
        "state_sha256": digest.hexdigest(),
        "sites": sites,
    }


def _required_gradient_prefixes() -> tuple[str, ...]:
    return (
        "camera_backbone", "camera_neck", "view_transform",
        "lidar_encoder.backbone.stem", "lidar_encoder.backbone.stage1",
        "lidar_encoder.backbone.conv_out", "fusion", "bev_neck", "head",
    )


def _trajectory_summary(windows: list[dict[str, Any]]) -> dict[str, Any]:
    losses = [float(window["loss"]) for window in windows]
    first = statistics.mean(losses[:16])
    final = statistics.mean(losses[-16:])
    timings = [float(window["wall_seconds"]) for window in windows[16:]]
    ordered = sorted(timings)
    p50 = ordered[(len(ordered) - 1) // 2]
    p95 = ordered[int((len(ordered) - 1) * 0.95)]
    return {
        "first_16_mean_loss": first,
        "last_16_mean_loss": final,
        "last_over_first_loss": final / first if first > 0.0 else None,
        "loss_decreased": final < first,
        "post_16_window_seconds_p50": p50,
        "post_16_window_seconds_p95": p95,
        "post_16_samples_per_second_from_p50": PHYSICAL_BATCH / p50,
    }


def _run_candidate(
    *, task, config, normalization: str, cell_id: str, tokens: tuple[str, ...],
    output: Path, source_sha: str, expected_parameter_sha256: str,
) -> dict[str, Any]:
    run_config = config.to_run_config()
    cell_dir = output / cell_id
    cell_dir.mkdir()
    (cell_dir / "resolved_config.json").write_bytes(config.canonical_bytes + b"\n")

    # Qualification model is intentionally destroyed; the scientific cell is
    # reconstructed from seed-0 W0 afterward.
    seed_everything(0)
    qualification_model = task.build_model(run_config)
    if _parameter_state_sha256(qualification_model) != expected_parameter_sha256:
        raise RuntimeError(f"{cell_id} qualification W0 parameter identity drift")
    qualification_model = qualification_model.to("cuda")
    qualification_criterion = task.build_criterion(run_config)
    qualification_loader = task.fixed_train_subset_loader(run_config, tokens[:PHYSICAL_BATCH])
    qualification_batch = next(iter(qualification_loader))
    if _batch_tokens(qualification_batch) != list(tokens[:PHYSICAL_BATCH]):
        raise RuntimeError(f"{cell_id} qualification token order drift")
    qualification = _qualify_scale(
        qualification_model, qualification_criterion, qualification_batch, torch.device("cuda")
    )
    _close_loader(qualification_loader)
    del qualification_model, qualification_criterion, qualification_loader, qualification_batch
    gc.collect()
    torch.cuda.empty_cache()

    seed_everything(0)
    model = task.build_model(run_config)
    initial_parameter_sha256 = _parameter_state_sha256(model)
    if initial_parameter_sha256 != expected_parameter_sha256:
        raise RuntimeError(f"{cell_id} post-qualification W0 parameter identity drift")
    initial_schema = _state_schema(model)
    model = model.to("cuda")
    criterion = task.build_criterion(run_config)
    optimizer = _build_optimizer(model, config)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _step: 1.0)
    scaler = make_grad_scaler(
        torch.device("cuda"),
        "fp16",
        init_scale=float(run_config["grad-scaler-init-scale"]),
    )
    state = TrainingState()
    diagnostics = PrecisionWindowDiagnostics(
        PrecisionDiagnosticsIdentity(
            source_sha=source_sha,
            resolved_config_sha256=config.sha256,
            model_mode="fusion",
            global_precision="fp16",
            sparse_conv_precision="fp32",
        ),
        max_windows=len(DIAGNOSTIC_WINDOWS),
        attempted_windows=DIAGNOSTIC_WINDOWS,
        capture_boundaries=True,
        capture_parameter_updates=True,
        fixture_identity={
            "cell": cell_id,
            "normalization": normalization,
            "split_manifest_sha256": EXPECTED_SPLIT_SHA256,
            "h256_ordered_token_sha256": EXPECTED_TOKEN_SHA256,
            "physical_microbatch": PHYSICAL_BATCH,
            "grad_scaler_init_scale": 32.0,
        },
    )
    loader = task.fixed_train_subset_loader(run_config, tokens)
    observed: list[str] = []
    windows: list[dict[str, Any]] = []
    norm_state_at_diagnostics = {}
    torch.cuda.reset_peak_memory_stats()
    seed_everything(0)
    for index, batch_cpu in enumerate(loader, start=1):
        if index > HORIZON:
            raise RuntimeError(f"{cell_id} loader yielded more than H256")
        expected = list(tokens[(index - 1) * PHYSICAL_BATCH:index * PHYSICAL_BATCH])
        actual = _batch_tokens(batch_cpu)
        if actual != expected:
            raise RuntimeError(f"{cell_id} batch {index} token order drift")
        observed.extend(actual)
        before = state.checkpoint_dict()
        scale_before = float(scaler.get_scale())
        torch.cuda.synchronize()
        started = time.perf_counter()
        metrics = train_one_epoch(
            model,
            _OneBatch(batch_cpu),
            criterion,
            optimizer,
            torch.device("cuda"),
            scheduler=scheduler,
            precision="fp16",
            grad_scaler=scaler,
            accumulation_steps=1,
            runtime_state=state,
            model_mode="fusion",
            exposure_multiplier=1,
            expected_global_microbatch_samples=PHYSICAL_BATCH,
            precision_diagnostics=diagnostics,
        )
        torch.cuda.synchronize()
        after = state.checkpoint_dict()
        accepted = after["optimizer_step"] - before["optimizer_step"] == 1
        windows.append({
            "attempted_window": index,
            "sample_tokens": actual,
            "loss": float(metrics["loss"]),
            "loss_finite": bool(metrics["nonfinite_loss_steps"] == 0),
            "accepted": accepted,
            "scale_before": scale_before,
            "scale_after": float(scaler.get_scale()),
            "wall_seconds": time.perf_counter() - started,
            "state_after": after,
        })
        if index in DIAGNOSTIC_WINDOWS:
            norm_state = _bn_running_state(model)
            if normalization == "batch_norm_1d":
                if norm_state["site_count"] != 21 or any(
                    site["num_batches_tracked"] != index or not site["all_finite"]
                    for site in norm_state["sites"].values()
                ):
                    raise RuntimeError(f"{cell_id} BN running-state contract drift at {index}")
            elif norm_state["applicable"]:
                raise RuntimeError(f"{cell_id} GN cell unexpectedly carries BN running state")
            norm_state_at_diagnostics[str(index)] = norm_state
    if len(windows) != HORIZON or observed != list(tokens):
        raise RuntimeError(f"{cell_id} incomplete H256 token consumption")
    if state.optimizer_step != HORIZON or state.attempted_windows != HORIZON:
        raise RuntimeError(f"{cell_id} did not complete 256 real optimizer updates")
    if state.invalid_windows or state.discarded_windows:
        raise RuntimeError(f"{cell_id} contains invalid or discarded windows")
    if len(diagnostics.records) != len(DIAGNOSTIC_WINDOWS):
        raise RuntimeError(f"{cell_id} diagnostic-window coverage drift")
    for record in diagnostics.records:
        gradients = record["parameter_gradients"]
        if not gradients["global"]["all_finite"]:
            raise RuntimeError(f"{cell_id} sampled nonfinite true-unscaled gradients")
        missing = [
            prefix for prefix in _required_gradient_prefixes()
            if gradients["by_prefix"].get(prefix, {}).get("complete_l2") in (None, 0.0)
        ]
        if missing:
            raise RuntimeError(f"{cell_id} missing required sampled gradients: {missing}")
        if not record["accepted"] or not record["counter_deltas_consistent"]:
            raise RuntimeError(f"{cell_id} sampled update/counter contract failed")

    diagnostic_path = cell_dir / "sampled_windows.jsonl"
    diagnostic_path.write_text(diagnostics.json_lines(), encoding="utf-8")
    trajectory = _trajectory_summary(windows)
    report = {
        "schema": SCHEMA,
        "cell": cell_id,
        "normalization": normalization,
        "source_sha": source_sha,
        "resolved_config_sha256": config.sha256,
        "initial_parameter_sha256": initial_parameter_sha256,
        "final_parameter_sha256": _parameter_state_sha256(model),
        "initial_state_schema": initial_schema,
        "qualification": qualification,
        "token_evidence": {
            "selection": "lowest SHA256(s10-c1b0-h256-v1\\0 || token)",
            "ordered_sha256": token_vector_sha256(observed),
            "samples": len(observed),
            "batches": len(windows),
        },
        "recipe": {
            "optimizer": "AdamW", "learning_rate": 1.0e-4, "weight_decay": 0.01,
            "scheduler": "constant_lambda_1", "grad_clip": None, "ema": None,
            "augmentation": None, "sampling": "uniform", "physical_microbatch": 4,
            "accumulation": 1, "precision": "global_fp16_SECOND_fp32_island",
            "grad_scaler_init_scale": 32.0,
        },
        "windows": windows,
        "trajectory": trajectory,
        "terminal_training_state": state.checkpoint_dict(),
        "diagnostic_windows": list(DIAGNOSTIC_WINDOWS),
        "sampled_diagnostics_sha256": _sha256_file(diagnostic_path),
        "normalization_state_at_diagnostics": norm_state_at_diagnostics,
        "memory": {
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
            "device_total_bytes": int(torch.cuda.get_device_properties(0).total_memory),
        },
        "hard_gate": "PASS",
        "interpretation_limits": [
            "single-seed H256 train-only health contrast; not convergence or architecture promotion",
            "no evaluator, D_select, D_audit, official val, checkpoint selection, or recipe sweep",
            "gradient magnitude alone is not a failure or a production-normalization decision",
        ],
    }
    _write_json(cell_dir / "cell_summary.json", report)
    _close_loader(loader)
    del model, criterion, optimizer, scheduler, scaler, loader
    gc.collect()
    torch.cuda.empty_cache()
    return report


def _artifact_manifest(root: Path) -> dict[str, Any]:
    files = {}
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = str(path.relative_to(root))
        if relative == "artifact_sha256s.json":
            continue
        files[relative] = {"sha256": _sha256_file(path), "bytes": path.stat().st_size}
    return {"schema": "fl_v3.s10.c1b0_artifacts.v1", "files": files}


def _execute(args, output: Path) -> None:
    if len(args.source_sha) != 40 or len(args.source_tree) != 40:
        raise RuntimeError("C1-B0 requires exact source commit and tree identities")
    runtime = _runtime_identity()
    base = json.loads(Path(args.config).read_text(encoding="utf-8"))
    configs = {
        normalization: _resolved_candidate(base, normalization)
        for _, normalization in CANDIDATES
    }
    for normalization, config in configs.items():
        _assert_config(config, normalization)
        verify_physical_data_identities(config)
    runtime_dependencies = verify_runtime_dependency_identity(
        configs["group_norm"].to_run_config()
    )
    binding = load_frozen_split_role(
        args.split_manifest,
        expected_manifest_sha256=EXPECTED_SPLIT_SHA256,
        role="D_low",
        expected_source_identities=_expected_sources(configs["group_norm"]),
    )
    tokens = select_h256_tokens(binding.sample_tokens)
    enforce_determinism(strict=False, precision="fp16")
    from fl_v3.training.tasks import NuScenesDetectionTask

    task = NuScenesDetectionTask()
    candidate_identity = {}
    shared_parameter_sha256 = None
    for cell_id, normalization in CANDIDATES:
        seed_everything(0)
        model = task.build_model(configs[normalization].to_run_config())
        parameter_sha256 = _parameter_state_sha256(model)
        schema = _state_schema(model)
        if shared_parameter_sha256 is None:
            shared_parameter_sha256 = parameter_sha256
        elif parameter_sha256 != shared_parameter_sha256:
            raise RuntimeError("C1-B0 GN/BN1d candidates do not share exact trainable W0")
        candidate_identity[cell_id] = {
            "normalization": normalization,
            "resolved_config_sha256": configs[normalization].sha256,
            "parameter_state_sha256": parameter_sha256,
            "state_schema": schema,
        }
        del model
    if candidate_identity[CANDIDATES[0][0]]["state_schema"] == candidate_identity[CANDIDATES[1][0]]["state_schema"]:
        raise RuntimeError("C1-B0 checkpoint state schemas did not distinguish GN from BN1d")

    identity = {
        "schema": SCHEMA,
        "source_sha": args.source_sha,
        "source_tree": args.source_tree,
        "runtime": runtime,
        "runtime_dependencies": runtime_dependencies,
        "split_binding": binding.identity(),
        "h256": {
            "selection_salt": TOKEN_SELECTION_SALT,
            "ordered_token_sha256": token_vector_sha256(tokens),
            "samples": len(tokens),
            "batches": HORIZON,
        },
        "candidate_identity": candidate_identity,
        "shared_parameter_state_sha256": shared_parameter_sha256,
        "cell_order": [cell for cell, _ in CANDIDATES],
    }
    _write_json(output / "execution_identity.json", identity)
    reports = []
    for cell_id, normalization in CANDIDATES:
        reports.append(_run_candidate(
            task=task,
            config=configs[normalization],
            normalization=normalization,
            cell_id=cell_id,
            tokens=tokens,
            output=output,
            source_sha=args.source_sha,
            expected_parameter_sha256=str(shared_parameter_sha256),
        ))
    by_cell = {report["cell"]: report for report in reports}
    summary = {
        "schema": SCHEMA,
        "status": "PASS",
        "source_sha": args.source_sha,
        "cell_order": [cell for cell, _ in CANDIDATES],
        "hard_gate": "PASS",
        "shared_parameter_state_sha256": shared_parameter_sha256,
        "h256_ordered_token_sha256": EXPECTED_TOKEN_SHA256,
        "descriptive_contrast": {
            cell: {
                "normalization": report["normalization"],
                "trajectory": report["trajectory"],
                "peak_allocated_bytes": report["memory"]["peak_allocated_bytes"],
                "peak_reserved_bytes": report["memory"]["peak_reserved_bytes"],
                "final_scaler": report["windows"][-1]["scale_after"],
            }
            for cell, report in by_cell.items()
        },
        "selection": "none; C1-B0 is observation-only and does not promote a normalization",
        "next_decision": "owner inspects matched H256 health before any C1-B1/full D_low continuation",
    }
    _write_json(output / "summary.json", summary)
    _write_json(output / "artifact_sha256s.json", _artifact_manifest(output))


def main() -> None:
    args = _parse_args()
    output = Path(args.output_dir).resolve()
    if output.exists():
        raise RuntimeError(f"C1-B0 output must be fresh: {output}")
    output.mkdir(parents=True)
    try:
        _execute(args, output)
    except BaseException as exc:
        _write_json(output / "failure_summary.json", {
            "schema": "fl_v3.s10.c1b0_failure.v1",
            "exception_type": type(exc).__name__,
            "exception_message": str(exc),
        })
        _write_json(output / "artifact_sha256s.json", _artifact_manifest(output))
        raise


if __name__ == "__main__":
    main()
