#!/usr/bin/env python3
"""Zero-update localization and diagnostic D_select peek for LiDAR epoch 4.

This entry is deliberately separate from the production Envelope-B runner.  It
loads the immutable epoch-4 recovery checkpoint, performs one explicitly
non-selectable D_select evaluation with fail-closed raw-head finite checks, and
then reproduces the first epoch-5 D_fit batch through a conditional numerical
localization sequence.  It never calls backward or an optimizer/scheduler step.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping

SCRIPT_DIR = Path(__file__).resolve().parent
SOURCE_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(SOURCE_ROOT / "fl_v3" / "src"))
sys.path.insert(0, str(SCRIPT_DIR))

import numpy as np
import torch

from fl_v3.config import load_resolved_config
from fl_v3.data.nuscenes.phase1 import build_phase1_train_data
from fl_v3.models.phase1_lidar import build_phase1_lidar_model
from fl_v3.models.phase1_swin import sha256_file, tensor_state_sha256
from fl_v3.training.checkpoint import load_checkpoint
from fl_v3.training.loop import _unpack_batch
from fl_v3.training.phase1 import (
    Phase1CyclicScheduler,
    build_phase1_optimizer,
)
from fl_v3.utils.runtime import (
    enforce_determinism,
    precision_autocast_context,
    seed_everything,
)

from s10_phase1_capability import (
    _atomic_write_bytes_once,
    _atomic_write_once,
    _build_components,
    _canonical_sha256,
    _cpu_resident_batch_fields,
    _evaluate_terminal,
    _read_json,
    _runtime_identity,
    _source_identity,
)


SCHEMA = "s10.phase1.lidar-epoch4-diagnostic.v1"
EXPECTED_CHECKPOINT_EPOCH = 4
EXPECTED_OPTIMIZER_STEP = 10_988
EXPECTED_EPOCH5_FIRST_TOKEN_SHA256 = (
    "cbf3cbef4f6659eb6420b97979af95f88e50a950cb2617d88ab469307d93fec4"
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _batch_sha256(value: Any) -> str:
    """Hash a CPU loader batch without modifying values or container order."""
    digest = hashlib.sha256()

    def part(payload: bytes) -> None:
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)

    def visit(item: Any, path: str) -> None:
        part(path.encode("utf-8"))
        part(type(item).__name__.encode("utf-8"))
        if torch.is_tensor(item):
            tensor = item.detach().resolve_conj().resolve_neg().cpu().contiguous()
            part(str(tensor.dtype).encode("ascii"))
            part(json.dumps(list(tensor.shape), separators=(",", ":")).encode("ascii"))
            try:
                payload = tensor.numpy().tobytes(order="C")
            except TypeError:
                payload = tensor.reshape(-1).view(torch.uint8).numpy().tobytes(order="C")
            part(payload)
            return
        if isinstance(item, np.ndarray):
            array = np.ascontiguousarray(item)
            part(str(array.dtype).encode("ascii"))
            part(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
            part(array.tobytes(order="C"))
            return
        if isinstance(item, Mapping):
            keys = sorted(item, key=lambda key: (type(key).__name__, repr(key)))
            part(json.dumps([f"{type(key).__name__}:{key!r}" for key in keys]).encode())
            for key in keys:
                visit(item[key], f"{path}.{type(key).__name__}:{key!r}")
            return
        if isinstance(item, (list, tuple)):
            part(str(len(item)).encode("ascii"))
            for index, child in enumerate(item):
                visit(child, f"{path}[{index}]")
            return
        if item is None or isinstance(item, (str, bool, int, float)):
            part(repr(item).encode("utf-8"))
            return
        raise TypeError(f"unsupported batch leaf {type(item)!r} at {path}")

    visit(value, "root")
    return digest.hexdigest()


def _floating_tensor_stats(value: Any) -> dict[str, Any]:
    """Return compact finite diagnostics for all floating tensors in a tree."""
    tensors = 0
    values = 0
    nonfinite = 0
    maxima: list[float] = []

    def visit(item: Any) -> None:
        nonlocal tensors, values, nonfinite
        if torch.is_tensor(item):
            if not (item.is_floating_point() or item.is_complex()):
                return
            tensor = item.detach()
            tensors += 1
            values += int(tensor.numel())
            finite = torch.isfinite(tensor)
            all_finite = bool(finite.all().item())
            if not all_finite:
                nonfinite += int((~finite).sum().item())
            if tensor.numel() and all_finite:
                maxima.append(float(tensor.abs().max().item()))
            return
        if isinstance(item, Mapping):
            for child in item.values():
                visit(child)
            return
        if isinstance(item, (list, tuple)):
            for child in item:
                visit(child)

    visit(value)
    return {
        "floating_tensor_count": tensors,
        "floating_value_count": values,
        "nonfinite_value_count": nonfinite,
        "all_finite": nonfinite == 0,
        "max_absolute_finite_tensor": max(maxima) if maxima else None,
    }


class _FiniteForward(torch.nn.Module):
    """Delegate to a production model and reject raw nonfinite head outputs."""

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model
        self.forward_calls = 0
        self.floating_values = 0
        self.max_absolute = 0.0

    def forward(self, batch):
        output = self.model(batch)
        stats = _floating_tensor_stats(output)
        self.forward_calls += 1
        self.floating_values += int(stats["floating_value_count"])
        maximum = stats["max_absolute_finite_tensor"]
        if maximum is not None:
            self.max_absolute = max(self.max_absolute, float(maximum))
        if not stats["all_finite"]:
            raise FloatingPointError(
                "epoch-4 D_select raw head output is nonfinite: "
                f"call={self.forward_calls}, nonfinite={stats['nonfinite_value_count']}"
            )
        return output

    def decode(self, *args, **kwargs):
        return self.model.decode(*args, **kwargs)


def _build_eager_components(config, device: torch.device, *, sdpa: bool):
    seed_everything(int(config.as_dict()["training"]["seed"]))
    model = build_phase1_lidar_model(config).to(device)
    criterion = model.build_criterion().to(device)
    model.set_phase1p_lidar_host_batch_offsets(True)
    patched = int(model.set_phase1p_lidar_sdpa(sdpa))
    _require(patched == (2 if sdpa else 0), "LiDAR SDPA diagnostic scope drift")
    optimizer = build_phase1_optimizer(model, config)
    scheduler = Phase1CyclicScheduler(optimizer, config)
    spec = config.as_dict()["precision"]["grad_scaler"]
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=bool(spec["enabled"] and device.type == "cuda"),
        init_scale=float(spec["init_scale"]),
        growth_factor=float(spec["growth_factor"]),
        backoff_factor=float(spec["backoff_factor"]),
        growth_interval=int(spec["growth_interval"]),
    )
    return model, criterion, optimizer, scheduler, scaler


def _load_exact_checkpoint(
    checkpoint: Path,
    record: Mapping[str, Any],
    *,
    model,
    optimizer,
    scheduler,
    scaler,
    config,
):
    state, identity = load_checkpoint(
        str(checkpoint),
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        grad_scaler=scaler,
        ema=None,
        config=config,
        map_location="cpu",
    )
    _require(identity == config.sha256, "diagnostic checkpoint config identity drift")
    _require(state.epoch == EXPECTED_CHECKPOINT_EPOCH, "diagnostic checkpoint epoch drift")
    _require(state.optimizer_step == EXPECTED_OPTIMIZER_STEP, "optimizer-step boundary drift")
    _require(int(record["epoch"]) == state.epoch, "epoch record/checkpoint mismatch")
    _require(
        tensor_state_sha256(model.state_dict()) == record["model_state_sha256"],
        "loaded epoch-4 model-state hash drift",
    )
    return state


def _stage_modules(model) -> list[tuple[str, torch.nn.Module]]:
    return [
        ("sparse_collapse", model.lidar_encoder),
        ("second_backbone", model.decoder_backbone),
        ("second_fpn", model.decoder_neck),
        ("head_shared_conv", model.head.shared_conv),
        ("head_heatmap", model.head.heatmap_head),
        ("query_class_encoding", model.head.class_encoding),
        ("query_position_encoding", model.head.decoder.query_position),
        ("key_position_encoding", model.head.decoder.key_position),
        ("self_attention", model.head.decoder.self_attn),
        ("self_attention_norm", model.head.decoder.norm1),
        ("cross_attention", model.head.decoder.cross_attn),
        ("cross_attention_norm", model.head.decoder.norm2),
        ("ffn_linear1", model.head.decoder.linear1),
        ("ffn_linear2", model.head.decoder.linear2),
        ("ffn_norm", model.head.decoder.norm3),
        ("query_prediction_head", model.head.prediction_head),
    ]


def _parameter_state_sha256(model: torch.nn.Module) -> str:
    return tensor_state_sha256(
        {name: parameter.detach() for name, parameter in model.named_parameters()}
    )


def _run_localization_cell(
    *,
    cell: str,
    config,
    checkpoint: Path,
    checkpoint_record: Mapping[str, Any],
    device: torch.device,
    precision: str,
    compiled: bool,
    sdpa: bool,
    expected_batch_sha256: str | None,
) -> dict[str, Any]:
    if compiled:
        model, _criterion, optimizer, scheduler, scaler = _build_components(
            config, "lidar", device
        )
    else:
        model, _criterion, optimizer, scheduler, scaler = _build_eager_components(
            config, device, sdpa=sdpa
        )
    bundle = build_phase1_train_data(config)
    hooks = []
    stages: list[dict[str, Any]] = []
    try:
        state = _load_exact_checkpoint(
            checkpoint,
            checkpoint_record,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            config=config,
        )
        bundle.set_epoch(EXPECTED_CHECKPOINT_EPOCH)
        iterator = iter(bundle.loader)
        try:
            batch = next(iterator)
        finally:
            del iterator
        tokens = list(batch["sample_token"])
        token_sha256 = _canonical_sha256(tokens)
        _require(
            token_sha256 == EXPECTED_EPOCH5_FIRST_TOKEN_SHA256,
            "epoch-5 first-batch token identity drift",
        )
        batch_sha256 = _batch_sha256(batch)
        if expected_batch_sha256 is not None:
            _require(
                batch_sha256 == expected_batch_sha256,
                f"{cell} augmented input differs from the production control",
            )
        moved, _ = _unpack_batch(
            batch,
            device,
            cpu_resident_batch_fields=_cpu_resident_batch_fields(config),
        )
        if not compiled:
            for name, module in _stage_modules(model):
                hooks.append(
                    module.register_forward_hook(
                        lambda _module, _inputs, output, stage=name: stages.append(
                            {"stage": stage, **_floating_tensor_stats(output)}
                        )
                    )
                )
        parameter_before = _parameter_state_sha256(model)
        scheduler_before = dict(scheduler.state_dict())
        scaler_before = dict(scaler.state_dict())
        model.train()
        with precision_autocast_context(precision, device):
            output = model(moved)
        output_stats = _floating_tensor_stats(output)
        stages.append({"stage": "model_output", **output_stats})
        _require(
            _parameter_state_sha256(model) == parameter_before,
            f"{cell} changed model parameters during a zero-update forward",
        )
        _require(scheduler.state_dict() == scheduler_before, f"{cell} advanced scheduler")
        _require(scaler.state_dict() == scaler_before, f"{cell} changed GradScaler")
        _require(state.optimizer_step == EXPECTED_OPTIMIZER_STEP, f"{cell} advanced optimizer state")
        first_bad = next(
            (item["stage"] for item in stages if not item["all_finite"]), None
        )
        return {
            "cell": cell,
            "precision": precision,
            "compiled": compiled,
            "sdpa": sdpa,
            "checkpoint_epoch": state.epoch,
            "optimizer_step_before_after": [state.optimizer_step, state.optimizer_step],
            "backward_executed": False,
            "optimizer_update_executed": False,
            "token_sha256": token_sha256,
            "batch_sha256": batch_sha256,
            "stages": stages,
            "all_finite": bool(output_stats["all_finite"]),
            "first_nonfinite_stage": first_bad,
        }
    finally:
        for hook in hooks:
            hook.remove()
        bundle.close()


def _run(args: argparse.Namespace) -> dict[str, Any]:
    config_path = Path(args.config).resolve()
    checkpoint = Path(args.checkpoint).resolve()
    epoch_record_path = Path(args.epoch_record).resolve()
    output_dir = Path(args.output_dir).resolve()
    _require(not output_dir.exists(), "diagnostic output must be fresh")
    output_dir.mkdir(parents=True)
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(output_dir / "torchinductor_cache")

    config = load_resolved_config(config_path)
    raw = config.as_dict()
    _require(raw["contract"]["branch"] == "lidar", "diagnostic requires LiDAR config")
    _require(checkpoint.is_file(), "epoch-4 checkpoint is missing")
    _require(epoch_record_path.is_file(), "epoch-4 record is missing")
    epoch_record = _read_json(epoch_record_path)
    _require(
        sha256_file(checkpoint) == epoch_record["checkpoint_sha256"],
        "epoch-4 checkpoint physical hash drift",
    )
    _require(epoch_record.get("selectable") is False, "epoch-4 recovery became selectable")

    source = _source_identity(args.source_sha)
    runtime, runtime_dependency_sha256 = _runtime_identity(config)
    enforce_determinism(strict=False, precision="fp16")
    seed_everything(int(raw["training"]["seed"]))
    device = torch.device("cuda", 0)

    _atomic_write_bytes_once(output_dir / "resolved_config.json", config.canonical_bytes)
    _atomic_write_once(
        output_dir / "scope.json",
        {
            "schema": SCHEMA,
            "source": source,
            "resolved_config_sha256": config.sha256,
            "runtime": runtime,
            "runtime_dependencies_sha256": runtime_dependency_sha256,
            "checkpoint": {
                "path": str(checkpoint),
                "sha256": epoch_record["checkpoint_sha256"],
                "model_state_sha256": epoch_record["model_state_sha256"],
                "epoch": EXPECTED_CHECKPOINT_EPOCH,
                "selectable": False,
            },
            "D_select": {
                "diagnostic_peek": True,
                "selectable": False,
                "early_stopping_allowed": False,
                "terminal_epoch20_execution_remains_reserved": True,
            },
            "D_audit_executed": False,
            "official_validation_executed": False,
            "backward_allowed": False,
            "optimizer_update_allowed": False,
        },
    )

    # Evaluate first on a clean eval-mode reconstruction.  The wrapper rejects
    # raw NaN/Inf before decode can filter invalid boxes into a misleading empty
    # submission.
    eval_model, _criterion, eval_optimizer, eval_scheduler, eval_scaler = _build_components(
        config, "lidar", device
    )
    eval_state = _load_exact_checkpoint(
        checkpoint,
        epoch_record,
        model=eval_model,
        optimizer=eval_optimizer,
        scheduler=eval_scheduler,
        scaler=eval_scaler,
        config=config,
    )
    checked_model = _FiniteForward(eval_model)
    d_select_root = output_dir / "epoch04_d_select"
    try:
        d_select_record = _evaluate_terminal(
            config,
            checked_model,
            device,
            d_select_root,
            checkpoint.parent,
            runtime_dependency_sha256,
        )
        d_select = {
            **d_select_record,
            "diagnostic_only": True,
            "selectable": False,
            "checkpoint_epoch": eval_state.epoch,
            "raw_head_forward_calls": checked_model.forward_calls,
            "raw_head_floating_values": checked_model.floating_values,
            "raw_head_all_finite": True,
            "raw_head_max_absolute": checked_model.max_absolute,
        }
    except FloatingPointError as exc:
        d_select = {
            "status": "FAILED_RAW_HEAD_NONFINITE",
            "diagnostic_only": True,
            "selectable": False,
            "checkpoint_epoch": eval_state.epoch,
            "error": str(exc),
            "raw_head_forward_calls": checked_model.forward_calls,
            "raw_head_floating_values": checked_model.floating_values,
            "raw_head_all_finite": False,
            "metrics_produced": False,
        }
        _atomic_write_once(output_dir / "epoch04_d_select_failure.json", d_select)

    del checked_model, eval_model, eval_optimizer, eval_scheduler, eval_scaler
    gc.collect()
    torch.cuda.empty_cache()

    production = _run_localization_cell(
        cell="production_compile_fp16",
        config=config,
        checkpoint=checkpoint,
        checkpoint_record=epoch_record,
        device=device,
        precision="fp16",
        compiled=True,
        sdpa=False,
        expected_batch_sha256=None,
    )
    cells = [production]
    if not production["all_finite"]:
        eager_fp16 = _run_localization_cell(
            cell="eager_reference_fp16",
            config=config,
            checkpoint=checkpoint,
            checkpoint_record=epoch_record,
            device=device,
            precision="fp16",
            compiled=False,
            sdpa=False,
            expected_batch_sha256=production["batch_sha256"],
        )
        cells.append(eager_fp16)
        if not eager_fp16["all_finite"]:
            cells.append(
                _run_localization_cell(
                    cell="eager_reference_fp32",
                    config=config,
                    checkpoint=checkpoint,
                    checkpoint_record=epoch_record,
                    device=device,
                    precision="fp32",
                    compiled=False,
                    sdpa=False,
                    expected_batch_sha256=production["batch_sha256"],
                )
            )
            if eager_fp16["first_nonfinite_stage"] in {
                "self_attention",
                "cross_attention",
            }:
                cells.append(
                    _run_localization_cell(
                        cell="eager_sdpa_fp16",
                        config=config,
                        checkpoint=checkpoint,
                        checkpoint_record=epoch_record,
                        device=device,
                        precision="fp16",
                        compiled=False,
                        sdpa=True,
                        expected_batch_sha256=production["batch_sha256"],
                    )
                )

    result = {
        "schema": SCHEMA,
        "status": "COMPLETE_DIAGNOSTIC",
        "source": source,
        "resolved_config_sha256": config.sha256,
        "checkpoint_sha256": epoch_record["checkpoint_sha256"],
        "checkpoint_epoch": EXPECTED_CHECKPOINT_EPOCH,
        "D_select": d_select,
        "localization_cells": cells,
        "backward_executed": False,
        "optimizer_update_executed": False,
        "D_audit_executed": False,
        "official_validation_executed": False,
        "automatic_recipe_promotion": False,
        "interpretation_limits": [
            "epoch-4 D_select is a disclosed diagnostic peek, not a selectable checkpoint",
            "no best-epoch, early-stopping, capability-completion, or generalization claim",
            "localization cells diagnose numerical behavior and do not promote a recipe",
        ],
    }
    _atomic_write_once(output_dir / "result.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--epoch-record", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--source-sha", required=True)
    result = _run(parser.parse_args())
    print(json.dumps({"status": result["status"]}, sort_keys=True))


if __name__ == "__main__":
    main()
