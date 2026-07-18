"""Opt-in, pre-step precision-window diagnostics for S08.

The facility is intentionally narrow: no module hooks, no checkpoint state, and
no model-output changes.  Parameter gradients are inspected after GradScaler
``unscale_``.  Retained activation gradients are *not* touched by ``unscale_``;
their statistics are computed from an FP64 copy divided by the window scale.
"""
from __future__ import annotations

from contextlib import contextmanager, nullcontext
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import random
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn

from fl_v3.training.runtime_state import TrainingState


WINDOW_DIAGNOSTICS_SCHEMA = "s08.window-diagnostics.v1"

_STATE_FIELDS = tuple(TrainingState.__dataclass_fields__)
_PARAMETER_PREFIXES = (
    "lidar_encoder.backbone.conv_out",
    "lidar_encoder.backbone.stage4",
    "lidar_encoder.backbone.down3",
    "lidar_encoder.backbone.stage3",
    "lidar_encoder.backbone.down2",
    "lidar_encoder.backbone.stage2",
    "lidar_encoder.backbone.down1",
    "lidar_encoder.backbone.stage1",
    "lidar_encoder.backbone.stem",
    "lidar_encoder.to_bev",
    "camera_backbone",
    "camera_neck",
    "view_transform",
    "lidar_backbone",
    "camera_adapter",
    "lidar_adapter",
    "fusion",
    "bev_neck",
    "head",
)


def _state_snapshot(state: TrainingState) -> dict[str, int]:
    return {name: int(getattr(state, name)) for name in _STATE_FIELDS}


def runtime_rng_state_sha256() -> str:
    """Hash Python/NumPy/Torch CPU and all CUDA RNG states without changing them."""
    digest = hashlib.sha256()
    digest.update(repr(random.getstate()).encode("utf-8"))
    numpy_state = np.random.get_state()
    digest.update(str(numpy_state[0]).encode("ascii"))
    digest.update(np.asarray(numpy_state[1]).tobytes(order="C"))
    digest.update(repr(tuple(numpy_state[2:])).encode("utf-8"))
    digest.update(torch.get_rng_state().detach().cpu().numpy().tobytes())
    for index, state in enumerate(torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []):
        digest.update(str(index).encode("ascii"))
        digest.update(state.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def _nonfinite_value(value: float) -> dict[str, Any]:
    if math.isnan(value):
        label = "nan"
    elif value > 0:
        label = "+inf"
    else:
        label = "-inf"
    return {"value": None, "nonfinite": label}


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else _nonfinite_value(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if torch.is_tensor(value):
        raise TypeError("diagnostic records must convert tensors before JSON finalization")
    raise TypeError(f"unsupported diagnostic record value {type(value)!r}")


def _assert_strict_json(value: Any) -> None:
    json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


@dataclass(frozen=True)
class PrecisionDiagnosticsIdentity:
    source_sha: str
    resolved_config_sha256: str
    model_mode: str
    global_precision: str
    sparse_conv_precision: str

    def __post_init__(self) -> None:
        hexchars = frozenset("0123456789abcdef")
        if len(self.source_sha) != 40 or any(c not in hexchars for c in self.source_sha):
            raise ValueError("source_sha must be an exact 40-character lowercase Git commit SHA")
        if len(self.resolved_config_sha256) != 64 or any(
            c not in hexchars for c in self.resolved_config_sha256
        ):
            raise ValueError("resolved_config_sha256 must be 64 lowercase hexadecimal characters")
        if self.model_mode not in {"camera_only", "lidar_only", "fusion"}:
            raise ValueError("diagnostic model_mode is invalid")
        if self.global_precision not in {"fp32", "fp16"}:
            raise ValueError("diagnostic global_precision is invalid")
        if self.sparse_conv_precision not in {"fp32", "fp16", "not_applicable"}:
            raise ValueError("diagnostic sparse_conv_precision is invalid")


@dataclass
class _WindowToken:
    index: int
    record: dict[str, Any]
    scale_before: float


def _parameter_prefix(name: str) -> str:
    for prefix in _PARAMETER_PREFIXES:
        if name == prefix or name.startswith(prefix + "."):
            return prefix
    return "other"


def _entry_tensors(gradient: torch.Tensor) -> dict[str, torch.Tensor]:
    detached = gradient.detach()
    finite_mask = torch.isfinite(detached)
    finite = detached[finite_mask].to(torch.float64)
    zero_f = torch.zeros((), device=detached.device, dtype=torch.float64)
    return {
        "finite": finite_mask.sum(dtype=torch.int64),
        "nan": torch.isnan(detached).sum(dtype=torch.int64),
        "posinf": torch.isposinf(detached).sum(dtype=torch.int64),
        "neginf": torch.isneginf(detached).sum(dtype=torch.int64),
        "sum_sq": torch.square(finite).sum() if finite.numel() else zero_f,
        "max_abs": finite.abs().max() if finite.numel() else zero_f,
    }


def _summarize_entries(entries: list[dict[str, Any]]) -> dict[str, Any]:
    total = sum(int(entry["numel"]) for entry in entries)
    if not entries:
        return {
            "total_elements": 0,
            "finite_elements": 0,
            "nonfinite_elements": 0,
            "nan_elements": 0,
            "positive_inf_elements": 0,
            "negative_inf_elements": 0,
            "all_finite": True,
            "stable_finite_l2": 0.0,
            "stable_finite_rms": 0.0,
            "max_abs_finite": 0.0,
            "complete_l2": 0.0,
        }
    finite = int(torch.stack([entry["finite"] for entry in entries]).sum().item())
    nan = int(torch.stack([entry["nan"] for entry in entries]).sum().item())
    posinf = int(torch.stack([entry["posinf"] for entry in entries]).sum().item())
    neginf = int(torch.stack([entry["neginf"] for entry in entries]).sum().item())
    sum_sq = float(torch.stack([entry["sum_sq"] for entry in entries]).sum().item())
    max_abs = float(torch.stack([entry["max_abs"] for entry in entries]).max().item())
    nonfinite = nan + posinf + neginf
    l2 = math.sqrt(max(0.0, sum_sq))
    rms = math.sqrt(max(0.0, sum_sq / finite)) if finite else 0.0
    return {
        "total_elements": total,
        "finite_elements": finite,
        "nonfinite_elements": nonfinite,
        "nan_elements": nan,
        "positive_inf_elements": posinf,
        "negative_inf_elements": neginf,
        "all_finite": nonfinite == 0 and finite == total,
        "stable_finite_l2": l2,
        "stable_finite_rms": rms,
        "max_abs_finite": max_abs,
        "complete_l2": l2 if nonfinite == 0 and finite == total else None,
    }


def parameter_gradient_statistics(model: nn.Module) -> dict[str, Any]:
    """Stable FP64 finite-only summaries after optimizer-gradient unscale."""
    entries: list[dict[str, Any]] = []
    missing: list[str] = []
    trainable_elements = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        trainable_elements += int(parameter.numel())
        if parameter.grad is None:
            missing.append(name)
            continue
        tensors = _entry_tensors(parameter.grad)
        entries.append({
            "name": name,
            "prefix": _parameter_prefix(name),
            "numel": int(parameter.grad.numel()),
            **tensors,
        })

    bad_counts = []
    if entries:
        bad_counts = torch.stack([
            entry["nan"] + entry["posinf"] + entry["neginf"] for entry in entries
        ]).detach().cpu().tolist()
    bad_names = [
        entry["name"] for entry, count in zip(entries, bad_counts, strict=True) if int(count) > 0
    ]
    groups: dict[str, list[dict[str, Any]]] = {}
    for entry in entries:
        groups.setdefault(str(entry["prefix"]), []).append(entry)
    return {
        "gradient_domain": "optimizer_unscaled",
        "pre_clip": True,
        "trainable_elements": trainable_elements,
        "parameters_with_grad": len(entries),
        "missing_grad_parameter_count": len(missing),
        "missing_grad_parameters": missing,
        "first_nonfinite_parameter_in_named_order": bad_names[0] if bad_names else None,
        "nonfinite_parameters_in_named_order": bad_names,
        "global": _summarize_entries(entries),
        "by_prefix": {
            prefix: _summarize_entries(group) for prefix, group in sorted(groups.items())
        },
    }


def _single_tensor_statistics(tensor: torch.Tensor) -> dict[str, Any]:
    entry = {"numel": int(tensor.numel()), **_entry_tensors(tensor)}
    return _summarize_entries([entry])


def boundary_gradient_statistics(
    boundaries: Mapping[str, torch.Tensor],
    *,
    scale_divisor: float,
    amp_enabled: bool,
) -> dict[str, Any]:
    """Summarize retained activation grads without mutating their scaled values."""
    if not math.isfinite(scale_divisor) or scale_divisor <= 0.0:
        raise ValueError("boundary gradient scale divisor must be finite and positive")
    result: dict[str, Any] = {}
    for name, tensor in sorted(boundaries.items()):
        item: dict[str, Any] = {
            "activation_shape": [int(value) for value in tensor.shape],
            "activation_dtype": str(tensor.dtype),
            "gradient_present": tensor.grad is not None,
            "raw_gradient_domain": "gradscaler_scaled" if amp_enabled else "unscaled",
            "unscaled_copy_divisor": float(scale_divisor),
        }
        if tensor.grad is not None:
            raw = tensor.grad.detach()
            item["raw_gradient_dtype"] = str(raw.dtype)
            item["raw_scaled"] = _single_tensor_statistics(raw)
            unscaled = raw.to(torch.float64) / float(scale_divisor)
            item["explicit_unscaled_fp64"] = _single_tensor_statistics(unscaled)
        result[name] = item
    return result


class PrecisionWindowDiagnostics:
    """Preallocated, opt-in records for a bounded S08 qualification cell."""

    def __init__(
        self,
        identity: PrecisionDiagnosticsIdentity,
        *,
        max_windows: int = 18,
        capture_boundaries: bool = True,
        attempted_windows: Sequence[int] | None = None,
        capture_parameter_updates: bool = False,
        fixture_identity: Mapping[str, Any] | None = None,
    ) -> None:
        if isinstance(max_windows, bool) or int(max_windows) < 1:
            raise ValueError("max_windows must be a positive integer")
        self.identity = identity
        self.max_windows = int(max_windows)
        self.capture_boundaries = bool(capture_boundaries)
        if attempted_windows is None:
            self.attempted_windows = None
        else:
            normalized = tuple(int(value) for value in attempted_windows)
            if (
                not normalized
                or any(value < 1 for value in normalized)
                or normalized != tuple(sorted(set(normalized)))
            ):
                raise ValueError("attempted_windows must be sorted unique positive integers")
            if len(normalized) != self.max_windows:
                raise ValueError("attempted_windows length must equal max_windows")
            self.attempted_windows = frozenset(normalized)
        self.capture_parameter_updates = bool(capture_parameter_updates)
        self.fixture_identity = _json_safe(dict(fixture_identity or {}))
        _assert_strict_json(self.fixture_identity)
        self._slots: list[dict[str, Any] | None] = [None] * self.max_windows
        self._next_index = 0
        self._active: _WindowToken | None = None
        self._parameter_before: dict[str, torch.Tensor] | None = None

    def should_capture(self, *, state: TrainingState, next_step: int) -> bool:
        """Return whether the next attempted optimizer window is predeclared."""
        del next_step
        attempted_window = int(state.attempted_windows) + 1
        return self.attempted_windows is None or attempted_window in self.attempted_windows

    @property
    def records(self) -> tuple[dict[str, Any], ...]:
        return tuple(record for record in self._slots if record is not None)

    @staticmethod
    def _validate_optimizer_coverage(model: nn.Module, optimizer) -> None:
        model_ids = [id(parameter) for parameter in model.parameters() if parameter.requires_grad]
        optimizer_ids = [
            id(parameter)
            for group in optimizer.param_groups
            for parameter in group["params"]
        ]
        if len(optimizer_ids) != len(set(optimizer_ids)):
            raise RuntimeError("diagnostics refuse duplicate optimizer parameters")
        if set(model_ids) != set(optimizer_ids):
            raise RuntimeError(
                "diagnostics require optimizer parameters to equal model trainable parameters"
            )

    def _validate_model_identity(self, model: nn.Module, precision: str) -> None:
        if precision != self.identity.global_precision:
            raise RuntimeError("diagnostic/global loop precision identity mismatch")
        if getattr(model, "model_mode", None) != self.identity.model_mode:
            raise RuntimeError("diagnostic/model mode identity mismatch")
        cfg = getattr(model, "cfg", None)
        if cfg is None:
            if self.capture_boundaries:
                raise RuntimeError("boundary diagnostics require a detector with explicit config")
            return
        second = bool(
            self.identity.model_mode in {"lidar_only", "fusion"}
            and getattr(cfg, "lidar_encoder", None) == "voxel"
        )
        if second:
            expected_requested = self.identity.sparse_conv_precision == "fp16"
            if self.identity.sparse_conv_precision == "not_applicable":
                raise RuntimeError("SECOND diagnostics require an explicit sparse partition")
            if bool(getattr(cfg, "sparse_conv_fp16", False)) != expected_requested:
                raise RuntimeError("diagnostic sparse partition/internal request mismatch")
        elif self.identity.sparse_conv_precision != "not_applicable":
            raise RuntimeError("non-SECOND diagnostics require sparse partition not_applicable")

    def begin_window(
        self,
        *,
        model: nn.Module,
        optimizer,
        state: TrainingState,
        scaler,
        scheduler,
        ema_model,
        accumulation_steps: int,
        precision: str,
    ) -> _WindowToken:
        if accumulation_steps != 1:
            raise RuntimeError("S08 window diagnostics require accumulation_steps == 1")
        if self._active is not None:
            raise RuntimeError("nested diagnostic windows are forbidden")
        if self._next_index >= self.max_windows:
            raise RuntimeError("diagnostic window capacity exhausted before forward")
        state.validate(checkpoint_boundary=True)
        self._validate_optimizer_coverage(model, optimizer)
        self._validate_model_identity(model, precision)
        scale_before = float(scaler.get_scale())
        if not math.isfinite(scale_before) or scale_before <= 0.0:
            raise RuntimeError("GradScaler scale must be finite and positive")
        scaler_enabled = bool(scaler.is_enabled())
        scheduler_before = None if scheduler is None else int(scheduler.last_epoch)
        record = {
            "schema": WINDOW_DIAGNOSTICS_SCHEMA,
            "window_index": self._next_index,
            "identity": asdict(self.identity),
            "fixture_identity": self.fixture_identity,
            "scaler": {
                "enabled": scaler_enabled,
                "scale_before": scale_before,
                "scale_after": None,
                "backoff_factor": (
                    float(scaler.get_backoff_factor())
                    if scaler_enabled and hasattr(scaler, "get_backoff_factor") else None
                ),
                "growth_factor": (
                    float(scaler.get_growth_factor())
                    if scaler_enabled and hasattr(scaler, "get_growth_factor") else None
                ),
                "growth_interval": (
                    int(scaler.get_growth_interval())
                    if scaler_enabled and hasattr(scaler, "get_growth_interval") else None
                ),
            },
            "counters_before": _state_snapshot(state),
            "counters_pre_step": None,
            "counters_after": None,
            "scheduler_last_epoch_before": scheduler_before,
            "scheduler_last_epoch_after": None,
            "scheduler_delta_consistent": None,
            "ema_enabled": ema_model is not None,
            "ema_updates_expected_before": (
                int(state.successful_windows) if ema_model is not None else None
            ),
            "ema_updates_expected_after": None,
            "ema_state_consistent": None,
            "rng_state_sha256_before_forward": runtime_rng_state_sha256(),
            "rng_state_sha256_after_forward_backward": None,
            "loss_finite": None,
            "loss": None,
            "loss_terms": None,
            "parameter_gradients_unscaled": None,
            "parameter_gradients": None,
            "parameter_updates": None,
            "boundary_gradients": None,
            "sparse_conv_fp16_requested": None,
            "sparse_conv_fp16_active": None,
            "sparse_runtime_consistent": None,
            "sparse_meta": None,
            "voxel_stats": None,
            "outcome": None,
            "accepted": None,
            "skipped": None,
            "counter_deltas_consistent": None,
        }
        token = _WindowToken(self._next_index, record, scale_before)
        self._active = token
        return token

    def _require_active(self, token: _WindowToken) -> None:
        if self._active is not token:
            raise RuntimeError("diagnostic window token is not active")

    @contextmanager
    def capture(self, model: nn.Module, token: _WindowToken):
        self._require_active(token)
        factory = getattr(model, "capture_training_boundaries", None)
        if self.capture_boundaries and factory is None:
            self._active = None
            raise RuntimeError("model does not expose explicit training-boundary capture")
        context = factory() if self.capture_boundaries else nullcontext({})
        try:
            with context as boundaries:
                yield boundaries
        finally:
            if self._active is token:
                self._active = None

    def _sparse_identity(self, model: nn.Module) -> tuple[bool | None, bool | None, bool | None]:
        cfg = getattr(model, "cfg", None)
        second = bool(
            cfg is not None
            and self.identity.model_mode in {"lidar_only", "fusion"}
            and getattr(cfg, "lidar_encoder", None) == "voxel"
        )
        if not second:
            return None, None, True
        encoder = getattr(model, "lidar_encoder", None)
        meta = None if encoder is None else encoder.last_sparse_meta
        if not isinstance(meta, dict):
            raise RuntimeError("SECOND diagnostics require current sparse runtime metadata")
        requested = bool(meta["sparse_conv_fp16_requested"])
        active = bool(meta["sparse_conv_fp16_active"])
        expected = self.identity.sparse_conv_precision == "fp16"
        if requested != expected:
            raise RuntimeError("sparse runtime request differs from resolved partition")
        parameter = next((p for p in model.parameters() if p.requires_grad), None)
        on_cuda = bool(parameter is not None and parameter.device.type == "cuda")
        consistent = requested == expected and (active == requested if on_cuda else not active)
        if not consistent:
            raise RuntimeError("sparse runtime active precision differs from requested partition")
        return requested, active, consistent

    def prepare_window(
        self,
        token: _WindowToken,
        *,
        model: nn.Module,
        criterion,
        boundaries: Mapping[str, torch.Tensor],
        state: TrainingState,
        loss_value: float,
        loss_finite: bool,
        parameters_unscaled: bool,
    ) -> dict[str, Any]:
        """Perform every tensor operation and validation before optimizer step."""
        self._require_active(token)
        expected_boundaries = {"head.input"}
        cfg = getattr(model, "cfg", None)
        second = bool(
            cfg is not None
            and self.identity.model_mode in {"lidar_only", "fusion"}
            and getattr(cfg, "lidar_encoder", None) == "voxel"
        )
        if second:
            expected_boundaries |= {"second.stem", "second.stage1", "second.output"}
        if self.capture_boundaries and set(boundaries) != expected_boundaries:
            raise RuntimeError(
                "training-boundary set mismatch: "
                f"expected={sorted(expected_boundaries)}, actual={sorted(boundaries)}"
            )

        requested, active, consistent = self._sparse_identity(model)
        record = token.record
        record["loss_finite"] = bool(loss_finite)
        record["loss"] = _json_safe(float(loss_value))
        diagnostic_terms = getattr(criterion, "diagnostic_terms", None)
        record["loss_terms"] = _json_safe(
            diagnostic_terms() if diagnostic_terms is not None else {}
        )
        record["parameter_gradients_unscaled"] = bool(parameters_unscaled)
        record["parameter_gradients"] = parameter_gradient_statistics(model)
        if self.capture_parameter_updates:
            self._parameter_before = {
                name: parameter.detach().clone()
                for name, parameter in model.named_parameters()
                if parameter.requires_grad
            }
            token._current_parameters = {
                name: parameter
                for name, parameter in model.named_parameters()
                if parameter.requires_grad
            }
        record["boundary_gradients"] = boundary_gradient_statistics(
            boundaries,
            scale_divisor=token.scale_before if bool(record["scaler"]["enabled"]) else 1.0,
            amp_enabled=bool(record["scaler"]["enabled"]),
        )
        record["sparse_conv_fp16_requested"] = requested
        record["sparse_conv_fp16_active"] = active
        record["sparse_runtime_consistent"] = consistent

        encoder = getattr(model, "lidar_encoder", None)
        sparse_meta = None if encoder is None else getattr(encoder, "last_sparse_meta", None)
        record["sparse_meta"] = _json_safe(sparse_meta)
        voxel_stats = None if encoder is None else getattr(encoder, "last_voxel_stats", None)
        if torch.is_tensor(voxel_stats):
            meta_fields = sparse_meta.get("voxel_stat_fields", ()) if isinstance(sparse_meta, dict) else ()
            record["voxel_stats"] = {
                "fields": list(meta_fields),
                "per_sample": voxel_stats.detach().to("cpu").tolist(),
            }
        record["counters_pre_step"] = _state_snapshot(state)
        record["rng_state_sha256_after_forward_backward"] = runtime_rng_state_sha256()
        _assert_strict_json(_json_safe(record))
        return record

    def finalize_window(
        self,
        token: _WindowToken,
        *,
        state: TrainingState,
        scheduler,
        ema_model,
        scaler_after: float,
        outcome: str,
    ) -> None:
        """Complete post-step state and optional realized-update summaries."""
        accepted = outcome == "accepted"
        record = token.record
        if self.capture_parameter_updates:
            if self._parameter_before is None:
                raise RuntimeError("parameter-update diagnostics have no pre-step snapshot")
            grouped: dict[str, dict[str, Any]] = {}
            current_parameters = getattr(token, "_current_parameters", None)
            if current_parameters is None:
                raise RuntimeError("parameter-update diagnostics lost current parameters")
            totals: dict[str, list[torch.Tensor | int]] = {}
            for name, before in self._parameter_before.items():
                current = current_parameters[name].detach()
                prefix = _parameter_prefix(name)
                weight_sq = before.to(torch.float64).square().sum()
                update_sq = (current.to(torch.float64) - before.to(torch.float64)).square().sum()
                slot = totals.setdefault(prefix, [weight_sq.new_zeros(()), update_sq.new_zeros(()), 0])
                slot[0] = slot[0] + weight_sq
                slot[1] = slot[1] + update_sq
                slot[2] = int(slot[2]) + int(before.numel())
            all_weight_sq = next(iter(totals.values()))[0].new_zeros(())
            all_update_sq = all_weight_sq.clone()
            all_numel = 0
            for prefix, (weight_sq, update_sq, numel) in sorted(totals.items()):
                weight_l2 = math.sqrt(max(0.0, float(weight_sq.item())))
                update_l2 = math.sqrt(max(0.0, float(update_sq.item())))
                grouped[prefix] = {
                    "numel": int(numel),
                    "weight_l2_before": weight_l2,
                    "realized_update_l2": update_l2,
                    "realized_update_over_weight": (
                        update_l2 / weight_l2 if weight_l2 > 0.0 else None
                    ),
                }
                all_weight_sq = all_weight_sq + weight_sq
                all_update_sq = all_update_sq + update_sq
                all_numel += int(numel)
            global_weight_l2 = math.sqrt(max(0.0, float(all_weight_sq.item())))
            global_update_l2 = math.sqrt(max(0.0, float(all_update_sq.item())))
            record["parameter_updates"] = {
                "domain": "realized_post_optimizer_step",
                "accepted": accepted,
                "global": {
                    "numel": all_numel,
                    "weight_l2_before": global_weight_l2,
                    "realized_update_l2": global_update_l2,
                    "realized_update_over_weight": (
                        global_update_l2 / global_weight_l2
                        if global_weight_l2 > 0.0 else None
                    ),
                },
                "by_prefix": grouped,
            }
            self._parameter_before = None
            del token._current_parameters
        record["scaler"]["scale_after"] = float(scaler_after)
        record["outcome"] = str(outcome)
        record["accepted"] = accepted
        record["skipped"] = not accepted
        after = _state_snapshot(state)
        record["counters_after"] = after
        record["scheduler_last_epoch_after"] = (
            None if scheduler is None else int(scheduler.last_epoch)
        )
        record["ema_updates_expected_after"] = (
            int(state.successful_windows) if ema_model is not None else None
        )
        before = record["counters_before"]
        scheduler_before = record["scheduler_last_epoch_before"]
        scheduler_after = record["scheduler_last_epoch_after"]
        record["scheduler_delta_consistent"] = bool(
            (scheduler_before is None and scheduler_after is None)
            or (
                scheduler_before is not None
                and scheduler_after is not None
                and scheduler_after - scheduler_before == int(accepted)
            )
        )
        ema_enabled = bool(record["ema_enabled"])
        record["ema_state_consistent"] = bool(
            ema_enabled == (ema_model is not None)
            and (
                (
                    not ema_enabled
                    and record["ema_updates_expected_before"] is None
                    and record["ema_updates_expected_after"] is None
                )
                or (
                    ema_enabled
                    and record["ema_updates_expected_before"]
                    == before["successful_windows"]
                    and record["ema_updates_expected_after"]
                    == after["successful_windows"]
                )
            )
        )
        record["counter_deltas_consistent"] = bool(
            after["optimizer_step"] - before["optimizer_step"] == int(accepted)
            and after["successful_windows"] - before["successful_windows"] == int(accepted)
            and after["attempted_windows"] - before["attempted_windows"] == 1
            and after["invalid_windows"] - before["invalid_windows"] == int(not accepted)
        )
        self._slots[token.index] = record
        self._next_index += 1
        self._active = None

    def json_lines(self) -> str:
        return "".join(
            json.dumps(_json_safe(record), sort_keys=True, allow_nan=False) + "\n"
            for record in self.records
        )
