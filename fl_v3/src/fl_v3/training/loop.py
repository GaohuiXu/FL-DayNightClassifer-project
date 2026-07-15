"""Generic local training / evaluation loop (fl_v3 T0).

**The criterion is injected, never hardcoded** — this is the decoupling vs the
fl_v2 oracle, whose ``train_local`` instantiated ``nn.CrossEntropyLoss()``
internally. Here the loss is a parameter, so the same loop trains a regression
task (MSE), a classification task (CE), or the AD detection task (composite
detection loss) without change. No accuracy is computed in the loop — utility
metrics are the Task's responsibility (``Task.evaluate``).

Optimizer is **Adam** (carried from the platform; do NOT switch to SGD — the
reproduction-fidelity decision). Only ``requires_grad`` params get optimizer
state, so frozen backbones (D1) allocate no Adam moments.
"""
from __future__ import annotations

import time
from typing import Any, Callable, Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from fl_v3.utils.runtime import (
    current_precision,
    make_grad_scaler,
    normalize_precision,
    precision_autocast_context,
)
from fl_v3.training.runtime_state import TrainingState, project_batch_for_mode

# A criterion maps (model_output, target) → scalar loss. ``target`` may be a tensor
# (the regression/classification tasks) OR the multimodal detection batch dict (the AD
# task) — the alias is widened from the T0 tensor-only signature so the SAME loop trains
# either. (Mirrors the widened alias in ``training/tasks.py``.)
Criterion = Callable[[Any, Any], torch.Tensor]


def _percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return float("nan")
    position = (len(ordered) - 1) * float(quantile)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _distribution(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"n": 0, "mean": None, "p50": None,
                "p95": None, "min": None, "max": None}
    return {
        "n": len(values),
        "mean": sum(values) / len(values),
        "p50": _percentile(values, 0.50),
        "p95": _percentile(values, 0.95),
        "min": min(values),
        "max": max(values),
    }


def _state_counters(state: TrainingState) -> dict[str, int]:
    return {
        name: int(getattr(state, name))
        for name in state.__dataclass_fields__
    }


class _ReadinessTiming:
    """Direct bounded stage timing; no hooks, retained tensors, or tensor math."""

    _STAGES = ("h2d", "forward", "loss", "backward", "optimizer")

    def __init__(self, device: torch.device, warmup_successful_windows: int) -> None:
        self.device = torch.device(device)
        self.use_cuda = self.device.type == "cuda"
        self.warmup_successful_windows = int(warmup_successful_windows)
        self.records: list[dict[str, Any]] = []
        self.active: dict[str, Any] | None = None
        self.measurement_started = False
        self.measurement_wall_start: float | None = None
        self.measurement_start_counters: dict[str, int] | None = None
        self.warmup_boundary_attempted_window: int | None = None

    def _mark(self) -> Any:
        if self.use_cuda:
            event = torch.cuda.Event(enable_timing=True)
            event.record(torch.cuda.current_stream(self.device))
            return event
        return time.perf_counter()

    def _elapsed_ms(self, start: Any, end: Any) -> float:
        if self.use_cuda:
            return float(start.elapsed_time(end))
        return float(end - start) * 1000.0

    def start_measurement(self, state: TrainingState) -> None:
        if self.measurement_started:
            raise RuntimeError("readiness timing measurement boundary was started twice")
        state.validate(checkpoint_boundary=True)
        if self.use_cuda:
            torch.cuda.synchronize(self.device)
            torch.cuda.reset_peak_memory_stats(self.device)
        self.measurement_started = True
        self.measurement_wall_start = time.perf_counter()
        self.measurement_start_counters = _state_counters(state)
        self.warmup_boundary_attempted_window = int(state.attempted_windows)

    def begin_window(
        self, *, attempted_window: int, data_wait_ms: float, scaler_before: float
    ) -> None:
        if self.active is not None:
            raise RuntimeError("nested readiness timing window")
        self.active = {
            "attempted_window": int(attempted_window),
            "data_wait_ms": float(data_wait_ms),
            "scaler_before": float(scaler_before),
            "measured": bool(self.measurement_started),
            "total_start": self._mark(),
            "pairs": {},
            "stage_start": None,
            "stage_name": None,
        }

    def begin_stage(self, name: str) -> None:
        if self.active is None or name not in self._STAGES:
            raise RuntimeError(f"invalid readiness timing stage start {name!r}")
        if self.active["stage_name"] is not None:
            raise RuntimeError("nested readiness timing stage")
        self.active["stage_name"] = name
        self.active["stage_start"] = self._mark()

    def end_stage(self, name: str) -> None:
        if self.active is None or self.active["stage_name"] != name:
            raise RuntimeError(f"readiness timing stage end mismatch {name!r}")
        self.active["pairs"][name] = (self.active["stage_start"], self._mark())
        self.active["stage_name"] = None
        self.active["stage_start"] = None

    def finish_window(
        self,
        *,
        outcome: str,
        scaler_after: float,
        global_samples: int,
    ) -> None:
        if self.active is None or self.active["stage_name"] is not None:
            raise RuntimeError("readiness timing window ended with an active stage")
        self.active["pairs"]["window"] = (
            self.active.pop("total_start"),
            self._mark(),
        )
        self.active["outcome"] = str(outcome)
        self.active["scaler_after"] = float(scaler_after)
        self.active["global_samples"] = int(global_samples)
        self.records.append(self.active)
        self.active = None

    def finalize(self, state: TrainingState) -> dict[str, Any]:
        if self.active is not None:
            raise RuntimeError("readiness timing finalized with an active window")
        if self.use_cuda:
            torch.cuda.synchronize(self.device)
        terminal = time.perf_counter()
        resolved_records: list[dict[str, Any]] = []
        for pending in self.records:
            pairs = pending.pop("pairs")
            pending.pop("stage_start")
            pending.pop("stage_name")
            pending["durations_ms"] = {
                name: self._elapsed_ms(start, end)
                for name, (start, end) in pairs.items()
            }
            resolved_records.append(pending)

        measured = [record for record in resolved_records if record["measured"]]
        accepted = [record for record in measured if record["outcome"] == "accepted"]
        stage_distributions = {
            name: _distribution([
                record["durations_ms"][name] for record in accepted
            ])
            for name in (*self._STAGES, "window")
        }
        data_wait = _distribution([record["data_wait_ms"] for record in measured])
        measured_wall_seconds = (
            None
            if self.measurement_wall_start is None
            else float(terminal - self.measurement_wall_start)
        )
        terminal_counters = _state_counters(state)
        start_counters = self.measurement_start_counters
        counter_delta = (
            None
            if start_counters is None
            else {
                key: terminal_counters[key] - start_counters[key]
                for key in terminal_counters
            }
        )
        successful_samples = 0 if counter_delta is None else counter_delta["exposure_samples"]
        successful_windows = 0 if counter_delta is None else counter_delta["successful_windows"]
        throughput = {
            "successful_windows_per_second": (
                None
                if not measured_wall_seconds
                else successful_windows / measured_wall_seconds
            ),
            "exposure_samples_per_second": (
                None
                if not measured_wall_seconds
                else successful_samples / measured_wall_seconds
            ),
        }

        if self.use_cuda and self.measurement_started:
            peak_allocated = int(torch.cuda.max_memory_allocated(self.device))
            peak_reserved = int(torch.cuda.max_memory_reserved(self.device))
            total_memory = int(torch.cuda.get_device_properties(self.device).total_memory)
        else:
            peak_allocated = peak_reserved = total_memory = 0
        window_p50 = stage_distributions["window"]["p50"]
        window_p95 = stage_distributions["window"]["p95"]
        return {
            "schema": "s09.direct-window-timing.v1",
            "clock": "cuda_event" if self.use_cuda else "host_perf_counter",
            "measurement_definition": (
                "data_wait_ms is host time blocked in next(DataLoader); durations_ms are "
                "CUDA-event stream durations on CUDA and synchronous host durations on CPU; "
                "window spans H2D through optimizer/scheduler/EMA"
            ),
            "warmup_successful_windows": self.warmup_successful_windows,
            "warmup_boundary_reached": self.measurement_started,
            "warmup_boundary_attempted_window": self.warmup_boundary_attempted_window,
            "measurement_wall_seconds": measured_wall_seconds,
            "measurement_start_counters": start_counters,
            "measurement_counter_delta": counter_delta,
            "terminal_counters": terminal_counters,
            "measured_attempted_windows": len(measured),
            "measured_accepted_windows": len(accepted),
            "measured_accepted_ratio": (
                len(accepted) / len(measured) if measured else None
            ),
            "data_wait_ms": data_wait,
            "accepted_stage_ms": stage_distributions,
            "window_p95_over_p50": (
                window_p95 / window_p50 if window_p50 and window_p50 > 0.0 else None
            ),
            "throughput": throughput,
            "memory": {
                "peak_allocated_bytes": peak_allocated,
                "peak_reserved_bytes": peak_reserved,
                "device_total_bytes": total_memory,
                "reserved_headroom_bytes": max(0, total_memory - peak_reserved),
            },
            "records": resolved_records,
        }


# ---------------------------------------------------------------------------
# Additive batch protocol (T2): the loop is task-agnostic over the batch SHAPE.
# A batch is either a ``(inputs, targets)`` tensor 2-tuple (the T0/regression path,
# byte-identical to before) OR a single ``dict`` (the detection batch — inputs AND
# targets are the same dict; the model reads its tensors, the criterion reads its GT).
# ---------------------------------------------------------------------------
def _move_to_device(obj: Any, device: torch.device) -> Any:
    """Recursively move tensors in a tensor / dict / list / tuple to ``device``.

    Non-tensors (str/int identity fields in the detection batch) pass through. A bare
    tensor takes the SAME ``.to(device)`` path as the original loop (byte-identical)."""
    if torch.is_tensor(obj):
        # non_blocking overlaps the HtoD with compute when the source is pinned (the loader sets
        # pin_memory=True). Timing-only — value-identical (incl. the fp32 byte-identity dev tool) since the
        # tensor is consumed on-device, never read on CPU before the copy lands.
        return obj.to(device, non_blocking=True)
    if isinstance(obj, dict):
        return {k: _move_to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_move_to_device(v, device) for v in obj)
    return obj


def _float_tensors(obj: Any) -> Any:
    """Recursively upcast tensors to fp32 for numerically sensitive losses/decode."""
    if torch.is_tensor(obj):
        return obj.float()
    if isinstance(obj, dict):
        return {k: _float_tensors(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_float_tensors(v) for v in obj)
    return obj


def _unpack_batch(batch: Any, device: torch.device) -> Tuple[Any, Any]:
    """Return ``(inputs, targets)`` both on ``device``.

    Tensor path: ``(X, y)`` → ``(X.to, y.to)`` — the exact original behavior. Dict path:
    the detection batch is moved ONCE and used as both inputs and targets."""
    if isinstance(batch, dict):
        batch = _move_to_device(batch, device)
        return batch, batch
    inputs, targets = batch
    return _move_to_device(inputs, device), _move_to_device(targets, device)


def _batch_size(targets: Any) -> int:
    """Examples in the batch: ``targets.size(0)`` (tensor) or ``len(gt_boxes)`` (dict)."""
    if torch.is_tensor(targets):
        return int(targets.size(0))
    if isinstance(targets, dict):
        return int(len(targets["gt_boxes"]))
    raise TypeError(f"cannot infer batch size from targets of type {type(targets)!r}")


def _grad_norm(model: nn.Module) -> float:
    norms = [
        p.grad.detach().float().norm(2)
        for p in model.parameters()
        if p.requires_grad and p.grad is not None
    ]
    if not norms:
        return 0.0
    return float(torch.linalg.vector_norm(torch.stack(norms), 2).detach().cpu())


def _gradients_finite(model: nn.Module) -> bool:
    return all(
        bool(torch.isfinite(p.grad).all().item())
        for p in model.parameters() if p.requires_grad and p.grad is not None
    )


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: Criterion,
    device: torch.device,
) -> Dict[str, float]:
    """Mean loss over ``dataloader`` (no accuracy assumption; tensor or dict batch)."""
    model.eval()
    precision = current_precision()
    total_loss, total_n = 0.0, 0
    with torch.no_grad():
        for batch in dataloader:
            inputs, targets = _unpack_batch(batch, device)
            with precision_autocast_context(precision, device):
                out = model(inputs)
            if precision == "fp16" and device.type == "cuda":
                out = _float_tensors(out)
            loss = criterion(out, targets)
            bs = _batch_size(targets)
            total_loss += float(loss.item()) * bs
            total_n += int(bs)
    return {"loss": total_loss / total_n if total_n else 0.0, "num_samples": float(total_n)}


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: Criterion,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip_norm: float = 0.0,
    scheduler: Optional[Any] = None,
    ema_model: Optional[Any] = None,
    max_steps: int = 0,
    precision: Optional[str] = None,
    grad_scaler: Optional[Any] = None,
    telemetry_interval: int = 0,
    accumulation_steps: int = 1,
    runtime_state: Optional[TrainingState] = None,
    max_optimizer_steps: int = 0,
    model_mode: Optional[str] = None,
    exposure_multiplier: int = 1,
    expected_global_microbatch_samples: int = 0,
    precision_diagnostics: Optional[Any] = None,
    readiness_timing: bool = False,
    readiness_warmup_successful_windows: int = 0,
) -> Dict[str, Any]:
    """One epoch of training with the injected criterion (tensor or dict batch).

    Precision is explicit: ``fp32`` runs without autocast/scaler, ``fp16`` uses CUDA
    autocast plus GradScaler(init_scale=512). Outputs are upcast to fp32 before the
    loss/head math. The per-step ``loss.item()`` CPU↔GPU sync is avoided for the
    mean-loss accumulator; telemetry fields expose scaler and finite-loss state.

    **Capability hooks (MCR Phase 1; all default-off ⇒ byte-identical for FL/gate callers):**
    ``grad_clip_norm>0`` clips the trainable grads (stability once the backbone is trained); ``scheduler``
    (if given) steps PER OPTIMIZER STEP (warmup+cosine over total steps); ``ema_model`` (an
    ``swa_utils.AveragedModel``) is updated after each step. None of these fire unless passed, so
    ``train_local`` / the determinism gate are unchanged.  S08's optional
    ``precision_diagnostics`` performs all tensor reductions after unscale and
    before clip/step; it is fail-closed to one-microbatch windows."""
    if accumulation_steps < 1:
        raise ValueError("accumulation_steps must be >= 1")
    if exposure_multiplier < 1:
        raise ValueError("exposure_multiplier must be >= 1")
    if max_steps < 0 or max_optimizer_steps < 0:
        raise ValueError("step limits must be non-negative")
    if expected_global_microbatch_samples < 0:
        raise ValueError("expected_global_microbatch_samples must be non-negative")
    if not isinstance(readiness_timing, bool):
        raise TypeError("readiness_timing must be boolean")
    if (
        isinstance(readiness_warmup_successful_windows, bool)
        or not isinstance(readiness_warmup_successful_windows, int)
        or readiness_warmup_successful_windows < 0
    ):
        raise ValueError("readiness_warmup_successful_windows must be a non-negative integer")
    if max_steps and max_steps % accumulation_steps:
        raise ValueError("max_steps must stop at a complete accumulation-window boundary")
    state = runtime_state if runtime_state is not None else TrainingState()
    state.validate(checkpoint_boundary=True)
    if readiness_timing and any(_state_counters(state).values()):
        raise RuntimeError("readiness timing requires a fresh non-resumed TrainingState")
    if state.discarded_windows:
        raise RuntimeError("training state contains a prior fail-closed discarded window")
    if max_optimizer_steps and max_optimizer_steps < state.optimizer_step:
        raise ValueError("max_optimizer_steps is behind the already executed optimizer step")
    try:
        known_batches = len(dataloader)
    except (TypeError, AttributeError):
        known_batches = None
    if known_batches is not None:
        planned_batches = min(known_batches, max_steps) if max_steps else known_batches
        if planned_batches % accumulation_steps:
            raise ValueError(
                f"loader/limit yields {planned_batches} microbatches, not divisible by "
                f"accumulation_steps={accumulation_steps}"
            )
    if expected_global_microbatch_samples:
        if expected_global_microbatch_samples % exposure_multiplier:
            raise ValueError(
                "expected global microbatch samples are not divisible by exposure_multiplier"
            )
        declared_local_batch = expected_global_microbatch_samples // exposure_multiplier
        loader_batch_size = getattr(dataloader, "batch_size", None)
        if loader_batch_size is not None and int(loader_batch_size) != declared_local_batch:
            raise ValueError(
                "loader batch_size differs from the resolved effective-batch identity: "
                f"expected={declared_local_batch}, actual={loader_batch_size}"
            )

    model.train()
    precision = normalize_precision(precision or current_precision())
    if precision_diagnostics is not None and accumulation_steps != 1:
        raise RuntimeError("S08 precision diagnostics require accumulation_steps == 1")
    if readiness_timing and accumulation_steps != 1:
        raise RuntimeError("S09 readiness timing requires accumulation_steps == 1")
    if readiness_timing and precision_diagnostics is not None:
        raise RuntimeError("S09 readiness timing cannot enable the S08 precision observer")
    if (
        readiness_timing
        and max_optimizer_steps
        and readiness_warmup_successful_windows >= max_optimizer_steps
    ):
        raise ValueError("readiness timing warm-up must be below max_optimizer_steps")
    scaler = grad_scaler if grad_scaler is not None else make_grad_scaler(device, precision)
    scaler_scale_at_start = float(scaler.get_scale()) if readiness_timing else None
    timing = (
        _ReadinessTiming(device, readiness_warmup_successful_windows)
        if readiness_timing
        else None
    )
    use_amp = precision == "fp16" and device.type == "cuda"
    do_clip = bool(grad_clip_norm and grad_clip_norm > 0)
    telemetry_interval = max(0, int(telemetry_interval))
    loss_sum = torch.zeros((), device=device, dtype=torch.float64)  # L4: device accumulate (no per-step sync)
    nonfinite_loss_count = torch.zeros((), device=device, dtype=torch.float64)
    total_n = 0
    step_count = 0
    optimizer_steps_at_start = state.optimizer_step
    scaler_skips = 0
    last_grad_norm = 0.0
    window_invalid = False
    window_nonfinite = False
    window_accounted = False
    fixed_microbatch_samples = (
        int(expected_global_microbatch_samples) if expected_global_microbatch_samples else None
    )
    _missing = object()
    old_record_terms = getattr(criterion, "record_terms", _missing)
    active_capture_context = None
    active_diagnostic_token = None
    active_boundaries = None

    def clear_window(*, discarded: bool = False) -> None:
        nonlocal window_invalid, window_nonfinite, window_accounted
        if discarded and state.pending_samples:
            state.discarded_windows += 1
            state.discarded_samples += state.pending_samples
        state.accumulation_phase = 0
        state.pending_samples = 0
        window_invalid = False
        window_nonfinite = False
        window_accounted = False
        optimizer.zero_grad(set_to_none=True)

    optimizer.zero_grad(set_to_none=True)
    if timing is not None and readiness_warmup_successful_windows == 0:
        timing.start_measurement(state)
    completed_normally = False
    try:
        batch_iterator = iter(dataloader)
        while True:
            if max_optimizer_steps and state.optimizer_step >= max_optimizer_steps:
                break
            if max_steps and step_count >= max_steps:
                break
            wait_started = time.perf_counter() if timing is not None else 0.0
            try:
                batch = next(batch_iterator)
            except StopIteration:
                break
            data_wait_ms = (
                (time.perf_counter() - wait_started) * 1000.0
                if timing is not None
                else 0.0
            )
            next_step = step_count + 1
            record_step = bool(telemetry_interval and next_step % telemetry_interval == 0)
            if old_record_terms is not _missing:
                criterion.record_terms = record_step
            if model_mode is not None:
                batch = project_batch_for_mode(batch, model_mode)
            if timing is not None:
                timing.begin_window(
                    attempted_window=state.attempted_windows + 1,
                    data_wait_ms=data_wait_ms,
                    scaler_before=float(scaler.get_scale()),
                )
                timing.begin_stage("h2d")
            inputs, targets = _unpack_batch(batch, device)
            if timing is not None:
                timing.end_stage("h2d")
            if precision_diagnostics is not None:
                active_diagnostic_token = precision_diagnostics.begin_window(
                    model=model,
                    optimizer=optimizer,
                    state=state,
                    scaler=scaler,
                    scheduler=scheduler,
                    ema_model=ema_model,
                    accumulation_steps=accumulation_steps,
                    precision=precision,
                )
                active_capture_context = precision_diagnostics.capture(
                    model, active_diagnostic_token
                )
                try:
                    active_boundaries = active_capture_context.__enter__()
                except BaseException:
                    active_capture_context = None
                    active_diagnostic_token = None
                    active_boundaries = None
                    raise
            if timing is not None:
                timing.begin_stage("forward")
            if use_amp:
                with precision_autocast_context(precision, device):
                    out = model(inputs)
                out = _float_tensors(out)
            else:
                out = model(inputs)
            if timing is not None:
                timing.end_stage("forward")
                timing.begin_stage("loss")
            loss = criterion(out, targets)
            bs = _batch_size(targets)
            global_bs = int(bs) * int(exposure_multiplier)
            if state.accumulation_phase == 0:
                state.attempted_windows += 1
            state.attempted_microbatches += 1
            state.attempted_samples += global_bs
            state.loss_evaluated_samples += global_bs
            state.accumulation_phase += 1
            state.pending_samples += global_bs
            if fixed_microbatch_samples is None:
                fixed_microbatch_samples = global_bs
            if global_bs != fixed_microbatch_samples:
                raise RuntimeError(
                    "microbatch sample count drift would change the effective update batch: "
                    f"expected={fixed_microbatch_samples}, actual={global_bs}"
                )
            finite_loss = bool(torch.isfinite(loss.detach()).item())
            diagnostic_loss_value = (
                float(loss.detach().item()) if precision_diagnostics is not None else 0.0
            )
            if timing is not None:
                timing.end_stage("loss")
                timing.begin_stage("backward")
            if not finite_loss:
                nonfinite_loss_count += 1
                window_invalid = True
                window_nonfinite = True
                optimizer.zero_grad(set_to_none=True)
            else:
                loss_sum += loss.detach().double() * bs
                total_n += int(bs)
                if not window_invalid:
                    scaled_loss = loss / float(accumulation_steps)
                    if scaler.is_enabled():
                        scaler.scale(scaled_loss).backward()
                    else:
                        scaled_loss.backward()
            if timing is not None:
                timing.end_stage("backward")
            step_count += 1
            if state.accumulation_phase < accumulation_steps:
                continue

            if state.accumulation_phase != accumulation_steps:
                raise RuntimeError("accumulation phase exceeded its fixed window")
            if timing is not None:
                timing.begin_stage("optimizer")
            successful = not window_invalid
            overflow = False
            if window_invalid:
                if precision_diagnostics is not None:
                    precision_diagnostics.prepare_window(
                        active_diagnostic_token,
                        model=model,
                        criterion=criterion,
                        boundaries=active_boundaries,
                        state=state,
                        loss_value=diagnostic_loss_value,
                        loss_finite=finite_loss,
                        parameters_unscaled=False,
                    )
            elif scaler.is_enabled():
                scale_before = float(scaler.get_scale())
                scaler.unscale_(optimizer)
                if precision_diagnostics is not None:
                    precision_diagnostics.prepare_window(
                        active_diagnostic_token,
                        model=model,
                        criterion=criterion,
                        boundaries=active_boundaries,
                        state=state,
                        loss_value=diagnostic_loss_value,
                        loss_finite=finite_loss,
                        parameters_unscaled=True,
                    )
                if do_clip:
                    grad_norm_t = torch.nn.utils.clip_grad_norm_(
                        [p for p in model.parameters() if p.requires_grad], grad_clip_norm)
                    if record_step:
                        last_grad_norm = float(grad_norm_t.detach().cpu())
                elif record_step:
                    last_grad_norm = _grad_norm(model)
                scaler.step(optimizer)
                scaler.update()
                skipped = float(scaler.get_scale()) < scale_before
                scaler_skips += int(skipped)
                successful = not skipped
                if skipped:
                    overflow = True
            else:
                if precision_diagnostics is not None:
                    precision_diagnostics.prepare_window(
                        active_diagnostic_token,
                        model=model,
                        criterion=criterion,
                        boundaries=active_boundaries,
                        state=state,
                        loss_value=diagnostic_loss_value,
                        loss_finite=finite_loss,
                        parameters_unscaled=True,
                    )
                if not _gradients_finite(model):
                    nonfinite_loss_count += 1
                    window_nonfinite = True
                    successful = False
                else:
                    successful = True
                if do_clip:
                    grad_norm_t = torch.nn.utils.clip_grad_norm_(
                        [p for p in model.parameters() if p.requires_grad], grad_clip_norm)
                    if record_step:
                        last_grad_norm = float(grad_norm_t.detach().cpu())
                elif record_step:
                    last_grad_norm = _grad_norm(model)
                if successful:
                    optimizer.step()
            if successful:
                state.optimizer_step += 1
                state.successful_windows += 1
                state.exposure_samples += state.pending_samples
                window_accounted = True
                if scheduler is not None:
                    scheduler.step()
                if ema_model is not None:
                    ema_model.update_parameters(model)
            else:
                state.invalid_windows += 1
                state.invalid_samples += state.pending_samples
                if window_nonfinite:
                    state.nonfinite_windows += 1
                elif overflow:
                    state.overflow_windows += 1
                else:
                    raise RuntimeError("invalid accumulation window has no recorded cause")
                window_accounted = True
            if timing is not None:
                timing.end_stage("optimizer")
            outcome = (
                "accepted"
                if successful
                else "nonfinite_loss"
                if window_nonfinite and not finite_loss
                else "nonfinite_gradients"
                if window_nonfinite
                else "overflow"
            )
            if precision_diagnostics is not None:
                precision_diagnostics.finalize_window(
                    active_diagnostic_token,
                    state=state,
                    scheduler=scheduler,
                    ema_model=ema_model,
                    scaler_after=float(scaler.get_scale()),
                    outcome=outcome,
                )
            clear_window()
            if timing is not None:
                timing.finish_window(
                    outcome=outcome,
                    scaler_after=float(scaler.get_scale()),
                    global_samples=global_bs,
                )
                if (
                    successful
                    and not timing.measurement_started
                    and state.successful_windows
                    == readiness_warmup_successful_windows
                ):
                    timing.start_measurement(state)
            if active_capture_context is not None:
                active_capture_context.__exit__(None, None, None)
                active_capture_context = None
                active_diagnostic_token = None
                active_boundaries = None
        completed_normally = True
    finally:
        if active_capture_context is not None:
            active_capture_context.__exit__(None, None, None)
            active_capture_context = None
            active_diagnostic_token = None
            active_boundaries = None
        had_partial = bool(state.accumulation_phase or state.pending_samples)
        if had_partial:
            clear_window(discarded=not window_accounted)
        if old_record_terms is not _missing:
            criterion.record_terms = old_record_terms
    if had_partial and completed_normally:
        state.validate(checkpoint_boundary=True)
        raise RuntimeError(
            "epoch/limit ended with a partial accumulation window; gradients were cleared and "
            "the epoch must not be marked successful"
        )
    state.validate(checkpoint_boundary=True)
    timing_report = timing.finalize(state) if timing is not None else None
    if timing_report is not None:
        n_averaged = getattr(ema_model, "n_averaged", None)
        timing_report["component_counters"] = {
            "optimizer_step": int(state.optimizer_step),
            "scheduler_last_epoch": (
                None if scheduler is None else int(getattr(scheduler, "last_epoch", -1))
            ),
            "ema_n_averaged": (
                None
                if n_averaged is None
                else int(n_averaged.detach().cpu().item())
                if torch.is_tensor(n_averaged)
                else int(n_averaged)
            ),
            "scaler_scale_at_start": scaler_scale_at_start,
            "scaler_scale_at_end": float(scaler.get_scale()),
            "scaler_skips": int(scaler_skips),
        }
    metrics: Dict[str, Any] = {
        "loss": float(loss_sum.item()) / total_n if total_n else 0.0,
        "num_samples": float(total_n),
        "steps": float(step_count),
        "optimizer_steps": float(state.optimizer_step - optimizer_steps_at_start),
        "optimizer_steps_total": float(state.optimizer_step),
        "exposure_samples": float(state.exposure_samples),
        "attempted_microbatches": float(state.attempted_microbatches),
        "attempted_samples": float(state.attempted_samples),
        "loss_evaluated_samples": float(state.loss_evaluated_samples),
        "attempted_windows": float(state.attempted_windows),
        "successful_windows": float(state.successful_windows),
        "invalid_windows": float(state.invalid_windows),
        "invalid_samples": float(state.invalid_samples),
        "discarded_windows": float(state.discarded_windows),
        "discarded_samples": float(state.discarded_samples),
        "precision": precision,
        "grad_scaler_enabled": float(scaler.is_enabled()),
        "grad_scaler_scale": float(scaler.get_scale()),
        "grad_scaler_skips": float(scaler_skips),
        "nonfinite_loss_steps": float(nonfinite_loss_count.item()),
        "last_grad_norm": float(last_grad_norm),
        "telemetry_interval": float(telemetry_interval),
    }
    if timing_report is not None:
        metrics["readiness_timing"] = timing_report
    return metrics


def train_local(
    model: nn.Module,
    trainloader: DataLoader,
    criterion: Criterion,
    device: torch.device,
    num_epochs: int = 1,
    learning_rate: float = 1e-3,
    weight_decay: float = 0.0,
    valloader: Optional[DataLoader] = None,
    grad_clip_norm: float = 0.0,
    backbone_lr_mult: float = 1.0,
    optimizer_name: str = "adam",
    precision: Optional[str] = None,
    grad_scaler_init_scale: float = 512.0,
    telemetry_interval: int = 0,
) -> Dict[str, float]:
    """Train locally for ``num_epochs``; evaluate on ``valloader`` (last epoch).

    Returns final train/val loss. Adam-family over the trainable params only.

    **FL recipe knobs (MCR Phase-3 / D17; all default-off ⇒ byte-identical to the pre-MCR FL path):**
    With a TRAINED camera backbone (bb02d), the FL client benefits from the same stability levers the
    centralized recipe uses — a separate LR group for the heavy Swin-T backbone (``backbone_lr_mult``,
    e.g. 0.1× the head LR), gradient clipping (``grad_clip_norm``, e.g. 35), and decoupled weight decay
    (``optimizer_name='adamw'``). The 2-group split fires **only when the model has trainable backbone
    params** (it does NOT for a frozen-backbone / dummy model → flat single-group Adam, the byte-identical
    determinism-gate path). Adam-family only (the standing reproduction-fidelity decision — no SGD).
    """
    precision = normalize_precision(precision or current_precision())
    # Fused Adam is a throughput path for fp16 CUDA runs; fp32 reference keeps the plain optimizer.
    _fused = (device.type == "cuda" and precision == "fp16")
    OptCls = torch.optim.AdamW if str(optimizer_name).lower() == "adamw" else torch.optim.Adam
    # Backbone LR group: only when the backbone is TRAINED (bb02d). Frozen/dummy → bb_params empty → the
    # exact pre-MCR flat path (byte-identical for the determinism gate + the old frozen-backbone FL configs).
    bb_params, rest_params = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (bb_params if n.startswith("camera_backbone.") else rest_params).append(p)
    if bb_params:
        optimizer = OptCls(
            [{"params": rest_params, "lr": learning_rate},
             {"params": bb_params, "lr": learning_rate * float(backbone_lr_mult)}],
            lr=learning_rate, weight_decay=weight_decay, fused=_fused,
        )
    else:
        optimizer = OptCls(rest_params, lr=learning_rate, weight_decay=weight_decay, fused=_fused)
    scaler = make_grad_scaler(device, precision, init_scale=float(grad_scaler_init_scale))
    final_train_loss = 0.0
    final_tm: Dict[str, float] = {}
    for _ in range(num_epochs):
        tm = train_one_epoch(model, trainloader, criterion, optimizer, device,
                             grad_clip_norm=grad_clip_norm, precision=precision,
                             grad_scaler=scaler,
                             telemetry_interval=telemetry_interval)
        final_train_loss = tm["loss"]
        final_tm = tm
    final_val_loss = 0.0
    if valloader is not None:
        final_val_loss = evaluate(model, valloader, criterion, device)["loss"]
    return {
        "final_train_loss": float(final_train_loss),
        "final_val_loss": float(final_val_loss),
        "grad_scaler_init_scale": float(grad_scaler_init_scale) if precision == "fp16" else 0.0,
        "grad_scaler_final_scale": float(scaler.get_scale()) if scaler.is_enabled() else 0.0,
        "grad_scaler_skips": float(final_tm.get("grad_scaler_skips", 0.0)),
        "optimizer_steps": float(final_tm.get("optimizer_steps", 0.0)),
    }
