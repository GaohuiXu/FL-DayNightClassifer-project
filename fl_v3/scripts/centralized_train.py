"""S09 resolved centralized trainer and bounded readiness entry point.

This entry point accepts the canonical ``s09.v1``/``s09.v2`` configs.  It constructs
one mode-aware loader, maps exact architecture enums to the reviewed stack,
and advances schedules by successful optimizer updates. ``train_eval`` writes one
complete boundary-safe checkpoint and evaluates it. ``readiness`` is explicitly
non-resumable, bounded, checkpoint-free, and evaluation-free. DDP remains fail
closed.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Callable

sys.path.insert(0, "fl_v3/src")

import torch

from fl_v3.config import load_resolved_config, verify_physical_data_identities
from fl_v3.training.checkpoint import load_checkpoint, save_checkpoint
from fl_v3.training.loop import train_one_epoch
from fl_v3.training.runtime_state import PersistentEpochIterator, TrainingState
from fl_v3.utils.runtime import (
    enforce_determinism,
    make_grad_scaler,
    seed_everything,
    verify_runtime_dependency_identity,
)


def _build_optimizer(model: torch.nn.Module, config) -> torch.optim.Optimizer:
    spec = config.data["optimizer"]
    cls = torch.optim.Adam if spec["name"] == "adam" else torch.optim.AdamW
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise RuntimeError("resolved model has no trainable parameters")
    return cls(params, lr=float(spec["learning_rate"]), weight_decay=float(spec["weight_decay"]))


def _build_ema(model: torch.nn.Module, decay):
    if decay is None:
        return None
    from torch.optim.swa_utils import AveragedModel

    def average(old, new, _count, d=float(decay)):
        return d * old + (1.0 - d) * new

    return AveragedModel(model, avg_fn=average, use_buffers=False)


def _checkpoint_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _event_number(event, *names: str) -> float:
    for name in names:
        value = getattr(event, name, None)
        if value is not None:
            return float(value)
    return 0.0


_EXPECTED_FUSION_PROFILE_RANGES = frozenset({
    "fl_v3::camera.preprocess",
    "fl_v3::camera.backbone",
    "fl_v3::camera.neck",
    "fl_v3::camera.view_transform",
    "fl_v3::lidar.encoder",
    "fl_v3::fusion.fuser",
    "fl_v3::shared.bev_neck",
    "fl_v3::shared.head",
})


def _json_profile_value(value):
    """Convert profiler shape metadata to a finite JSON-native value."""
    if value is None:
        return []
    if isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (list, tuple)):
        return [_json_profile_value(item) for item in value]
    return str(value)


class BoundedOperatorProfiler:
    """One bounded ``torch.profiler`` cycle advanced once per attempted window."""

    def __init__(self, spec: dict, output_dir: Path) -> None:
        self.spec = dict(spec)
        self.output_dir = output_dir
        self.trace_path = output_dir / "trace.json"
        self.summary_path = output_dir / "summary.json"
        self._profiler = None
        self._handler_calls = 0
        self._step_calls = 0

    def _trace_ready(self, profiler) -> None:
        self._handler_calls += 1
        if self._handler_calls != 1:
            raise RuntimeError("operator profiler emitted more than one trace cycle")
        profiler.export_chrome_trace(str(self.trace_path))
        rows = []
        for event in profiler.key_averages(
            group_by_input_shape=bool(self.spec["record_shapes"])
        ):
            rows.append({
                "key": str(event.key),
                "count": int(event.count),
                "input_shapes": _json_profile_value(
                    getattr(event, "input_shapes", None)
                ),
                "self_cpu_time_total_us": _event_number(event, "self_cpu_time_total"),
                "cpu_time_total_us": _event_number(event, "cpu_time_total"),
                "self_device_time_total_us": _event_number(
                    event, "self_device_time_total", "self_cuda_time_total"
                ),
                "device_time_total_us": _event_number(
                    event, "device_time_total", "cuda_time_total"
                ),
                "self_cpu_memory_usage_bytes": int(
                    getattr(event, "self_cpu_memory_usage", 0)
                ),
                "cpu_memory_usage_bytes": int(getattr(event, "cpu_memory_usage", 0)),
                "self_device_memory_usage_bytes": int(
                    getattr(event, "self_device_memory_usage", 0)
                ),
                "device_memory_usage_bytes": int(
                    getattr(event, "device_memory_usage", 0)
                ),
            })
        rows.sort(
            key=lambda row: (
                -row["self_device_time_total_us"],
                -row["self_cpu_time_total_us"],
                row["key"],
            )
        )
        range_rows = [
            row for row in rows if row["key"].startswith("fl_v3::")
        ]
        operator_rows = [
            row for row in rows if not row["key"].startswith("fl_v3::")
        ]
        observed_ranges = frozenset(row["key"] for row in range_rows)
        missing_ranges = sorted(_EXPECTED_FUSION_PROFILE_RANGES - observed_ranges)
        if missing_ranges:
            raise RuntimeError(
                "operator profile omitted required F-U module ranges: "
                f"{missing_ranges}"
            )
        _write_json(self.summary_path, {
            "schema": "s09.operator-profile-summary.v2",
            "units": {"time": "microseconds", "memory": "bytes"},
            "schedule": self.spec,
            "all_row_count": len(rows),
            "range_row_count": len(range_rows),
            "operator_row_count": len(operator_rows),
            "expected_range_keys": sorted(_EXPECTED_FUSION_PROFILE_RANGES),
            "observed_range_keys": sorted(observed_ranges),
            "missing_range_keys": missing_ranges,
            # Module ranges are never subject to the operator top-k cap. This
            # preserves the complete forward decomposition even when many aten
            # kernels have larger self-device time.
            "range_rows": range_rows,
            "operator_rows": operator_rows[: int(self.spec["row_limit"])],
        })

    def __enter__(self):
        self.output_dir.mkdir(parents=True, exist_ok=False)
        activities = [torch.profiler.ProfilerActivity.CPU]
        if not torch.cuda.is_available():
            raise RuntimeError("S09 operator profiling requires CUDA")
        activities.append(torch.profiler.ProfilerActivity.CUDA)
        self._profiler = torch.profiler.profile(
            activities=activities,
            schedule=torch.profiler.schedule(
                wait=int(self.spec["wait_attempted_windows"]),
                warmup=int(self.spec["warmup_attempted_windows"]),
                active=int(self.spec["active_attempted_windows"]),
                repeat=1,
            ),
            on_trace_ready=self._trace_ready,
            record_shapes=bool(self.spec["record_shapes"]),
            profile_memory=bool(self.spec["profile_memory"]),
            with_stack=False,
            with_modules=False,
        )
        self._profiler.__enter__()
        return self

    def step(self) -> None:
        if self._profiler is None:
            raise RuntimeError("operator profiler is not active")
        self._step_calls += 1
        self._profiler.step()

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        if self._profiler is None:
            return False
        suppress = self._profiler.__exit__(exc_type, exc_value, traceback)
        self._profiler = None
        if exc_type is None and self._handler_calls != 1:
            raise RuntimeError(
                "operator profiler did not complete its one configured trace cycle"
            )
        return bool(suppress)

    def report(self) -> dict:
        if self._handler_calls != 1:
            raise RuntimeError("operator profiler report requested before trace completion")
        return {
            "schema": "s09.operator-profile-artifacts.v2",
            "schedule": self.spec,
            "attempted_window_step_calls": self._step_calls,
            "trace": {
                "path": str(self.trace_path),
                "sha256": _checkpoint_sha256(self.trace_path),
                "bytes": self.trace_path.stat().st_size,
            },
            "summary": {
                "path": str(self.summary_path),
                "sha256": _checkpoint_sha256(self.summary_path),
                "bytes": self.summary_path.stat().st_size,
            },
        }


def readiness_evidence_errors(
    report: object,
    *,
    expect_operator_profile: bool,
    verify_profile_artifacts: bool = False,
) -> list[str]:
    """Return fail-closed S09 readiness evidence violations.

    This is shared by the producer and the immutable Slurm wrapper so a process
    exit code alone can never classify a numerically or counter-invalid cell as
    PASS. Dynamic-scaler overflows are allowed; direct nonfinite windows are not.
    """
    errors: list[str] = []
    if not isinstance(report, dict):
        return ["readiness report is not an object"]

    def require(condition: bool, message: str) -> None:
        if not condition:
            errors.append(message)

    try:
        state = TrainingState.from_checkpoint(report["terminal_training_state"])
        training = report["recipe"]["training"]
        metrics = report["training_metrics"]
        timing = metrics["readiness_timing"]
        counters = timing["component_counters"]
        target = report["target_successful_windows"]
        cap = report["max_attempted_windows"]

        require(state.optimizer_step == target, "optimizer_step did not reach target")
        require(state.attempted_windows <= cap, "attempted-window cap was exceeded")
        require(state.nonfinite_windows == 0, "direct nonfinite windows are forbidden")
        require(state.discarded_windows == 0, "discarded readiness windows are forbidden")
        require(
            training["world_size"] == training["accumulation_steps"] == 1,
            "readiness requires world_size=accumulation_steps=1",
        )
        batch = training["micro_batch_size"]
        require(
            state.attempted_microbatches == state.attempted_windows,
            "attempted microbatch/window accounting drifted",
        )
        require(
            state.exposure_samples == state.successful_windows * batch,
            "successful-window exposure accounting drifted",
        )
        require(
            state.attempted_samples == state.attempted_windows * batch,
            "attempted-window sample accounting drifted",
        )
        for key, expected in {
            "optimizer_steps_total": state.optimizer_step,
            "exposure_samples": state.exposure_samples,
            "attempted_samples": state.attempted_samples,
            "attempted_windows": state.attempted_windows,
            "successful_windows": state.successful_windows,
            "grad_scaler_skips": state.overflow_windows,
            "nonfinite_loss_steps": 0,
        }.items():
            require(
                float(metrics[key]) == float(expected),
                f"training_metrics.{key} drifted",
            )
        for key, expected in {
            "optimizer_step": state.optimizer_step,
            "scheduler_last_epoch": state.optimizer_step,
            "scaler_skips": state.overflow_windows,
        }.items():
            require(counters[key] == expected, f"{key} drifted")
        expected_ema = None if training["ema_decay"] is None else state.optimizer_step
        require(counters["ema_n_averaged"] == expected_ema, "ema_n_averaged drifted")
        require(timing["warmup_boundary_reached"] is True, "timing warm-up was not reached")
        require(timing["measured_accepted_windows"] > 0, "no measured accepted windows")

        profile = report.get("operator_profile")
        if not expect_operator_profile:
            require(profile is None, "unexpected operator-profile artifacts")
            return errors
        if not isinstance(profile, dict):
            errors.append("operator profile was required but is missing")
            return errors
        require(
            profile["schema"] == "s09.operator-profile-artifacts.v2",
            "operator-profile artifact schema drifted",
        )
        require(
            profile["attempted_window_step_calls"] == state.attempted_windows,
            "operator profiler was not stepped once per attempted window",
        )
        schedule = profile["schedule"]
        profile_windows = sum(
            schedule[key] for key in (
                "wait_attempted_windows",
                "warmup_attempted_windows",
                "active_attempted_windows",
            )
        )
        require(
            timing["warmup_boundary_attempted_window"] >= profile_windows,
            "operator-profile cycle overlaps throughput windows",
        )

        for artifact_name in ("trace", "summary"):
            artifact = profile[artifact_name]
            path = Path(artifact["path"])
            require(
                isinstance(artifact["sha256"], str)
                and len(artifact["sha256"]) == 64
                and artifact["bytes"] > 0,
                f"operator-profile {artifact_name} identity is invalid",
            )
            if verify_profile_artifacts:
                require(path.is_file(), f"operator-profile {artifact_name} is missing")
                if path.is_file():
                    require(
                        path.stat().st_size == artifact["bytes"],
                        f"operator-profile {artifact_name} size drifted",
                    )
                    require(
                        _checkpoint_sha256(path) == artifact["sha256"],
                        f"operator-profile {artifact_name} hash drifted",
                    )
        if verify_profile_artifacts:
            summary = json.loads(Path(profile["summary"]["path"]).read_text())
            expected_ranges = sorted(_EXPECTED_FUSION_PROFILE_RANGES)
            require(
                summary["schema"] == "s09.operator-profile-summary.v2",
                "operator-profile summary schema drifted",
            )
            require(
                summary["expected_range_keys"] == expected_ranges
                and set(expected_ranges).issubset(summary["observed_range_keys"])
                and summary["missing_range_keys"] == [],
                "operator profile omitted required F-U ranges",
            )
    except (KeyError, TypeError, ValueError, RuntimeError, OSError, json.JSONDecodeError) as exc:
        errors.append(f"readiness evidence is malformed: {exc}")
    return errors


def _percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("cannot summarize an empty timing sample")
    position = (len(ordered) - 1) * float(quantile)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def readiness_performance_gate(
    report: object,
    *,
    expected_train_samples: int,
    accepted_ratio_min: float,
    window_p95_p50_max: float,
    data_wait_share_max: float,
    peak_reserved_bytes_max: int,
    epoch_hours_max: float,
    combined_p50_limit_ms: float,
    combined_p95_limit_ms: float,
) -> dict[str, object]:
    """Compute and fail-close bounded readiness performance acceptance gates."""
    errors: list[str] = []
    metrics: dict[str, float | int] = {}
    if not isinstance(report, dict):
        return {"errors": ["readiness report is not an object"], "metrics": metrics}

    def require(condition: bool, message: str) -> None:
        if not condition:
            errors.append(message)

    try:
        training_metrics = report["training_metrics"]
        timing = training_metrics["readiness_timing"]
        measured = [record for record in timing["records"] if record["measured"]]
        accepted = [record for record in measured if record["outcome"] == "accepted"]
        if not measured or not accepted:
            raise ValueError("performance gate has no measured accepted windows")

        combined_ms = [
            float(record["data_wait_ms"])
            + float(record["durations_ms"]["window"])
            for record in accepted
        ]
        combined_p50_ms = _percentile(combined_ms, 0.50)
        combined_p95_ms = _percentile(combined_ms, 0.95)
        if combined_p50_ms <= 0.0:
            raise ValueError("combined p50 must be positive")
        window_ratio = combined_p95_ms / combined_p50_ms
        data_wait_mean_ms = sum(float(record["data_wait_ms"]) for record in measured) / len(
            measured
        )
        cuda_window_mean_ms = sum(
            float(record["durations_ms"]["window"]) for record in measured
        ) / len(measured)
        integrated_mean_ms = data_wait_mean_ms + cuda_window_mean_ms
        if integrated_mean_ms <= 0.0:
            raise ValueError("integrated mean window must be positive")
        data_wait_share = data_wait_mean_ms / integrated_mean_ms

        delta = timing["measurement_counter_delta"]
        measurement_wall_seconds = float(timing["measurement_wall_seconds"])
        attempted_samples = float(delta["attempted_samples"])
        exposure_samples = float(delta["exposure_samples"])
        if measurement_wall_seconds <= 0.0 or attempted_samples <= 0.0 or exposure_samples <= 0.0:
            raise ValueError("measured wall time and sample counters must be positive")
        attempted_rate = attempted_samples / measurement_wall_seconds
        accepted_rate = exposure_samples / measurement_wall_seconds
        train_samples = int(report["partition"]["unique_train_tokens"])
        dataset_traversal_hours = train_samples / attempted_rate / 3600.0
        accepted_exposure_hours = train_samples / accepted_rate / 3600.0
        peak_reserved_bytes = int(timing["memory"]["peak_reserved_bytes"])
        accepted_ratio = float(timing["measured_accepted_ratio"])
        aggregate_loss = float(training_metrics["loss"])

        metrics = {
            "accepted_combined_p50_ms": combined_p50_ms,
            "accepted_combined_p95_ms": combined_p95_ms,
            "combined_p95_over_p50": window_ratio,
            "measured_accepted_ratio": accepted_ratio,
            "measured_data_wait_share": data_wait_share,
            "peak_reserved_bytes": peak_reserved_bytes,
            "dataset_traversal_hours": dataset_traversal_hours,
            "accepted_exposure_equivalent_hours": accepted_exposure_hours,
            "material_regression_p50_limit_ms": float(combined_p50_limit_ms),
            "material_regression_p95_limit_ms": float(combined_p95_limit_ms),
        }
        require(train_samples == expected_train_samples, "production train-token count drifted")
        require(accepted_ratio >= accepted_ratio_min, "post-warm accepted-window ratio is below threshold")
        require(window_ratio <= window_p95_p50_max, "combined window p95/p50 exceeds threshold")
        require(data_wait_share <= data_wait_share_max, "measured data-wait share exceeds threshold")
        require(peak_reserved_bytes <= peak_reserved_bytes_max, "peak reserved memory exceeds threshold")
        require(
            dataset_traversal_hours <= epoch_hours_max
            and accepted_exposure_hours <= epoch_hours_max,
            "a steady epoch estimate exceeds threshold",
        )
        require(math.isfinite(aggregate_loss), "aggregate loss is not finite")
        require(
            combined_p50_ms <= combined_p50_limit_ms
            and combined_p95_ms <= combined_p95_limit_ms,
            "steady combined latency exceeds the material-regression bound",
        )
    except (KeyError, TypeError, ValueError, ZeroDivisionError) as exc:
        errors.append(f"readiness performance evidence is malformed: {exc}")
    return {"errors": errors, "metrics": metrics}


def _timing_distribution(values: list[float]) -> dict[str, float | int]:
    if not values:
        raise ValueError("loader timing distribution is empty")
    return {
        "n": len(values),
        "mean": sum(values) / len(values),
        "p50": _percentile(values, 0.50),
        "p95": _percentile(values, 0.95),
        "min": min(values),
        "max": max(values),
    }


def _digest_batch_value(digest, value: object) -> None:
    """Content-address one bounded production batch without retaining it."""
    if torch.is_tensor(value):
        tensor = value.detach().cpu().contiguous()
        digest.update(b"tensor\0")
        digest.update(str(tensor.dtype).encode("ascii") + b"\0")
        digest.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode("ascii"))
        digest.update(b"\0")
        digest.update(tensor.numpy().tobytes())
        return
    if isinstance(value, dict):
        digest.update(f"dict:{len(value)}\0".encode("ascii"))
        for key in sorted(value):
            digest.update(str(key).encode("utf-8") + b"\0")
            _digest_batch_value(digest, value[key])
        return
    if isinstance(value, (list, tuple)):
        kind = "list" if isinstance(value, list) else "tuple"
        digest.update(f"{kind}:{len(value)}\0".encode("ascii"))
        for item in value:
            _digest_batch_value(digest, item)
        return
    if value is None or isinstance(value, (str, int, float, bool)):
        digest.update(b"scalar\0")
        digest.update(
            json.dumps(value, ensure_ascii=False, allow_nan=False).encode("utf-8") + b"\0"
        )
        return
    raise TypeError(f"unsupported loader-digest value type {type(value)!r}")


def _batch_size_for_profile(batch: object) -> int:
    if isinstance(batch, dict):
        value = batch.get("batch_size")
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise RuntimeError("production loader batch has invalid batch_size")
        return value
    try:
        size = len(batch)  # type: ignore[arg-type]
    except TypeError as exc:
        raise RuntimeError("cannot determine loader-profile batch size") from exc
    if size < 1:
        raise RuntimeError("loader profile produced an empty batch")
    return int(size)


def _close_loader_dataset(dataset: object) -> None:
    seen: set[int] = set()
    current = dataset
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        close = getattr(current, "close", None)
        if callable(close):
            close()
            return
        current = getattr(current, "dataset", None)


def run_production_loader_profile(
    *,
    task,
    run_config: dict,
    infos: list[dict],
    tokens: list[str],
    profile_spec,
) -> dict:
    """Measure the exact production loader path; never select a worker in-job."""
    profiles = []
    all_digests: list[str] = []
    workers = [int(value) for value in profile_spec["workers"]]
    for num_workers in workers:
        cell_config = dict(run_config)
        cell_config["num-workers"] = num_workers
        loader = task._make_loader(cell_config, infos, tokens, shuffle=True)
        dataset = loader.dataset
        repeats = []
        try:
            for repeat in range(int(profile_spec["repeats"])):
                sampler = getattr(loader, "sampler", None)
                if sampler is None or not hasattr(sampler, "set_epoch"):
                    raise RuntimeError("loader profile requires the production epoch sampler")
                sampler.set_epoch(0)
                iterator_started = time.perf_counter()
                iterator = iter(loader)
                iterator_create_ms = (time.perf_counter() - iterator_started) * 1000.0
                digest = hashlib.sha256()
                audit_waits = []
                for _ in range(int(profile_spec["determinism_batches"])):
                    started = time.perf_counter()
                    batch = next(iterator)
                    audit_waits.append((time.perf_counter() - started) * 1000.0)
                    _digest_batch_value(digest, batch)
                content_sha256 = digest.hexdigest()
                all_digests.append(content_sha256)
                for _ in range(int(profile_spec["warmup_batches"])):
                    next(iterator)
                waits = []
                measured_samples = 0
                wall_started = time.perf_counter()
                for _ in range(int(profile_spec["measured_batches"])):
                    started = time.perf_counter()
                    batch = next(iterator)
                    waits.append((time.perf_counter() - started) * 1000.0)
                    measured_samples += _batch_size_for_profile(batch)
                wall_seconds = time.perf_counter() - wall_started
                if num_workers == 0:
                    cache_state = (
                        "single-process-first" if repeat == 0 else "single-process-repeat"
                    )
                else:
                    cache_state = (
                        "cold-worker-start" if repeat == 0 else "persistent-worker-warm"
                    )
                repeats.append({
                    "repeat": repeat,
                    "cache_state": cache_state,
                    "iterator_create_ms": iterator_create_ms,
                    "determinism_content_sha256": content_sha256,
                    "determinism_wait_ms": _timing_distribution(audit_waits),
                    "measured_batches": int(profile_spec["measured_batches"]),
                    "measured_samples": measured_samples,
                    "measured_wall_seconds": wall_seconds,
                    "samples_per_second": measured_samples / wall_seconds,
                    "batch_wait_ms": _timing_distribution(waits),
                })
                del iterator
        finally:
            del loader
            _close_loader_dataset(dataset)
            gc.collect()
        profiles.append({"num_workers": num_workers, "repeats": repeats})
    return {
        "schema": "s09.production-loader-profile.v1",
        "spec": {
            "workers": workers,
            "repeats": int(profile_spec["repeats"]),
            "determinism_batches": int(profile_spec["determinism_batches"]),
            "warmup_batches": int(profile_spec["warmup_batches"]),
            "measured_batches": int(profile_spec["measured_batches"]),
        },
        "measurement_definition": (
            "batch_wait_ms is host time blocked in next(production DataLoader); "
            "worker cells are observational and do not alter training.num_workers"
        ),
        "training_num_workers": int(run_config["num-workers"]),
        "profiles": profiles,
        "content_sha256_identical": len(set(all_digests)) == 1,
        "content_sha256": all_digests[0],
    }


def run_strict_official_evaluation(
    *,
    config,
    run_config: dict,
    runtime_dependencies: dict,
    task,
    model,
    optimizer,
    scheduler,
    scaler,
    ema,
    checkpoint: Path,
    device: torch.device,
    output_dir: Path,
    decode_fn: Callable | None = None,
    official_eval_fn: Callable | None = None,
    nusc_factory: Callable | None = None,
) -> dict:
    """Evaluate one strict checkpoint through one token-complete official path.

    Loading deliberately reuses the production checkpoint loader (including
    config/data identity checks and rollback).  The raw/EMA choice and timing
    collection are both fields of the hashed resolved config.
    """
    _, checkpoint_identity = load_checkpoint(
        str(checkpoint), model=model, optimizer=optimizer, scheduler=scheduler,
        grad_scaler=scaler, ema=ema, config=config, map_location="cpu",
    )
    if checkpoint_identity != config.sha256:
        raise RuntimeError(
            "checkpoint identity is not the exact resolved-config identity: "
            f"checkpoint={checkpoint_identity}, config={config.sha256}"
        )
    weights = str(run_config["evaluation-checkpoint-weights"])
    if weights == "ema":
        if ema is None or not hasattr(ema, "module"):
            raise RuntimeError("EMA evaluation requested but checkpoint/runtime has no EMA model")
        model.load_state_dict(ema.module.state_dict(), strict=True)
    elif weights != "raw":
        raise RuntimeError(f"unknown strict evaluation checkpoint policy {weights!r}")
    model.to(device)

    from fl_v3.data.nuscenes import paths as nuscenes_paths
    from fl_v3.data.nuscenes.class_map import DETECTION_NAMES
    from fl_v3.eval.detection_eval import (
        VERSION_EVAL_SET, decode_eval_set, run_detection_eval,
    )
    from fl_v3.eval.box_to_global import NUSCENES_MAX_BOXES_PER_SAMPLE

    version = str(run_config["nuscenes-version"])
    split = str(run_config["nuscenes-val-split"])
    infos, _ = task._load_info(run_config, split)
    all_tokens = sorted(str(info["sample_token"]) for info in infos)
    if len(all_tokens) != len(set(all_tokens)):
        raise RuntimeError("resolved validation cache contains duplicate sample tokens")
    loader = task._make_loader(run_config, infos, all_tokens, shuffle=False)
    timing: dict | None = {} if bool(run_config["evaluation-timing"]) else None
    decoder = decode_fn or decode_eval_set
    decodes = decoder(model, loader, device, run_config, timing)
    decoded_tokens = [str(item.sample_token) for item in decodes]
    if len(decoded_tokens) != len(set(decoded_tokens)):
        raise RuntimeError("strict evaluation decoded a sample token more than once")
    if set(decoded_tokens) != set(all_tokens) or len(decoded_tokens) != len(all_tokens):
        missing = sorted(set(all_tokens) - set(decoded_tokens))[:3]
        extra = sorted(set(decoded_tokens) - set(all_tokens))[:3]
        raise RuntimeError(
            "strict evaluation is not token-complete: "
            f"expected={len(all_tokens)}, decoded={len(decoded_tokens)}, "
            f"missing={missing}, extra={extra}"
        )
    over_cap = [
        str(item.sample_token) for item in decodes
        if len(item.boxes) > NUSCENES_MAX_BOXES_PER_SAMPLE
    ]
    if over_cap:
        raise RuntimeError(
            "strict evaluation decode exceeded official per-sample box cap "
            f"{NUSCENES_MAX_BOXES_PER_SAMPLE}: {over_cap[:3]}"
        )

    eval_run_config = dict(run_config)
    eval_run_config.update({
        "checkpoint-sha256": _checkpoint_sha256(checkpoint),
        "checkpoint-weights": weights,
        "runtime-dependencies-sha256": _canonical_sha256(runtime_dependencies),
    })
    factory = nusc_factory or nuscenes_paths.create_nuscenes
    nusc = factory(version, run_config["nuscenes-dataroot"], verbose=False)
    evaluator = official_eval_fn or run_detection_eval
    metrics = evaluator(
        nusc, decodes, VERSION_EVAL_SET[version], version,
        str(output_dir / "official_detection_eval"), DETECTION_NAMES,
        all_eval_tokens=all_tokens, run_config=eval_run_config, verbose=False,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "official_metrics.json").write_text(
        json.dumps(metrics, sort_keys=True) + "\n", encoding="utf-8",
    )
    if timing is not None:
        (output_dir / "evaluation_timing.json").write_text(
            json.dumps(timing, sort_keys=True) + "\n", encoding="utf-8",
        )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="strict s09.v1/s09.v2 JSON")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    config = load_resolved_config(args.config)
    execution = config.data["execution"]
    report_schema = (
        "s09.readiness-report.v2"
        if config.as_dict()["schema_version"] == "s09.v2"
        else "s09.readiness-report.v1"
    )
    execution_mode = config.execution_mode
    readiness = execution_mode == "readiness"
    out_dir = Path(args.out_dir)
    if readiness:
        if args.resume:
            raise RuntimeError("execution.mode='readiness' is non-resumable")
        if out_dir.exists():
            raise RuntimeError(
                "readiness requires a fresh absent output directory: "
                f"{out_dir}"
            )
        out_dir.mkdir(parents=True)

    startup_started = time.perf_counter() if readiness else 0.0
    identity_started = time.perf_counter() if readiness else 0.0
    runtime_dependencies = verify_runtime_dependency_identity(config.to_run_config())
    print(json.dumps({"runtime_dependencies": runtime_dependencies}, sort_keys=True), flush=True)
    verify_physical_data_identities(config)
    startup_phases = {}
    if readiness:
        startup_phases["runtime_and_data_identity_seconds"] = (
            time.perf_counter() - identity_started
        )
    train_spec = config.data["training"]
    declared_world = int(train_spec["world_size"])
    actual_world = int(os.environ.get("WORLD_SIZE", "1"))
    if actual_world != declared_world:
        raise RuntimeError(f"WORLD_SIZE identity drift: config={declared_world}, runtime={actual_world}")
    if actual_world != 1:
        raise RuntimeError("S07-B integration required for DDP wrapping and distributed sampler wiring")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed = int(train_spec["seed"])
    seed_everything(seed)
    enforce_determinism(precision=config.precision, strict=(config.precision == "fp32"))

    from fl_v3.training.tasks import get_task

    run_config = config.to_run_config()
    task = get_task("nuscenes_detection")
    data_started = time.perf_counter() if readiness else 0.0
    train_split = str(run_config["nuscenes-train-split"])
    infos, _ = task._load_info(run_config, train_split)
    part = task._partition(run_config)
    tokens = sorted({token for shard in part["client_tokens"].values() for token in shard})
    if readiness:
        startup_phases["info_and_partition_seconds"] = time.perf_counter() - data_started

    loader_profile = None
    profile_spec = execution["loader_profile"]
    if readiness and profile_spec is not None:
        profile_started = time.perf_counter()
        loader_profile = run_production_loader_profile(
            task=task,
            run_config=run_config,
            infos=infos,
            tokens=tokens,
            profile_spec=profile_spec,
        )
        startup_phases["loader_profile_seconds"] = time.perf_counter() - profile_started
        if not loader_profile["content_sha256_identical"]:
            startup_phases["total_before_training_seconds"] = (
                time.perf_counter() - startup_started
            )
            (out_dir / "resolved_config.json").write_bytes(config.canonical_bytes + b"\n")
            _write_json(out_dir / "runtime_dependencies.json", runtime_dependencies)
            _write_json(out_dir / "readiness.json", {
                "schema": report_schema,
                "status": "FAIL",
                "terminal_reason": (
                    "bounded production-loader content hashes differ across "
                    "declared worker/repeat cells"
                ),
                "resolved_config_sha256": config.sha256,
                "execution_sha256": _canonical_sha256(config.as_dict()["execution"]),
                "data_identities": config.data_identities,
                "runtime_dependencies_sha256": _canonical_sha256(runtime_dependencies),
                "startup_phase_seconds": startup_phases,
                "loader_profile": loader_profile,
                "model_constructed": False,
                "training_started": False,
                "checkpoint_written": False,
                "official_evaluation_executed": False,
            })
            raise RuntimeError(
                "production loader profile content identity differs across cells; "
                "training was not started"
            )

    loader_started = time.perf_counter() if readiness else 0.0
    loader = task._make_loader(run_config, infos, tokens, shuffle=True)
    stream = PersistentEpochIterator(loader)
    if readiness:
        startup_phases["fixed_training_loader_seconds"] = (
            time.perf_counter() - loader_started
        )

    model_started = time.perf_counter() if readiness else 0.0
    model = task.build_model(run_config).to(device)
    criterion = task.build_criterion(run_config)
    optimizer = _build_optimizer(model, config)
    # Constant scheduler is still serialized and advances exactly once per
    # successful update; later schedule enums belong in an approved schema bump.
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _step: 1.0)
    scaler = make_grad_scaler(
        device,
        config.precision,
        init_scale=float(run_config.get("grad-scaler-init-scale", 512.0)),
    )
    ema = _build_ema(model, train_spec["ema_decay"])
    state = TrainingState()
    if readiness:
        startup_phases["model_and_training_components_seconds"] = (
            time.perf_counter() - model_started
        )

    if readiness:
        max_updates = int(train_spec["max_optimizer_steps"])
        max_attempted = int(execution["max_attempted_windows"])
        timing_warmup = int(execution["timing_warmup_successful_windows"])
        operator_profile_spec = execution.get("operator_profile")
        operator_profiler = (
            BoundedOperatorProfiler(
                dict(operator_profile_spec), out_dir / "operator_profile"
            )
            if operator_profile_spec is not None
            else None
        )
        training_started = time.perf_counter()
        train_kwargs = {
            "scheduler": scheduler,
            "ema_model": ema,
            "precision": config.precision,
            "grad_scaler": scaler,
            "accumulation_steps": int(train_spec["accumulation_steps"]),
            "runtime_state": state,
            "max_steps": max_attempted,
            "max_optimizer_steps": max_updates,
            "model_mode": config.model_mode,
            "exposure_multiplier": actual_world,
            "expected_global_microbatch_samples": (
                int(train_spec["micro_batch_size"]) * actual_world
            ),
            "readiness_timing": True,
            "readiness_warmup_successful_windows": timing_warmup,
        }
        profiler_context = operator_profiler if operator_profiler is not None else nullcontext()
        with profiler_context as active_profiler:
            if active_profiler is not None:
                if not hasattr(model, "operator_profile_ranges"):
                    raise RuntimeError(
                        "resolved detector does not expose bounded operator-profile ranges"
                    )
                train_kwargs["attempted_window_callback"] = active_profiler.step
                range_context = model.operator_profile_ranges()
            else:
                range_context = nullcontext()
            with range_context:
                mode_context = (
                    model.serialized_mode(True)
                    if hasattr(model, "serialized_mode")
                    else nullcontext()
                )
                with mode_context:
                    metrics = train_one_epoch(
                        model, stream.batches(0), criterion, optimizer, device,
                        **train_kwargs,
                    )
        operator_profile_report = (
            operator_profiler.report() if operator_profiler is not None else None
        )
        training_wall_seconds = time.perf_counter() - training_started
        startup_phases["total_before_training_seconds"] = (
            training_started - startup_started
        )
        terminal_state = state.checkpoint_dict()
        passed = (
            state.optimizer_step == max_updates
            and state.attempted_windows <= max_attempted
            and state.discarded_windows == 0
        )
        terminal_reason = (
            "successful-update target reached within the attempted-window cap"
            if passed
            else (
                "successful-update target not reached before loader exhaustion or "
                "attempted-window cap"
            )
        )
        report = {
            "schema": report_schema,
            "status": "PASS" if passed else "FAIL",
            "terminal_reason": terminal_reason,
            "resolved_config_sha256": config.sha256,
            "execution_sha256": _canonical_sha256(config.as_dict()["execution"]),
            "data_identities": config.data_identities,
            "runtime_dependencies": runtime_dependencies,
            "runtime_dependencies_sha256": _canonical_sha256(runtime_dependencies),
            "device": (
                {
                    "type": "cuda",
                    "index": int(torch.cuda.current_device()),
                    "name": torch.cuda.get_device_name(device),
                    "compute_capability": list(torch.cuda.get_device_capability(device)),
                    "total_memory_bytes": int(
                        torch.cuda.get_device_properties(device).total_memory
                    ),
                }
                if device.type == "cuda"
                else {"type": "cpu"}
            ),
            "model_mode": config.model_mode,
            "precision": config.precision,
            "sparse_conv_precision": config.sparse_conv_precision,
            "camera_activation_checkpoint": bool(
                run_config["det-camera-activation-checkpoint"]
            ),
            "recipe": {
                "optimizer": config.as_dict()["optimizer"],
                "training": config.as_dict()["training"],
                "scheduler": "constant_lambda_1",
                "gradient_clipping": None,
            },
            "partition": {
                "mode": str(part["mode"]),
                "num_clients": int(part["num_clients"]),
                "unique_train_tokens": len(tokens),
            },
            "fixed_training_num_workers": int(train_spec["num_workers"]),
            "loader_profile": loader_profile,
            "operator_profile": operator_profile_report,
            "startup_phase_seconds": startup_phases,
            "training_wall_seconds": training_wall_seconds,
            "training_metrics": metrics,
            "terminal_training_state": terminal_state,
            "target_successful_windows": max_updates,
            "max_attempted_windows": max_attempted,
            "model_constructed": True,
            "training_started": True,
            "checkpoint_written": False,
            "official_evaluation_executed": False,
            "interpretation_limits": [
                "engineering readiness only; not convergence, mAP/NDS, or model quality",
                "loader profile cells are observational and did not select num_workers in-job",
                (
                    "operator-profile active windows are diagnostic and excluded from "
                    "post-warmup throughput interpretation"
                    if operator_profile_report is not None
                    else "no operator profile was requested"
                ),
                "no checkpoint or official evaluation was produced",
            ],
        }
        evidence_errors = readiness_evidence_errors(
            report,
            expect_operator_profile=operator_profile_report is not None,
            verify_profile_artifacts=True,
        )
        if evidence_errors:
            evidence_reason = "; ".join(evidence_errors)
            terminal_reason = (
                f"readiness evidence validation failed: {evidence_reason}"
                if passed
                else f"{terminal_reason}; evidence validation: {evidence_reason}"
            )
            passed = False
        report["status"] = "PASS" if passed else "FAIL"
        report["terminal_reason"] = terminal_reason
        report["evidence_validation_errors"] = evidence_errors
        (out_dir / "resolved_config.json").write_bytes(config.canonical_bytes + b"\n")
        _write_json(out_dir / "runtime_dependencies.json", runtime_dependencies)
        _write_json(out_dir / "readiness.json", report)
        print(json.dumps({"readiness": report}, sort_keys=True, allow_nan=False), flush=True)
        if not passed:
            raise RuntimeError(terminal_reason)
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = out_dir / "checkpoint.pt"
    if args.resume:
        if not checkpoint.is_file():
            raise RuntimeError(f"--resume requested but checkpoint is missing: {checkpoint}")
        state, _ = load_checkpoint(
            str(checkpoint), model=model, optimizer=optimizer, scheduler=scheduler,
            grad_scaler=scaler, ema=ema, config=config, map_location="cpu",
        )
    elif checkpoint.exists():
        raise RuntimeError(f"output checkpoint already exists; use --resume or a new output: {checkpoint}")

    max_updates = int(train_spec["max_optimizer_steps"])
    max_epochs = int(train_spec["max_epochs"])
    while state.optimizer_step < max_updates and state.epoch < max_epochs:
        mode_context = model.serialized_mode(True) if hasattr(model, "serialized_mode") else None
        if mode_context is None:
            metrics = train_one_epoch(
                model, stream.batches(state.epoch), criterion, optimizer, device,
                scheduler=scheduler, ema_model=ema, precision=config.precision,
                grad_scaler=scaler, accumulation_steps=int(train_spec["accumulation_steps"]),
                runtime_state=state, max_optimizer_steps=max_updates,
                model_mode=config.model_mode, exposure_multiplier=actual_world,
                expected_global_microbatch_samples=(
                    int(train_spec["micro_batch_size"]) * actual_world
                ),
            )
        else:
            with mode_context:
                metrics = train_one_epoch(
                    model, stream.batches(state.epoch), criterion, optimizer, device,
                    scheduler=scheduler, ema_model=ema, precision=config.precision,
                    grad_scaler=scaler, accumulation_steps=int(train_spec["accumulation_steps"]),
                    runtime_state=state, max_optimizer_steps=max_updates,
                    model_mode=config.model_mode, exposure_multiplier=actual_world,
                    expected_global_microbatch_samples=(
                        int(train_spec["micro_batch_size"]) * actual_world
                    ),
                )
        state.epoch += 1
        save_checkpoint(
            str(checkpoint), model=model, optimizer=optimizer, scheduler=scheduler,
            grad_scaler=scaler, ema=ema, state=state, config=config,
            checkpoint_identity=config.sha256,
        )
        print(json.dumps({"epoch": state.epoch, **metrics}, sort_keys=True), flush=True)

    if state.optimizer_step != max_updates:
        raise RuntimeError(
            f"executed-update budget not reached: {state.optimizer_step}/{max_updates}; "
            "nonfinite/overflow/epoch stop remains negative evidence"
        )
    (out_dir / "resolved_config.json").write_bytes(config.canonical_bytes + b"\n")
    (out_dir / "runtime_dependencies.json").write_text(
        json.dumps(runtime_dependencies, sort_keys=True) + "\n", encoding="utf-8"
    )
    (out_dir / "checkpoint.sha256").write_text(_checkpoint_sha256(checkpoint) + "\n", encoding="utf-8")
    metrics = run_strict_official_evaluation(
        config=config, run_config=run_config, runtime_dependencies=runtime_dependencies,
        task=task, model=model, optimizer=optimizer, scheduler=scheduler,
        scaler=scaler, ema=ema, checkpoint=checkpoint, device=device,
        output_dir=out_dir,
    )
    print(json.dumps({"official_evaluation": metrics}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
