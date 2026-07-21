"""Pure evidence helpers for the S10 Phase I-P two-GH200 DDP qualifier."""
from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Mapping

import numpy as np
import torch

from fl_v3.training.s10_observation import compare_tensor_tree_tensors


WORLD_SIZE = 2
LOCAL_BATCH = 16
EFFECTIVE_BATCH = 32
WARMUP_WINDOWS = 16
MEASURED_WINDOWS = 256
CONTINUATION_WINDOWS = 8
BLOCK_WINDOWS = 16
SPEEDUP_LOWER_BOUND = 1.60
MAX_CHARGED_RATIO = 1.25
MAX_RESERVED_FRACTION = 0.85


class Phase1PDDPError(RuntimeError):
    """DDP evidence is incomplete or violates the frozen measurement contract."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise Phase1PDDPError(message)


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def batch_sha256(value: Any) -> str:
    """Hash a complete CPU loader batch without changing tensor values."""
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
            part(canonical_bytes([int(size) for size in tensor.shape]))
            try:
                payload = tensor.numpy().tobytes(order="C")
            except TypeError:
                payload = tensor.reshape(-1).view(torch.uint8).numpy().tobytes()
            part(payload)
            return
        if isinstance(item, np.ndarray):
            array = np.ascontiguousarray(item)
            part(str(array.dtype).encode("ascii"))
            part(canonical_bytes([int(size) for size in array.shape]))
            part(array.tobytes(order="C"))
            return
        if isinstance(item, Mapping):
            keys = sorted(item, key=lambda key: (type(key).__name__, repr(key)))
            part(canonical_bytes([f"{type(key).__name__}:{key!r}" for key in keys]))
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
        raise TypeError(f"unsupported loader-batch leaf {type(item)!r} at {path}")

    visit(value, "root")
    return digest.hexdigest()


class BatchDigestLoader:
    """Transparent loader proxy that records exact input identities."""

    def __init__(self, loader: Any, records: list[str], *, limit: int) -> None:
        self.loader = loader
        self.records = records
        self.limit = int(limit)
        self.batch_size = getattr(loader, "batch_size", None)

    def __len__(self) -> int:
        return len(self.loader)

    def __iter__(self):
        for batch in self.loader:
            if len(self.records) < self.limit:
                self.records.append(batch_sha256(batch))
            yield batch


def capture_state(value: Any) -> tuple[dict[str, torch.Tensor], str]:
    """Capture tensor leaves on CPU plus a non-value structural identity."""
    tensors: dict[str, torch.Tensor] = {}

    def visit(item: Any, path: str) -> Any:
        if torch.is_tensor(item):
            tensors[path] = item.detach().cpu().clone()
            return {
                "tensor": True,
                "dtype": str(item.dtype),
                "shape": [int(size) for size in item.shape],
                "layout": str(item.layout),
            }
        if isinstance(item, Mapping):
            children = []
            for key in sorted(
                item, key=lambda child: (type(child).__name__, repr(child))
            ):
                label = f"{type(key).__name__}:{key!r}"
                children.append([label, visit(item[key], f"{path}.{label}")])
            return {"mapping": type(item).__name__, "children": children}
        if isinstance(item, (list, tuple)):
            return {
                "sequence": type(item).__name__,
                "children": [
                    visit(child, f"{path}[{index}]")
                    for index, child in enumerate(item)
                ],
            }
        if item is None or isinstance(item, (str, bool, int, float)):
            return {"scalar_type": type(item).__name__, "value": item}
        raise TypeError(f"unsupported checkpoint state leaf {type(item)!r} at {path}")

    structure = visit(value, "root")
    return tensors, canonical_sha256(structure)


def capture_sha256(capture: tuple[dict[str, torch.Tensor], str]) -> str:
    tensors, structure = capture
    digest = hashlib.sha256()
    digest.update(structure.encode("ascii"))
    for name in sorted(tensors):
        tensor = tensors[name].contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(canonical_bytes([int(size) for size in tensor.shape]))
        try:
            payload = tensor.numpy().tobytes(order="C")
        except TypeError:
            payload = tensor.reshape(-1).view(torch.uint8).numpy().tobytes()
        digest.update(hashlib.sha256(payload).digest())
    return digest.hexdigest()


def compare_state_captures(
    reference: tuple[dict[str, torch.Tensor], str],
    candidate: tuple[dict[str, torch.Tensor], str],
    *,
    rtol: float,
    atol: float,
) -> dict[str, Any]:
    """Return the standard Phase-I-P per-tensor comparison record."""
    reference_tensors, reference_structure = reference
    candidate_tensors, candidate_structure = candidate
    numerical = compare_tensor_tree_tensors(reference_tensors, candidate_tensors)
    floating_failures = []
    discrete_failures = []
    floating_names = []
    discrete_names = []
    for name in sorted(set(reference_tensors) & set(candidate_tensors)):
        left = reference_tensors[name]
        right = candidate_tensors[name]
        if left.shape != right.shape or left.dtype != right.dtype:
            continue
        if left.is_floating_point() or left.is_complex():
            floating_names.append(name)
            if not torch.allclose(left, right, rtol=rtol, atol=atol, equal_nan=False):
                floating_failures.append(name)
        else:
            discrete_names.append(name)
            if not torch.equal(left, right):
                discrete_failures.append(name)
    gate = bool(
        reference_structure == candidate_structure
        and numerical["name_set_equal"]
        and not numerical["shape_mismatch_tensors"]
        and not numerical["dtype_mismatch_tensors"]
        and numerical["global"]["all_finite"]
        and not floating_failures
        and not discrete_failures
    )
    return {
        "reference_structure_sha256": reference_structure,
        "candidate_structure_sha256": candidate_structure,
        "structure_equal": reference_structure == candidate_structure,
        "allclose_rtol": float(rtol),
        "allclose_atol": float(atol),
        "floating_allclose_failures": floating_failures,
        "discrete_exact_failures": discrete_failures,
        "floating_tensor_names": floating_names,
        "discrete_tensor_names": discrete_names,
        "numerical": numerical,
        "gate_pass": gate,
    }


def capture_stack(model: Any, optimizer: Any, scheduler: Any, scaler: Any) -> dict[str, Any]:
    return {
        "model": capture_state(model.state_dict()),
        "optimizer": capture_state(optimizer.state_dict()),
        "scheduler": capture_state(scheduler.state_dict()),
        "scaler": capture_state(scaler.state_dict()),
    }


def compare_stacks(
    reference: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    rtol: float,
    atol: float,
) -> dict[str, Any]:
    expected = {"model", "optimizer", "scheduler", "scaler"}
    _require(set(reference) == set(candidate) == expected, "stack state kinds drift")
    return {
        name: compare_state_captures(
            reference[name], candidate[name], rtol=rtol, atol=atol
        )
        for name in sorted(expected)
    }


def rng_state() -> dict[str, Any]:
    device = torch.cuda.current_device() if torch.cuda.is_available() else None
    return {
        "python": __import__("random").getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda_local": (
            torch.cuda.get_rng_state(device) if device is not None else None
        ),
    }


def restore_rng_state(state: Mapping[str, Any]) -> None:
    expected = {"python", "numpy", "torch", "cuda_local"}
    _require(set(state) == expected, "DDP RNG sidecar fields drift")
    __import__("random").setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda_local"] is not None:
        _require(torch.cuda.is_available(), "DDP RNG sidecar requires CUDA")
        torch.cuda.set_rng_state(state["cuda_local"], torch.cuda.current_device())


def rng_state_sha256(state: Mapping[str, Any]) -> str:
    """Hash one rank's RNG state by value, independent of pickle internals."""
    _require(
        set(state) == {"python", "numpy", "torch", "cuda_local"},
        "DDP RNG hash fields drift",
    )
    digest = hashlib.sha256()
    digest.update(repr(state["python"]).encode("utf-8"))
    numpy_state = state["numpy"]
    digest.update(str(numpy_state[0]).encode("ascii"))
    digest.update(np.asarray(numpy_state[1]).tobytes(order="C"))
    digest.update(repr(tuple(numpy_state[2:])).encode("utf-8"))
    digest.update(state["torch"].detach().cpu().numpy().tobytes(order="C"))
    cuda_state = state["cuda_local"]
    if cuda_state is not None:
        digest.update(cuda_state.detach().cpu().numpy().tobytes(order="C"))
    return digest.hexdigest()


def rng_sha256() -> str:
    return rng_state_sha256(rng_state())


def aggregate_rank_measurements(records: list[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate two lockstep rank reports using the slower-rank wall clock."""
    _require(len(records) == WORLD_SIZE, "DDP measurement requires exactly two ranks")
    ordered = sorted(records, key=lambda item: int(item["rank"]))
    _require(
        [int(item["rank"]) for item in ordered] == [0, 1],
        "DDP rank measurement identities drift",
    )
    timings = [item["metrics"]["readiness_timing"] for item in ordered]
    expected_samples = MEASURED_WINDOWS * EFFECTIVE_BATCH
    walls = []
    warmup_walls = []
    startup_walls = []
    rank_health = []
    for record, timing in zip(ordered, timings):
        delta = timing["measurement_counter_delta"]
        memory = timing["memory"]
        checks = {
            "measured_windows": (
                int(timing["measured_accepted_windows"]) == MEASURED_WINDOWS
                and int(timing["measured_attempted_windows"]) == MEASURED_WINDOWS
            ),
            "global_exposure": int(delta["exposure_samples"]) == expected_samples,
            "zero_invalid": int(delta["invalid_windows"]) == 0,
            "zero_discarded": int(delta["discarded_windows"]) == 0,
            "zero_overflow": int(delta["overflow_windows"]) == 0,
            "memory_under_limit": float(memory["peak_reserved_fraction"])
            <= MAX_RESERVED_FRACTION,
            "no_reserved_growth": not bool(
                memory["monotonic_reserved_growth_over_64mib"]
            ),
            "no_steady_recompile": not bool(
                record["compile_evidence"]["unexpected_steady_state_recompile"]
            ),
        }
        wall = float(timing["measurement_wall_seconds"])
        _require(math.isfinite(wall) and wall > 0.0, "DDP rank wall time is invalid")
        walls.append(wall)
        training_wall = float(record["training_wall_seconds_including_warmup"])
        startup_wall = float(record["startup_seconds"]["before_training_total"])
        _require(
            math.isfinite(training_wall) and training_wall >= wall,
            "DDP rank training/warm-up wall time is invalid",
        )
        _require(
            math.isfinite(startup_wall) and startup_wall >= 0.0,
            "DDP rank startup wall time is invalid",
        )
        warmup_walls.append(training_wall - wall)
        startup_walls.append(startup_wall)
        rank_health.append(
            {"rank": int(record["rank"]), "checks": checks, "gate_pass": all(checks.values())}
        )
    blocks = []
    for index in range(MEASURED_WINDOWS // BLOCK_WINDOWS):
        rank_blocks = [
            timing["throughput_blocks"]["records"][index] for timing in timings
        ]
        for block in rank_blocks:
            _require(
                int(block["accepted_windows"]) == BLOCK_WINDOWS
                and int(block["exposure_samples"]) == BLOCK_WINDOWS * EFFECTIVE_BATCH,
                f"DDP throughput block {index} exposure drift",
            )
        blocks.append(
            {
                "accepted_windows": BLOCK_WINDOWS,
                "exposure_samples": BLOCK_WINDOWS * EFFECTIVE_BATCH,
                "wall_seconds": max(float(block["wall_seconds"]) for block in rank_blocks),
            }
        )
    wall = max(walls)
    return {
        "measurement_wall_seconds": wall,
        "startup_seconds": {"before_training_total": max(startup_walls)},
        "compile_evidence": {
            "warmup_including_compile_seconds": max(warmup_walls),
        },
        "exposure_samples": expected_samples,
        "exposure_samples_per_second": expected_samples / wall,
        "throughput_blocks": blocks,
        "rank_health": rank_health,
        "gate_pass": all(item["gate_pass"] for item in rank_health),
        "rank_memory": [
            {"rank": int(record["rank"]), **timing["memory"]}
            for record, timing in zip(ordered, timings)
        ],
        "rank_devices": [
            {"rank": int(record["rank"]), **record["device"]}
            for record in ordered
        ],
    }


__all__ = [
    "BatchDigestLoader",
    "BLOCK_WINDOWS",
    "CONTINUATION_WINDOWS",
    "EFFECTIVE_BATCH",
    "LOCAL_BATCH",
    "MAX_CHARGED_RATIO",
    "MAX_RESERVED_FRACTION",
    "MEASURED_WINDOWS",
    "Phase1PDDPError",
    "SPEEDUP_LOWER_BOUND",
    "WARMUP_WINDOWS",
    "WORLD_SIZE",
    "aggregate_rank_measurements",
    "batch_sha256",
    "canonical_bytes",
    "canonical_sha256",
    "capture_sha256",
    "capture_stack",
    "capture_state",
    "compare_stacks",
    "compare_state_captures",
    "restore_rng_state",
    "rng_sha256",
    "rng_state",
    "rng_state_sha256",
]
