#!/usr/bin/env python3
"""Read-only reassessment of an immutable Phase I-P profiler result."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any


_GATE_PATH = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "fl_v3"
    / "training"
    / "phase1_checkpoint_gate.py"
)
_SPEC = importlib.util.spec_from_file_location("s10_phase1_checkpoint_gate", _GATE_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError(f"cannot load checkpoint gate module at {_GATE_PATH}")
_GATE_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_GATE_MODULE)
evaluate_calibrated_continuation_gate = (
    _GATE_MODULE.evaluate_calibrated_continuation_gate
)


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object at {path}")
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", required=True)
    parser.add_argument("--output")
    args = parser.parse_args()
    result_path = Path(args.result).resolve()
    result = _read_object(result_path)
    if "effective_runtime_config_sha256" in result:
        config_path = result_path.parent / "effective_runtime_config.json"
        expected_config_sha256 = result["effective_runtime_config_sha256"]
    else:
        config_path = result_path.parent / "resolved_config.json"
        expected_config_sha256 = result["resolved_config_sha256"]
    config = _read_object(config_path)
    if _sha256_file(config_path) != expected_config_sha256:
        raise RuntimeError("resolved config hash does not match profiler result")
    precision = str(config["precision"]["global_autocast"])
    tolerances = {
        "fp32": {"relative_l2": 1e-4, "max_absolute": 1e-6},
        "fp16": {"relative_l2": 2e-3, "max_absolute": 2e-4},
    }[precision]
    checkpoint = result["checkpoint"]
    fresh_process = {
        "continuation": checkpoint["continuation"],
        "input_stream": checkpoint["input_stream"],
        "training_state_equal": checkpoint["training_state_equal"],
        "rng_state_equal": checkpoint["rng_state_equal"],
    }
    gate = evaluate_calibrated_continuation_gate(
        restored_boundary=checkpoint["restored_boundary"],
        same_process=checkpoint["same_process_replay"],
        fresh_process=fresh_process,
        relative_l2_tolerance=tolerances["relative_l2"],
        max_absolute_tolerance=tolerances["max_absolute"],
    )
    reassessment = {
        "schema": "s10.phase1p.checkpoint-reassessment.v1",
        "immutable_result": {
            "path": str(result_path),
            "sha256": _sha256_file(result_path),
            "original_status": result["status"],
        },
        "resolved_config_sha256": expected_config_sha256,
        "precision": precision,
        "continuation_gate": gate,
    }
    reassessment["reassessment_sha256"] = hashlib.sha256(
        _canonical_bytes(reassessment)
    ).hexdigest()
    payload = _canonical_bytes(reassessment) + b"\n"
    if args.output:
        output_path = Path(args.output).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("xb") as stream:
            stream.write(payload)
    else:
        print(payload.decode("utf-8"), end="")


if __name__ == "__main__":
    main()
