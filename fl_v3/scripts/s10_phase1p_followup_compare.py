#!/usr/bin/env python3
"""Seal the exact same-allocation Camera-B16 IP-E3 conservative screen."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, "fl_v3/src")

from fl_v3.training.phase1p_compare import (
    compare_b16_batched_rotation_output_dirs,
    compare_b16_followup_output_dirs,
)


def _canonical_bytes(value) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )


def _write_once(path: Path, value) -> str:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite follow-up summary {path}")
    partial = path.with_name(path.name + ".partial")
    if partial.exists():
        raise FileExistsError(f"stale follow-up-summary partial exists: {partial}")
    payload = _canonical_bytes(value)
    with partial.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(partial, path)
    return hashlib.sha256(payload).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-dir", required=True)
    parser.add_argument("--candidate-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--candidate-kind",
        choices=("conservative", "batched-rotation"),
        default="conservative",
    )
    arguments = parser.parse_args()
    if arguments.candidate_kind == "conservative":
        result = compare_b16_followup_output_dirs(
            arguments.reference_dir,
            arguments.candidate_dir,
        )
        gate = result["conservative_followup_gate"]
        compact = {
            "status": "COMPLETE_B16_FOLLOWUP_COMPARISON",
            "verdict": gate["verdict"],
            "conditional_batched_rotation_implementation_eligible": gate[
                "conditional_batched_rotation_implementation_eligible"
            ],
        }
    else:
        result = compare_b16_batched_rotation_output_dirs(
            arguments.reference_dir,
            arguments.candidate_dir,
        )
        gate = result["batched_rotation_gate"]
        compact = {
            "status": "COMPLETE_B16_BATCHED_ROTATION_COMPARISON",
            "verdict": gate["verdict"],
            "positive_screen": gate["positive_screen"],
        }
    digest = _write_once(Path(arguments.output), result)
    compact["summary_sha256"] = digest
    print(
        json.dumps(
            compact,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
