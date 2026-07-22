#!/usr/bin/env python3
"""Seal one exact same-allocation IP-E2 sustained comparison."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, "fl_v3/src")

from fl_v3.training.phase1p_compare import (
    compare_lidar_e2_output_dirs,
    compare_output_dirs,
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
        raise FileExistsError(f"refusing to overwrite paired summary {path}")
    partial = path.with_name(path.name + ".partial")
    if partial.exists():
        raise FileExistsError(f"stale paired-summary partial exists: {partial}")
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
    parser.add_argument("--lidar-e2", action="store_true")
    arguments = parser.parse_args()
    compare = compare_lidar_e2_output_dirs if arguments.lidar_e2 else compare_output_dirs
    result = compare(arguments.reference_dir, arguments.candidate_dir)
    digest = _write_once(Path(arguments.output), result)
    print(
        json.dumps(
            {
                "status": "COMPLETE_PAIRED_COMPARISON",
                "speed_verdict": result["throughput"]["speed_verdict"],
                "candidate_screen_gate_pass": result["candidate_screen_gate_pass"],
                "summary_sha256": digest,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
