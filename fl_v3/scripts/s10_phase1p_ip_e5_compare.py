#!/usr/bin/env python3
"""Seal the exact same-allocation IP-E5 one-GPU versus two-GPU result."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, "fl_v3/src")

from fl_v3.training.phase1p_compare import compare_ip_e5_ddp_output_dirs


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
        raise FileExistsError(f"refusing to overwrite IP-E5 summary {path}")
    partial = path.with_name(path.name + ".partial")
    if partial.exists():
        raise FileExistsError(f"stale IP-E5-summary partial exists: {partial}")
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
    arguments = parser.parse_args()
    result = compare_ip_e5_ddp_output_dirs(
        arguments.reference_dir, arguments.candidate_dir
    )
    digest = _write_once(Path(arguments.output), result)
    print(
        json.dumps(
            {
                "status": "COMPLETE_IP_E5_DDP_COMPARISON",
                "verdict": result["verdict"],
                "qualification_gate_pass": result["qualification_gate"][
                    "gate_pass"
                ],
                "production_promotion_authorized": result[
                    "production_promotion_authorized"
                ],
                "summary_sha256": digest,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
