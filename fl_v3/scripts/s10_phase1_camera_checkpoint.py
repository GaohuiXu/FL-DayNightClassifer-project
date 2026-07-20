#!/usr/bin/env python3
"""Validate and atomically accept the one owner-approved Swin acquisition."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, "fl_v3/src")

import torch

from fl_v3.config import load_resolved_config
from fl_v3.models.phase1_camera import Phase1CameraDetector
from fl_v3.models.phase1_swin import (
    seal_validated_swin_checkpoint,
    sha256_file,
    validate_and_load_original_swin,
)


REDIRECT_HOST = "release-assets.githubusercontent.com"
REDIRECT_PATH = (
    "/github-production-release-asset/357198522/"
    "fd006b80-9bd3-11eb-8445-769d89efab4e"
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--result", required=True)
    args = parser.parse_args()

    config = load_resolved_config(args.config)
    raw = config.as_dict()
    if raw["contract"]["branch"] != "camera":
        raise ValueError("checkpoint acceptance requires the Phase-I Camera config")
    initialization = raw["initialization"]
    if initialization["status"] != "pending_acquisition":
        raise ValueError("checkpoint acceptance requires pending_acquisition status")
    torch.manual_seed(int(raw["contract"]["seed"]))
    model = Phase1CameraDetector(pool_backend="fallback")
    final_path = Path(initialization["final_path"]).resolve()
    quarantine_path = Path(initialization["quarantine_path"]).resolve()
    mapping_report_path = Path(initialization["mapping_report_path"]).resolve()
    if final_path.is_file():
        if quarantine_path.exists() or not mapping_report_path.is_file():
            raise RuntimeError("existing accepted Swin artifacts have an invalid lifecycle")
        observed = validate_and_load_original_swin(model, final_path)
        report = json.loads(mapping_report_path.read_text(encoding="utf-8"))
        if (
            report.get("physical_path") != str(final_path)
            or report.get("physical_sha256") != observed["physical_sha256"]
            or report.get("initialization_state_sha256")
            != observed["initialization_state_sha256"]
            or report.get("acquisition", {}).get("count") != 1
            or report.get("acquisition", {}).get("redirect_host") != REDIRECT_HOST
            or report.get("acquisition", {}).get("redirect_path") != REDIRECT_PATH
        ):
            raise RuntimeError("existing Swin mapping report differs from revalidation")
        accepted = {
            **observed,
            "mapping_report_path": str(mapping_report_path),
            "mapping_report_sha256": sha256_file(mapping_report_path),
        }
    else:
        if mapping_report_path.exists():
            raise RuntimeError("Swin mapping report exists without accepted checkpoint")
        accepted = seal_validated_swin_checkpoint(
            model,
            quarantine_path=quarantine_path,
            final_path=final_path,
            mapping_report_path=mapping_report_path,
            acquisition_redirect_host=REDIRECT_HOST,
            acquisition_redirect_path=REDIRECT_PATH,
        )
    result_path = Path(args.result).resolve()
    result_path.parent.mkdir(parents=True, exist_ok=True)
    if result_path.exists():
        raise FileExistsError(f"checkpoint result path already exists: {result_path}")
    compact = {
        "schema": "s10.phase1.swin-acceptance-result.v1",
        "physical_path": accepted["physical_path"],
        "physical_bytes": accepted["physical_bytes"],
        "physical_sha256": accepted["physical_sha256"],
        "mapping_report_path": accepted["mapping_report_path"],
        "mapping_report_sha256": accepted["mapping_report_sha256"],
        "initialization_state_sha256": accepted["initialization_state_sha256"],
        "mapped_tensor_count": accepted["mapped_tensor_count"],
        "loaded_missing_keys": accepted["loaded_missing_keys"],
        "loaded_unexpected_keys": accepted["loaded_unexpected_keys"],
    }
    result_path.write_text(
        json.dumps(compact, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    result_path.chmod(0o400)
    compact["result_path"] = str(result_path)
    compact["result_sha256"] = sha256_file(result_path)
    print(json.dumps(compact, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
