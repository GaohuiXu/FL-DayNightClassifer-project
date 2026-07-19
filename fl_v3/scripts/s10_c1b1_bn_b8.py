#!/usr/bin/env python3
"""O-141 BN1d physical-B8 operational completion.

This is one practical candidate run, not an isolated batch-size causal claim.
It keeps the frozen D_low exposure and D_select evaluator while jointly changing
the BN1d physical microbatch to eight and the initial loss scale to eight.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any

sys.path.insert(0, "fl_v3/src")
sys.path.insert(0, "fl_v3/scripts")

from fl_v3.config import resolve_config, verify_physical_data_identities
from fl_v3.data.nuscenes.s10_binding import load_frozen_split_role
from fl_v3.utils.runtime import verify_runtime_dependency_identity
from s10_c1b1_capability import (
    _artifact_manifest,
    _manifest_select_logs,
    _paired_log_evidence,
    _runtime_identity,
)
from s10_stop_c0_health import (
    EXPECTED_SPLIT_SHA256,
    _canonical_sha256,
    _run_cell,
    _sha256_file,
    _source_identities,
    _write_json,
)


SCHEMA = "fl_v3.s10.c1b1_bn_b8.v1"
CELL = "C1-B1-BN1D-B8-DLOW"
GN_REFERENCE_CELL = "C1-B1-CUR-A1-GN-DLOW"
BN_B4_REFERENCE_CELL = "C1-B1-CUR-A1-BN1D-DLOW"
PHYSICAL_BATCH = 8
EVAL_PHYSICAL_BATCH = 4
HORIZON = 769
EXPECTED_D_LOW_SAMPLES = 6155
EXPECTED_D_SELECT_SAMPLES = 4626
EXPECTED_D_SELECT_LOGS = 8
EXPECTED_W0_SHA256 = "87be0d2416b3ed06e2d1e9214e11ad3ac25bc275993b0865d918af6f332829d1"
EXPECTED_TOKEN_ORDER_SHA256 = "947dc9bc8441267587df6b0b88d16efc84ab3c7ff0a1a152481ac2697f0a2eb1"
EXPECTED_REMAINDER_SHA256 = "7495cdbec472ce49f29e8f19abe08fc9431a258b437a5db05ab89fae0db60443"
BOUNDARIES = (1, 4, 16, 64, 256, 512, 769)
DIAGNOSTICS = (1, 4, 16, 64, 256, 512, 769)
REFERENCE_FILES = {
    GN_REFERENCE_CELL: {
        "cell_summary.json": "81e31258dd783f47e8775a1b1327dbac66b0cf9b005fcf6e5f4249c98d61ea85",
        "D_select_results.json": "7fc24fd757d9302096c27208c58469fdd335f22fe363a70bb32ab76875f1e549",
    },
    BN_B4_REFERENCE_CELL: {
        "cell_summary.json": "5abc990577ec99eb04f2b9fc063ecba648d342891ce3e50159b4adff62537517",
        "D_select_results.json": "124eddeee78d5fd3495a3f1cff820a5ab82f5aebaef4cfd4ab9996a926966268",
    },
}


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--split-manifest", required=True)
    parser.add_argument("--reference-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--source-sha", required=True)
    parser.add_argument("--source-tree", required=True)
    return parser.parse_args()


def _assert_config(config) -> None:
    if str(config.data["schema_version"]) != "s10.v1":
        raise RuntimeError("BN-B8 requires the s10.v1 schema")
    model = config.data["model"]
    expected_model = {
        "mode": "fusion",
        "camera_arch": "swin_t_stride8",
        "camera_pretrained": True,
        "camera_activation_checkpoint": False,
        "lidar_arch": "second_075",
        "second_normalization": "batch_norm_1d",
        "fusion_arch": "conv_fuser_256",
        "head_arch": "centerhead_multitask",
    }
    if {key: model[key] for key in expected_model} != expected_model:
        raise RuntimeError("BN-B8 graph/initialization contract drift")
    training = config.data["training"]
    if (
        int(training["max_optimizer_steps"]) != HORIZON
        or int(training["micro_batch_size"]) != PHYSICAL_BATCH
        or int(training["world_size"]) != 1
        or int(training["accumulation_steps"]) != 1
        or int(training["effective_global_batch"]) != PHYSICAL_BATCH
        or int(training["seed"]) != 0
        or int(training["max_epochs"]) != 1
        or int(training["num_workers"]) != 8
        or float(training["grad_scaler_init_scale"]) != 8.0
        or training["ema_decay"] is not None
        or str(training["sampling"]) != "uniform"
    ):
        raise RuntimeError("BN-B8 training/scale/exposure contract drift")
    optimizer = config.data["optimizer"]
    if (
        str(optimizer["name"]) != "adamw"
        or float(optimizer["learning_rate"]) != 1.0e-4
        or float(optimizer["weight_decay"]) != 0.01
    ):
        raise RuntimeError("BN-B8 optimizer/constant-scheduler contract drift")
    if config.precision != "fp16" or config.sparse_conv_precision != "fp32":
        raise RuntimeError("BN-B8 precision partition drift")


def _spec() -> dict[str, Any]:
    return {
        "id": CELL,
        "mode": "fusion",
        "camera_pretrained": True,
        "normalization": "batch_norm_1d",
        "attempted_windows": HORIZON,
        "physical_microbatch": PHYSICAL_BATCH,
        "eval_physical_microbatch": EVAL_PHYSICAL_BATCH,
        "boundaries": BOUNDARIES,
        "diagnostics": DIAGNOSTICS,
        "evaluate": True,
        "operator_profile": False,
        "fail_fast_numerical": True,
        "grad_scaler_init_scale": 8.0,
        "report_schema": SCHEMA,
        "expected_initial_parameter_sha256": EXPECTED_W0_SHA256,
        "interpretation_limits": [
            "single-seed D_low one-epoch internal evidence; not the S10 full claim",
            "BN1d physical-B8 plus scale-8 is a joint operational candidate, not an isolated batch-size causal contrast",
            "B8 has half as many optimizer updates as B4 at the same 6,152-sample exposure",
            "D_select is train-only proxy evidence; D_audit and official val remain sealed",
            "no operator profiler, recipe sweep or automatic normalization promotion",
        ],
    }


def _load_references(root: Path) -> tuple[dict[str, dict[str, Any]], dict[str, Path]]:
    reports: dict[str, dict[str, Any]] = {}
    results: dict[str, Path] = {}
    for cell, files in REFERENCE_FILES.items():
        cell_root = root / cell
        for name, expected in files.items():
            path = cell_root / name
            if not path.is_file() or _sha256_file(path) != expected:
                raise RuntimeError(f"sealed C1-B1 reference identity drift: {cell}/{name}")
        report = json.loads((cell_root / "cell_summary.json").read_text(encoding="utf-8"))
        if (
            report["cell"]["id"] != cell
            or report["initial_parameter_sha256"] != EXPECTED_W0_SHA256
            or report["training_token_evidence"]["consumed_sample_tokens_ordered_sha256"]
            != EXPECTED_TOKEN_ORDER_SHA256
            or report["training_token_evidence"]["drop_last_remainder_tokens_sorted_sha256"]
            != EXPECTED_REMAINDER_SHA256
            or int(report["D_select_evaluation"]["n_samples"]) != EXPECTED_D_SELECT_SAMPLES
        ):
            raise RuntimeError(f"sealed C1-B1 reference semantic drift: {cell}")
        reports[cell] = report
        results[cell] = cell_root / "D_select_results.json"
    return reports, results


def _point_metrics(report: dict[str, Any]) -> dict[str, float]:
    evaluation = report["D_select_evaluation"]
    result = {
        "NDS": float(evaluation["internal_subset_NDS"]),
        "mAP": float(evaluation["internal_subset_mAP"]),
    }
    if not all(math.isfinite(value) for value in result.values()):
        raise RuntimeError("BN-B8 produced nonfinite capability metrics")
    return result


def _train_runtime(report: dict[str, Any]) -> dict[str, float]:
    wall = sum(float(chunk["wall_seconds"]) for chunk in report["chunks"])
    attempted_samples = int(report["training_token_evidence"]["consumed_sample_count"])
    return {
        "wall_seconds": wall,
        "attempted_samples_per_second": attempted_samples / wall,
    }


def _execute(args, output: Path) -> None:
    if len(args.source_sha) != 40 or len(args.source_tree) != 40:
        raise RuntimeError("BN-B8 requires exact source commit and tree identities")
    split_manifest = Path(args.split_manifest).resolve()
    if _sha256_file(split_manifest) != EXPECTED_SPLIT_SHA256:
        raise RuntimeError("BN-B8 split manifest identity drift")
    base = json.loads(Path(args.config).read_text(encoding="utf-8"))
    if base.get("scheduler") is not None:
        raise RuntimeError("BN-B8 requires the unchanged constant scheduler")
    config = resolve_config(base)
    _assert_config(config)
    verify_physical_data_identities(config)
    runtime_dependencies = verify_runtime_dependency_identity(config.to_run_config())
    expected_sources = _source_identities(config)
    low = load_frozen_split_role(
        split_manifest, expected_manifest_sha256=EXPECTED_SPLIT_SHA256,
        role="D_low", expected_source_identities=expected_sources,
    )
    select = load_frozen_split_role(
        split_manifest, expected_manifest_sha256=EXPECTED_SPLIT_SHA256,
        role="D_select", expected_source_identities=expected_sources,
    )
    if len(low.sample_tokens) != EXPECTED_D_LOW_SAMPLES or len(select.sample_tokens) != EXPECTED_D_SELECT_SAMPLES:
        raise RuntimeError("BN-B8 frozen role sample counts drifted")
    log_tokens = _manifest_select_logs(split_manifest)
    if len(log_tokens) != EXPECTED_D_SELECT_LOGS:
        raise RuntimeError("BN-B8 D_select log count drifted")
    references, reference_results = _load_references(Path(args.reference_root).resolve())

    _write_json(output / "execution_identity.json", {
        "schema": SCHEMA,
        "source_sha": args.source_sha,
        "source_tree": args.source_tree,
        "runtime": _runtime_identity(),
        "runtime_dependencies": runtime_dependencies,
        "runtime_dependencies_sha256": _canonical_sha256(runtime_dependencies),
        "resolved_config_sha256": config.sha256,
        "split_manifest": {"path": str(split_manifest), "sha256": EXPECTED_SPLIT_SHA256},
        "roles": {"D_low": low.identity(), "D_select": select.identity()},
        "D_select_log_tokens": list(log_tokens),
        "sealed_references": REFERENCE_FILES,
        "declared_change": {
            "normalization": "batch_norm_1d",
            "physical_microbatch": PHYSICAL_BATCH,
            "optimizer_updates": HORIZON,
            "grad_scaler_init_scale": 8.0,
            "eval_physical_microbatch": EVAL_PHYSICAL_BATCH,
        },
    })

    report = _run_cell(
        base_config=base,
        spec=_spec(),
        source_sha=args.source_sha,
        split_manifest=split_manifest,
        output_dir=output,
        runtime_dependencies=runtime_dependencies,
    )
    state = report["terminal_training_state"]
    if (
        int(state["attempted_windows"]) != HORIZON
        or int(state["optimizer_step"]) != HORIZON
        or int(state["invalid_windows"]) != 0
        or int(state["nonfinite_windows"]) != 0
        or int(state["overflow_windows"]) != 0
        or int(state["discarded_windows"]) != 0
        or report["health"]["hard_errors"]
    ):
        raise RuntimeError("BN-B8 failed the exact 769-update numerical gate")
    if any(float(chunk["metrics"]["grad_scaler_scale"]) != 8.0 for chunk in report["chunks"]):
        raise RuntimeError("BN-B8 GradScaler did not remain at the frozen scale 8")
    tokens = report["training_token_evidence"]
    if (
        int(tokens["consumed_sample_count"]) != 6152
        or int(tokens["drop_last_remainder_count"]) != 3
        or tokens["consumed_sample_tokens_ordered_sha256"] != EXPECTED_TOKEN_ORDER_SHA256
        or tokens["drop_last_remainder_tokens_sorted_sha256"] != EXPECTED_REMAINDER_SHA256
    ):
        raise RuntimeError("BN-B8 actual token exposure drifted from the sealed B4 order")
    if int(report["D_select_evaluation"]["n_samples"]) != EXPECTED_D_SELECT_SAMPLES:
        raise RuntimeError("BN-B8 D_select evaluator sample count drifted")

    new_result = output / CELL / "D_select_results.json"
    accepted = {**references, CELL: report}
    manifest_identity = {
        "path": str(split_manifest), "sha256": EXPECTED_SPLIT_SHA256,
        "role": "D_select", "source_identities": expected_sources,
    }
    from fl_v3.data.nuscenes import paths as nuscenes_paths

    nusc = nuscenes_paths.create_nuscenes(
        "v1.0-trainval", config.to_run_config()["nuscenes-dataroot"], verbose=False,
    )
    versus_gn = _paired_log_evidence(
        nusc=nusc,
        result_paths={GN_REFERENCE_CELL: reference_results[GN_REFERENCE_CELL], CELL: new_result},
        sample_tokens=select.sample_tokens,
        log_tokens=log_tokens,
        manifest_identity=manifest_identity,
        accepted_reports=accepted,
        candidate_cells=(GN_REFERENCE_CELL, CELL),
        delta_direction="BN1d_B8_minus_GN_B4",
    )
    versus_bn_b4 = _paired_log_evidence(
        nusc=nusc,
        result_paths={BN_B4_REFERENCE_CELL: reference_results[BN_B4_REFERENCE_CELL], CELL: new_result},
        sample_tokens=select.sample_tokens,
        log_tokens=log_tokens,
        manifest_identity=manifest_identity,
        accepted_reports=accepted,
        candidate_cells=(BN_B4_REFERENCE_CELL, CELL),
        delta_direction="BN1d_B8_minus_BN1d_B4_incomplete",
    )
    _write_json(output / "paired_vs_gn_b4.json", versus_gn)
    _write_json(output / "paired_vs_bn_b4.json", versus_bn_b4)
    summary = {
        "schema": SCHEMA,
        "status": "PASS",
        "hard_gate": "PASS",
        "source_sha": args.source_sha,
        "cell": CELL,
        "initial_parameter_sha256": report["initial_parameter_sha256"],
        "training_tokens": {
            "consumed_ordered_sha256": EXPECTED_TOKEN_ORDER_SHA256,
            "drop_last_remainder_sorted_sha256": EXPECTED_REMAINDER_SHA256,
            "consumed_samples": 6152,
            "dropped_samples": 3,
        },
        "numerical_gate": {
            "attempted_updates": HORIZON,
            "accepted_updates": HORIZON,
            "overflow_windows": 0,
            "grad_scaler_scale": 8.0,
        },
        "capability": {
            "BN1d_B8": _point_metrics(report),
            "GN_B4_reference": _point_metrics(references[GN_REFERENCE_CELL]),
            "BN1d_B4_incomplete_reference": _point_metrics(references[BN_B4_REFERENCE_CELL]),
            "paired_vs_gn_b4": versus_gn,
            "paired_vs_bn_b4": versus_bn_b4,
        },
        "training_runtime": {
            "BN1d_B8": _train_runtime(report),
            "GN_B4_reference": _train_runtime(references[GN_REFERENCE_CELL]),
            "BN1d_B4_incomplete_reference": _train_runtime(references[BN_B4_REFERENCE_CELL]),
        },
        "memory": report["memory"],
        "scientific_selection": "OWNER_DECISION_REQUIRED",
        "automatic_promotion": False,
        "interpretation_limits": _spec()["interpretation_limits"],
    }
    _write_json(output / "summary.json", summary)
    _write_json(output / "artifact_sha256s.json", _artifact_manifest(output))


def main() -> None:
    args = _parse_args()
    output = Path(args.output_dir).resolve()
    if output.exists():
        raise RuntimeError(f"BN-B8 output must be fresh: {output}")
    output.mkdir(parents=True)
    try:
        _execute(args, output)
    except BaseException as exc:
        _write_json(output / "failure_summary.json", {
            "schema": "fl_v3.s10.c1b1_bn_b8_failure.v1",
            "exception_type": type(exc).__name__,
            "exception_message": str(exc),
        })
        _write_json(output / "artifact_sha256s.json", _artifact_manifest(output))
        raise


if __name__ == "__main__":
    main()
