#!/usr/bin/env python3
"""C1-B1 matched one-epoch GN/BN1d capability and D_select evaluation.

This runner is deliberately not an automatic architecture selector.  It binds
the full D_low training exposure, terminal-only checkpoints, exact D_select
evaluation and paired leave-one-log-out evidence.  Numeric promotion margins
remain an owner decision after the evidence exists.
"""
from __future__ import annotations

import argparse
import copy
import gc
import json
import math
import os
from pathlib import Path
import platform
import statistics
import sys
from typing import Any, Iterable, Mapping

sys.path.insert(0, "fl_v3/src")
sys.path.insert(0, "fl_v3/scripts")

import torch

from fl_v3.config import resolve_config, verify_physical_data_identities
from fl_v3.data.nuscenes.s10_binding import load_frozen_split_role
from fl_v3.eval.subset_detection_eval import (
    _evaluate_official_math,
    _restrict_eval_boxes,
    bound_detection_config,
    evaluate_subset_tokens,
    strict_load_json,
)
from fl_v3.utils.runtime import enforce_determinism, seed_everything, verify_runtime_dependency_identity
from s10_c1b0_fusion_health import _parameter_state_sha256, _state_schema
from s10_stop_c0_health import (
    EXPECTED_SPLIT_SHA256,
    FULL_BOUNDARIES,
    FULL_DIAGNOSTICS,
    _canonical_sha256,
    _run_cell,
    _sha256_file,
    _source_identities,
    _write_json,
)


SCHEMA = "fl_v3.s10.c1b1_capability.v1"
HORIZON = 1538
PHYSICAL_BATCH = 4
EXPECTED_D_LOW_SAMPLES = 6155
EXPECTED_D_SELECT_SAMPLES = 4626
EXPECTED_D_SELECT_LOGS = 8
T_CRITICAL_95_DF7 = 2.364624251
CANDIDATES = (
    ("C1-B1-CUR-A1-GN-DLOW", "group_norm"),
    ("C1-B1-CUR-A1-BN1D-DLOW", "batch_norm_1d"),
)


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--split-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--source-sha", required=True)
    parser.add_argument("--source-tree", required=True)
    return parser.parse_args()


def _runtime_identity() -> dict[str, Any]:
    if platform.machine() != "aarch64":
        raise RuntimeError("C1-B1 requires an aarch64 Arrhenius compute node")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("C1-B1 requires exactly one visible CUDA device")
    properties = torch.cuda.get_device_properties(0)
    if "GH200" not in properties.name.upper():
        raise RuntimeError(f"C1-B1 expected GH200, got {properties.name!r}")
    return {
        "machine": platform.machine(),
        "node": platform.node(),
        "device_name": properties.name,
        "device_total_memory": int(properties.total_memory),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_job_gpus": os.environ.get("SLURM_JOB_GPUS"),
        "slurm_gpus_on_node": os.environ.get("SLURM_GPUS_ON_NODE"),
    }


def _artifact_manifest(root: Path) -> dict[str, Any]:
    files = {}
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = str(path.relative_to(root))
        if relative == "artifact_sha256s.json":
            continue
        files[relative] = {"sha256": _sha256_file(path), "bytes": path.stat().st_size}
    return {"schema": "fl_v3.s10.c1b1_artifacts.v1", "files": files}


def _resolved_candidate(base: Mapping[str, Any], normalization: str):
    raw = copy.deepcopy(dict(base))
    raw["model"]["second_normalization"] = normalization
    return resolve_config(raw)


def _assert_config(config, normalization: str) -> None:
    model = dict(config.data["model"])
    training = config.data["training"]
    if str(config.data["schema_version"]) != "s10.v1":
        raise RuntimeError("C1-B1 requires explicit s10.v1 config")
    if model != {
        "mode": "fusion",
        "camera_arch": "swin_t_stride8",
        "camera_pretrained": True,
        "camera_activation_checkpoint": False,
        "lidar_arch": "second_075",
        "second_normalization": normalization,
        "fusion_arch": "conv_fuser_256",
        "head_arch": "centerhead_multitask",
    }:
        raise RuntimeError("C1-B1 graph/initialization/normalization config drift")
    if config.precision != "fp16" or config.sparse_conv_precision != "fp32":
        raise RuntimeError("C1-B1 requires global FP16 plus SECOND FP32 island")
    if (
        int(training["max_optimizer_steps"]) != HORIZON
        or int(training["micro_batch_size"]) != PHYSICAL_BATCH
        or int(training["world_size"]) != 1
        or int(training["accumulation_steps"]) != 1
        or int(training["effective_global_batch"]) != PHYSICAL_BATCH
        or int(training["seed"]) != 0
        or int(training["max_epochs"]) != 1
        or float(training["grad_scaler_init_scale"]) != 32.0
        or training["ema_decay"] is not None
        or str(training["sampling"]) != "uniform"
    ):
        raise RuntimeError("C1-B1 B4/one-epoch/seed/scale/no-EMA contract drift")
    optimizer = config.data["optimizer"]
    if (
        str(optimizer["name"]) != "adamw"
        or float(optimizer["learning_rate"]) != 1.0e-4
        or float(optimizer["weight_decay"]) != 0.01
    ):
        raise RuntimeError("C1-B1 optimizer/constant-scheduler contract drift")


def _candidate_spec(
    cell: str, normalization: str, expected_initial_parameter_sha256: str,
) -> dict[str, Any]:
    return {
        "id": cell,
        "mode": "fusion",
        "camera_pretrained": True,
        "normalization": normalization,
        "attempted_windows": HORIZON,
        "boundaries": FULL_BOUNDARIES,
        "diagnostics": FULL_DIAGNOSTICS,
        "evaluate": True,
        "operator_profile": False,
        "grad_scaler_init_scale": 32.0,
        "report_schema": SCHEMA,
        "expected_initial_parameter_sha256": expected_initial_parameter_sha256,
        "interpretation_limits": [
            "single-seed D_low one-epoch internal evidence; not the S10 full claim",
            "fixed engineering baseline recipe; not production-recipe acceptance",
            "D_select is train-only proxy evidence; D_audit and official val remain sealed",
            "no operator profiler or synchronized bottleneck attribution runs in C1-B1",
            "no numeric normalization-promotion margin was owner-approved",
        ],
    }


def jackknife_interval(full_delta: float, leave_one_out_deltas: Iterable[float]) -> dict[str, Any]:
    """Return paired delete-one-cluster jackknife uncertainty for eight logs."""
    values = tuple(float(value) for value in leave_one_out_deltas)
    if len(values) != EXPECTED_D_SELECT_LOGS or not all(math.isfinite(v) for v in values):
        raise ValueError("C1-B1 jackknife requires eight finite leave-one-log-out deltas")
    point = float(full_delta)
    if not math.isfinite(point):
        raise ValueError("C1-B1 full delta must be finite")
    n = len(values)
    pseudovalues = tuple(n * point - (n - 1) * value for value in values)
    pseudo_mean = statistics.mean(pseudovalues)
    standard_error = math.sqrt(
        sum((value - pseudo_mean) ** 2 for value in pseudovalues) / (n * (n - 1))
    )
    radius = T_CRITICAL_95_DF7 * standard_error
    return {
        "full_point_delta": point,
        "leave_one_log_out_deltas": list(values),
        "jackknife_bias_corrected_delta": pseudo_mean,
        "jackknife_standard_error": standard_error,
        "t_critical_95_df7": T_CRITICAL_95_DF7,
        "point_centered_95_interval": [point - radius, point + radius],
        "clusters": n,
        "interpretation": "descriptive paired-log uncertainty; not an automatic promotion margin",
    }


def _metric_projection(metrics: Any) -> dict[str, Any]:
    raw = metrics.serialize()
    per_class = {str(key): float(value) for key, value in raw["mean_dist_aps"].items()}
    result = {
        "NDS": float(raw["nd_score"]),
        "mAP": float(raw["mean_ap"]),
        "per_class_mAP": per_class,
    }
    if not all(math.isfinite(value) for value in [result["NDS"], result["mAP"], *per_class.values()]):
        raise RuntimeError("C1-B1 evaluator returned a nonfinite capability metric")
    return result


def _manifest_select_logs(path: Path) -> tuple[str, ...]:
    raw = strict_load_json(str(path))
    record = raw.get("roles", {}).get("D_select", {})
    logs = record.get("log_tokens")
    if (
        not isinstance(logs, list)
        or len(logs) != EXPECTED_D_SELECT_LOGS
        or logs != sorted(logs)
        or len(logs) != len(set(logs))
        or any(not isinstance(token, str) or not token for token in logs)
    ):
        raise RuntimeError("C1-B1 requires the exact sorted eight-log D_select role")
    return tuple(logs)


def _tokens_by_log(nusc, sample_tokens: Iterable[str], expected_logs: Iterable[str]) -> dict[str, tuple[str, ...]]:
    grouped: dict[str, list[str]] = {str(log): [] for log in expected_logs}
    for token in sample_tokens:
        sample = nusc.get("sample", str(token))
        scene = nusc.get("scene", sample["scene_token"])
        log_token = str(scene["log_token"])
        if log_token not in grouped:
            raise RuntimeError("D_select sample resolved outside the frozen log ownership")
        grouped[log_token].append(str(token))
    result = {key: tuple(sorted(value)) for key, value in grouped.items()}
    if any(not value for value in result.values()):
        raise RuntimeError("D_select contains an empty declared log cluster")
    flattened = sorted(token for values in result.values() for token in values)
    if flattened != sorted(str(token) for token in sample_tokens):
        raise RuntimeError("D_select log-cluster token accounting drift")
    return result


def _paired_log_evidence(
    *, nusc, result_paths: Mapping[str, Path], sample_tokens: tuple[str, ...],
    log_tokens: tuple[str, ...], manifest_identity: Mapping[str, Any],
    accepted_reports: Mapping[str, Mapping[str, Any]],
    candidate_cells: tuple[str, str] | None = None,
    delta_direction: str = "BN1d_minus_GN",
) -> dict[str, Any]:
    cfg, _ = bound_detection_config()
    grouped = _tokens_by_log(nusc, sample_tokens, log_tokens)
    full = {}
    leave_one_out = {}
    for cell, path in result_paths.items():
        evaluation = evaluate_subset_tokens(
            nusc,
            str(path),
            parent_split="train",
            sample_tokens=sample_tokens,
            manifest_identity=manifest_identity,
        )
        projection = _metric_projection(evaluation.metrics)
        accepted = accepted_reports[cell]["D_select_evaluation"]
        if (
            projection["NDS"] != float(accepted["internal_subset_NDS"])
            or projection["mAP"] != float(accepted["internal_subset_mAP"])
        ):
            raise RuntimeError(f"{cell} repeated full D_select evaluator parity drift")
        full[cell] = projection
        rows = {}
        for omitted in log_tokens:
            omitted_tokens = set(grouped[omitted])
            kept = tuple(token for token in sample_tokens if token not in omitted_tokens)
            gt_boxes = _restrict_eval_boxes(evaluation.gt_boxes, kept)
            pred_boxes = _restrict_eval_boxes(evaluation.pred_boxes, kept)
            metrics, _ = _evaluate_official_math(gt_boxes, pred_boxes, cfg)
            rows[omitted] = _metric_projection(metrics)
        leave_one_out[cell] = rows

    baseline_cell, candidate_cell = (
        candidate_cells
        if candidate_cells is not None
        else tuple(cell for cell, _ in CANDIDATES)
    )
    metric_names = ("NDS", "mAP")
    paired = {}
    for metric in metric_names:
        full_delta = full[candidate_cell][metric] - full[baseline_cell][metric]
        deltas = [
            leave_one_out[candidate_cell][log][metric]
            - leave_one_out[baseline_cell][log][metric]
            for log in log_tokens
        ]
        paired[metric] = jackknife_interval(full_delta, deltas)
    classes = sorted(full[baseline_cell]["per_class_mAP"])
    per_class = {}
    for class_name in classes:
        full_delta = (
            full[candidate_cell]["per_class_mAP"][class_name]
            - full[baseline_cell]["per_class_mAP"][class_name]
        )
        deltas = [
            leave_one_out[candidate_cell][log]["per_class_mAP"][class_name]
            - leave_one_out[baseline_cell][log]["per_class_mAP"][class_name]
            for log in log_tokens
        ]
        per_class[class_name] = jackknife_interval(full_delta, deltas)
    return {
        "schema": "fl_v3.s10.c1b1_paired_log.v1",
        "delta_direction": str(delta_direction),
        "candidate_cells": {
            "baseline": baseline_cell,
            "candidate": candidate_cell,
        },
        "cluster_unit": "frozen D_select log_token",
        "log_tokens": list(log_tokens),
        "samples_per_log": {key: len(value) for key, value in grouped.items()},
        "full_metrics": full,
        "leave_one_log_out_metrics": leave_one_out,
        "paired_deltas": paired,
        "paired_per_class_mAP_deltas": per_class,
        "scientific_selection": "OWNER_DECISION_REQUIRED",
        "selection_reason": (
            "single seed and eight held-out train logs; no owner-approved numeric "
            "superiority/non-inferiority margin exists"
        ),
    }


def _execute(args, output: Path) -> None:
    if len(args.source_sha) != 40 or len(args.source_tree) != 40:
        raise RuntimeError("C1-B1 requires exact source commit and tree identities")
    split_manifest = Path(args.split_manifest).resolve()
    if _sha256_file(split_manifest) != EXPECTED_SPLIT_SHA256:
        raise RuntimeError("C1-B1 split manifest identity drift")
    base = json.loads(Path(args.config).read_text(encoding="utf-8"))
    if base.get("scheduler") is not None:
        raise RuntimeError("C1-B1 requires the unchanged constant scheduler")
    configs = {
        normalization: _resolved_candidate(base, normalization)
        for _, normalization in CANDIDATES
    }
    for normalization, config in configs.items():
        _assert_config(config, normalization)
        verify_physical_data_identities(config)
    runtime_dependencies = verify_runtime_dependency_identity(configs["group_norm"].to_run_config())
    expected_sources = _source_identities(configs["group_norm"])
    low = load_frozen_split_role(
        split_manifest, expected_manifest_sha256=EXPECTED_SPLIT_SHA256,
        role="D_low", expected_source_identities=expected_sources,
    )
    select = load_frozen_split_role(
        split_manifest, expected_manifest_sha256=EXPECTED_SPLIT_SHA256,
        role="D_select", expected_source_identities=expected_sources,
    )
    if len(low.sample_tokens) != EXPECTED_D_LOW_SAMPLES or len(select.sample_tokens) != EXPECTED_D_SELECT_SAMPLES:
        raise RuntimeError("C1-B1 frozen role sample counts drifted")
    log_tokens = _manifest_select_logs(split_manifest)
    runtime = _runtime_identity()
    enforce_determinism(strict=False, precision="fp16")

    from fl_v3.training.tasks import NuScenesDetectionTask

    task = NuScenesDetectionTask()
    candidate_identity = {}
    shared_parameter_sha256 = None
    for cell, normalization in CANDIDATES:
        seed_everything(0)
        model = task.build_model(configs[normalization].to_run_config())
        parameter_sha256 = _parameter_state_sha256(model)
        schema = _state_schema(model)
        if shared_parameter_sha256 is None:
            shared_parameter_sha256 = parameter_sha256
        elif parameter_sha256 != shared_parameter_sha256:
            raise RuntimeError("C1-B1 GN/BN1d candidates do not share exact trainable W0")
        candidate_identity[cell] = {
            "normalization": normalization,
            "resolved_config_sha256": configs[normalization].sha256,
            "parameter_state_sha256": parameter_sha256,
            "state_schema": schema,
        }
        del model
    if candidate_identity[CANDIDATES[0][0]]["state_schema"] == candidate_identity[CANDIDATES[1][0]]["state_schema"]:
        raise RuntimeError("C1-B1 GN/BN1d checkpoint schemas were not distinguished")
    gc.collect()

    _write_json(output / "execution_identity.json", {
        "schema": SCHEMA,
        "source_sha": args.source_sha,
        "source_tree": args.source_tree,
        "runtime": runtime,
        "runtime_dependencies": runtime_dependencies,
        "runtime_dependencies_sha256": _canonical_sha256(runtime_dependencies),
        "split_manifest": {"path": str(split_manifest), "sha256": EXPECTED_SPLIT_SHA256},
        "roles": {"D_low": low.identity(), "D_select": select.identity()},
        "D_select_log_tokens": list(log_tokens),
        "candidate_identity": candidate_identity,
        "shared_parameter_state_sha256": shared_parameter_sha256,
        "cell_order": [cell for cell, _ in CANDIDATES],
        "scale_qualification_dependency": {
            "job": "504508",
            "scope": "same graph/W0/precision/physical-B4 recipe except longer declared horizon",
            "result": "both candidates passed no-update scale-32 qualification",
        },
    })

    reports = []
    for cell, normalization in CANDIDATES:
        report = _run_cell(
            base_config=base,
            spec=_candidate_spec(cell, normalization, str(shared_parameter_sha256)),
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
            raise RuntimeError(f"{cell} failed the exact 1538-update numerical gate")
        if int(report["D_select_evaluation"]["n_samples"]) != EXPECTED_D_SELECT_SAMPLES:
            raise RuntimeError(f"{cell} D_select evaluator sample count drifted")
        reports.append(report)
        print(json.dumps({
            "cell": cell,
            "mAP": report["D_select_evaluation"]["internal_subset_mAP"],
            "NDS": report["D_select_evaluation"]["internal_subset_NDS"],
        }, sort_keys=True), flush=True)

    by_cell = {report["cell"]["id"]: report for report in reports}
    gn_cell, bn_cell = (cell for cell, _ in CANDIDATES)
    for field in (
        "consumed_sample_tokens_ordered_sha256",
        "drop_last_remainder_tokens_sorted_sha256",
    ):
        if by_cell[gn_cell]["training_token_evidence"][field] != by_cell[bn_cell]["training_token_evidence"][field]:
            raise RuntimeError(f"C1-B1 matched training-token identity drift for {field}")

    from fl_v3.data.nuscenes import paths as nuscenes_paths

    nusc = nuscenes_paths.create_nuscenes(
        "v1.0-trainval", configs["group_norm"].to_run_config()["nuscenes-dataroot"],
        verbose=False,
    )
    paired = _paired_log_evidence(
        nusc=nusc,
        result_paths={cell: output / cell / "D_select_results.json" for cell, _ in CANDIDATES},
        sample_tokens=select.sample_tokens,
        log_tokens=log_tokens,
        manifest_identity={
            "path": str(split_manifest), "sha256": EXPECTED_SPLIT_SHA256,
            "role": "D_select", "source_identities": expected_sources,
        },
        accepted_reports=by_cell,
    )
    _write_json(output / "paired_log_evidence.json", paired)
    summary = {
        "schema": SCHEMA,
        "status": "PASS",
        "hard_gate": "PASS",
        "source_sha": args.source_sha,
        "cell_order": [cell for cell, _ in CANDIDATES],
        "shared_parameter_state_sha256": shared_parameter_sha256,
        "matched_training_tokens": {
            "consumed_ordered_sha256": by_cell[gn_cell]["training_token_evidence"]["consumed_sample_tokens_ordered_sha256"],
            "drop_last_remainder_sorted_sha256": by_cell[gn_cell]["training_token_evidence"]["drop_last_remainder_tokens_sorted_sha256"],
            "consumed_samples": 6152,
            "dropped_samples": 3,
        },
        "capability": paired,
        "scientific_selection": "OWNER_DECISION_REQUIRED",
        "automatic_promotion": False,
        "next_decision": "inspect capability, paired-log uncertainty and numerical health before A2 or MIT repair",
        "interpretation_limits": [
            "single-seed D_low one-epoch train-only contrast; not official validation or a full-run claim",
            "D_select is held-out train-only proxy evidence; D_audit and official val remain sealed",
            "no numeric superiority/non-inferiority margin was owner-approved, so this runner cannot select a winner",
            "no intermediate checkpoint selection, recipe sweep, A2, MIT repair or later-stop continuation",
        ],
    }
    _write_json(output / "summary.json", summary)
    _write_json(output / "artifact_sha256s.json", _artifact_manifest(output))


def main() -> None:
    args = _parse_args()
    output = Path(args.output_dir).resolve()
    if output.exists():
        raise RuntimeError(f"C1-B1 output must be fresh: {output}")
    output.mkdir(parents=True)
    try:
        _execute(args, output)
    except BaseException as exc:
        _write_json(output / "failure_summary.json", {
            "schema": "fl_v3.s10.c1b1_failure.v1",
            "exception_type": type(exc).__name__,
            "exception_message": str(exc),
        })
        _write_json(output / "artifact_sha256s.json", _artifact_manifest(output))
        raise


if __name__ == "__main__":
    main()
