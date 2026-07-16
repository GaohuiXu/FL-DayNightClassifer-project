#!/usr/bin/env python3
"""O-130 STOP-B stochastic/runtime variation decomposition.

This is a fixed-W0, no-update observation over one frozen B4 token vector.  It
separates repeated fixed-seed variation from varying-seed training stochasticity
for the current camera-only, LiDAR-only and fusion component graphs.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import statistics
import sys
import time
from typing import Any, Mapping, Sequence

sys.path.insert(0, "fl_v3/src")

import torch

from fl_v3.config import load_resolved_config, verify_physical_data_identities
from fl_v3.data.nuscenes.s10_binding import (
    load_frozen_split_role,
    load_frozen_stop_b_panel,
)
from fl_v3.training.loop import _move_to_device
from fl_v3.training.precision_diagnostics import runtime_rng_state_sha256
from fl_v3.training.s10_observation import (
    capture_parameter_gradient_tensors,
    capture_tensor_tree_tensors,
    classify_stop_b_randomness,
    compare_parameter_gradient_tensors,
    compare_tensor_tree_tensors,
    module_state_sha256,
    strict_json_value,
    zero_model_gradients,
)
from fl_v3.training.tasks import NuScenesDetectionTask
from fl_v3.utils.runtime import (
    enforce_determinism,
    precision_autocast_context,
    seed_everything,
    verify_runtime_dependency_identity,
)


MODE_ORDER = ("C-STR8", "L-S075", "F-U")
FIXED_SEEDS = (10000, 10000, 10000, 10000, 10000)
VARYING_SEEDS = (11000, 11001, 11002, 11003, 11004)
FUSION_W0_SHA256 = "e58bcd46d588c68b31335fe87cc5fbff06cbc0fbcdae7e88b0b1ed70d1d65395"


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera-config", required=True)
    parser.add_argument("--lidar-config", required=True)
    parser.add_argument("--fusion-config", required=True)
    parser.add_argument("--split-manifest", required=True)
    parser.add_argument("--split-sha256", required=True)
    parser.add_argument("--panel-manifest", required=True)
    parser.add_argument("--panel-file-sha256", required=True)
    parser.add_argument("--panel-content-sha256", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--source-sha", required=True)
    parser.add_argument("--source-tree", required=True)
    return parser.parse_args()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(strict_json_value(value), sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _append_jsonl(path: Path, value: Any) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(
            json.dumps(strict_json_value(value), sort_keys=True, allow_nan=False) + "\n"
        )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tensor_map_sha256(values: Mapping[str, torch.Tensor | None]) -> str:
    digest = hashlib.sha256()
    for name in sorted(values):
        digest.update(name.encode("utf-8") + b"\0")
        value = values[name]
        if value is None:
            digest.update(b"MISSING\0")
            continue
        tensor = value.detach().contiguous().cpu()
        digest.update(str(tensor.dtype).encode("ascii") + b"\0")
        digest.update(
            json.dumps(list(tensor.shape), separators=(",", ":")).encode("ascii")
            + b"\0"
        )
        digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _runtime_assertions() -> dict[str, Any]:
    if platform.machine() != "aarch64":
        raise RuntimeError("STOP-B B-RAND requires an aarch64 Arrhenius node")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("STOP-B B-RAND requires exactly one visible CUDA device")
    device = torch.device("cuda", 0)
    properties = torch.cuda.get_device_properties(device)
    if "GH200" not in properties.name.upper():
        raise RuntimeError(f"STOP-B B-RAND expected GH200, got {properties.name!r}")
    return {
        "machine": platform.machine(),
        "node": platform.node(),
        "device": str(device),
        "device_name": properties.name,
        "device_total_memory": int(properties.total_memory),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_job_gpus": os.environ.get("SLURM_JOB_GPUS"),
        "slurm_gpus_on_node": os.environ.get("SLURM_GPUS_ON_NODE"),
    }


def _expected_split_sources(config) -> dict[str, Any]:
    data = config.data["data"]
    return {
        "detection_config_sha256": "217f96cca4e80f790c4674ef72257a6863ee9a85b0ce185bc56488afc32c7a0b",
        "version": data["version"],
        "n_sweeps": data["n_sweeps"],
        "train_cache_logical_sha256": data["caches"]["train"]["logical_sha256"],
        "train_cache_pickle_sha256": data["caches"]["train"]["pickle_sha256"],
        "train_cache_sidecar_sha256": data["caches"]["train"]["sidecar_sha256"],
        "val_cache_logical_sha256": data["caches"]["val"]["logical_sha256"],
        "val_cache_pickle_sha256": data["caches"]["val"]["pickle_sha256"],
        "val_cache_sidecar_sha256": data["caches"]["val"]["sidecar_sha256"],
        "zip_manifest_logical_sha256": data["zip_manifest"]["logical_sha256"],
        "zip_manifest_file_sha256": data["zip_manifest"]["file_sha256"],
    }


def _assert_mode_configs(configs) -> None:
    expected_models = {
        "C-STR8": {
            "mode": "camera_only",
            "camera_arch": "swin_t_stride8",
            "camera_pretrained": False,
            "camera_activation_checkpoint": False,
            "lidar_arch": "none",
            "fusion_arch": "none",
            "head_arch": "centerhead_multitask",
        },
        "L-S075": {
            "mode": "lidar_only",
            "camera_arch": "none",
            "camera_pretrained": None,
            "camera_activation_checkpoint": False,
            "lidar_arch": "second_075",
            "fusion_arch": "none",
            "head_arch": "centerhead_multitask",
        },
        "F-U": {
            "mode": "fusion",
            "camera_arch": "swin_t_stride8",
            "camera_pretrained": False,
            "camera_activation_checkpoint": False,
            "lidar_arch": "second_075",
            "fusion_arch": "conv_fuser_256",
            "head_arch": "centerhead_multitask",
        },
    }
    if tuple(configs) != MODE_ORDER:
        raise RuntimeError("STOP-B B-RAND mode order drift")
    for label, config in configs.items():
        if config.precision != "fp32":
            raise RuntimeError(f"{label} must remain uniform FP32")
        if dict(config.data["model"]) != expected_models[label]:
            raise RuntimeError(f"{label} model graph drift")
        expected_sparse = "not_applicable" if label == "C-STR8" else "fp32"
        if config.sparse_conv_precision != expected_sparse:
            raise RuntimeError(f"{label} sparse precision drift")
        training = config.data["training"]
        if (
            int(training["micro_batch_size"]) != 4
            or int(training["world_size"]) != 1
            or int(training["accumulation_steps"]) != 1
            or int(training["effective_global_batch"]) != 4
            or int(training["seed"]) != 0
            or str(training["sampling"]) != "uniform"
            or training["ema_decay"] is not None
        ):
            raise RuntimeError(f"{label} B4/W0/uniform/no-EMA config drift")
    reference_data = configs["F-U"].data["data"]
    for label, config in configs.items():
        if config.data["data"] != reference_data:
            raise RuntimeError(f"{label} data identity drift")


def _batch_tokens(batch: Mapping[str, Any], expected: Sequence[str]) -> None:
    actual = [str(value) for value in batch["sample_token"]]
    if actual != list(expected):
        raise RuntimeError(
            f"STOP-B B-RAND token order drift: expected={list(expected)}, actual={actual}"
        )
    if int(batch["batch_size"]) != 4:
        raise RuntimeError("STOP-B B-RAND requires physical B4")


def _gradient_integrity(values: Mapping[str, torch.Tensor | None]) -> dict[str, Any]:
    missing = sorted(name for name, value in values.items() if value is None)
    nonfinite = []
    elements = 0
    for name, value in values.items():
        if value is None:
            continue
        elements += int(value.numel())
        if not bool(torch.isfinite(value).all().item()):
            nonfinite.append(name)
    return {
        "parameter_count": len(values),
        "elements_with_grad": elements,
        "missing_gradients": missing,
        "nonfinite_parameters": nonfinite,
        "all_finite": not nonfinite,
    }


def _output_integrity(values: Mapping[str, torch.Tensor]) -> dict[str, Any]:
    bad = [name for name, value in values.items() if not bool(torch.isfinite(value).all().item())]
    return {
        "tensor_count": len(values),
        "elements": sum(int(value.numel()) for value in values.values()),
        "nonfinite_tensors": bad,
        "all_finite": not bad,
    }


def _run_once(model, criterion, batch_cpu, *, device, seed, capture):
    seed_everything(seed)
    zero_model_gradients(model)
    batch = _move_to_device(batch_cpu, device)
    started = time.perf_counter()
    with precision_autocast_context("fp32", device):
        output = model(batch)
        loss = criterion(output, batch)
    if not bool(torch.isfinite(loss).item()):
        raise RuntimeError("STOP-B B-RAND loss is nonfinite")
    loss.backward()
    torch.cuda.synchronize(device)
    record = {
        "seed": int(seed),
        "loss": float(loss.detach().item()),
        "rng_state_sha256_after": runtime_rng_state_sha256(),
        "wall_seconds": time.perf_counter() - started,
        "optimizer_constructed": False,
        "optimizer_update": False,
    }
    output_tensors = None
    gradient_tensors = None
    if capture:
        output_tensors = capture_tensor_tree_tensors(output)
        gradient_tensors = capture_parameter_gradient_tensors(model)
        record.update({
            "output_sha256": _tensor_map_sha256(output_tensors),
            "parameter_gradients_sha256": _tensor_map_sha256(gradient_tensors),
            "output_integrity": _output_integrity(output_tensors),
            "gradient_integrity": _gradient_integrity(gradient_tensors),
        })
    zero_model_gradients(model)
    del output, loss, batch
    return record, output_tensors, gradient_tensors


def _compact_gradient_report(report):
    return {
        "name_set_equal": report["name_set_equal"],
        "missing_gradient_sets_equal": report["missing_gradient_sets_equal"],
        "reference_missing_count": len(report["reference_missing_gradients"]),
        "candidate_missing_count": len(report["candidate_missing_gradients"]),
        "shape_mismatch_count": len(report["shape_mismatch_parameters"]),
        "dtype_mismatch_count": len(report["dtype_mismatch_parameters"]),
        "nonfinite_parameter_count": len(report["nonfinite_parameters"]),
        "allclose_failure_count": len(report["allclose_failure_parameters"]),
        "global": report["global"],
        "by_prefix": report["by_prefix"],
    }


def _compare_runs(reference, candidate, reference_output, candidate_output,
                  reference_gradients, candidate_gradients):
    output_report = compare_tensor_tree_tensors(reference_output, candidate_output)
    gradient_report = compare_parameter_gradient_tensors(
        reference_gradients,
        candidate_gradients,
        scale_divisor=1.0,
    )
    denominator = max(abs(float(reference["loss"])), 1e-12)
    return {
        "reference_seed": int(reference["seed"]),
        "candidate_seed": int(candidate["seed"]),
        "loss_absolute_difference": abs(
            float(candidate["loss"]) - float(reference["loss"])
        ),
        "loss_relative_difference": abs(
            float(candidate["loss"]) - float(reference["loss"])
        ) / denominator,
        "output": {
            "name_set_equal": output_report["name_set_equal"],
            "shape_mismatch_count": len(output_report["shape_mismatch_tensors"]),
            "dtype_mismatch_count": len(output_report["dtype_mismatch_tensors"]),
            "global": output_report["global"],
        },
        "parameter_gradients": _compact_gradient_report(gradient_report),
        "output_sha256_equal": (
            reference["output_sha256"] == candidate["output_sha256"]
        ),
        "parameter_gradients_sha256_equal": (
            reference["parameter_gradients_sha256"]
            == candidate["parameter_gradients_sha256"]
        ),
    }


def _assert_comparison_integrity(report, *, label, group, index):
    output = report["output"]
    gradients = report["parameter_gradients"]
    if not (
        output["name_set_equal"]
        and output["shape_mismatch_count"] == 0
        and output["dtype_mismatch_count"] == 0
        and output["global"]["all_finite"]
        and gradients["name_set_equal"]
        and gradients["missing_gradient_sets_equal"]
        and gradients["shape_mismatch_count"] == 0
        and gradients["dtype_mismatch_count"] == 0
        and gradients["nonfinite_parameter_count"] == 0
        and gradients["global"]["all_finite"]
    ):
        raise RuntimeError(
            f"{label} {group} comparison {index} failed structural/finite integrity"
        )


def _percentile(values: Sequence[float], quantile: float) -> float:
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _distribution(values: Sequence[float]) -> dict[str, Any]:
    if not values:
        raise RuntimeError("STOP-B B-RAND distribution is empty")
    return {
        "count": len(values),
        "median": float(statistics.median(values)),
        "p95": _percentile(values, 0.95),
        "min": min(values),
        "max": max(values),
    }


def _group_summary(pairs):
    return {
        "reference_comparison_count": len(pairs),
        "loss_relative_difference": _distribution([
            item["loss_relative_difference"] for item in pairs
        ]),
        "output_relative_l2": _distribution([
            item["output"]["global"]["relative_l2_error"] for item in pairs
        ]),
        "output_cosine": _distribution([
            item["output"]["global"]["cosine_similarity"] for item in pairs
        ]),
        "gradient_relative_l2": _distribution([
            item["parameter_gradients"]["global"]["relative_l2_error"]
            for item in pairs
        ]),
        "gradient_cosine": _distribution([
            item["parameter_gradients"]["global"]["cosine_similarity"]
            for item in pairs
        ]),
    }


def _run_group(
    *,
    label,
    group,
    seeds,
    model,
    criterion,
    batch,
    device,
    raw,
):
    reference = None
    reference_output = None
    reference_gradients = None
    records = []
    pairs = []
    for index, seed in enumerate(seeds):
        record, output_tensors, gradient_tensors = _run_once(
            model, criterion, batch, device=device, seed=seed, capture=True
        )
        record.update({
            "mode": label,
            "group": group,
            "run_index": index,
            "role": "reference" if index == 0 else "candidate",
        })
        _append_jsonl(raw / "runs.jsonl", record)
        records.append(record)
        if output_tensors is None or gradient_tensors is None:
            raise RuntimeError("STOP-B B-RAND measured tensor capture is missing")
        if index == 0:
            reference = record
            reference_output = output_tensors
            reference_gradients = gradient_tensors
            continue
        pair = _compare_runs(
            reference,
            record,
            reference_output,
            output_tensors,
            reference_gradients,
            gradient_tensors,
        )
        pair.update({"mode": label, "group": group, "candidate_run_index": index})
        _assert_comparison_integrity(
            pair, label=label, group=group, index=index
        )
        _append_jsonl(raw / "comparisons.jsonl", pair)
        pairs.append(pair)
        del output_tensors, gradient_tensors
        gc.collect()
    if len(records) != 5 or len(pairs) != 4:
        raise RuntimeError("STOP-B B-RAND group cardinality drift")
    del reference_output, reference_gradients
    gc.collect()
    return records, pairs, _group_summary(pairs)


def _stochastic_depth_registry(model):
    records = []
    for name, module in model.named_modules():
        if type(module).__name__ != "StochasticDepth":
            continue
        records.append({
            "name": name,
            "probability": float(module.p),
            "mode": str(module.mode),
            "training": bool(module.training),
        })
    return records


def _artifact_manifest(root: Path) -> dict[str, Any]:
    files = {}
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = str(path.relative_to(root))
        if relative == "artifact_sha256s.json":
            continue
        files[relative] = {"sha256": _sha256_file(path), "bytes": path.stat().st_size}
    return {"schema": "fl_v3.s10.stop_b_rand_artifacts.v1", "files": files}


def _execute(args, output: Path, raw: Path) -> None:
    runtime = _runtime_assertions()
    configs = {
        "C-STR8": load_resolved_config(args.camera_config),
        "L-S075": load_resolved_config(args.lidar_config),
        "F-U": load_resolved_config(args.fusion_config),
    }
    _assert_mode_configs(configs)
    verify_physical_data_identities(configs["F-U"])
    runtime_dependencies = verify_runtime_dependency_identity(
        configs["F-U"].to_run_config()
    )
    binding = load_frozen_split_role(
        args.split_manifest,
        expected_manifest_sha256=args.split_sha256,
        role="D_low",
        expected_source_identities=_expected_split_sources(configs["F-U"]),
    )
    panel, panel_report = load_frozen_stop_b_panel(
        args.panel_manifest,
        expected_file_sha256=args.panel_file_sha256,
        expected_content_sha256=args.panel_content_sha256,
        binding=binding,
    )
    tokens = list(panel["batches_b4"]["P_core"][0])
    identity = {
        "schema": "fl_v3.s10.stop_b_rand_execution.v1",
        "source_sha": args.source_sha,
        "source_tree": args.source_tree,
        "runtime": runtime,
        "runtime_dependencies": runtime_dependencies,
        "configs": {
            label: {
                "path": str(Path(path).resolve()),
                "resolved_sha256": configs[label].sha256,
            }
            for label, path in (
                ("C-STR8", args.camera_config),
                ("L-S075", args.lidar_config),
                ("F-U", args.fusion_config),
            )
        },
        "split_binding": binding.identity(),
        "panel": panel_report,
        "tokens": tokens,
        "fixed_seeds": list(FIXED_SEEDS),
        "varying_seeds": list(VARYING_SEEDS),
        "optimizer_constructed": False,
        "official_evaluation_executed": False,
    }
    _write_json(output / "execution_identity.json", identity)

    enforce_determinism(strict=True, precision="fp32")
    device = torch.device("cuda", 0)
    task = NuScenesDetectionTask()
    mode_summaries = {}
    total_runs = 0
    total_started = time.perf_counter()
    for label in MODE_ORDER:
        config = configs[label]
        run_config = config.to_run_config()
        loader = task.fixed_train_subset_loader(run_config, tokens)
        batches = list(loader)
        if len(batches) != 1:
            raise RuntimeError(f"{label} fixed loader must emit exactly one B4")
        batch = batches[0]
        _batch_tokens(batch, tokens)

        seed_everything(0)
        model = task.build_model(run_config)
        w0_hash = module_state_sha256(model)
        seed_everything(0)
        probe = task.build_model(run_config)
        probe_hash = module_state_sha256(probe)
        del probe
        gc.collect()
        if probe_hash != w0_hash:
            raise RuntimeError(f"{label} seed-0 W0 construction is not repeatable")
        if label == "F-U" and w0_hash != FUSION_W0_SHA256:
            raise RuntimeError("F-U W0 drift from accepted STOP-B identity")
        model.to(device)
        model.train()
        criterion = task.build_criterion(run_config)
        if module_state_sha256(model) != w0_hash:
            raise RuntimeError(f"{label} GPU model does not match W0")
        stochastic_depth = _stochastic_depth_registry(model)

        warmup, warmup_output, warmup_gradients = _run_once(
            model, criterion, batch, device=device, seed=9000, capture=True
        )
        warmup.update({"mode": label, "group": "warmup", "run_index": 0})
        _append_jsonl(raw / "runs.jsonl", warmup)
        del warmup_output, warmup_gradients
        gc.collect()
        total_runs += 1
        if module_state_sha256(model) != w0_hash:
            raise RuntimeError(f"{label} warm-up mutated W0")

        fixed_records, fixed_pairs, fixed_summary = _run_group(
            label=label,
            group="fixed_seed",
            seeds=FIXED_SEEDS,
            model=model,
            criterion=criterion,
            batch=batch,
            device=device,
            raw=raw,
        )
        total_runs += len(fixed_records)
        if len({item["rng_state_sha256_after"] for item in fixed_records}) != 1:
            raise RuntimeError(f"{label} fixed-seed RNG-state hashes drift")
        if module_state_sha256(model) != w0_hash:
            raise RuntimeError(f"{label} fixed-seed group mutated W0")

        varying_records, varying_pairs, varying_summary = _run_group(
            label=label,
            group="varying_seed",
            seeds=VARYING_SEEDS,
            model=model,
            criterion=criterion,
            batch=batch,
            device=device,
            raw=raw,
        )
        total_runs += len(varying_records)
        if module_state_sha256(model) != w0_hash:
            raise RuntimeError(f"{label} varying-seed group mutated W0")

        all_records = [warmup, *fixed_records, *varying_records]
        missing_sets = {
            tuple(item["gradient_integrity"]["missing_gradients"])
            for item in all_records
        }
        integrity_pass = bool(
            len(missing_sets) == 1
            and all(item["output_integrity"]["all_finite"] for item in all_records)
            and all(item["gradient_integrity"]["all_finite"] for item in all_records)
        )
        if not integrity_pass:
            raise RuntimeError(f"{label} measured-run integrity failed")
        mode_summaries[label] = {
            "mode": config.model_mode,
            "config_sha256": config.sha256,
            "W0_state_dict_sha256": w0_hash,
            "W0_probe_sha256": probe_hash,
            "stochastic_depth": stochastic_depth,
            "groups": {
                "fixed_seed": fixed_summary,
                "varying_seed": varying_summary,
            },
            "run_count_including_warmup": 11,
            "comparison_count": len(fixed_pairs) + len(varying_pairs),
            "missing_gradient_set": list(next(iter(missing_sets))),
            "integrity_pass": integrity_pass,
        }
        _write_json(raw / f"{label.lower().replace('-', '_')}_summary.json",
                    mode_summaries[label])
        del criterion, model, batch, batches, loader
        gc.collect()
        torch.cuda.empty_cache()

    if total_runs != 33:
        raise RuntimeError(f"STOP-B B-RAND expected 33 runs, got {total_runs}")
    classification = classify_stop_b_randomness(mode_summaries)
    summary = {
        "schema": "fl_v3.s10.stop_b_rand_summary.v1",
        "status": "PASS",
        "integrity_gate": "PASS",
        "total_forward_backward_runs": total_runs,
        "physical_microbatch": 4,
        "mode_summaries": mode_summaries,
        "classification": classification,
        "total_wall_seconds": time.perf_counter() - total_started,
        "optimizer_constructed": False,
        "optimizer_updates": 0,
        "official_evaluation_executed": False,
        "numerical_equality_is_acceptance_gate": False,
        "allowed_interpretation": (
            "bounded operational candidate-source triage for STOP-B redesign"
        ),
        "forbidden_interpretation": [
            "kernel or module causality",
            "large-gradient explanation",
            "convergence",
            "capability",
            "production recipe",
            "architecture acceptance",
            "automatic localization continuation",
        ],
    }
    _write_json(output / "summary.json", summary)
    _write_json(output / "artifact_sha256s.json", _artifact_manifest(output))


def main() -> None:
    args = _parse_args()
    output = Path(args.output_dir).resolve()
    if output.exists():
        raise RuntimeError(f"STOP-B B-RAND output must be fresh: {output}")
    output.mkdir(parents=True)
    raw = output / "raw"
    raw.mkdir()
    try:
        _execute(args, output, raw)
    except BaseException as exc:
        _write_json(output / "failure_summary.json", {
            "schema": "fl_v3.s10.stop_b_rand_failure.v1",
            "exception_type": type(exc).__name__,
            "exception_message": str(exc),
        })
        _write_json(output / "artifact_sha256s.json", _artifact_manifest(output))
        raise


if __name__ == "__main__":
    main()
