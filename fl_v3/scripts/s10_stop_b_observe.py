#!/usr/bin/env python3
"""Execute the approved S10 STOP-B main observation allocation.

The script freezes the deterministic D_low panel before model construction, then
runs fixed-W0 FP32 and global-FP16/SECOND-FP32 observations without constructing
an optimizer or advancing any training state.
"""
from __future__ import annotations

import argparse
import copy
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
    build_stop_b_panel,
    load_frozen_split_role,
    validate_stop_b_panel,
    write_canonical_json,
)
from fl_v3.training.loop import _move_to_device
from fl_v3.training.precision_diagnostics import runtime_rng_state_sha256
from fl_v3.training.s10_observation import (
    LIDAR_BACKWARD_CHAIN,
    MAIN_BOUNDARIES,
    STOP_B_SCHEMA,
    StopBObservationRecorder,
    attribute_term_gradients,
    loss_term_snapshot,
    module_state_sha256,
    parameter_gradient_snapshot,
    parameter_gradients_sha256,
    recompose_from_sample_terms,
    strict_json_value,
    tensor_tree_sha256,
    zero_model_gradients,
)
from fl_v3.training.tasks import NuScenesDetectionTask
from fl_v3.utils.runtime import (
    enforce_determinism,
    precision_autocast_context,
    seed_everything,
    verify_runtime_dependency_identity,
)


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fp32-config", required=True)
    parser.add_argument("--fp16-config", required=True)
    parser.add_argument("--split-manifest", required=True)
    parser.add_argument("--split-sha256", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--source-sha", required=True)
    parser.add_argument("--source-tree", required=True)
    parser.add_argument("--loss-scale", type=float, default=64.0)
    return parser.parse_args()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(strict_json_value(value), sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _append_jsonl(path: Path, value: Any) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(strict_json_value(value), sort_keys=True, allow_nan=False) + "\n")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _runtime_assertions(loss_scale: float) -> dict[str, Any]:
    if platform.machine() != "aarch64":
        raise RuntimeError("STOP-B requires an aarch64 Arrhenius compute node")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("STOP-B requires exactly one visible CUDA device")
    if not math.isfinite(loss_scale) or loss_scale != 64.0:
        raise RuntimeError("STOP-B accepted FP16 loss scale is exactly 64")
    device = torch.device("cuda", 0)
    properties = torch.cuda.get_device_properties(device)
    if "GH200" not in properties.name.upper():
        raise RuntimeError(f"STOP-B expected GH200, got {properties.name!r}")
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


def _assert_config_pair(fp32, fp16) -> None:
    if fp32.precision != "fp32" or fp32.sparse_conv_precision != "fp32":
        raise RuntimeError("STOP-B FP32 config partition drift")
    if fp16.precision != "fp16" or fp16.sparse_conv_precision != "fp32":
        raise RuntimeError("STOP-B AMP config must be global FP16 plus SECOND FP32")
    left, right = fp32.as_dict(), fp16.as_dict()
    left.pop("precision")
    right.pop("precision")
    if left != right:
        raise RuntimeError("STOP-B configs may differ only in global precision")
    model = fp32.data["model"]
    required = {
        "mode": "fusion",
        "camera_arch": "swin_t_stride8",
        "camera_pretrained": False,
        "camera_activation_checkpoint": False,
        "lidar_arch": "second_075",
        "fusion_arch": "conv_fuser_256",
        "head_arch": "centerhead_multitask",
    }
    if dict(model) != required:
        raise RuntimeError(f"STOP-B current graph config drift: {dict(model)}")
    training = fp32.data["training"]
    if (
        int(training["micro_batch_size"]) != 4
        or int(training["world_size"]) != 1
        or int(training["accumulation_steps"]) != 1
        or int(training["effective_global_batch"]) != 4
        or int(training["seed"]) != 0
        or str(training["sampling"]) != "uniform"
        or training["ema_decay"] is not None
    ):
        raise RuntimeError("STOP-B B4/W0/uniform/no-EMA config drift")


def _batch_tokens(batch: Mapping[str, Any], expected: Sequence[str]) -> None:
    actual = [str(value) for value in batch["sample_token"]]
    if actual != list(expected):
        raise RuntimeError(f"STOP-B batch token order drift: expected={list(expected)}, actual={actual}")
    if int(batch["batch_size"]) != 4:
        raise RuntimeError("STOP-B scientific observations require physical B4")


def _voxel_snapshot(model) -> dict[str, Any]:
    encoder = model.lidar_encoder
    meta = copy.deepcopy(encoder.last_sparse_meta)
    stats = encoder.last_voxel_stats
    if not isinstance(meta, dict) or not torch.is_tensor(stats):
        raise RuntimeError("STOP-B sparse runtime metadata is missing")
    return {
        "metadata": meta,
        "fields": list(meta["voxel_stat_fields"]),
        "per_sample": stats.detach().cpu().tolist(),
    }


def _run_once(
    model,
    criterion,
    batch_cpu,
    *,
    device: torch.device,
    precision: str,
    scale: float,
    diagnostic: bool,
    seed: int,
    term_attribution: bool = False,
) -> dict[str, Any]:
    seed_everything(seed)
    zero_model_gradients(model)
    batch = _move_to_device(batch_cpu, device)
    recorder = StopBObservationRecorder() if diagnostic else None
    started = time.perf_counter()
    if diagnostic:
        model_context = model.capture_s10_observation(recorder)
        loss_context = criterion.capture_s10_terms()
    else:
        from contextlib import nullcontext

        model_context = nullcontext()
        loss_context = nullcontext()
    with model_context:
        with loss_context:
            with precision_autocast_context(precision, device):
                output = model(batch)
                loss = criterion(output, batch)
            output_sha = tensor_tree_sha256(output)
            if term_attribution:
                if precision != "fp32" or not diagnostic:
                    raise RuntimeError("term attribution is FP32 diagnostic-only")
                bundle = criterion.s10_term_bundle()
                attribution = attribute_term_gradients(bundle, recorder)
                record = {
                    "loss": float(loss.detach().item()),
                    "output_sha256": output_sha,
                    "loss_terms": loss_term_snapshot(bundle),
                    "forward_observation": recorder.forward_snapshot(),
                    "term_attribution": attribution,
                    "voxel": _voxel_snapshot(model),
                }
            else:
                scaled = loss * float(scale)
                scaled.backward()
                record = {
                    "loss": float(loss.detach().item()),
                    "scaled_loss": float(scaled.detach().item()),
                    "output_sha256": output_sha,
                    "parameter_gradients_sha256": parameter_gradients_sha256(model),
                }
                if diagnostic:
                    bundle = criterion.s10_term_bundle()
                    record.update({
                        "loss_terms": loss_term_snapshot(bundle),
                        "forward_observation": recorder.forward_snapshot(),
                        "boundary_gradients": recorder.gradient_snapshot(scale_divisor=scale),
                        "parameter_gradients": parameter_gradient_snapshot(
                            model, scale_divisor=scale
                        ),
                        "voxel": _voxel_snapshot(model),
                    })
    torch.cuda.synchronize(device)
    record.update({
        "schema": STOP_B_SCHEMA,
        "precision": precision,
        "sparse_conv_precision": "fp32",
        "diagnostic": bool(diagnostic),
        "optimizer_constructed": False,
        "optimizer_update": False,
        "scheduler_update": False,
        "ema_update": False,
        "grad_scaler_update": False,
        "fixed_loss_scale": float(scale),
        "seed": int(seed),
        "sample_tokens": [str(value) for value in batch_cpu["sample_token"]],
        "rng_state_sha256_after": runtime_rng_state_sha256(),
        "wall_seconds": time.perf_counter() - started,
    })
    zero_model_gradients(model)
    criterion.last_s10_terms = {}
    for child in criterion.losses:
        child.last_s10_terms = {}
        child._s10_focal = {}
    del output, loss, batch, recorder
    return record


def _parity_cells(model, criterion, task, run_config, panel, *, device, precision, scale):
    tokens = [*panel["batches_b4"]["P_core"][0], *panel["batches_b4"]["P_term"][0]]
    loader = task.fixed_train_subset_loader(run_config, tokens)
    records = []
    expected_batches = [panel["batches_b4"]["P_core"][0], panel["batches_b4"]["P_term"][0]]
    for index, (batch, expected) in enumerate(zip(loader, expected_batches, strict=True)):
        _batch_tokens(batch, expected)
        seed = 10000 + index
        before_state = module_state_sha256(model)
        off = _run_once(
            model, criterion, batch, device=device, precision=precision,
            scale=scale, diagnostic=False, seed=seed,
        )
        after_off_state = module_state_sha256(model)
        on = _run_once(
            model, criterion, batch, device=device, precision=precision,
            scale=scale, diagnostic=True, seed=seed,
        )
        after_on_state = module_state_sha256(model)
        parity = {
            "schema": STOP_B_SCHEMA,
            "cell": f"B-PARITY-{precision.upper()}",
            "batch_index": index,
            "sample_tokens": list(expected),
            "output_sha256_equal": off["output_sha256"] == on["output_sha256"],
            "parameter_gradients_sha256_equal": (
                off["parameter_gradients_sha256"] == on["parameter_gradients_sha256"]
            ),
            "loss_exact_equal": off["loss"] == on["loss"],
            "rng_state_sha256_equal": (
                off["rng_state_sha256_after"] == on["rng_state_sha256_after"]
            ),
            "model_state_before": before_state,
            "model_state_after_off": after_off_state,
            "model_state_after_on": after_on_state,
            "model_state_equal": before_state == after_off_state == after_on_state,
            "off": off,
            "on": on,
        }
        parity["status"] = "PASS" if all((
            parity["output_sha256_equal"],
            parity["parameter_gradients_sha256_equal"],
            parity["loss_exact_equal"],
            parity["rng_state_sha256_equal"],
            parity["model_state_equal"],
        )) else "FAIL"
        if parity["status"] != "PASS":
            raise RuntimeError(f"STOP-B diagnostic parity failed for {precision} batch {index}")
        records.append(parity)
    if len(records) != 2:
        raise RuntimeError("STOP-B parity loader did not emit exactly two B4 batches")
    return records


def _broad_cells(model, criterion, task, run_config, panel, *, device, precision, scale):
    tokens = [*panel["tokens"]["P_core"], *panel["tokens"]["P_term"]]
    expected = [*panel["batches_b4"]["P_core"], *panel["batches_b4"]["P_term"]]
    loader = task.fixed_train_subset_loader(run_config, tokens)
    records = []
    for index, (batch, expected_tokens) in enumerate(zip(loader, expected, strict=True)):
        _batch_tokens(batch, expected_tokens)
        record = _run_once(
            model, criterion, batch, device=device, precision=precision,
            scale=scale, diagnostic=True, seed=20000 + index,
        )
        record.update({
            "cell": f"B-BROAD-{precision.upper()}",
            "batch_index": index,
            "stratum": "P_core" if index < 12 else "P_term",
        })
        records.append(record)
    if len(records) != 16:
        raise RuntimeError("STOP-B broad loader did not emit exactly sixteen B4 batches")
    return records


def _term_cells(model, criterion, task, run_config, panel, *, device):
    tokens = panel["tokens"]["P_term"]
    expected = panel["batches_b4"]["P_term"]
    loader = task.fixed_train_subset_loader(run_config, tokens)
    records = []
    for index, (batch, expected_tokens) in enumerate(zip(loader, expected, strict=True)):
        _batch_tokens(batch, expected_tokens)
        record = _run_once(
            model, criterion, batch, device=device, precision="fp32",
            scale=1.0, diagnostic=True, seed=30000 + index, term_attribution=True,
        )
        record.update({"cell": "B-TERM-FP32", "batch_index": index, "stratum": "P_term"})
        records.append(record)
    if len(records) != 4:
        raise RuntimeError("STOP-B term loader did not emit exactly four B4 batches")
    return records


def _aggregation_cell(model, criterion, task, run_config, panel, *, device):
    expected = panel["batches_b4"]["P_core"][0]
    loader = task.fixed_train_subset_loader(run_config, expected)
    batch_cpu = next(iter(loader))
    _batch_tokens(batch_cpu, expected)
    seed_everything(40000)
    zero_model_gradients(model)
    batch = _move_to_device(batch_cpu, device)
    recorder = StopBObservationRecorder()
    b1_criterion = copy.deepcopy(criterion)
    with model.capture_s10_observation(recorder):
        with criterion.capture_s10_terms():
            output = model(batch)
            actual = criterion(output, batch)
            bundle = criterion.s10_term_bundle()
            recomposed_from_b4_terms, _ = recompose_from_sample_terms(bundle)

            # This is the sole approved B1 diagnostic.  The detector still runs
            # exactly once at physical B4; each sample's prediction/target pair
            # is then evaluated by a fresh criterion at logical B1.  Recombining
            # the four raw numerator/denominator records tests the real B1 target
            # and loss path without making B1 a model-capacity or cost baseline.
            b1_losses = []
            b1_bundles = []
            for sample in range(4):
                sample_output = [
                    {name: value[sample : sample + 1] for name, value in task_output.items()}
                    for task_output in output
                ]
                sample_batch = {
                    "gt_boxes": [batch["gt_boxes"][sample]],
                    "gt_labels": [batch["gt_labels"][sample]],
                    "gt_velocity": [batch["gt_velocity"][sample]],
                }
                with b1_criterion.capture_s10_terms():
                    b1_loss = b1_criterion(sample_output, sample_batch)
                b1_losses.append(b1_loss)
                b1_bundles.append(b1_criterion.s10_term_bundle())

            recomposed_tasks = []
            for task_index in range(6):
                task_records = [item["tasks"][task_index] for item in b1_bundles]
                hm_numerator = torch.stack([
                    item["tensors"]["hm_sample_numerators"][0]
                    for item in task_records
                ]).sum()
                hm_denominator = torch.stack([
                    item["tensors"]["hm_sample_denominators"][0]
                    for item in task_records
                ]).sum().clamp_min(1.0)
                reg_numerator = torch.stack([
                    item["tensors"]["reg_sample_numerators"][0]
                    for item in task_records
                ]).sum()
                reg_denominator_raw = torch.stack([
                    item["tensors"]["reg_sample_denominators"][0]
                    for item in task_records
                ]).sum()
                if bool(task_records[0]["metadata"]["class_weighted_regression"]):
                    reg_denominator = reg_denominator_raw.clamp_min(1e-6)
                else:
                    reg_denominator = reg_denominator_raw.clamp_min(1.0)
                recomposed_tasks.append(
                    hm_numerator / hm_denominator
                    + float(task_records[0]["metadata"]["reg_weight"])
                    * reg_numerator / reg_denominator
                )
            recomposed_from_four_b1 = torch.stack(recomposed_tasks).sum()
            tensors = recorder.tensors_in_order()
            actual_grads = torch.autograd.grad(actual, tensors, retain_graph=True, allow_unused=True)
            recomposed_grads = torch.autograd.grad(
                recomposed_from_four_b1, tensors, retain_graph=True, allow_unused=True
            )
            if any(value is None for value in actual_grads) or any(value is None for value in recomposed_grads):
                raise RuntimeError("B4/B1 recomposition has a missing boundary gradient")
            gradient_checks = {}
            for name, left, right in zip(
                MAIN_BOUNDARIES, actual_grads, recomposed_grads, strict=True
            ):
                delta = left.detach() - right.detach()
                ref = math.sqrt(float(torch.square(left.detach()).sum(dtype=torch.float64).item()))
                err = math.sqrt(float(torch.square(delta).sum(dtype=torch.float64).item()))
                gradient_checks[name] = {
                    "allclose_rtol_1e-5_atol_1e-7": bool(
                        torch.allclose(left, right, rtol=1e-5, atol=1e-7)
                    ),
                    "relative_l2_error": err / ref if ref > 0.0 else err,
                }
            naive_mean = torch.stack(b1_losses).mean()
            record = {
                "schema": STOP_B_SCHEMA,
                "cell": "B-AGG-FP32",
                "sample_tokens": list(expected),
                "full_detector_B1_forwards": 0,
                "B4_detector_forwards": 1,
                "B1_criterion_evaluations": 4,
                "actual_B4_loss": float(actual.detach().item()),
                "recomposed_from_B4_sample_terms": float(
                    recomposed_from_b4_terms.detach().item()
                ),
                "recomposed_from_four_actual_B1_raw_terms": float(
                    recomposed_from_four_b1.detach().item()
                ),
                "four_actual_B1_losses": [
                    float(value.detach().item()) for value in b1_losses
                ],
                "loss_allclose_rtol_1e-5_atol_1e-7": bool(
                    torch.allclose(actual, recomposed_from_b4_terms, rtol=1e-5, atol=1e-7)
                    and torch.allclose(
                        actual, recomposed_from_four_b1, rtol=1e-5, atol=1e-7
                    )
                ),
                "naive_mean_of_four_independently_normalized_B1_losses": float(
                    naive_mean.detach().item()
                ),
                "naive_mean_minus_actual_B4": float((naive_mean - actual).detach().item()),
                "boundary_gradient_checks": gradient_checks,
                "loss_terms": loss_term_snapshot(bundle),
                "optimizer_constructed": False,
                "optimizer_update": False,
            }
    if not record["loss_allclose_rtol_1e-5_atol_1e-7"] or not all(
        value["allclose_rtol_1e-5_atol_1e-7"]
        for value in record["boundary_gradient_checks"].values()
    ):
        raise RuntimeError("STOP-B B4/B1 raw-term reconstruction failed")
    zero_model_gradients(model)
    criterion.last_s10_terms = {}
    b1_criterion.last_s10_terms = {}
    return record


def _median(values):
    return float(statistics.median(values)) if values else None


def _localization_summary(broad_by_precision, term_records):
    interval_order = [
        f"{left}->{right}" for left, right in zip(
            LIDAR_BACKWARD_CHAIN[:-1], LIDAR_BACKWARD_CHAIN[1:], strict=True
        )
    ]
    precision_summary = {}
    qualifying = {}
    for precision, records in broad_by_precision.items():
        core = [record for record in records if record["stratum"] == "P_core"]
        ratios = {name: [] for name in interval_order}
        top_counts = {name: 0 for name in interval_order}
        for record in core:
            gradients = record["boundary_gradients"]
            per_interval_per_sample = {name: [] for name in interval_order}
            for left, right in zip(
                LIDAR_BACKWARD_CHAIN[:-1], LIDAR_BACKWARD_CHAIN[1:], strict=True
            ):
                key = f"{left}->{right}"
                left_samples = gradients[left]["true_unscaled_per_sample"]
                right_samples = gradients[right]["true_unscaled_per_sample"]
                for left_stats, right_stats in zip(left_samples, right_samples, strict=True):
                    denominator = float(left_stats["stable_finite_rms"])
                    numerator = float(right_stats["stable_finite_rms"])
                    if denominator > 0.0:
                        ratio = numerator / denominator
                    elif numerator > 0.0:
                        ratio = float("inf")
                    else:
                        ratio = 1.0
                    ratios[key].append(ratio)
                    per_interval_per_sample[key].append(ratio)
            for sample in range(4):
                winner = max(
                    interval_order,
                    key=lambda name: per_interval_per_sample[name][sample],
                )
                top_counts[winner] += 1
        metrics = {}
        accepted = []
        for name in interval_order:
            values = ratios[name]
            median = _median(values)
            count_gt2 = sum(value > 2.0 for value in values)
            metrics[name] = {
                "sample_count": len(values),
                "median_upstream_over_downstream_grad_rms": median,
                "count_ratio_gt_2": count_gt2,
                "top_interval_count": top_counts[name],
                "gate_median_ge_4": bool(median is not None and median >= 4.0),
                "gate_36_of_48_ratio_gt_2": count_gt2 >= 36,
                "gate_36_of_48_top": top_counts[name] >= 36,
            }
            if all((
                metrics[name]["gate_median_ge_4"],
                metrics[name]["gate_36_of_48_ratio_gt_2"],
                metrics[name]["gate_36_of_48_top"],
            )):
                accepted.append(name)
        precision_summary[precision] = {"intervals": metrics, "qualifying_intervals": accepted}
        qualifying[precision] = set(accepted)

    shared = []
    for name in interval_order:
        if name not in qualifying.get("fp32", set()) or name not in qualifying.get("fp16", set()):
            continue
        fp32_median = precision_summary["fp32"]["intervals"][name][
            "median_upstream_over_downstream_grad_rms"
        ]
        fp16_median = precision_summary["fp16"]["intervals"][name][
            "median_upstream_over_downstream_grad_rms"
        ]
        if math.isinf(fp32_median) and math.isinf(fp16_median):
            ratio = 1.0
        elif math.isinf(fp32_median) or math.isinf(fp16_median):
            ratio = float("inf")
        else:
            ratio = max(fp32_median, fp16_median) / min(fp32_median, fp16_median)
        if ratio <= 2.0:
            shared.append(name)
    primary = shared[0] if shared else None

    dominant = None
    term_summary = {}
    if primary is not None:
        upstream = primary.split("->", 1)[1]
        top_names = []
        top_shares = []
        for record in term_records:
            sources = record["term_attribution"]["sources"]
            winner = max(sources, key=lambda name: sources[name][upstream]["projection_share"])
            top_names.append(winner)
            top_shares.append(float(sources[winner][upstream]["projection_share"]))
        counts = {name: top_names.count(name) for name in sorted(set(top_names))}
        winner = max(counts, key=lambda name: (counts[name], name)) if counts else None
        winner_shares = [
            share for name, share in zip(top_names, top_shares, strict=True) if name == winner
        ]
        median_share = _median(winner_shares)
        if winner is not None and counts[winner] >= 3 and median_share is not None and median_share >= 0.5:
            dominant = winner
        term_summary = {
            "boundary": upstream,
            "top_source_by_batch": top_names,
            "top_projection_share_by_batch": top_shares,
            "top_source_counts": counts,
            "candidate_dominant_source": winner,
            "candidate_median_projection_share": median_share,
            "dominant_source_gate": dominant,
        }

    residual_reconstruction_pass = all(
        all(
            item["allclose_rtol_1e-5_atol_1e-7"]
            for item in record["term_attribution"]["aggregate_gradient_reconstruction"].values()
        )
        for record in term_records
    )
    verdict = "LOCALIZED" if primary is not None and residual_reconstruction_pass else "INCONCLUSIVE"
    refine_intervals = {
        "fusion.lidar_input->second.output",
        "second.stage4->second.down3",
        "second.stage3->second.down2",
        "second.stage2->second.down1",
        "second.stage1->second.stem",
    }
    return {
        "verdict": verdict,
        "primary_interval": primary,
        "shared_qualifying_intervals": shared,
        "precision": precision_summary,
        "term_attribution": term_summary,
        "dominant_source": dominant,
        "term_gradient_reconstruction_pass": residual_reconstruction_pass,
        "refinement_recommended": bool(primary in refine_intervals),
        "interpretation": (
            "operational adjacent-boundary localization only; mechanism causality requires STOP-C counterfactual"
            if verdict == "LOCALIZED"
            else "predeclared localization gate not met; do not expand the panel or hypothesis family"
        ),
    }


def _artifact_manifest(root: Path) -> dict[str, Any]:
    files = {}
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = str(path.relative_to(root))
        if relative == "artifact_sha256s.json":
            continue
        files[relative] = {"sha256": _sha256_file(path), "bytes": path.stat().st_size}
    return {"schema": "fl_v3.s10.stop_b_artifacts.v1", "files": files}


def main() -> None:
    args = _parse_args()
    output = Path(args.output_dir).resolve()
    if output.exists():
        raise RuntimeError(f"STOP-B output must be fresh: {output}")
    output.mkdir(parents=True)
    raw = output / "raw"
    raw.mkdir()
    runtime = _runtime_assertions(args.loss_scale)
    fp32 = load_resolved_config(args.fp32_config)
    fp16 = load_resolved_config(args.fp16_config)
    _assert_config_pair(fp32, fp16)
    verify_physical_data_identities(fp16)
    runtime_dependencies = verify_runtime_dependency_identity(fp16.to_run_config())
    binding = load_frozen_split_role(
        args.split_manifest,
        expected_manifest_sha256=args.split_sha256,
        role="D_low",
        expected_source_identities=_expected_split_sources(fp16),
    )

    # PRE-MODEL HARD PHASE: no model has been constructed above this line.
    task = NuScenesDetectionTask()
    train_info, train_meta = task.load_production_train_info(fp16.to_run_config())
    panel = build_stop_b_panel(binding, train_info)
    validate_stop_b_panel(panel, binding)
    panel_path = output / "panel_manifest.json"
    write_canonical_json(panel_path, panel)
    panel_file_sha = _sha256_file(panel_path)
    panel_path.chmod(0o444)
    pre_model = {
        "schema": "fl_v3.s10.stop_b_pre_model_freeze.v1",
        "model_constructed_before_panel_freeze": False,
        "model_output_observed_before_panel_freeze": False,
        "split_binding": binding.identity(),
        "panel_content_sha256": panel["panel_sha256"],
        "panel_file_sha256": panel_file_sha,
        "train_cache_meta": train_meta,
    }
    _write_json(output / "pre_model_freeze.json", pre_model)

    identity = {
        "schema": "fl_v3.s10.stop_b_execution_identity.v1",
        "source_sha": args.source_sha,
        "source_tree": args.source_tree,
        "fp32_config_path": str(Path(args.fp32_config).resolve()),
        "fp32_config_sha256": fp32.sha256,
        "fp16_config_path": str(Path(args.fp16_config).resolve()),
        "fp16_config_sha256": fp16.sha256,
        "split_manifest_sha256": args.split_sha256,
        "panel_content_sha256": panel["panel_sha256"],
        "panel_file_sha256": panel_file_sha,
        "fixed_amp_loss_scale": args.loss_scale,
        "runtime": runtime,
        "runtime_dependencies": runtime_dependencies,
        "optimizer_constructed": False,
        "official_evaluation_executed": False,
        "D_select_observed": False,
        "D_audit_observed": False,
        "official_val_observed": False,
    }
    _write_json(output / "execution_identity.json", identity)

    # Freeze one CPU FP32 W0 state and load the exact bytes into both precision cells.
    enforce_determinism(strict=True, precision="fp32")
    seed_everything(0)
    base_model = task.build_model(fp32.to_run_config())
    w0_hash = module_state_sha256(base_model)
    w0_state = {
        name: value.detach().cpu().clone() for name, value in base_model.state_dict().items()
    }
    del base_model
    gc.collect()
    _write_json(output / "w0_identity.json", {
        "schema": "fl_v3.s10.stop_b_w0.v1",
        "seed": 0,
        "initialization": "all_scratch_A0_current_graph",
        "camera_pretrained": False,
        "checkpoint_loaded": False,
        "state_dict_sha256": w0_hash,
    })

    device = torch.device("cuda", 0)
    parity_records = []
    broad_by_precision = {}
    term_records = []
    aggregation = None
    cell_timing = {}
    total_started = time.perf_counter()
    for precision, config in (("fp32", fp32), ("fp16", fp16)):
        enforce_determinism(strict=(precision == "fp32"), precision=precision)
        seed_everything(0)
        model = task.build_model(config.to_run_config())
        model.load_state_dict(w0_state, strict=True)
        model.to(device)
        model.train()
        criterion = task.build_criterion(config.to_run_config())
        if module_state_sha256(model) != w0_hash:
            raise RuntimeError(f"{precision} model does not match frozen W0")
        torch.cuda.reset_peak_memory_stats(device)
        phase_start = time.perf_counter()
        parity = _parity_cells(
            model, criterion, task, config.to_run_config(), panel,
            device=device, precision=precision, scale=(1.0 if precision == "fp32" else args.loss_scale),
        )
        parity_records.extend(parity)
        for record in parity:
            _append_jsonl(raw / "parity.jsonl", record)
        broad = _broad_cells(
            model, criterion, task, config.to_run_config(), panel,
            device=device, precision=precision, scale=(1.0 if precision == "fp32" else args.loss_scale),
        )
        broad_by_precision[precision] = broad
        for record in broad:
            _append_jsonl(raw / "broad.jsonl", record)
        if precision == "fp32":
            term_records = _term_cells(
                model, criterion, task, config.to_run_config(), panel, device=device
            )
            for record in term_records:
                _append_jsonl(raw / "terms.jsonl", record)
            aggregation = _aggregation_cell(
                model, criterion, task, config.to_run_config(), panel, device=device
            )
            _write_json(raw / "aggregation.json", aggregation)
        final_state = module_state_sha256(model)
        if final_state != w0_hash:
            raise RuntimeError(f"STOP-B {precision} mutated model state")
        torch.cuda.synchronize(device)
        cell_timing[precision] = {
            "wall_seconds": time.perf_counter() - phase_start,
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
            "final_state_dict_sha256": final_state,
        }
        del criterion, model
        gc.collect()
        torch.cuda.empty_cache()

    if aggregation is None:
        raise RuntimeError("STOP-B aggregation cell is missing")
    localization = _localization_summary(broad_by_precision, term_records)
    parity_pass = all(record["status"] == "PASS" for record in parity_records)
    summary = {
        "schema": "fl_v3.s10.stop_b_summary.v1",
        "status": "PASS" if parity_pass and localization["term_gradient_reconstruction_pass"] else "FAIL",
        "stop_b_verdict": localization["verdict"],
        "split_binding": binding.identity(),
        "panel_content_sha256": panel["panel_sha256"],
        "panel_file_sha256": panel_file_sha,
        "W0_state_dict_sha256": w0_hash,
        "parity_pass": parity_pass,
        "aggregation_pass": bool(
            aggregation["loss_allclose_rtol_1e-5_atol_1e-7"]
            and all(
                value["allclose_rtol_1e-5_atol_1e-7"]
                for value in aggregation["boundary_gradient_checks"].values()
            )
        ),
        "localization": localization,
        "cell_timing": cell_timing,
        "total_wall_seconds": time.perf_counter() - total_started,
        "optimizer_constructed": False,
        "optimizer_updates": 0,
        "full_detector_B1_forwards": 0,
        "full_detector_forwards": 45,
        "broad_samples_per_precision": 64,
        "term_samples": 16,
        "official_evaluation_executed": False,
        "allowed_interpretation": "current-W0 numerical localization or honest INCONCLUSIVE only",
        "forbidden_interpretation": [
            "convergence", "capability", "production recipe", "official-val performance",
            "causal architecture proof", "trained-checkpoint numerical health",
        ],
    }
    if not summary["aggregation_pass"]:
        summary["status"] = "FAIL"
    _write_json(output / "summary.json", summary)
    _write_json(output / "artifact_sha256s.json", _artifact_manifest(output))
    if summary["status"] != "PASS":
        raise RuntimeError("STOP-B hard gate failed")


if __name__ == "__main__":
    main()
