#!/usr/bin/env python3
"""O-131 STOP-C0 integrated training-health entry rung.

This is one bounded scientific runner, not a recipe search.  It trains exact
manifest roles with the unchanged engineering baseline while sampling true
unscaled gradients and realized optimizer updates at predeclared windows.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import copy
import gc
import hashlib
import json
import math
from pathlib import Path
import sys
import time
from typing import Any, Iterator

sys.path.insert(0, "fl_v3/src")
sys.path.insert(0, "fl_v3/scripts")

import torch

from centralized_train import BoundedOperatorProfiler, _build_optimizer
from fl_v3.config import resolve_config, verify_physical_data_identities
from fl_v3.data.nuscenes.s10_binding import load_frozen_split_role
from fl_v3.eval.detection_eval import build_results_dict, decode_eval_set
from fl_v3.eval.subset_detection_eval import run_internal_manifest_eval, write_strict_json
from fl_v3.training.checkpoint import save_checkpoint
from fl_v3.training.loop import train_one_epoch
from fl_v3.training.precision_diagnostics import (
    PrecisionDiagnosticsIdentity,
    PrecisionWindowDiagnostics,
)
from fl_v3.training.runtime_state import TrainingState
from fl_v3.utils.runtime import (
    enforce_determinism,
    make_grad_scaler,
    seed_everything,
    verify_runtime_dependency_identity,
)


SCHEMA = "fl_v3.s10.stop_c0_health.v1"
EXPECTED_SPLIT_SHA256 = "7e84a1d4f4a099c31a1d5194f17ba77d278fdc92f52462e72517b494dc5223a8"
EXPECTED_SWINT_SHA256 = "704ceda373461b0a224fcdddd75cd2a5e9f8064512ed47adbddef7f343fd147b"
FULL_BOUNDARIES = (64, 384, 768, 1152, 1538)
FULL_DIAGNOSTICS = (1, 4, 16, 64, 256, 768, 1538)
SHORT_BOUNDARIES = (16, 64)
SHORT_DIAGNOSTICS = (1, 4, 16, 64)
PROFILE_SPEC = {
    "wait_attempted_windows": 16,
    "warmup_attempted_windows": 2,
    "active_attempted_windows": 10,
    "record_shapes": False,
    "profile_memory": True,
    "row_limit": 100,
}

CELL_SPECS = (
    {
        "id": "C0-F-A1",
        "mode": "fusion",
        "camera_pretrained": True,
        "attempted_windows": 1538,
        "boundaries": FULL_BOUNDARIES,
        "diagnostics": FULL_DIAGNOSTICS,
        "evaluate": True,
        "operator_profile": True,
    },
    {
        "id": "C0-L-A0",
        "mode": "lidar_only",
        "camera_pretrained": None,
        "attempted_windows": 1538,
        "boundaries": FULL_BOUNDARIES,
        "diagnostics": FULL_DIAGNOSTICS,
        "evaluate": True,
        "operator_profile": False,
    },
    {
        "id": "C0-F-A0-P64",
        "mode": "fusion",
        "camera_pretrained": False,
        "attempted_windows": 64,
        "boundaries": SHORT_BOUNDARIES,
        "diagnostics": SHORT_DIAGNOSTICS,
        "evaluate": False,
        "operator_profile": False,
    },
)


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _source_identities(config) -> dict[str, Any]:
    identities = config.data_identities
    return {
        "version": "v1.0-trainval",
        "n_sweeps": 10,
        "train_cache_logical_sha256": identities["train_cache_logical_sha256"],
        "train_cache_pickle_sha256": identities["train_cache_pickle_sha256"],
        "train_cache_sidecar_sha256": identities["train_cache_sidecar_sha256"],
        "val_cache_logical_sha256": identities["val_cache_logical_sha256"],
        "val_cache_pickle_sha256": identities["val_cache_pickle_sha256"],
        "val_cache_sidecar_sha256": identities["val_cache_sidecar_sha256"],
        "zip_manifest_logical_sha256": identities["zip_manifest_logical_sha256"],
        "zip_manifest_file_sha256": identities["zip_manifest_file_sha256"],
        "detection_config_sha256": (
            "217f96cca4e80f790c4674ef72257a6863ee9a85b0ce185bc56488afc32c7a0b"
        ),
    }


def _resolved_for_cell(base: dict[str, Any], spec: dict[str, Any]):
    raw = copy.deepcopy(base)
    raw["training"]["max_optimizer_steps"] = int(spec["attempted_windows"])
    if spec["mode"] == "lidar_only":
        raw["model"].update({
            "mode": "lidar_only",
            "camera_arch": "none",
            "camera_pretrained": None,
            "camera_activation_checkpoint": False,
            "lidar_arch": "second_075",
            "fusion_arch": "none",
        })
    else:
        raw["model"].update({
            "mode": "fusion",
            "camera_arch": "swin_t_stride8",
            "camera_pretrained": bool(spec["camera_pretrained"]),
            "camera_activation_checkpoint": False,
            "lidar_arch": "second_075",
            "fusion_arch": "conv_fuser_256",
        })
    return resolve_config(raw)


class _ExactChunk:
    """Expose exactly N batches from one shared epoch iterator."""

    def __init__(self, source: Iterator[Any], count: int) -> None:
        self.source = source
        self.remaining = int(count)

    def __iter__(self):
        return self

    def __next__(self):
        if self.remaining == 0:
            raise StopIteration
        try:
            value = next(self.source)
        except StopIteration as exc:
            raise RuntimeError("D_low loader exhausted before its declared chunk") from exc
        self.remaining -= 1
        return value


def _dropped_tokens(tokens: tuple[str, ...], seed: int, batch_size: int) -> list[str]:
    generator = torch.Generator().manual_seed(int(seed))
    order = torch.randperm(len(tokens), generator=generator).tolist()
    remainder = len(tokens) % int(batch_size)
    return [tokens[index] for index in order[-remainder:]] if remainder else []


def _close_loader(loader: object) -> None:
    iterator = getattr(loader, "_iterator", None)
    shutdown = getattr(iterator, "_shutdown_workers", None)
    if callable(shutdown):
        shutdown()
    dataset = getattr(loader, "dataset", None)
    base = getattr(dataset, "dataset", dataset)
    close = getattr(base, "close", None)
    if callable(close):
        close()


def _required_prefixes(mode: str) -> tuple[str, ...]:
    lidar = (
        "lidar_encoder.backbone.stem",
        "lidar_encoder.backbone.stage1",
        "lidar_encoder.to_bev",
    )
    common = ("bev_neck", "head")
    if mode == "lidar_only":
        return lidar + common
    return (
        "camera_backbone",
        "camera_neck",
        "view_transform",
        *lidar,
        "fusion",
        *common,
    )


def _cell_health(
    spec: dict[str, Any],
    chunks: list[dict[str, Any]],
    records: tuple[dict[str, Any], ...],
    terminal: dict[str, int],
    evaluation: dict[str, Any] | None,
) -> dict[str, Any]:
    first_state = chunks[0]["state_after"]
    post_warmup_invalid = int(terminal["invalid_windows"] - first_state["invalid_windows"])
    post_warmup = [
        record for record in records
        if int(record["counters_before"]["attempted_windows"]) + 1 >= 64
    ]
    prefix_seen: dict[str, bool] = {prefix: False for prefix in _required_prefixes(spec["mode"])}
    post_warmup_finite = True
    lidar_ratios: list[float] = []
    head_ratios: list[float] = []
    for record in post_warmup:
        gradients = record["parameter_gradients"]
        post_warmup_finite &= bool(gradients["global"]["all_finite"])
        for prefix in prefix_seen:
            summary = gradients["by_prefix"].get(prefix)
            if summary and summary["complete_l2"] is not None and summary["complete_l2"] > 0.0:
                prefix_seen[prefix] = True
        updates = record.get("parameter_updates") or {}
        for prefix, row in updates.get("by_prefix", {}).items():
            ratio = row.get("realized_update_over_weight")
            if ratio is None:
                continue
            if prefix.startswith("lidar_encoder"):
                lidar_ratios.append(float(ratio))
            elif prefix == "head":
                head_ratios.append(float(ratio))

    first_loss = float(chunks[0]["metrics"]["loss"])
    final_loss = float(chunks[-1]["metrics"]["loss"])
    eval_finite = True
    if evaluation is not None:
        eval_finite = all(
            math.isfinite(float(evaluation[key]))
            for key in ("internal_subset_mAP", "internal_subset_NDS")
        )
    hard_errors = []
    if int(terminal["optimizer_step"]) <= 0:
        hard_errors.append("no accepted optimizer update")
    if int(terminal["discarded_windows"]) != 0:
        hard_errors.append("discarded window")
    if spec["attempted_windows"] > 64 and post_warmup_invalid != 0:
        hard_errors.append("post-warmup invalid window")
    if not post_warmup_finite:
        hard_errors.append("post-warmup nonfinite sampled gradient")
    missing = sorted(prefix for prefix, seen in prefix_seen.items() if not seen)
    if missing:
        hard_errors.append(f"required trainable prefixes lacked sampled gradients: {missing}")
    if not eval_finite:
        hard_errors.append("nonfinite internal evaluation")

    max_lidar_ratio = max(lidar_ratios, default=0.0)
    median_head_ratio = (
        sorted(head_ratios)[len(head_ratios) // 2] if head_ratios else 0.0
    )
    harm_indicators = {
        "post_warmup_invalid": post_warmup_invalid > 0,
        "extreme_lidar_realized_update": bool(
            max_lidar_ratio >= 1.0e-2
            and (median_head_ratio == 0.0 or max_lidar_ratio >= 10.0 * median_head_ratio)
        ),
        "adverse_loss_trajectory": final_loss > 1.25 * first_loss,
    }
    harm_count = sum(bool(value) for value in harm_indicators.values())
    if hard_errors:
        label = "HARD_FAIL"
    elif final_loss < first_loss:
        label = "NUMERICALLY_HEALTHY_WITH_TRAINING_SIGNAL"
    else:
        label = "NUMERICALLY_HEALTHY_AMBIGUOUS_TRAJECTORY"
    return {
        "label": label,
        "hard_errors": hard_errors,
        "post_warmup_invalid_windows": post_warmup_invalid,
        "required_prefix_gradient_seen": prefix_seen,
        "first_chunk_loss": first_loss,
        "final_chunk_loss": final_loss,
        "final_over_first_loss": final_loss / first_loss if first_loss > 0.0 else None,
        "max_sampled_lidar_update_over_weight": max_lidar_ratio,
        "median_sampled_head_update_over_weight": median_head_ratio,
        "large_lidar_gradient_harm_indicators": harm_indicators,
        "large_lidar_gradient_harm_label": (
            "CORRELATED_HARM_SIGNAL" if harm_count >= 2 else "NOT_ESTABLISHED"
        ),
        "interpretation": (
            "gradient magnitude alone is not a failure; the harm label requires at "
            "least two independent overflow/update/trajectory indicators"
        ),
    }


def _run_cell(
    *,
    base_config: dict[str, Any],
    spec: dict[str, Any],
    source_sha: str,
    split_manifest: Path,
    output_dir: Path,
    runtime_dependencies: dict[str, Any],
) -> dict[str, Any]:
    config = _resolved_for_cell(base_config, spec)
    verify_physical_data_identities(config)
    run_config = config.to_run_config()
    expected_sources = _source_identities(config)
    low = load_frozen_split_role(
        split_manifest,
        expected_manifest_sha256=EXPECTED_SPLIT_SHA256,
        role="D_low",
        expected_source_identities=expected_sources,
    )
    select = load_frozen_split_role(
        split_manifest,
        expected_manifest_sha256=EXPECTED_SPLIT_SHA256,
        role="D_select",
        expected_source_identities=expected_sources,
    )
    if len(low.sample_tokens) != 6155 or len(select.sample_tokens) != 4626:
        raise RuntimeError("accepted S10 role counts drifted")

    cell_dir = output_dir / spec["id"]
    cell_dir.mkdir(parents=True, exist_ok=False)
    (cell_dir / "resolved_config.json").write_bytes(config.canonical_bytes + b"\n")
    seed = int(config.data["training"]["seed"])
    seed_everything(seed)
    enforce_determinism(precision=config.precision, strict=False)

    from fl_v3.training.tasks import get_task

    task = get_task("nuscenes_detection")
    loader = task.manifest_train_subset_loader(
        run_config, low.sample_tokens, shuffle=True, drop_last=True,
    )
    if len(loader) != 1538:
        raise RuntimeError(f"D_low B4 drop-last loader length drifted: {len(loader)}")
    dropped = _dropped_tokens(low.sample_tokens, seed, 4)
    if len(dropped) != 3:
        raise RuntimeError("D_low B4 dropped-token count drifted")

    device = torch.device("cuda")
    model = task.build_model(run_config).to(device)
    criterion = task.build_criterion(run_config)
    optimizer = _build_optimizer(model, config)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _step: 1.0)
    scaler = make_grad_scaler(device, config.precision, init_scale=512.0)
    state = TrainingState()
    diagnostics = PrecisionWindowDiagnostics(
        PrecisionDiagnosticsIdentity(
            source_sha=source_sha,
            resolved_config_sha256=config.sha256,
            model_mode=config.model_mode,
            global_precision=config.precision,
            sparse_conv_precision=config.sparse_conv_precision,
        ),
        max_windows=len(spec["diagnostics"]),
        attempted_windows=spec["diagnostics"],
        capture_boundaries=True,
        capture_parameter_updates=True,
        fixture_identity={
            "cell": spec["id"],
            "split_manifest_sha256": EXPECTED_SPLIT_SHA256,
            "D_low_sample_tokens_sha256": low.sample_tokens_sha256,
            "dropped_tokens": dropped,
        },
    )
    profiler = (
        BoundedOperatorProfiler(PROFILE_SPEC, cell_dir / "operator_profile")
        if spec["operator_profile"] else None
    )
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)

    epoch_iterator = iter(loader)
    chunks: list[dict[str, Any]] = []
    start = 0
    for index, boundary in enumerate(spec["boundaries"]):
        count = int(boundary) - start
        chunk = _ExactChunk(epoch_iterator, count)
        before = state.checkpoint_dict()
        torch.cuda.synchronize(device)
        wall_started = time.perf_counter()
        profile_context = profiler if index == 0 and profiler is not None else nullcontext()
        with profile_context as active_profiler:
            range_context = (
                model.operator_profile_ranges()
                if active_profiler is not None else nullcontext()
            )
            with range_context:
                mode_context = (
                    model.serialized_mode(True)
                    if hasattr(model, "serialized_mode") else nullcontext()
                )
                with mode_context:
                    metrics = train_one_epoch(
                        model,
                        chunk,
                        criterion,
                        optimizer,
                        device,
                        scheduler=scheduler,
                        precision=config.precision,
                        grad_scaler=scaler,
                        accumulation_steps=1,
                        runtime_state=state,
                        model_mode=config.model_mode,
                        exposure_multiplier=1,
                        expected_global_microbatch_samples=4,
                        precision_diagnostics=diagnostics,
                        attempted_window_callback=(
                            active_profiler.step if active_profiler is not None else None
                        ),
                    )
        torch.cuda.synchronize(device)
        wall_seconds = time.perf_counter() - wall_started
        after = state.checkpoint_dict()
        chunks.append({
            "index": index,
            "attempted_window_start_exclusive": start,
            "attempted_window_end_inclusive": int(boundary),
            "declared_windows": count,
            "wall_seconds": wall_seconds,
            "exposure_samples_per_second": (
                (after["exposure_samples"] - before["exposure_samples"]) / wall_seconds
            ),
            "state_before": before,
            "state_after": after,
            "metrics": metrics,
        })
        start = int(boundary)
    try:
        next(epoch_iterator)
    except StopIteration:
        pass
    else:
        raise RuntimeError("D_low one-epoch cell did not consume the exact drop-last loader")
    if state.attempted_windows != int(spec["attempted_windows"]):
        raise RuntimeError("cell attempted-window horizon drifted")
    if len(diagnostics.records) != len(spec["diagnostics"]):
        raise RuntimeError("sampled diagnostic record count drifted")

    state.epoch = 1 if spec["attempted_windows"] == 1538 else 0
    terminal = state.checkpoint_dict()
    diagnostic_path = cell_dir / "sampled_windows.jsonl"
    diagnostic_path.write_text(diagnostics.json_lines(), encoding="utf-8")
    operator_profile = profiler.report() if profiler is not None else None
    if operator_profile is not None:
        for artifact_name in ("trace", "summary"):
            artifact = operator_profile[artifact_name]
            artifact["path"] = str(
                Path(artifact["path"]).resolve().relative_to(output_dir)
            )
    checkpoint_path = None
    checkpoint_sha256 = None
    evaluation = None
    evaluation_timing = None
    if spec["evaluate"]:
        checkpoint_path = cell_dir / "checkpoint.pt"
        save_checkpoint(
            str(checkpoint_path),
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            grad_scaler=scaler,
            ema=None,
            state=state,
            config=config,
            checkpoint_identity=config.sha256,
        )
        checkpoint_sha256 = _sha256_file(checkpoint_path)
        eval_loader = task.manifest_train_subset_loader(
            run_config, select.sample_tokens, shuffle=False, drop_last=False,
        )
        timing: dict[str, Any] = {}
        eval_started = time.perf_counter()
        decodes = decode_eval_set(model, eval_loader, device, run_config, timing)
        decoded_tokens = [item.sample_token for item in decodes]
        if decoded_tokens != list(select.sample_tokens):
            raise RuntimeError("D_select decode token vector drifted")
        from fl_v3.data.nuscenes.class_map import DETECTION_NAMES
        from fl_v3.data.nuscenes import paths as nuscenes_paths

        eval_run_config = dict(run_config)
        eval_run_config.update({
            "checkpoint-sha256": checkpoint_sha256,
            "checkpoint-weights": "raw",
            "runtime-dependencies-sha256": _canonical_sha256(runtime_dependencies),
        })
        submission = build_results_dict(
            decodes,
            DETECTION_NAMES,
            select.sample_tokens,
            run_config=eval_run_config,
        )
        result_path = cell_dir / "D_select_results.json"
        write_strict_json(str(result_path), submission)
        nusc = nuscenes_paths.create_nuscenes(
            "v1.0-trainval", run_config["nuscenes-dataroot"], verbose=False,
        )
        evaluation = run_internal_manifest_eval(
            nusc,
            str(result_path),
            str(split_manifest),
            "D_select",
            str(cell_dir / "D_select_metrics.json"),
            expected_manifest_sha256=EXPECTED_SPLIT_SHA256,
            expected_parent_version="v1.0-trainval",
            expected_parent_split="train",
            expected_source_identities=expected_sources,
        )
        evaluation_timing = {
            "total_wall_seconds": time.perf_counter() - eval_started,
            "decode": timing,
        }
        _close_loader(eval_loader)

    health = _cell_health(
        spec, chunks, diagnostics.records, terminal, evaluation,
    )
    report = {
        "schema": SCHEMA,
        "cell": spec,
        "source_sha": source_sha,
        "resolved_config_sha256": config.sha256,
        "runtime_dependencies_sha256": _canonical_sha256(runtime_dependencies),
        "roles": {"D_low": low.identity(), "D_select": select.identity()},
        "recipe": {
            "optimizer": "AdamW",
            "learning_rate": 1.0e-4,
            "weight_decay": 0.01,
            "scheduler": "constant_lambda_1",
            "grad_clip": None,
            "ema": None,
            "augmentation": None,
            "sampling": "uniform",
            "physical_microbatch": 4,
            "accumulation": 1,
            "precision": "global_fp16_SECOND_fp32_island",
            "grad_scaler_init_scale": 512.0,
        },
        "dropped_tokens": dropped,
        "chunks": chunks,
        "terminal_training_state": terminal,
        "diagnostic_windows": list(spec["diagnostics"]),
        "sampled_diagnostics_path": str(diagnostic_path.relative_to(output_dir)),
        "sampled_diagnostics_sha256": _sha256_file(diagnostic_path),
        "operator_profile": operator_profile,
        "memory": {
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
            "device_total_bytes": int(torch.cuda.get_device_properties(device).total_memory),
        },
        "checkpoint_path": (
            None if checkpoint_path is None else str(checkpoint_path.relative_to(output_dir))
        ),
        "checkpoint_sha256": checkpoint_sha256,
        "D_select_evaluation": evaluation,
        "D_select_evaluation_timing": evaluation_timing,
        "health": health,
        "interpretation_limits": [
            "single-seed limited-rung internal evidence; not the S10 full claim",
            "fixed engineering baseline recipe; not production-recipe acceptance",
            "D_select is train-only proxy evidence; official val remains sealed",
            "operator profiling covers one early A1 window only and is not STOP-E",
            "large-gradient causality is not established by gradient magnitude alone",
        ],
    }
    _write_json(cell_dir / "cell_summary.json", report)
    _close_loader(loader)
    del model, criterion, optimizer, scheduler, scaler, loader
    gc.collect()
    torch.cuda.empty_cache()
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--split-manifest", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--source-sha", required=True)
    parser.add_argument("--swin-weights", required=True)
    args = parser.parse_args()
    if len(args.source_sha) != 40 or any(ch not in "0123456789abcdef" for ch in args.source_sha):
        raise ValueError("--source-sha must be an exact lowercase Git SHA")
    split_manifest = Path(args.split_manifest).resolve()
    if _sha256_file(split_manifest) != EXPECTED_SPLIT_SHA256:
        raise RuntimeError("STOP-A split manifest identity drift")
    if _sha256_file(args.swin_weights) != EXPECTED_SWINT_SHA256:
        raise RuntimeError("ImageNet Swin-T checkpoint identity drift")
    output_dir = Path(args.out_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    base_config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    base_resolved = resolve_config(base_config)
    runtime_dependencies = verify_runtime_dependency_identity(base_resolved.to_run_config())
    _write_json(output_dir / "runtime_dependencies.json", runtime_dependencies)

    reports = []
    for spec in CELL_SPECS:
        report = _run_cell(
            base_config=base_config,
            spec=spec,
            source_sha=args.source_sha,
            split_manifest=split_manifest,
            output_dir=output_dir,
            runtime_dependencies=runtime_dependencies,
        )
        reports.append(report)
        print(json.dumps({
            "cell": spec["id"],
            "health": report["health"]["label"],
            "optimizer_steps": report["terminal_training_state"]["optimizer_step"],
        }, sort_keys=True), flush=True)

    by_id = {report["cell"]["id"]: report for report in reports}
    fusion_eval = by_id["C0-F-A1"]["D_select_evaluation"]
    lidar_eval = by_id["C0-L-A0"]["D_select_evaluation"]
    hard_failures = {
        cell_id: report["health"]["hard_errors"]
        for cell_id, report in by_id.items()
        if report["health"]["hard_errors"]
    }
    summary = {
        "schema": SCHEMA,
        "status": "FAIL" if hard_failures else "PASS",
        "source_sha": args.source_sha,
        "cell_order": [spec["id"] for spec in CELL_SPECS],
        "hard_failures": hard_failures,
        "cell_summary_sha256": {
            cell_id: _sha256_file(output_dir / cell_id / "cell_summary.json")
            for cell_id in by_id
        },
        "one_epoch_internal_fusion_contribution": {
            "fusion_minus_lidar_mAP": (
                float(fusion_eval["internal_subset_mAP"])
                - float(lidar_eval["internal_subset_mAP"])
            ),
            "fusion_minus_lidar_NDS": (
                float(fusion_eval["internal_subset_NDS"])
                - float(lidar_eval["internal_subset_NDS"])
            ),
            "interpretation": (
                "descriptive one-epoch D_select delta only; not the primary full "
                "fusion-contribution claim and not an architecture promotion gate"
            ),
        },
        "next_decision": (
            "If both long cells are numerically healthy, large absolute LiDAR gradients "
            "are not treated as a bug without correlated harm; continue STOP-C strong contrast. "
            "A correlated-harm or hard-failure result may consume at most one predeclared "
            "single-factor C counterfactual after owner review."
        ),
    }
    _write_json(output_dir / "c0_summary.json", summary)
    if hard_failures:
        raise RuntimeError(f"C0 hard gate failed: {hard_failures}")


if __name__ == "__main__":
    main()
