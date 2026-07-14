"""Bounded GH200 Q1 precision qualification for the exact replay-frozen mini fixture.

The single test runs all eight predeclared primary cells and writes raw JSON
evidence.  A numerical cell may fail its qualification gate without aborting the
remaining cells.  Identity, lifecycle, dependency, OOM, or record failures are
hard errors.  This is not a performance, convergence, or metric test.
"""
from __future__ import annotations

import copy
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random

import numpy as np
import pytest
import torch

from fl_v3.config import resolve_config
from fl_v3.data.nuscenes import dataset as DS
from fl_v3.data.nuscenes import info_cache as IC
from fl_v3.models.fusion.collate import detection_collate_fn
from fl_v3.models.fusion.preprocess import (
    AUGMENTATION_PARAM_FIELDS,
    ImageAugmentationConfig,
    _sample_parameters,
)
from fl_v3.source_identity import build_source_state
from fl_v3.training.loop import train_one_epoch
from fl_v3.training.precision_diagnostics import (
    PrecisionDiagnosticsIdentity,
    PrecisionWindowDiagnostics,
    runtime_rng_state_sha256,
)
from fl_v3.training.runtime_state import TrainingState
from fl_v3.training.tasks import get_task
from fl_v3.utils.runtime import (
    enforce_determinism,
    make_grad_scaler,
    precision_state,
    seed_everything,
    verify_runtime_dependency_identity,
)


SAMPLE_TOKEN = "00889f8a9549450aa2f32cf310a3e305"
KEY_LIDAR_TOKEN = "5933b3acbae44b6a92b36327b134e56c"
SCENE_TOKEN = "2fc3753772e241f2ab2cd16a784cc680"
TIMESTAMP = 1535657119649820
RAW_LIDAR_SHA256 = "db897ac315c8ef88e3714e30618ae692acf348e24b4adea78e5523fb84ca1123"
POINT_PREFIX_SHA256 = "a2826b1c34470a0cc69be4bb572379378686f59d2242ca017840cfaa196e1713"
SEED = 20260713
SPCONV_SOURCE_STATE = build_source_state([{
    "status": " M",
    "path": "pyproject.toml",
    "sha256": "e2c84544b5b5d6fd8e149d88539c3a6e989a1824637fd6b0006891955cb7a7e9",
}])
CUMM_SOURCE_STATE = build_source_state([])

CELLS = (
    ("C1", "C-STR8", "camera_only", "swin_t_stride8", "none", "none", "fp32", "not_applicable"),
    ("C2", "C-STR8", "camera_only", "swin_t_stride8", "none", "none", "fp16", "not_applicable"),
    ("L1", "L-S075", "lidar_only", "none", "second_075", "none", "fp32", "fp32"),
    ("L2", "L-S075", "lidar_only", "none", "second_075", "none", "fp16", "fp16"),
    ("L3", "L-S075", "lidar_only", "none", "second_075", "none", "fp16", "fp32"),
    ("F1", "F-U", "fusion", "swin_t_stride8", "second_075", "conv_fuser_256", "fp32", "fp32"),
    ("F2", "F-U", "fusion", "swin_t_stride8", "second_075", "conv_fuser_256", "fp16", "fp16"),
    ("F3", "F-U", "fusion", "swin_t_stride8", "second_075", "conv_fuser_256", "fp16", "fp32"),
)


class _OneBatch:
    batch_size = 1

    def __init__(self, batch):
        self.batch = batch

    def __len__(self):
        return 1

    def __iter__(self):
        yield self.batch


def _canonical_bytes(value) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False,
    ).encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tensor_raw_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().to("cpu").contiguous()
    return hashlib.sha256(tensor.numpy().tobytes(order="C")).hexdigest()


def _state_dict_sha256(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        value = state[name].detach().to("cpu").contiguous()
        metadata = _canonical_bytes({
            "name": name,
            "dtype": str(value.dtype),
            "shape": [int(item) for item in value.shape],
        })
        payload = value.numpy().tobytes(order="C")
        digest.update(len(metadata).to_bytes(8, "big"))
        digest.update(metadata)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def _tensor_manifest(value, path="batch") -> dict[str, dict]:
    output = {}
    if torch.is_tensor(value):
        output[path] = {
            "shape": [int(item) for item in value.shape],
            "dtype": str(value.dtype),
            "sha256": _tensor_raw_sha256(value),
        }
    elif isinstance(value, dict):
        for key in sorted(value):
            output.update(_tensor_manifest(value[key], f"{path}.{key}"))
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            output.update(_tensor_manifest(item, f"{path}[{index}]"))
    return output


def _rng_snapshot() -> dict:
    return {
        "python": copy.deepcopy(random.getstate()),
        "numpy": copy.deepcopy(np.random.get_state()),
        "torch": torch.get_rng_state().clone(),
        "cuda": [state.clone() for state in torch.cuda.get_rng_state_all()],
    }


def _restore_rng(snapshot: dict) -> None:
    random.setstate(copy.deepcopy(snapshot["python"]))
    np.random.set_state(copy.deepcopy(snapshot["numpy"]))
    torch.set_rng_state(snapshot["torch"].clone())
    torch.cuda.set_rng_state_all([state.clone() for state in snapshot["cuda"]])


def _required_env(name: str, length: int) -> str:
    value = os.environ.get(name, "")
    if len(value) != length or any(c not in "0123456789abcdef" for c in value):
        raise RuntimeError(f"{name} must be an exact lowercase {length}-hex identity")
    return value


def _dependency_fields(second: bool) -> dict:
    fields = {
        "torch": str(torch.__version__),
        "torch_build_sha256": _required_env("S08_TORCH_BUILD_SHA256", 64),
        "torch_source_sha": _required_env("S08_TORCH_SOURCE_SHA", 40),
        "spconv": None,
        "spconv_build_sha256": None,
        "spconv_source_sha": None,
        "spconv_source_state": None,
        "cumm": None,
        "cumm_build_sha256": None,
        "cumm_source_sha": None,
        "cumm_source_state": None,
    }
    if second:
        if importlib.metadata.version("spconv") != "2.3.8":
            raise RuntimeError("Q1 requires spconv==2.3.8")
        if importlib.metadata.version("cumm") != "0.7.13":
            raise RuntimeError("Q1 requires cumm==0.7.13")
        fields.update(
            spconv="2.3.8",
            spconv_build_sha256=_required_env("S08_SPCONV_BUILD_SHA256", 64),
            spconv_source_sha=_required_env("S08_SPCONV_SOURCE_SHA", 40),
            spconv_source_state=SPCONV_SOURCE_STATE,
            cumm="0.7.13",
            cumm_build_sha256=_required_env("S08_CUMM_BUILD_SHA256", 64),
            cumm_source_sha=_required_env("S08_CUMM_SOURCE_SHA", 40),
            cumm_source_state=CUMM_SOURCE_STATE,
        )
    return fields


def _resolved_cell_config(cell, dataroot: str, fixture_sha256: str):
    cell_id, _tag, mode, camera, lidar, fusion, precision, partition = cell
    bypass = f"/S08_FIXTURE_BYPASS/{fixture_sha256}/{cell_id}"
    return resolve_config({
        "schema_version": "s08.v1",
        "model": {
            "mode": mode,
            "camera_arch": camera,
            "camera_pretrained": False if camera != "none" else None,
            "lidar_arch": lidar,
            "fusion_arch": fusion,
            "head_arch": "centerhead_multitask",
        },
        "precision": precision,
        "sparse_conv_precision": partition,
        "optimizer": {"name": "adamw", "learning_rate": 0.0001, "weight_decay": 0.01},
        "training": {
            "max_optimizer_steps": 3,
            "micro_batch_size": 1,
            "world_size": 1,
            "accumulation_steps": 1,
            "effective_global_batch": 1,
            "seed": SEED,
            "max_epochs": 1,
            "num_workers": 0,
            "ema_decay": None,
            "sampling": "uniform",
        },
        "data": {
            "dataroot": dataroot,
            "version": "v1.0-mini",
            "train_split": "mini_train",
            "val_split": "mini_val",
            "n_sweeps": 10,
            "caches": {
                "train": {
                    "format": "t1.v2", "path": bypass + "/train.pkl",
                    "sidecar_path": bypass + "/train.meta.json",
                    "logical_sha256": fixture_sha256, "pickle_sha256": fixture_sha256,
                    "sidecar_sha256": fixture_sha256,
                },
                "val": {
                    "format": "t1.v2", "path": bypass + "/val.pkl",
                    "sidecar_path": bypass + "/val.meta.json",
                    "logical_sha256": fixture_sha256, "pickle_sha256": fixture_sha256,
                    "sidecar_sha256": fixture_sha256,
                },
            },
            "zip_manifest": {
                "path": bypass + "/manifest.sqlite",
                "logical_sha256": fixture_sha256,
                "file_sha256": fixture_sha256,
            },
        },
        "dependencies": _dependency_fields(lidar == "second_075"),
        "evaluation": {"timing": False, "checkpoint_weights": "raw"},
    })


def _prepare_fixture(nusc_mini, dataroot: str):
    split_tokens = IC.split_sample_tokens(nusc_mini, "mini_train")
    if SAMPLE_TOKEN not in split_tokens:
        raise RuntimeError("frozen Q1 sample is not in mini_train")
    info = IC.build_info_list(nusc_mini, [SAMPLE_TOKEN], dataroot, n_sweeps=10)[0]
    sample = nusc_mini.get("sample", SAMPLE_TOKEN)
    if sample["data"]["LIDAR_TOP"] != KEY_LIDAR_TOKEN:
        raise RuntimeError("frozen key LiDAR token drift")
    if info["scene_token"] != SCENE_TOKEN or int(info["timestamp"]) != TIMESTAMP:
        raise RuntimeError("frozen scene/timestamp drift")
    if len(info["lidar_sweeps"]) != 9:
        raise RuntimeError("frozen fixture no longer has nine previous sweeps")
    raw_path = Path(dataroot) / info["lidar_rel_path"]
    if _sha256_file(raw_path) != RAW_LIDAR_SHA256:
        raise RuntimeError("frozen raw keyframe LiDAR payload drift")
    if raw_path.stat().st_size // (5 * 4) != 34688:
        raise RuntimeError("frozen keyframe point count drift")

    dataset = DS.NuScenesMultimodalDataset(
        [info], dataroot, sample_tokens=[SAMPLE_TOKEN], n_sweeps=10, model_mode="fusion",
    )
    loader = DS.make_loader(
        dataset, batch_size=1, shuffle=False, num_workers=0, seed=SEED,
        collate_fn=detection_collate_fn,
    )
    try:
        batch = next(iter(loader))
    finally:
        dataset.close()
    batch["lidar_points"] = batch["lidar_points"][:4096].contiguous()
    points = batch["lidar_points"]
    if tuple(points.shape) != (4096, 7) or points.dtype != torch.float32:
        raise RuntimeError("frozen collated point-prefix shape/dtype drift")
    if _tensor_raw_sha256(points) != POINT_PREFIX_SHA256:
        raise RuntimeError("frozen 4096-point prefix drift")
    if not bool(torch.all(points[:, 6] == 0.0)):
        raise RuntimeError("frozen 4096-point prefix is no longer keyframe-only")

    generator = torch.Generator(device="cpu").manual_seed(SEED)
    height, width = (int(value) for value in batch["images"].shape[-2:])
    augmentation = _sample_parameters(
        ImageAugmentationConfig(), True, 1, 6, height, width, 256, 704, generator,
    )
    batch["augmentation_params"] = augmentation
    tensor_manifest = _tensor_manifest(batch)
    batch_sha256 = hashlib.sha256(_canonical_bytes(tensor_manifest)).hexdigest()
    fixture = {
        "schema": "s08.replay-frozen-fixture.v1",
        "fixture_directory_data_route_executed": True,
        "resolved_config_cache_manifest_route_executed": False,
        "resolved_config_data_fields_are_fixture_identity_only": True,
        "dataset": "nuScenes v1.0-mini / mini_train / directory backend",
        "dataroot": dataroot,
        "sample_token": SAMPLE_TOKEN,
        "key_lidar_token": KEY_LIDAR_TOKEN,
        "scene_token": SCENE_TOKEN,
        "timestamp": TIMESTAMP,
        "raw_lidar_relative_path": info["lidar_rel_path"],
        "raw_lidar_sha256": RAW_LIDAR_SHA256,
        "keyframe_point_count": 34688,
        "n_sweeps_metadata": 10,
        "previous_sweeps": 9,
        "point_prefix_count": 4096,
        "point_prefix_sha256": POINT_PREFIX_SHA256,
        "point_prefix_dt_histogram": {"0.0": 4096},
        "batch_tensor_manifest": tensor_manifest,
        "batch_tensor_manifest_sha256": batch_sha256,
        "augmentation_param_fields": list(AUGMENTATION_PARAM_FIELDS),
        "augmentation_params": augmentation.tolist(),
        "augmentation_params_sha256": _tensor_raw_sha256(augmentation),
        "recipe": {
            "optimizer": "AdamW", "learning_rate": 0.0001, "weight_decay": 0.01,
            "scheduler": "constant LambdaLR", "ema": False, "gradient_clip": False,
            "dataset_3d_augmentation": False, "gt_paste": False,
            "batch_size": 1, "num_workers": 0, "world_size": 1,
            "accumulation_steps": 1, "seed": SEED,
        },
        "interpretation": "D1-derived numerical isolation; not full production data execution",
    }
    fixture_sha256 = hashlib.sha256(_canonical_bytes(fixture)).hexdigest()
    return batch, fixture, fixture_sha256, batch_sha256


def _cell_pass(precision: str, records, state: TrainingState, post_accept_skip: bool) -> bool:
    outcomes = [record["outcome"] for record in records]
    counters_ok = all(bool(record["counter_deltas_consistent"]) for record in records)
    accepted_records = [record for record in records if record["accepted"]]
    accepted_numerics_ok = all(
        bool(record["loss_finite"])
        and bool(record["parameter_gradients"]["global"]["all_finite"])
        and record["parameter_gradients"]["missing_grad_parameter_count"] == 0
        and bool(record["sparse_runtime_consistent"])
        and all(
            boundary["gradient_present"]
            and boundary["explicit_unscaled_fp64"]["all_finite"]
            for boundary in record["boundary_gradients"].values()
        )
        for record in accepted_records
    )
    if precision == "fp32":
        shape_ok = len(records) == 3 and outcomes == ["accepted"] * 3
    else:
        shape_ok = (
            len(records) <= 18
            and outcomes[-3:] == ["accepted"] * 3
            and not post_accept_skip
        )
    return bool(
        shape_ok and counters_ok and accepted_numerics_ok and state.optimizer_step == 3
        and state.successful_windows == 3 and state.exposure_samples == 3
    )


def _write_json(path: Path, value) -> None:
    path.write_text(
        json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


@pytest.mark.slow
def test_s08_q1_primary_precision_qualification(nusc_mini, dataroot):
    assert torch.cuda.is_available(), "Q1 requires one GH200"
    assert torch.cuda.device_count() == 1, "Q1 must expose exactly one GPU"
    assert torch.cuda.get_device_name(0) == "NVIDIA GH200 120GB"
    source_sha = _required_env("S08_SOURCE_SHA", 40)
    raw_dir = Path(os.environ.get("S08_Q1_RAW_DIR", ""))
    if not raw_dir.is_dir() or any(raw_dir.iterdir()):
        raise RuntimeError("S08_Q1_RAW_DIR must exist and be empty before Q1")

    device = torch.device("cuda:0")
    batch, fixture, fixture_sha256, batch_sha256 = _prepare_fixture(nusc_mini, dataroot)
    resolved = {
        cell[0]: _resolved_cell_config(cell, dataroot, fixture_sha256) for cell in CELLS
    }
    runtime_dependencies = verify_runtime_dependency_identity(resolved["L1"].to_run_config())
    _write_json(raw_dir / "fixture_manifest.json", fixture)
    _write_json(
        raw_dir / "resolved_configs.json",
        {cell_id: config.as_dict() for cell_id, config in resolved.items()},
    )
    records_path = raw_dir / "window_records.jsonl"
    records_path.write_text("", encoding="utf-8")
    task = get_task("nuscenes_detection")
    summaries = []

    for tag in ("C-STR8", "L-S075", "F-U"):
        mode_cells = [cell for cell in CELLS if cell[1] == tag]
        reference = resolved[mode_cells[0][0]]
        enforce_determinism(strict=True, precision="fp32")
        seed_everything(SEED)
        master = task.build_model(reference.to_run_config())
        canonical_state = {
            name: value.detach().to("cpu").clone()
            for name, value in master.state_dict().items()
        }
        canonical_state_sha256 = _state_dict_sha256(canonical_state)
        forward_rng = _rng_snapshot()
        forward_rng_sha256 = runtime_rng_state_sha256()
        del master
        gc.collect()

        for cell in mode_cells:
            cell_id, _tag, mode, _camera, _lidar, _fusion, precision, partition = cell
            config = resolved[cell_id]
            run_config = config.to_run_config()
            enforce_determinism(strict=True, precision=precision)
            model = criterion = optimizer = scheduler = scaler = None
            try:
                model = task.build_model(run_config)
                model.load_state_dict(canonical_state, strict=True)
                loaded_state_sha256 = _state_dict_sha256(model.state_dict())
                if loaded_state_sha256 != canonical_state_sha256:
                    raise RuntimeError(f"{cell_id} canonical state load drift")
                model.to(device)
                criterion = task.build_criterion(run_config)
                parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
                optimizer = torch.optim.AdamW(parameters, lr=0.0001, weight_decay=0.01)
                scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _step: 1.0)
                scaler = make_grad_scaler(device, precision, init_scale=512.0)
                state = TrainingState()
                diagnostics = PrecisionWindowDiagnostics(
                    PrecisionDiagnosticsIdentity(
                        source_sha=source_sha,
                        resolved_config_sha256=config.sha256,
                        model_mode=mode,
                        global_precision=precision,
                        sparse_conv_precision=partition,
                    ),
                    max_windows=3 if precision == "fp32" else 18,
                    fixture_identity={
                        "fixture_manifest_sha256": fixture_sha256,
                        "batch_tensor_manifest_sha256": batch_sha256,
                        "canonical_state_sha256": canonical_state_sha256,
                        "augmentation_params_sha256": fixture["augmentation_params_sha256"],
                        "replayed_forward_rng_sha256": forward_rng_sha256,
                    },
                )
                first_accept = None
                post_accept_skip = False
                limit = 3 if precision == "fp32" else 18
                for attempt in range(1, limit + 1):
                    _restore_rng(forward_rng)
                    if runtime_rng_state_sha256() != forward_rng_sha256:
                        raise RuntimeError(f"{cell_id} forward RNG restore drift")
                    before = len(diagnostics.records)
                    train_one_epoch(
                        model,
                        _OneBatch(batch),
                        criterion,
                        optimizer,
                        device,
                        scheduler=scheduler,
                        ema_model=None,
                        max_steps=1,
                        precision=precision,
                        grad_scaler=scaler,
                        accumulation_steps=1,
                        runtime_state=state,
                        max_optimizer_steps=3,
                        model_mode=mode,
                        exposure_multiplier=1,
                        expected_global_microbatch_samples=1,
                        precision_diagnostics=diagnostics,
                    )
                    if len(diagnostics.records) != before + 1:
                        raise RuntimeError(f"{cell_id} missing attempted-window record")
                    latest = diagnostics.records[-1]
                    if latest["rng_state_sha256_before_forward"] != forward_rng_sha256:
                        raise RuntimeError(f"{cell_id} recorded forward RNG identity drift")
                    if latest["outcome"] == "accepted":
                        first_accept = attempt if first_accept is None else first_accept
                    elif first_accept is not None:
                        post_accept_skip = True
                        break
                    if state.optimizer_step == 3:
                        break

                records = list(diagnostics.records)
                passed = _cell_pass(precision, records, state, post_accept_skip)
                summary = {
                    "cell_id": cell_id,
                    "mode_tag": tag,
                    "model_mode": mode,
                    "global_precision": precision,
                    "sparse_conv_precision": partition,
                    "resolved_config_sha256": config.sha256,
                    "canonical_state_sha256": canonical_state_sha256,
                    "loaded_state_sha256": loaded_state_sha256,
                    "attempted_windows": state.attempted_windows,
                    "accepted_windows": state.successful_windows,
                    "overflow_windows": state.overflow_windows,
                    "nonfinite_windows": state.nonfinite_windows,
                    "optimizer_steps": state.optimizer_step,
                    "scheduler_last_epoch": int(scheduler.last_epoch),
                    "exposure_samples": state.exposure_samples,
                    "first_accepted_attempt": first_accept,
                    "post_first_accept_skip": post_accept_skip,
                    "scale_trace": [
                        [record["scaler"]["scale_before"], record["scaler"]["scale_after"]]
                        for record in records
                    ],
                    "outcomes": [record["outcome"] for record in records],
                    "qualification_pass": passed,
                }
                summaries.append(summary)
                with records_path.open("a", encoding="utf-8") as stream:
                    for record in records:
                        stream.write(json.dumps(
                            {"cell_id": cell_id, **record},
                            sort_keys=True,
                            allow_nan=False,
                        ) + "\n")
                _write_json(raw_dir / "q1_partial_summary.json", {
                    "schema": "s08.q1-partial-summary.v1",
                    "source_sha": source_sha,
                    "requested_cell_order": [item[0] for item in CELLS],
                    "completed_cell_order": [item["cell_id"] for item in summaries],
                    "cells": summaries,
                })
                print("S08_Q1_CELL=" + json.dumps(summary, sort_keys=True), flush=True)
            finally:
                del scaler, scheduler, optimizer, criterion, model
                gc.collect()
                torch.cuda.empty_cache()
        del canonical_state
        gc.collect()

    requested = [cell[0] for cell in CELLS]
    completed = [summary["cell_id"] for summary in summaries]
    if completed != requested:
        raise RuntimeError(f"Q1 cell order/completeness drift: {completed} != {requested}")
    output = {
        "schema": "s08.q1-summary.v1",
        "source_sha": source_sha,
        "fixture_manifest_sha256": fixture_sha256,
        "runtime_dependencies": runtime_dependencies,
        "precision_state_final": precision_state(),
        "requested_cell_order": requested,
        "runner_complete": True,
        "maximum_attempted_windows": 99,
        "maximum_accepted_updates": 24,
        "cells": summaries,
        "all_primary_cells_pass": all(summary["qualification_pass"] for summary in summaries),
        "allowed_interpretation": "bounded accepted optimizer windows on one replay-frozen mini fixture",
        "forbidden_interpretation": "convergence, capability, performance, mAP/NDS, or production-data readiness",
    }
    _write_json(raw_dir / "q1_summary.json", output)
    print("S08_Q1_SUMMARY=" + json.dumps(output, sort_keys=True), flush=True)
