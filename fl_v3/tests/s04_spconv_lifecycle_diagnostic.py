"""Bounded subprocess-isolated diagnostics for S04 sparse-fp16 eval lifecycle.

This is not a correctness test or a workaround.  Each cell runs in a fresh Python
process so one spconv tuner failure cannot suppress the remaining observations.
The matrix command succeeds only when every cell returns a structured envelope;
individual cell outcomes may be ``success`` or ``error``.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import subprocess
import sys
import traceback
from pathlib import Path

import torch


CELLS = (
    "fresh_fp16_eval_6",
    "train_to_eval_no_backward_6",
    "train_to_eval_after_backward_6",
    "fresh_fp16_eval_large",
    "fresh_fp32_eval_6",
    "fp32_then_fp16_eval_6",
    "fp16_train_then_fresh_fp16_eval_6",
)


def _cfg():
    from fl_v3.models.fusion.bev_grid import BEVConfig

    return BEVConfig(
        point_cloud_range=(-4.0, -4.0, -5.0, 4.0, 4.0, 3.0),
        bev_voxel=(0.5, 0.5),
        out_size_factor=8,
    )


def _encoder(*, fp16: bool):
    from fl_v3.models.fusion.sparse_voxel_encoder import SparseVoxelEncoder

    return SparseVoxelEncoder(
        out_channels=16,
        cfg=_cfg(),
        z_voxel=0.2,
        max_voxels_train=128,
        max_voxels_eval=192,
        max_points_per_voxel=3,
        sparse_conv_fp16=fp16,
    ).cuda()


def _six_voxel_points() -> torch.Tensor:
    # Seven valid points occupying six voxels across two samples, exactly matching
    # the Job-336718 failing fixture (plus one out-of-range point).
    return torch.tensor(
        [
            [0, -3.75, -3.75, -4.9, 0.10, 0],
            [0, -3.70, -3.70, -4.8, 0.20, 0],
            [0, -1.20, 0.20, 0.10, 0.30, 0],
            [0, 2.20, 2.20, 2.10, 0.40, 0],
            [1, 1.20, -1.10, -0.60, 0.50, 0],
            [1, 2.70, 2.90, 1.20, 0.60, 0],
            [1, -2.20, 1.80, -2.20, 0.70, 0],
            [1, 9.00, 0.00, 0.00, 0.80, 0],
        ],
        device="cuda",
        dtype=torch.float32,
    )


def _large_points(per_sample: int = 128) -> torch.Tensor:
    i = torch.arange(per_sample, device="cuda", dtype=torch.int64)
    xidx = i % 16
    yidx = torch.div(i, 16, rounding_mode="floor") % 16
    zidx = (i * 7) % 40
    x = -4.0 + (xidx.float() + 0.5) * 0.5
    y = -4.0 + (yidx.float() + 0.5) * 0.5
    z = -5.0 + (zidx.float() + 0.5) * 0.2
    intensity = (i % 31).float() / 31.0
    ring = (i % 16).float()
    rows = []
    for batch in range(2):
        rows.append(
            torch.stack((torch.full_like(x, batch), x, y, z, intensity, ring), dim=1)
        )
    return torch.cat(rows, dim=0)


def _trace_implicit_gemm(events: list[dict]):
    import spconv.pytorch.ops as ops

    original = ops.implicit_gemm

    def traced(*args, **kwargs):
        event = {
            "features_shape": list(args[0].shape),
            "features_dtype": str(args[0].dtype),
            "filters_shape": list(args[1].shape),
            "filters_dtype": str(args[1].dtype),
            "num_activate_out": int(args[5]),
            "is_train": bool(args[7]),
            "is_subm": bool(args[8]),
            "output_dtype_argument": str(kwargs.get("output_dtype")),
            "outcome": "entered",
        }
        events.append(event)
        try:
            result = original(*args, **kwargs)
        except Exception as exc:
            event["outcome"] = "error"
            event["exception"] = f"{type(exc).__name__}: {exc}"
            raise
        event["outcome"] = "success"
        output = result[0] if isinstance(result, tuple) else result
        event["actual_output_dtype"] = str(output.dtype)
        return result

    ops.implicit_gemm = traced
    return original


def _forward(model, points: torch.Tensor, *, backward: bool) -> dict:
    output = model(points, B=2)
    record = {
        "output_shape": list(output.shape),
        "output_dtype": str(output.dtype),
        "output_finite": bool(torch.isfinite(output).all().detach().cpu()),
    }
    if backward:
        loss = output.float().square().mean()
        loss.backward()
        grads = [p.grad for p in model.parameters() if p.requires_grad]
        record.update(
            loss=float(loss.detach().cpu()),
            all_gradients_present=bool(grads) and all(g is not None for g in grads),
            all_gradients_finite=bool(grads)
            and all(g is not None and torch.isfinite(g).all() for g in grads),
        )
    return record


def _run_cell(name: str) -> dict:
    import spconv.pytorch.ops as ops

    torch.manual_seed(20260711)
    torch.cuda.manual_seed_all(20260711)
    events: list[dict] = []
    original = _trace_implicit_gemm(events)
    phases: list[dict] = []
    result = {
        "cell": name,
        "pid": os.getpid(),
        "events": events,
        "phases": phases,
    }
    try:
        six = _six_voxel_points()
        if name == "fresh_fp16_eval_6":
            model = _encoder(fp16=True).eval()
            with torch.no_grad():
                phases.append({"phase": "fresh_eval", **_forward(model, six, backward=False)})
        elif name == "train_to_eval_no_backward_6":
            model = _encoder(fp16=True).train()
            phases.append({"phase": "train_forward", **_forward(model, six, backward=False)})
            model.eval()
            with torch.no_grad():
                phases.append({"phase": "same_model_eval", **_forward(model, six, backward=False)})
        elif name == "train_to_eval_after_backward_6":
            model = _encoder(fp16=True).train()
            phases.append({"phase": "train_forward_backward", **_forward(model, six, backward=True)})
            model.eval()
            with torch.no_grad():
                phases.append({"phase": "same_model_eval", **_forward(model, six, backward=False)})
        elif name == "fresh_fp16_eval_large":
            model = _encoder(fp16=True).eval()
            with torch.no_grad():
                phases.append(
                    {"phase": "fresh_eval_large", **_forward(model, _large_points(), backward=False)}
                )
        elif name == "fresh_fp32_eval_6":
            model = _encoder(fp16=False).eval()
            with torch.no_grad():
                phases.append({"phase": "fresh_fp32_eval", **_forward(model, six, backward=False)})
        elif name == "fp32_then_fp16_eval_6":
            fp32 = _encoder(fp16=False).eval()
            with torch.no_grad():
                phases.append({"phase": "fp32_eval", **_forward(fp32, six, backward=False)})
            fp16 = _encoder(fp16=True).eval()
            with torch.no_grad():
                phases.append({"phase": "fresh_fp16_eval", **_forward(fp16, six, backward=False)})
        elif name == "fp16_train_then_fresh_fp16_eval_6":
            train_model = _encoder(fp16=True).train()
            phases.append(
                {"phase": "separate_model_train_forward", **_forward(train_model, six, backward=False)}
            )
            eval_model = _encoder(fp16=True).eval()
            with torch.no_grad():
                phases.append(
                    {"phase": "fresh_model_fp16_eval", **_forward(eval_model, six, backward=False)}
                )
        else:
            raise ValueError(f"unknown cell: {name}")
        result["status"] = "success"
    except Exception as exc:
        result.update(
            status="error",
            exception_type=type(exc).__name__,
            exception=str(exc),
            traceback=traceback.format_exc(),
        )
    finally:
        ops.implicit_gemm = original
    result["cuda_device_count"] = torch.cuda.device_count()
    return result


def _run_matrix(output: Path) -> None:
    records = []
    for cell in CELLS:
        command = [sys.executable, str(Path(__file__).resolve()), "--cell", cell]
        try:
            completed = subprocess.run(command, capture_output=True, text=True, timeout=120)
            payload = None
            for line in completed.stdout.splitlines():
                if line.startswith("S04_DIAG_RESULT="):
                    payload = json.loads(line.split("=", 1)[1])
            records.append(
                {
                    "cell": cell,
                    "process_returncode": completed.returncode,
                    "result": payload,
                    "stdout": completed.stdout,
                    "stderr": completed.stderr,
                }
            )
        except subprocess.TimeoutExpired as exc:
            records.append(
                {
                    "cell": cell,
                    "process_returncode": None,
                    "result": None,
                    "timed_out": True,
                    "stdout": exc.stdout,
                    "stderr": exc.stderr,
                }
            )

    matrix = {
        "schema": "s04.spconv-lifecycle-diagnostic.v1",
        "cells": records,
        "dependency_versions": {
            name: importlib.metadata.version(name) for name in ("torch", "spconv", "cumm")
        },
        "scientific_metric": False,
        "optimizer_or_parameter_update": False,
    }
    output.write_text(json.dumps(matrix, indent=2, sort_keys=True) + "\n")
    complete = (
        [record["cell"] for record in records] == list(CELLS)
        and all(record["process_returncode"] == 0 for record in records)
        and all(
            record["result"] is not None
            and record["result"].get("cell") == record["cell"]
            and record["result"].get("status") in {"success", "error"}
            for record in records
        )
    )
    summary = {
        record["cell"]: None if record["result"] is None else record["result"].get("status")
        for record in records
    }
    print("S04_DIAG_MATRIX=" + json.dumps(summary, sort_keys=True))
    if not complete:
        raise SystemExit("incomplete S04 lifecycle diagnostic matrix")


def main() -> None:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--cell", choices=CELLS)
    group.add_argument("--matrix", type=Path)
    args = parser.parse_args()
    if args.cell:
        result = _run_cell(args.cell)
        print("S04_DIAG_RESULT=" + json.dumps(result, sort_keys=True))
    else:
        _run_matrix(args.matrix)


if __name__ == "__main__":
    main()
