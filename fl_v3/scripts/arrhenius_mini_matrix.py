"""Arrhenius mini tiny-overfit matrix for engineering correctness.

This harness is not a scientific evaluation. It fixes a tiny set of mini
nuScenes tokens, reuses one batch, and checks that the supported LiDAR/precision
cells can run eval+train steps with finite losses, gradients, and sparse metadata.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List


CANONICAL_MATRIX_CELLS = {
    "voxel_fp16_main": {"det-lidar-encoder": "voxel", "precision": "fp16"},
    "voxel_fp32_ref": {"det-lidar-encoder": "voxel", "precision": "fp32"},
    "pillar_fp32_legacy": {"det-lidar-encoder": "pillar", "precision": "fp32"},
}
LEGACY_MATRIX_ALIASES = {
    "voxel_fp16": "voxel_fp16_main",
    "voxel_fp32": "voxel_fp32_ref",
    "pillar_fp32": "pillar_fp32_legacy",
}
MATRIX_CELLS = {
    **CANONICAL_MATRIX_CELLS,
    **{alias: CANONICAL_MATRIX_CELLS[target] for alias, target in LEGACY_MATRIX_ALIASES.items()},
}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _jsonable(obj: Any) -> Any:
    """Best-effort conversion for tensors, tuples, Paths, and scalars."""
    try:
        import torch

        if torch.is_tensor(obj):
            if obj.numel() == 1:
                return obj.detach().cpu().item()
            return obj.detach().cpu().tolist()
    except Exception:
        pass
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
            return str(obj)
        return obj
    return str(obj)


def _git_rev() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=_repo_root(),
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return ""


def _canonical_cell_name(name: str) -> str:
    return LEGACY_MATRIX_ALIASES.get(name, name)


def _read_cfg(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _build_base_cfg(args: argparse.Namespace) -> dict:
    cfg = _read_cfg(args.config)
    cfg["task-type"] = "nuscenes_detection"
    cfg["device"] = "cuda"
    cfg["seed"] = int(args.seed)
    cfg["precision"] = "fp32"
    cfg["output-dir"] = str(Path(args.output_dir).resolve())
    cfg["nuscenes-dataroot"] = str(Path(args.dataroot).resolve())
    cfg["nuscenes-cache-dir"] = str(Path(args.cache_dir).resolve())
    cfg["nuscenes-version"] = "v1.0-mini"
    cfg["nuscenes-train-split"] = "mini_train"
    cfg["nuscenes-val-split"] = "mini_val"
    cfg["nuscenes-partition-mode"] = "iid"
    cfg["nuscenes-num-clients"] = 1
    cfg["min-keyframes-per-client"] = 1
    cfg["det-eval-limit"] = int(args.num_tokens)
    cfg["batch-size"] = int(args.batch_size)
    cfg["num-workers"] = int(args.num_workers)
    cfg["det-camera-backbone"] = str(args.backbone)
    cfg["det-freeze-backbone"] = True
    cfg["det-pretrained-backbone"] = bool(args.pretrained_backbone)
    cfg["det-aug-bev"] = False
    cfg["det-gt-paste"] = False
    cfg["det-lidar-sweeps"] = int(args.lidar_sweeps)
    cfg["wandb-enabled"] = False
    cfg["wandb-mode"] = "disabled"
    return cfg


def _select_tokens(info_list: List[dict], count: int) -> List[str]:
    tokens = sorted(str(i["sample_token"]) for i in info_list)
    if not tokens:
        raise RuntimeError("mini train info-cache contains no sample tokens")
    return tokens[: max(1, int(count))]


def _make_fixed_batch(cfg: dict, tokens: List[str]):
    from fl_v3.data.nuscenes import paths as P
    from fl_v3.data.nuscenes.dataset import NuScenesMultimodalDataset, make_loader
    from fl_v3.models.fusion.collate import detection_collate_fn
    from fl_v3.training.tasks import get_task

    task = get_task("nuscenes_detection")
    info_list, train_meta = task._load_info(cfg, str(cfg["nuscenes-train-split"]))
    ds = NuScenesMultimodalDataset(
        info_list,
        P.get_dataroot(cfg),
        sample_tokens=tokens,
        n_sweeps=int(cfg.get("det-lidar-sweeps", 1)),
        augment=None,
        gtpaste=None,
    )
    loader = make_loader(
        ds,
        batch_size=int(cfg["batch-size"]),
        shuffle=False,
        num_workers=int(cfg.get("num-workers", 0)),
        seed=int(cfg.get("seed", 42)),
        collate_fn=detection_collate_fn,
    )
    return next(iter(loader)), train_meta


def _grad_norm(model) -> float:
    import torch

    norms = [
        p.grad.detach().float().norm(2)
        for p in model.parameters()
        if p.requires_grad and p.grad is not None
    ]
    if not norms:
        return 0.0
    return float(torch.linalg.vector_norm(torch.stack(norms), 2).detach().cpu())


def _loss_terms(criterion) -> Dict[str, Any]:
    return dict(getattr(criterion, "last_terms", {}) or {})


def _sparse_meta(model) -> Dict[str, Any]:
    enc = getattr(model, "lidar_encoder", None)
    return dict(getattr(enc, "last_sparse_meta", {}) or {})


def _validate_sparse_meta(meta: dict, cell: str) -> List[str]:
    warnings: List[str] = []
    if not meta:
        warnings.append(f"{cell}: voxel cell did not record sparse metadata")
        return warnings
    if meta.get("coord_order") != "bzyx":
        warnings.append(f"{cell}: unexpected coord_order={meta.get('coord_order')!r}")
    if meta.get("indices_dtype") != "torch.int32":
        warnings.append(f"{cell}: unexpected indices_dtype={meta.get('indices_dtype')!r}")
    if "spatial_shape" not in meta:
        warnings.append(f"{cell}: sparse spatial_shape missing")
    if int(meta.get("num_voxels", -1)) < 0:
        warnings.append(f"{cell}: invalid num_voxels={meta.get('num_voxels')!r}")
    return warnings


def _run_cell(
    cell_name: str,
    base_cfg: dict,
    fixed_batch_cpu: dict,
    output_dir: Path,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    import torch

    from fl_v3.training.loop import _float_tensors, _move_to_device
    from fl_v3.training.tasks import get_task
    from fl_v3.utils.runtime import (
        enforce_determinism,
        make_grad_scaler,
        precision_autocast_context,
        precision_state,
        seed_everything,
        validate_sparse_precision,
    )

    if cell_name not in MATRIX_CELLS:
        raise ValueError(f"unknown matrix cell {cell_name!r}; valid: {sorted(MATRIX_CELLS)}")

    cfg = copy.deepcopy(base_cfg)
    cfg.update(MATRIX_CELLS[cell_name])
    cfg["output-dir"] = str(output_dir / cell_name)
    Path(cfg["output-dir"]).mkdir(parents=True, exist_ok=True)
    validate_sparse_precision(cfg["precision"], cfg["det-lidar-encoder"])

    seed_everything(int(cfg.get("seed", 42)))
    enforce_determinism(strict=False, precision=str(cfg["precision"]))
    device = torch.device("cuda:0")
    torch.cuda.set_device(0)
    torch.cuda.empty_cache()

    task = get_task("nuscenes_detection")
    model = task.build_model(cfg).to(device)
    if cfg.get("det-lidar-encoder") == "voxel":
        getattr(model, "lidar_encoder").record_debug = True
    criterion = task.build_criterion(cfg)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
    )
    scaler = make_grad_scaler(device, str(cfg["precision"]))
    batch = _move_to_device(fixed_batch_cpu, device)

    cell: Dict[str, Any] = {
        "cell": cell_name,
        "precision": cfg["precision"],
        "lidar_encoder": cfg["det-lidar-encoder"],
        "steps_requested": int(args.steps),
        "optimizer_steps": 0,
        "grad_scaler_enabled": bool(scaler.is_enabled()),
        "grad_scaler_init_scale": float(scaler.get_scale()),
        "warnings": [],
        "precision_state": precision_state(),
        "sample_tokens": list(batch.get("sample_token", [])),
        "batch_size": int(batch.get("batch_size", len(batch.get("gt_boxes", [])))),
    }

    step_path = output_dir / f"{cell_name}_steps.jsonl"
    losses: List[float] = []
    start = time.perf_counter()
    model.eval()
    with torch.no_grad():
        torch.cuda.synchronize()
        eval_t0 = time.perf_counter()
        with precision_autocast_context(str(cfg["precision"]), device):
            out = model(batch)
        if str(cfg["precision"]) == "fp16":
            out = _float_tensors(out)
        eval_loss = criterion(out, batch)
        torch.cuda.synchronize()
        cell["eval_batch"] = {
            "loss": float(eval_loss.detach().cpu()),
            "finite": bool(torch.isfinite(eval_loss.detach()).item()),
            "seconds": time.perf_counter() - eval_t0,
            "terms": _loss_terms(criterion),
            "sparse_meta": _sparse_meta(model),
        }
    if not cell["eval_batch"]["finite"]:
        raise RuntimeError(f"{cell_name}: non-finite eval loss {cell['eval_batch']['loss']}")

    model.train()
    with open(step_path, "w", encoding="utf-8") as sf:
        for step in range(int(args.steps)):
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with precision_autocast_context(str(cfg["precision"]), device):
                out = model(batch)
            if str(cfg["precision"]) == "fp16":
                out = _float_tensors(out)
            loss = criterion(out, batch)
            finite_loss = bool(torch.isfinite(loss.detach()).item())
            if not finite_loss:
                raise RuntimeError(f"{cell_name}: step {step} non-finite loss {float(loss.detach().cpu())}")
            scale_before = float(scaler.get_scale())
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = _grad_norm(model)
            if not math.isfinite(grad_norm):
                raise RuntimeError(f"{cell_name}: step {step} non-finite grad_norm {grad_norm}")
            scaler.step(optimizer)
            scaler.update()
            scale_after = float(scaler.get_scale())
            skipped = bool(scaler.is_enabled() and scale_after < scale_before)
            cell["optimizer_steps"] += int(not skipped)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - t0
            loss_value = float(loss.detach().cpu())
            losses.append(loss_value)
            rec = {
                "cell": cell_name,
                "step": step,
                "loss": loss_value,
                "finite_loss": finite_loss,
                "grad_norm": grad_norm,
                "grad_scaler_scale_before": scale_before,
                "grad_scaler_scale_after": scale_after,
                "grad_scaler_skipped": skipped,
                "seconds": elapsed,
                "terms": _loss_terms(criterion),
            }
            if step == 0 or step == int(args.steps) - 1:
                rec["sparse_meta"] = _sparse_meta(model)
            sf.write(json.dumps(_jsonable(rec), sort_keys=True) + "\n")
            sf.flush()
            print(
                "[tiny-overfit]",
                cell_name,
                "step",
                step,
                "loss",
                f"{loss_value:.6g}",
                "grad_norm",
                f"{grad_norm:.6g}",
                "scale",
                f"{scale_after:.1f}",
                "skipped",
                skipped,
                flush=True,
            )

    cell["seconds"] = time.perf_counter() - start
    cell["step_log"] = str(step_path)
    cell["loss_first"] = float(losses[0]) if losses else None
    cell["loss_last"] = float(losses[-1]) if losses else None
    cell["loss_min"] = float(min(losses)) if losses else None
    cell["loss_decreased"] = bool(losses and min(losses) < losses[0])
    cell["loss_last_le_first"] = bool(losses and losses[-1] <= losses[0])
    cell["grad_scaler_final_scale"] = float(scaler.get_scale())
    if scaler.is_enabled() and int(cell["optimizer_steps"]) <= 0:
        raise RuntimeError(f"{cell_name}: GradScaler skipped every optimizer step")
    if not cell["loss_decreased"]:
        cell["warnings"].append(
            "tiny-overfit loss did not decrease below the first step; treated as engineering warning, not a scientific failure"
        )
    if cfg.get("det-lidar-encoder") == "voxel":
        meta = _sparse_meta(model) or cell["eval_batch"].get("sparse_meta", {})
        cell["sparse_meta_final"] = meta
        meta_warnings = _validate_sparse_meta(meta, cell_name)
        cell["warnings"].extend(meta_warnings)
        if meta_warnings:
            raise RuntimeError("; ".join(meta_warnings))
    return cell


def _parse_matrix(raw: str) -> List[str]:
    cells = [c.strip() for c in raw.split(",") if c.strip()]
    if not cells:
        raise ValueError("matrix is empty")
    unknown = [c for c in cells if c not in MATRIX_CELLS]
    if unknown:
        raise ValueError(
            f"unknown matrix cells {unknown}; valid canonical cells: {sorted(CANONICAL_MATRIX_CELLS)}; "
            f"legacy aliases: {sorted(LEGACY_MATRIX_ALIASES)}"
        )
    out: List[str] = []
    seen = set()
    for cell in cells:
        canonical = _canonical_cell_name(cell)
        if canonical not in seen:
            out.append(canonical)
            seen.add(canonical)
    return out


def _torch_env() -> Dict[str, Any]:
    import torch

    env = {
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
    }
    if torch.cuda.is_available():
        env["gpu0"] = torch.cuda.get_device_name(0)
        env["gpu0_cc"] = ".".join(str(x) for x in torch.cuda.get_device_capability(0))
    try:
        import spconv

        env["spconv"] = getattr(spconv, "__version__", "")
        env["spconv_file"] = getattr(spconv, "__file__", "")
    except Exception as exc:
        env["spconv_import_error"] = f"{type(exc).__name__}: {exc}"
    return env


def main() -> None:
    default_root = os.environ.get(
        "ARRHENIUS_OUTPUT_ROOT",
        "/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs",
    )
    default_dataroot = _repo_root() / "data" / "nuscenes_mini"
    default_cache = Path(default_root) / "nuscenes" / "info_cache_mini_from_main"

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(_repo_root() / "fl_v3" / "configs" / "t4_mini_smoke.json"))
    ap.add_argument("--dataroot", default=os.environ.get("ARRHENIUS_NUSCENES_DATAROOT", str(default_dataroot)))
    ap.add_argument("--cache-dir", default=os.environ.get("ARRHENIUS_NUSCENES_CACHE", str(default_cache)))
    ap.add_argument("--output-dir", default=str(Path(default_root) / "stop_c_mini_tiny_overfit"))
    ap.add_argument("--matrix", default="voxel_fp16_main,voxel_fp32_ref")
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--num-tokens", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--learning-rate", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--backbone", default="resnet18", choices=["resnet18", "swin_t"])
    ap.add_argument("--pretrained-backbone", action="store_true")
    ap.add_argument("--lidar-sweeps", type=int, default=1)
    args = ap.parse_args()

    cells = _parse_matrix(args.matrix)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Import heavy runtime only after argparse so --help works on login nodes.
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("Stop C mini matrix requires CUDA; submit via Arrhenius Slurm")

    from fl_v3.data.nuscenes import paths as P
    from fl_v3.training.tasks import get_task

    base_cfg = _build_base_cfg(args)
    dataroot = P.get_dataroot(base_cfg)
    cache_dir = Path(str(base_cfg["nuscenes-cache-dir"]))
    if not Path(dataroot).exists():
        raise FileNotFoundError(f"mini dataroot not found: {dataroot}")
    if not cache_dir.exists():
        raise FileNotFoundError(f"mini info-cache dir not found: {cache_dir}")

    task = get_task("nuscenes_detection")
    train_info, train_meta = task._load_info(base_cfg, str(base_cfg["nuscenes-train-split"]))
    tokens = _select_tokens(train_info, int(args.num_tokens))
    fixed_batch, batch_meta = _make_fixed_batch(base_cfg, tokens)

    manifest = {
        "kind": "arrhenius_stop_c_mini_tiny_overfit",
        "scientific_claim": False,
        "engineering_scope": "mini data/cached one-batch eval and tiny-overfit only",
        "git_rev": _git_rev(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
        "repo": str(_repo_root()),
        "output_dir": str(output_dir),
        "config": str(Path(args.config).resolve()),
        "dataroot": dataroot,
        "cache_dir": str(cache_dir),
        "train_cache_meta": train_meta,
        "batch_cache_meta": batch_meta,
        "selected_tokens": tokens,
        "matrix": cells,
        "args": vars(args),
        "env": _torch_env(),
        "cells": [],
    }
    manifest_path = output_dir / "mini_tiny_overfit_summary.json"
    print("[mini-matrix] output", output_dir, flush=True)
    print("[mini-matrix] cells", ",".join(cells), "tokens", len(tokens), "steps", args.steps, flush=True)
    print("[mini-matrix] dataroot", dataroot, flush=True)
    print("[mini-matrix] cache", cache_dir, flush=True)

    ok = True
    for cell in cells:
        try:
            result = _run_cell(cell, base_cfg, fixed_batch, output_dir, args)
            result["status"] = "ok"
            manifest["cells"].append(result)
        except Exception as exc:
            ok = False
            manifest["cells"].append({
                "cell": cell,
                "status": "failed",
                "error_type": type(exc).__name__,
                "error": str(exc),
            })
            print(f"[mini-matrix] FAIL {cell}: {type(exc).__name__}: {exc}", flush=True)
            break
        finally:
            with open(manifest_path, "w", encoding="utf-8") as f:
                json.dump(_jsonable(manifest), f, indent=2, sort_keys=True)

    print("[mini-matrix] summary", manifest_path, flush=True)
    for result in manifest["cells"]:
        print(
            "[mini-matrix] cell",
            result.get("cell"),
            result.get("status"),
            "first",
            result.get("loss_first"),
            "last",
            result.get("loss_last"),
            "min",
            result.get("loss_min"),
            "opt_steps",
            result.get("optimizer_steps"),
            "warnings",
            len(result.get("warnings", [])),
            flush=True,
        )
    if not ok:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
