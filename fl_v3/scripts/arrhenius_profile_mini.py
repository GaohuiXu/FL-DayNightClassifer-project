"""Arrhenius mini module profiler for engineering smoke runs.

This is a synchronized stage profiler for mini-data engineering only. It reports
where time and memory go on GH200, but it is not a throughput claim and not a
scientific mAP/NDS/ASR result.
"""
from __future__ import annotations

import argparse
import copy
import csv
import gc
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple


CANONICAL_MATRIX_CELLS = {
    "voxel_fp16_main": {"det-lidar-encoder": "voxel", "precision": "fp16", "det-sparse-conv-fp16": False},
    "voxel_fp16_sparseconv_fp16_exp": {
        "det-lidar-encoder": "voxel",
        "precision": "fp16",
        "det-sparse-conv-fp16": True,
    },
    "voxel_fp32_ref": {"det-lidar-encoder": "voxel", "precision": "fp32", "det-sparse-conv-fp16": False},
    "pillar_fp32_legacy": {"det-lidar-encoder": "pillar", "precision": "fp32", "det-sparse-conv-fp16": False},
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
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return str(obj)
    if isinstance(obj, (str, int, float, bool)) or obj is None:
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
    cfg["det-max-pillars"] = int(args.max_pillars)
    cfg["det-max-points-per-pillar"] = int(args.max_points_per_pillar)
    cfg["wandb-enabled"] = False
    cfg["wandb-mode"] = "disabled"
    return cfg


def _select_tokens(info_list: List[dict], count: int) -> List[str]:
    tokens = sorted(str(i["sample_token"]) for i in info_list)
    if not tokens:
        raise RuntimeError("mini train info-cache contains no sample tokens")
    return tokens[: max(1, int(count))]


def _make_loader(cfg: dict, tokens: List[str], args: argparse.Namespace):
    import torch
    from torch.utils.data import DataLoader

    from fl_v3.data.nuscenes import paths as P
    from fl_v3.data.nuscenes.dataset import NuScenesMultimodalDataset
    from fl_v3.models.fusion.collate import detection_collate_fn
    from fl_v3.training.tasks import get_task
    from fl_v3.utils.runtime import seeded_worker_init

    task = get_task("nuscenes_detection")
    info_list, meta = task._load_info(cfg, str(cfg["nuscenes-train-split"]))
    ds_t0 = time.perf_counter()
    ds = NuScenesMultimodalDataset(
        info_list,
        P.get_dataroot(cfg),
        sample_tokens=tokens,
        n_sweeps=int(cfg.get("det-lidar-sweeps", 1)),
        augment=None,
        gtpaste=None,
    )
    dataset_init_ms = (time.perf_counter() - ds_t0) * 1000.0
    generator = torch.Generator()
    generator.manual_seed(int(cfg.get("seed", 42)))
    extra = {}
    if int(args.num_workers) > 0:
        extra["persistent_workers"] = bool(args.persistent_workers)
        extra["prefetch_factor"] = int(args.prefetch_factor)
    loader = DataLoader(
        ds,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        worker_init_fn=seeded_worker_init,
        generator=generator,
        drop_last=False,
        collate_fn=detection_collate_fn,
        pin_memory=bool(args.pin_memory),
        **extra,
    )
    return loader, meta, dataset_init_ms


def _next_batch(loader, state: dict) -> dict:
    if "it" not in state:
        state["initializations"] = int(state.get("initializations", 0)) + 1
        state["it"] = iter(loader)
    try:
        return next(state["it"])
    except StopIteration:
        state["resets"] = int(state.get("resets", 0)) + 1
        state["it"] = iter(loader)
        return next(state["it"])


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


def _sparse_profile(model) -> Dict[str, Any]:
    enc = getattr(model, "lidar_encoder", None)
    return dict(getattr(enc, "last_profile_times", {}) or {})


class CudaTimer:
    def __init__(self, device):
        self.device = device

    def measure(self, fn):
        import torch

        torch.cuda.synchronize(self.device)
        t0 = time.perf_counter()
        result = fn()
        torch.cuda.synchronize(self.device)
        return result, (time.perf_counter() - t0) * 1000.0


class GpuSampler:
    def __init__(self, output_dir: Path, cell: str, interval_ms: int):
        self.output_dir = output_dir
        self.cell = cell
        self.interval_ms = int(interval_ms)
        self.csv_path = output_dir / f"{cell}_gpu_telemetry.csv"
        self.err_path = output_dir / f"{cell}_gpu_telemetry.err"
        self.proc = None
        self._out = None
        self._err = None

    def start(self) -> None:
        if self.interval_ms <= 0 or shutil.which("nvidia-smi") is None:
            return
        fields = (
            "timestamp,index,name,utilization.gpu,utilization.memory,"
            "memory.used,memory.total,power.draw,temperature.gpu"
        )
        cmd = [
            "nvidia-smi",
            f"--query-gpu={fields}",
            "--format=csv,nounits",
            f"--loop-ms={self.interval_ms}",
        ]
        self._out = open(self.csv_path, "w", encoding="utf-8")
        self._err = open(self.err_path, "w", encoding="utf-8")
        self.proc = subprocess.Popen(cmd, stdout=self._out, stderr=self._err)

    def stop(self) -> None:
        if self.proc is not None and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait(timeout=5)
        for handle in (self._out, self._err):
            if handle is not None:
                handle.close()

    def summary(self) -> Dict[str, Any]:
        return _summarize_gpu_csv(self.csv_path)


def _clean_csv_key(raw: str) -> str:
    return raw.split("[")[0].strip().lower().replace(" ", "_").replace(".", "_")


def _to_float(raw: str) -> float | None:
    s = str(raw).strip()
    if not s or "not supported" in s.lower():
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _summarize_gpu_csv(path: Path) -> Dict[str, Any]:
    if not path.exists() or path.stat().st_size == 0:
        return {"available": False, "path": str(path), "samples": 0}
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({_clean_csv_key(k): v for k, v in row.items()})
    if not rows:
        return {"available": False, "path": str(path), "samples": 0}
    summary: Dict[str, Any] = {"available": True, "path": str(path), "samples": len(rows)}
    for key in (
        "utilization_gpu",
        "utilization_memory",
        "memory_used",
        "memory_total",
        "power_draw",
        "temperature_gpu",
    ):
        vals = [_to_float(r.get(key, "")) for r in rows]
        nums = [v for v in vals if v is not None]
        if nums:
            summary[f"{key}_avg"] = sum(nums) / len(nums)
            summary[f"{key}_max"] = max(nums)
            summary[f"{key}_min"] = min(nums)
    if rows and "name" in rows[0]:
        summary["name"] = rows[0]["name"]
    return summary


def _nvidia_smi_static() -> Dict[str, Any]:
    if shutil.which("nvidia-smi") is None:
        return {"available": False}
    fields = "name,memory.total,driver_version,power.limit"
    try:
        out = subprocess.check_output(
            ["nvidia-smi", f"--query-gpu={fields}", "--format=csv,nounits,noheader"],
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
        return {"available": True, "query": fields, "raw": out}
    except Exception as exc:
        return {"available": False, "error": f"{type(exc).__name__}: {exc}"}


def _profile_forward(model, batch: dict, timer: CudaTimer) -> Tuple[dict, Dict[str, float]]:
    times: Dict[str, float] = {}

    pre, times["preprocess"] = timer.measure(
        lambda: model.preprocess(batch["images"], batch["lidar2img"], batch["cam_intrinsics"])
    )
    imgs = pre["images"]
    B, N = imgs.shape[0], imgs.shape[1]
    feats, times["camera_backbone"] = timer.measure(
        lambda: model.camera_backbone(imgs.reshape(B * N, *imgs.shape[2:]))
    )
    camfeat, times["camera_neck"] = timer.measure(lambda: model.camera_neck(feats))
    vt, times["view_transform_lss"] = timer.measure(lambda: model.view_transform(camfeat, pre["lidar2img"], B, N))
    camera_bev = vt["bev"]
    lidar_bev, times["lidar_encoder_total"] = timer.measure(lambda: model.lidar_encoder(batch["lidar_points"], B))
    for name, ms in _sparse_profile(model).items():
        times[f"sparse_voxel_{name}"] = float(ms)
    if model.lidar_backbone is not None:
        lidar_bev, times["lidar_backbone"] = timer.measure(lambda: model.lidar_backbone(lidar_bev))
    fused, times["fusion"] = timer.measure(lambda: model.fusion(camera_bev, lidar_bev))
    neck, times["bev_neck"] = timer.measure(lambda: model.bev_neck(fused))
    out, times["head"] = timer.measure(lambda: model.head(neck))
    return out, times


def _profile_train_step(
    model,
    criterion,
    optimizer,
    scaler,
    batch_cpu,
    precision: str,
    device,
) -> Dict[str, Any]:
    import torch

    from fl_v3.training.loop import _float_tensors, _move_to_device
    from fl_v3.utils.runtime import precision_autocast_context

    timer = CudaTimer(device)
    torch.cuda.reset_peak_memory_stats(device)
    record: Dict[str, Any] = {"stage_ms": {}}

    batch, record["stage_ms"]["h2d"] = timer.measure(lambda: _move_to_device(batch_cpu, device))
    optimizer.zero_grad(set_to_none=True)

    def forward_body():
        with precision_autocast_context(precision, device):
            return _profile_forward(model, batch, timer)

    (out, forward_stages), forward_total_ms = timer.measure(forward_body)
    record["stage_ms"]["forward_total"] = forward_total_ms
    record["stage_ms"].update(forward_stages)
    if precision == "fp16":
        out, record["stage_ms"]["fp16_output_upcast"] = timer.measure(lambda: _float_tensors(out))

    loss, record["stage_ms"]["loss"] = timer.measure(lambda: criterion(out, batch))
    finite_loss = bool(torch.isfinite(loss.detach()).item())
    if not finite_loss:
        raise RuntimeError(f"non-finite loss: {float(loss.detach().cpu())}")
    scale_before = float(scaler.get_scale())
    _, record["stage_ms"]["backward"] = timer.measure(lambda: scaler.scale(loss).backward())

    def unscale_and_grad():
        scaler.unscale_(optimizer)
        return _grad_norm(model)

    grad_norm, record["stage_ms"]["unscale_grad_norm"] = timer.measure(unscale_and_grad)
    if not math.isfinite(grad_norm):
        raise RuntimeError(f"non-finite grad_norm: {grad_norm}")

    def optimizer_step():
        scaler.step(optimizer)
        scaler.update()

    _, record["stage_ms"]["optimizer_step"] = timer.measure(optimizer_step)
    scale_after = float(scaler.get_scale())
    skipped = bool(scaler.is_enabled() and scale_after < scale_before)

    record["loss"] = float(loss.detach().cpu())
    record["finite_loss"] = finite_loss
    record["loss_terms"] = _loss_terms(criterion)
    record["grad_norm"] = float(grad_norm)
    record["grad_scaler_scale_before"] = scale_before
    record["grad_scaler_scale_after"] = scale_after
    record["grad_scaler_skipped"] = skipped
    record["optimizer_step_counted"] = bool(not skipped)
    record["memory_allocated_mib"] = torch.cuda.max_memory_allocated(device) / (1024.0 ** 2)
    record["memory_reserved_mib"] = torch.cuda.max_memory_reserved(device) / (1024.0 ** 2)
    record["sparse_meta"] = _sparse_meta(model)
    record["stage_ms"]["train_step_total"] = sum(
        float(record["stage_ms"].get(k, 0.0))
        for k in ("h2d", "forward_total", "fp16_output_upcast", "loss", "backward", "unscale_grad_norm", "optimizer_step")
    )
    return record


def _aggregate(records: List[dict], batch_size: int) -> Dict[str, Any]:
    measured = [r for r in records if r.get("phase") == "measured"]
    if not measured:
        return {"measured_steps": 0}
    measured_no_reset = [r for r in measured if not r.get("data_iterator_reset", False)]
    keys = sorted({k for r in measured for k in r.get("stage_ms", {})})
    stage_mean, stage_max = {}, {}
    for key in keys:
        vals = [float(r["stage_ms"][key]) for r in measured if key in r.get("stage_ms", {})]
        if vals:
            stage_mean[key] = sum(vals) / len(vals)
            stage_max[key] = max(vals)
    compute_vals = [float(r["stage_ms"]["train_step_total"]) for r in measured]
    end_to_end_vals = [
        float(r["stage_ms"].get(
            "end_to_end_step_total",
            r["stage_ms"]["train_step_total"] + r["stage_ms"].get("data_fetch", 0.0),
        ))
        for r in measured
    ]
    losses = [float(r["loss"]) for r in measured]
    opt_steps = sum(int(r.get("optimizer_step_counted", False)) for r in measured)
    skips = sum(int(r.get("grad_scaler_skipped", False)) for r in measured)
    mean_compute = sum(compute_vals) / len(compute_vals)
    mean_end_to_end = sum(end_to_end_vals) / len(end_to_end_vals)
    out = {
        "measured_steps": len(measured),
        "data_iterator_resets_all": sum(int(r.get("data_iterator_reset", False)) for r in records),
        "data_iterator_resets_measured": sum(int(r.get("data_iterator_reset", False)) for r in measured),
        "loss_first_measured": losses[0],
        "loss_last_measured": losses[-1],
        "loss_min_measured": min(losses),
        "optimizer_steps": opt_steps,
        "grad_scaler_skips": skips,
        "compute_step_total_ms_mean": mean_compute,
        "compute_step_total_ms_max": max(compute_vals),
        "end_to_end_step_total_ms_mean": mean_end_to_end,
        "end_to_end_step_total_ms_max": max(end_to_end_vals),
        "compute_samples_per_sec_mean": (float(batch_size) * 1000.0 / mean_compute) if mean_compute > 0 else 0.0,
        "end_to_end_samples_per_sec_mean": (
            float(batch_size) * 1000.0 / mean_end_to_end if mean_end_to_end > 0 else 0.0
        ),
        # Back-compat alias for early Stop-D drafts: this is compute-only and excludes data_fetch.
        "train_step_total_ms_mean": mean_compute,
        "train_step_total_ms_max": max(compute_vals),
        "stage_ms_mean": stage_mean,
        "stage_ms_max": stage_max,
        "memory_allocated_mib_peak": max(float(r["memory_allocated_mib"]) for r in measured),
        "memory_reserved_mib_peak": max(float(r["memory_reserved_mib"]) for r in measured),
    }
    if measured_no_reset:
        data_fetch_vals = [
            float(r.get("stage_ms", {}).get("data_fetch", r.get("data_fetch_ms", 0.0)))
            for r in measured_no_reset
        ]
        e2e_no_reset_vals = [
            float(r["stage_ms"].get(
                "end_to_end_step_total",
                r["stage_ms"]["train_step_total"] + r["stage_ms"].get("data_fetch", 0.0),
            ))
            for r in measured_no_reset
        ]
        mean_e2e_no_reset = sum(e2e_no_reset_vals) / len(e2e_no_reset_vals)
        out.update({
            "measured_steps_no_iterator_reset": len(measured_no_reset),
            "data_fetch_ms_mean_no_iterator_reset": sum(data_fetch_vals) / len(data_fetch_vals),
            "data_fetch_ms_max_no_iterator_reset": max(data_fetch_vals),
            "end_to_end_step_total_ms_mean_no_iterator_reset": mean_e2e_no_reset,
            "end_to_end_samples_per_sec_mean_no_iterator_reset": (
                float(batch_size) * 1000.0 / mean_e2e_no_reset if mean_e2e_no_reset > 0 else 0.0
            ),
        })
    else:
        out["measured_steps_no_iterator_reset"] = 0
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
        "nvidia_smi": _nvidia_smi_static(),
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


def _run_cell(
    cell_name: str,
    base_cfg: dict,
    loader,
    output_dir: Path,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    import torch

    from fl_v3.training.tasks import get_task
    from fl_v3.utils.runtime import (
        enforce_determinism,
        make_grad_scaler,
        precision_state,
        seed_everything,
        validate_sparse_precision,
    )

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
    build_t0 = time.perf_counter()
    model = task.build_model(cfg).to(device).train()
    model_build_ms = (time.perf_counter() - build_t0) * 1000.0
    if cfg.get("det-lidar-encoder") == "voxel":
        getattr(model, "lidar_encoder").record_debug = True
        getattr(model, "lidar_encoder").record_profile = True
    criterion = task.build_criterion(cfg)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
    )
    scaler = make_grad_scaler(device, str(cfg["precision"]))

    sampler = GpuSampler(output_dir, cell_name, int(args.gpu_sample_ms))
    sampler.start()
    state: dict = {}
    records: List[dict] = []
    step_path = output_dir / f"{cell_name}_profile_steps.jsonl"
    ok = True
    error = ""
    try:
        with open(step_path, "w", encoding="utf-8") as sf:
            total_steps = int(args.warmup_iters) + int(args.profile_iters)
            for step in range(total_steps):
                fetch_t0 = time.perf_counter()
                resets_before = int(state.get("resets", 0))
                batch_cpu = _next_batch(loader, state)
                reset_this_step = int(state.get("resets", 0)) > resets_before
                fetch_ms = (time.perf_counter() - fetch_t0) * 1000.0
                rec = _profile_train_step(
                    model,
                    criterion,
                    optimizer,
                    scaler,
                    batch_cpu,
                    str(cfg["precision"]),
                    device,
                )
                rec.update({
                    "cell": cell_name,
                    "step": step,
                    "phase": "warmup" if step < int(args.warmup_iters) else "measured",
                    "data_fetch_ms": fetch_ms,
                    "data_iterator_reset": reset_this_step,
                    "data_iterator_resets_total": int(state.get("resets", 0)),
                    "data_iterator_initializations_total": int(state.get("initializations", 0)),
                    "batch_size": int(batch_cpu.get("batch_size", len(batch_cpu.get("gt_boxes", [])))),
                    "sample_tokens": list(batch_cpu.get("sample_token", [])),
                })
                rec["stage_ms"]["data_fetch"] = fetch_ms
                rec["stage_ms"]["end_to_end_step_total"] = rec["stage_ms"]["train_step_total"] + fetch_ms
                sf.write(json.dumps(_jsonable(rec), sort_keys=True) + "\n")
                sf.flush()
                records.append(rec)
                print(
                    "[profile]",
                    cell_name,
                    rec["phase"],
                    "step",
                    step,
                    "loss",
                    f"{rec['loss']:.6g}",
                    "compute_ms",
                    f"{rec['stage_ms']['train_step_total']:.2f}",
                    "e2e_ms",
                    f"{rec['stage_ms']['end_to_end_step_total']:.2f}",
                    "gpu_mem_mib",
                    f"{rec['memory_allocated_mib']:.1f}",
                    "fetch_ms",
                    f"{fetch_ms:.2f}",
                    "reset",
                    reset_this_step,
                    "scale",
                    f"{rec['grad_scaler_scale_after']:.1f}",
                    "skip",
                    rec["grad_scaler_skipped"],
                    flush=True,
                )
    except Exception as exc:
        ok = False
        error = f"{type(exc).__name__}: {exc}"
    finally:
        sampler.stop()

    aggregate = _aggregate(records, int(args.batch_size))
    result = {
        "cell": cell_name,
        "status": "ok" if ok else "failed",
        "error": error,
        "precision": cfg["precision"],
        "lidar_encoder": cfg["det-lidar-encoder"],
        "sparse_conv_fp16": bool(cfg.get("det-sparse-conv-fp16", False)),
        "model_build_ms": model_build_ms,
        "precision_state": precision_state(),
        "grad_scaler_enabled": bool(scaler.is_enabled()),
        "grad_scaler_final_scale": float(scaler.get_scale()),
        "step_log": str(step_path),
        "gpu_telemetry": sampler.summary(),
        "aggregate": aggregate,
    }
    if records:
        result["sparse_meta_last"] = records[-1].get("sparse_meta", {})
    del model, criterion, optimizer, scaler
    gc.collect()
    torch.cuda.empty_cache()
    if not ok:
        raise RuntimeError(f"{cell_name}: {error}")
    return result


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
    ap.add_argument("--output-dir", default=str(Path(default_root) / "stop_d_profile_mini"))
    ap.add_argument("--matrix", default="voxel_fp16_main")
    ap.add_argument("--warmup-iters", type=int, default=4)
    ap.add_argument("--profile-iters", type=int, default=8)
    ap.add_argument("--num-tokens", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--pin-memory", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--persistent-workers", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--prefetch-factor", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--learning-rate", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--backbone", default="resnet18", choices=["resnet18", "swin_t"])
    ap.add_argument("--pretrained-backbone", action="store_true")
    ap.add_argument("--lidar-sweeps", type=int, default=1)
    ap.add_argument("--max-pillars", type=int, default=30000)
    ap.add_argument("--max-points-per-pillar", type=int, default=32)
    ap.add_argument("--gpu-sample-ms", type=int, default=200)
    args = ap.parse_args()

    cells = _parse_matrix(args.matrix)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("Stop D mini profiler requires CUDA; submit via Arrhenius Slurm")

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
    cache_t0 = time.perf_counter()
    train_info, train_meta = task._load_info(base_cfg, str(base_cfg["nuscenes-train-split"]))
    cache_load_ms = (time.perf_counter() - cache_t0) * 1000.0
    token_count = max(int(args.num_tokens), int(args.batch_size))
    tokens = _select_tokens(train_info, token_count)
    loader, loader_meta, dataset_init_ms = _make_loader(base_cfg, tokens, args)
    total_profile_steps = int(args.warmup_iters) + int(args.profile_iters)
    tokens_for_full_batches_without_reset = int(args.batch_size) * total_profile_steps
    data_window = {
        "selected_token_count": len(tokens),
        "batch_size": int(args.batch_size),
        "warmup_iters": int(args.warmup_iters),
        "profile_iters": int(args.profile_iters),
        "total_steps": total_profile_steps,
        "tokens_for_full_batches_without_iterator_reset": tokens_for_full_batches_without_reset,
        "iterator_reset_expected": len(tokens) < tokens_for_full_batches_without_reset,
    }

    manifest = {
        "kind": "arrhenius_stop_d_mini_profile",
        "scientific_claim": False,
        "engineering_scope": "mini module/stage profiling only",
        "git_rev": _git_rev(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
        "repo": str(_repo_root()),
        "output_dir": str(output_dir),
        "config": str(Path(args.config).resolve()),
        "dataroot": dataroot,
        "cache_dir": str(cache_dir),
        "train_cache_meta": train_meta,
        "loader_cache_meta": loader_meta,
        "selected_tokens": tokens,
        "matrix": cells,
        "args": vars(args),
        "data_window": data_window,
        "data_timing_ms": {
            "cache_load": cache_load_ms,
            "dataset_init": dataset_init_ms,
        },
        "env": _torch_env(),
        "cells": [],
    }
    manifest_path = output_dir / "profile_summary.json"
    print("[profile] output", output_dir, flush=True)
    print("[profile] cells", ",".join(cells), "warmup", args.warmup_iters, "iters", args.profile_iters, flush=True)
    print("[profile] dataroot", dataroot, flush=True)
    print("[profile] cache", cache_dir, flush=True)
    print("[profile] tokens", len(tokens), "batch_size", args.batch_size, "workers", args.num_workers, flush=True)
    print(
        "[profile] data_window",
        "tokens_for_full_batches_without_reset",
        tokens_for_full_batches_without_reset,
        "iterator_reset_expected",
        data_window["iterator_reset_expected"],
        flush=True,
    )

    ok = True
    for cell in cells:
        try:
            result = _run_cell(cell, base_cfg, loader, output_dir, args)
            manifest["cells"].append(result)
        except Exception as exc:
            ok = False
            manifest["cells"].append({
                "cell": cell,
                "status": "failed",
                "error_type": type(exc).__name__,
                "error": str(exc),
            })
            print(f"[profile] FAIL {cell}: {type(exc).__name__}: {exc}", flush=True)
            break
        finally:
            with open(manifest_path, "w", encoding="utf-8") as f:
                json.dump(_jsonable(manifest), f, indent=2, sort_keys=True)

    print("[profile] summary", manifest_path, flush=True)
    for result in manifest["cells"]:
        agg = result.get("aggregate", {})
        print(
            "[profile] cell",
            result.get("cell"),
            result.get("status"),
            "mean_compute_ms",
            agg.get("compute_step_total_ms_mean"),
            "mean_e2e_ms",
            agg.get("end_to_end_step_total_ms_mean"),
            "resets_measured",
            agg.get("data_iterator_resets_measured"),
            "mean_e2e_no_reset_ms",
            agg.get("end_to_end_step_total_ms_mean_no_iterator_reset"),
            "e2e_samples_per_sec",
            agg.get("end_to_end_samples_per_sec_mean"),
            "peak_alloc_mib",
            agg.get("memory_allocated_mib_peak"),
            "gpu_util_avg",
            result.get("gpu_telemetry", {}).get("utilization_gpu_avg"),
            flush=True,
        )
    if not ok:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
