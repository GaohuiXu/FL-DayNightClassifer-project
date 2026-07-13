"""Arrhenius GH200 environment smoke tests.

This is an environment/integration harness, not a scientific result. It covers:
import sanity, single-GPU spconv sparse-conv FP32/FP16-AMP, optional nuScenes
data/cache preflight, a tiny proxy eval, and a minimal real-data train smoke if
the nuScenes dataroot/cache have been staged.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
from itertools import islice, cycle
from pathlib import Path
from typing import Any


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


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
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)


def _sparse_meta(model) -> dict:
    enc = getattr(model, "lidar_encoder", None)
    return dict(getattr(enc, "last_sparse_meta", {}) or {})


def _load_cfg(path: str, args: argparse.Namespace) -> dict:
    from fl_v3.utils.runtime import validate_sparse_precision

    cfg = json.load(open(path, encoding="utf-8")) if path else {
        "task-type": "nuscenes_detection",
        "seed": 42,
        "device": "cuda",
        "output-dir": "./fl_outputs/nuscenes/arrhenius_smoke",
        "nuscenes-dataroot": "",
        "nuscenes-cache-dir": "./fl_outputs/nuscenes/info_cache",
        "nuscenes-version": "v1.0-mini",
        "nuscenes-train-split": "mini_train",
        "nuscenes-val-split": "mini_val",
        "nuscenes-partition-mode": "log_group",
        "nuscenes-num-clients": 4,
        "min-keyframes-per-client": 1,
        "num-server-rounds": 1,
        "fraction-train": 1.0,
        "num-local-epochs": 1,
        "batch-size": 1,
        "num-workers": 0,
        "det-camera-backbone": "resnet18",
        "det-lidar-encoder": "voxel",
        "det-sparse-conv-fp16": False,
        "det-freeze-backbone": True,
        "det-pretrained-backbone": True,
        "det-eval-limit": 2,
        "wandb-enabled": False,
        "wandb-mode": "disabled",
    }
    if args.dataroot:
        cfg["nuscenes-dataroot"] = args.dataroot
    if args.cache_dir:
        cfg["nuscenes-cache-dir"] = args.cache_dir
    if args.output_dir:
        cfg["output-dir"] = args.output_dir
    cfg["device"] = "cuda"
    cfg["precision"] = args.precision
    cfg["det-eval-limit"] = int(args.eval_limit)
    cfg["num-workers"] = int(args.num_workers)
    cfg["batch-size"] = int(args.batch_size)
    cfg["det-camera-backbone"] = args.backbone
    cfg["det-pretrained-backbone"] = True
    if args.lidar_encoder:
        cfg["det-lidar-encoder"] = args.lidar_encoder
    if args.sparse_conv_fp16 is not None:
        cfg["det-sparse-conv-fp16"] = bool(args.sparse_conv_fp16)
    else:
        cfg["det-sparse-conv-fp16"] = bool(cfg.get("det-sparse-conv-fp16", False))
    validate_sparse_precision(cfg.get("precision"), cfg.get("det-lidar-encoder", "pillar"))
    if int(args.min_keyframes_per_client) > 0:
        cfg["min-keyframes-per-client"] = int(args.min_keyframes_per_client)
    cfg["wandb-enabled"] = False
    cfg["wandb-mode"] = "disabled"
    return cfg


def import_sanity() -> dict:
    import numpy
    import scipy
    import torch
    import torchvision
    import flwr
    import ray
    import sklearn
    import matplotlib
    import cumm
    import spconv
    import spconv.pytorch as spconv_torch
    import fl_v3

    print("[import] machine", platform.machine())
    print("[import] python", sys.version.replace("\n", " "))
    print("[import] torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("[import] gpu0", torch.cuda.get_device_name(0), "cap", torch.cuda.get_device_capability(0))
    print("[import] torchvision", torchvision.__version__)
    print("[import] numpy", numpy.__version__, "scipy", scipy.__version__)
    print("[import] flwr", flwr.__version__, "ray", ray.__version__, "sklearn", sklearn.__version__)
    print("[import] matplotlib", matplotlib.__version__)
    print("[import] cumm", getattr(cumm, "__version__", "?"), cumm.__file__)
    print("[import] spconv", getattr(spconv, "__version__", "?"), spconv_torch)
    print("[import] fl_v3", getattr(fl_v3, "__version__", "?"))
    print("[import] OK")
    return {
        "machine": platform.machine(),
        "python": sys.version.replace("\n", " "),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
        "gpu0": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "",
        "torchvision": torchvision.__version__,
        "numpy": numpy.__version__,
        "scipy": scipy.__version__,
        "flwr": flwr.__version__,
        "ray": ray.__version__,
        "sklearn": sklearn.__version__,
        "matplotlib": matplotlib.__version__,
        "cumm": getattr(cumm, "__version__", "?"),
        "cumm_file": getattr(cumm, "__file__", ""),
        "spconv": getattr(spconv, "__version__", "?"),
    }


def spconv_smoke() -> dict:
    import torch
    import spconv.pytorch as spconv

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for spconv_smoke")
    dev = torch.device("cuda:0")
    torch.cuda.set_device(dev)
    torch.manual_seed(12345)

    def make_input(seed: int):
        torch.manual_seed(seed)
        d, h, w = 16, 64, 64
        n, c = 2048, 16
        lin = torch.randperm(d * h * w, device=dev)[:n]
        z = lin // (h * w)
        y = (lin // w) % h
        x = lin % w
        idx = torch.stack([torch.zeros_like(z), z, y, x], dim=1).to(torch.int32)
        feat = torch.randn(n, c, device=dev, dtype=torch.float32, requires_grad=True)
        return spconv.SparseConvTensor(feat, idx, [d, h, w], 1), feat

    def make_net():
        return spconv.SparseSequential(
            spconv.SubMConv3d(16, 32, kernel_size=3, padding=1, bias=False),
            torch.nn.ReLU(),
            spconv.SparseConv3d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            torch.nn.ReLU(),
            spconv.SubMConv3d(64, 64, kernel_size=3, padding=1, bias=False),
            torch.nn.ReLU(),
        ).to(dev)

    net = make_net()
    x, feat = make_input(100)
    y = net(x)
    loss = y.features.float().square().mean() * 1_000_000
    loss.backward()
    torch.cuda.synchronize()
    fp32_shape = tuple(y.features.shape)
    fp32_dtype = str(y.features.dtype)
    fp32_loss = float(loss.detach().cpu())
    fp32_grad = float(feat.grad.detach().float().norm().cpu())
    print("[spconv] FP32_OK", "out", fp32_shape, "dtype", y.features.dtype,
          "loss", fp32_loss, "feat_grad", fp32_grad)

    net = make_net()
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3)
    scaler = torch.amp.GradScaler("cuda", init_scale=512.0)
    out_dtypes = []
    for step in range(2):
        opt.zero_grad(set_to_none=True)
        x, _ = make_input(200 + step)
        with torch.autocast("cuda", dtype=torch.float16):
            y = net(x)
            loss = y.features.float().square().mean() * 1_000_000
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        torch.cuda.synchronize()
        out_dtypes.append(str(y.features.dtype))
    print("[spconv] FP16_AMP_GRADSCALER_OK", "scale", scaler.get_scale(), "out_dtypes", out_dtypes)
    return {
        "fp32": {
            "out_shape": fp32_shape,
            "out_dtype": fp32_dtype,
            "loss": fp32_loss,
            "feat_grad_norm": fp32_grad,
        },
        "fp16_amp": {
            "grad_scaler_final_scale": float(scaler.get_scale()),
            "out_dtypes": out_dtypes,
        },
    }


def sparse_lidar_smoke() -> dict:
    import torch
    from fl_v3.models.fusion.bev_grid import BEVConfig
    from fl_v3.models.fusion.sparse_voxel_encoder import SparseVoxelEncoder

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for sparse_lidar_smoke")
    dev = torch.device("cuda:0")
    torch.cuda.set_device(dev)
    torch.manual_seed(20260701)

    cfg = BEVConfig(point_cloud_range=(-4.0, -4.0, -2.0, 4.0, 4.0, 2.0), bev_voxel=(1.0, 1.0))
    pts = torch.tensor(
        [
            [0, -3.5, -3.5, -1.5, 0.10, 0],
            [1,  1.2, -1.1, -0.6, 0.40, 0],
            [0, -3.4, -3.4, -1.4, 0.20, 0],
            [1,  2.7,  2.9,  1.2, 0.50, 0],
            [0, -0.2,  0.2,  0.1, 0.30, 0],
            [1,  9.0,  0.0,  0.0, 0.60, 0],  # out of range; should be dropped
        ],
        device=dev,
        dtype=torch.float32,
    )

    def run_case(name: str, sparse_conv_fp16: bool) -> dict:
        enc = SparseVoxelEncoder(
            out_channels=8,
            cfg=cfg,
            max_voxels=128,
            max_points_per_voxel=8,
            sparse_conv_fp16=sparse_conv_fp16,
        ).to(dev).train()
        enc.record_debug = True
        y = enc(pts, B=2)
        meta = enc.last_sparse_meta or {}
        expected_feature_dtype = "torch.float16" if sparse_conv_fp16 else "torch.float32"
        assert y.shape == (2, 8, cfg.ny, cfg.nx), y.shape
        assert torch.isfinite(y).all()
        assert meta.get("coord_order") == "bzyx", meta
        assert meta.get("indices_dtype") == "torch.int32", meta
        assert meta.get("features_dtype") == expected_feature_dtype, meta
        assert meta.get("vfe_features_dtype") == "torch.float32", meta
        assert meta.get("sparse_conv_fp16_active") is sparse_conv_fp16, meta
        assert meta.get("point_grouping") == "sorted", meta
        assert meta.get("spatial_shape") == (enc.nz, cfg.ny, cfg.nx), meta
        assert meta.get("batch_index_min") == 0 and meta.get("batch_index_max") == 1, meta
        (y.float().square().mean()).backward()
        grad_norm = torch.nn.utils.clip_grad_norm_([p for p in enc.parameters() if p.requires_grad], 1e9)

        occ = enc.occupancy(pts, B=2)
        assert occ.shape == (2, cfg.ny, cfg.nx)
        assert float(occ.sum().detach().cpu()) == 5.0

        enc.zero_grad(set_to_none=True)
        empty = enc(pts[:0], B=2)
        assert empty.shape == (2, 8, cfg.ny, cfg.nx)
        assert torch.count_nonzero(empty.detach()) == 0
        empty.sum().backward()
        assert all(p.grad is not None for p in enc.parameters() if p.requires_grad)
        print(
            "[sparse-lidar] OK",
            name,
            "out",
            tuple(y.shape),
            "meta",
            meta,
            "grad_norm",
            float(grad_norm),
        )
        return {
            "out_shape": tuple(y.shape),
            "out_dtype": str(y.dtype),
            "finite": bool(torch.isfinite(y).all().item()),
            "sparse_meta": meta,
            "occupancy_sum": float(occ.sum().detach().cpu()),
            "empty_output_shape": tuple(empty.shape),
            "empty_nonzero": int(torch.count_nonzero(empty.detach()).cpu()),
            "grad_norm": float(grad_norm),
        }

    return {
        "sparse_fp32_ref": run_case("sparse_fp32_ref", sparse_conv_fp16=False),
        "sparse_fp16_amp": run_case("sparse_fp16_amp", sparse_conv_fp16=True),
    }


def data_preflight(cfg: dict) -> dict:
    from fl_v3.data.nuscenes import paths as P
    from fl_v3.data.nuscenes import info_cache as IC

    dataroot = P.get_dataroot(cfg)
    version = str(cfg["nuscenes-version"])
    train_split = str(cfg["nuscenes-train-split"])
    val_split = str(cfg["nuscenes-val-split"])
    print("[data] dataroot", dataroot)
    print("[data] cache", cfg["nuscenes-cache-dir"])
    ok = True
    report = {
        "ok": True,
        "dataroot": dataroot,
        "cache_dir": cfg["nuscenes-cache-dir"],
        "version": version,
        "splits": {},
    }
    try:
        dataset_report = P.verify_dataset(version, dataroot)
        report["verify_dataset"] = dataset_report
        print("[data] verify_dataset", dataset_report)
    except Exception as exc:
        ok = False
        report["verify_dataset_error"] = f"{type(exc).__name__}: {exc}"
        print("[data] verify_dataset FAIL", type(exc).__name__, exc)
    for split in (train_split, val_split):
        try:
            info, meta = IC.load_cache(str(cfg["nuscenes-cache-dir"]), version, split)
            report["splits"][split] = {
                "ok": True,
                "n": len(info),
                "content_hash": meta.get("content_hash", ""),
                "meta": meta,
            }
            print("[data] cache OK", split, "n", len(info), "hash", meta.get("content_hash", "")[:16])
        except Exception as exc:
            ok = False
            report["splits"][split] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
            print("[data] cache FAIL", split, type(exc).__name__, exc)
    report["ok"] = ok
    return report


def eval_smoke(cfg: dict) -> dict:
    import torch
    from fl_v3.training.tasks import get_task
    from fl_v3.utils.runtime import enforce_determinism, seed_everything

    seed_everything(int(cfg.get("seed", 42)))
    enforce_determinism(strict=False, precision=str(cfg.get("precision", "fp16")))
    task = get_task("nuscenes_detection")
    device = torch.device("cuda")
    model = task.build_model(cfg).to(device).eval()
    if cfg.get("det-lidar-encoder") == "voxel":
        getattr(model, "lidar_encoder").record_debug = True
    crit = task.build_criterion(cfg)
    metrics = task.evaluate(model, task.eval_loader(cfg), crit, device, cfg)
    print("[eval] OK", metrics)
    return {
        "metrics": metrics,
        "precision": str(cfg.get("precision", "fp16")),
        "lidar_encoder": str(cfg.get("det-lidar-encoder", "pillar")),
        "sparse_meta": _sparse_meta(model),
    }


def train_smoke(cfg: dict, steps: int) -> dict:
    import torch
    from fl_v3.training.tasks import get_task, _aug_from_run
    from fl_v3.data.nuscenes.dataset import NuScenesMultimodalDataset, make_loader
    from fl_v3.data.nuscenes import paths as P
    from fl_v3.models.fusion.collate import detection_collate_fn
    from fl_v3.training.loop import _float_tensors, _move_to_device
    from fl_v3.utils.runtime import (
        enforce_determinism,
        make_grad_scaler,
        precision_autocast_context,
        seed_everything,
    )

    seed_everything(int(cfg.get("seed", 42)))
    enforce_determinism(strict=False, precision=str(cfg.get("precision", "fp16")))
    task = get_task("nuscenes_detection")
    part = task._partition(cfg)
    toks = sorted({t for ts in part["client_tokens"].values() for t in ts})[: max(int(cfg["batch-size"]) * steps, 1)]
    info, _ = task._load_info(cfg, str(cfg["nuscenes-train-split"]))
    ds = NuScenesMultimodalDataset(
        info,
        P.get_dataroot(cfg),
        sample_tokens=toks,
        n_sweeps=int(cfg.get("det-lidar-sweeps", 1)),
        augment=_aug_from_run(cfg),
    )
    loader = make_loader(
        ds,
        batch_size=int(cfg["batch-size"]),
        shuffle=True,
        num_workers=int(cfg.get("num-workers", 0)),
        seed=int(cfg.get("seed", 42)),
        collate_fn=detection_collate_fn,
    )
    device = torch.device("cuda")
    model = task.build_model(cfg).to(device).train()
    if cfg.get("det-lidar-encoder") == "voxel":
        getattr(model, "lidar_encoder").record_debug = True
    crit = task.build_criterion(cfg)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
    precision = str(cfg.get("precision", "fp16"))
    scaler = make_grad_scaler(device, precision)
    print("[train] start", "steps", steps, "batch", cfg["batch-size"], "tokens", len(toks), "precision", cfg["precision"])
    print("[train] lidar_encoder", cfg.get("det-lidar-encoder", "pillar"))
    ok = True
    step_records = []
    for i, batch in enumerate(islice(cycle(loader), steps)):
        opt.zero_grad(set_to_none=True)
        batch = _move_to_device(batch, device)
        with precision_autocast_context(precision, device):
            out = model(batch)
        out = _float_tensors(out)
        loss = crit(out, batch)
        before = scaler.get_scale()
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        grad_norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1e9)
        scaler.step(opt)
        scaler.update()
        finite = bool(torch.isfinite(loss).item())
        skipped = bool(scaler.get_scale() < before)
        ok = ok and finite
        print("[train] step", i, "loss", float(loss.detach().cpu()), "finite", finite,
              "grad_norm", float(grad_norm), "scale", scaler.get_scale(), "skipped", skipped)
        step_records.append({
            "step": i,
            "loss": float(loss.detach().cpu()),
            "finite": finite,
            "grad_norm": float(grad_norm),
            "grad_scaler_scale_before": float(before),
            "grad_scaler_scale_after": float(scaler.get_scale()),
            "grad_scaler_skipped": skipped,
            "sparse_meta": _sparse_meta(model),
        })
    if not ok:
        raise RuntimeError("train smoke saw non-finite loss")
    print("[train] OK")
    return {
        "precision": precision,
        "lidar_encoder": str(cfg.get("det-lidar-encoder", "pillar")),
        "steps_requested": int(steps),
        "tokens": toks,
        "grad_scaler_enabled": bool(scaler.is_enabled()),
        "grad_scaler_final_scale": float(scaler.get_scale()),
        "grad_scaler_skips": sum(int(r["grad_scaler_skipped"]) for r in step_records),
        "finite": all(bool(r["finite"]) for r in step_records),
        "steps": step_records,
    }


def dummy_train_smoke() -> dict:
    from fl_v3.engine.local_runner import run_clean_rounds

    cfg = {
        "task-type": "dummy_regression",
        "seed": 42,
        "device": "cuda",
        "num-clients": 2,
        "num-local-epochs": 1,
        "batch-size": 16,
        "learning-rate": 0.01,
        "weight-decay": 0.0,
        "precision": "fp32",
    }
    result = run_clean_rounds(cfg, num_rounds=1, fraction_train=1.0)
    print("[dummy-train] OK", result["final_checksum"][:16], result["final_eval"])
    return {
        "final_checksum": result["final_checksum"],
        "final_eval": result["final_eval"],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--config",
        default="",
        help="optional flat JSON config; empty uses the built-in clean mini smoke config",
    )
    ap.add_argument("--dataroot", default=os.environ.get("ARRHENIUS_NUSCENES_DATAROOT", ""))
    ap.add_argument("--cache-dir", default=os.environ.get("ARRHENIUS_NUSCENES_CACHE", ""))
    ap.add_argument("--output-dir", default=os.environ.get("ARRHENIUS_OUTPUT_ROOT", ""))
    ap.add_argument("--precision", default="fp16", choices=["fp16", "fp32"])
    ap.add_argument("--eval-limit", type=int, default=2)
    ap.add_argument("--train-steps", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--min-keyframes-per-client", type=int, default=0)
    ap.add_argument("--backbone", default="resnet18", choices=["resnet18", "swin_t"])
    ap.add_argument("--lidar-encoder", default="", choices=["", "pillar", "voxel"])
    ap.add_argument("--sparse-conv-fp16", action=argparse.BooleanOptionalAction, default=None,
                    help="voxel only; default follows config and remains off in canonical mini smoke")
    ap.add_argument("--require-data", action="store_true")
    ap.add_argument("modes", nargs="*", default=["import", "spconv", "data", "dummy-train"])
    args = ap.parse_args()

    cfg = _load_cfg(args.config, args)
    if not cfg.get("nuscenes-cache-dir"):
        cfg["nuscenes-cache-dir"] = str(Path(args.output_dir or ".") / "nuscenes" / "info_cache")
    output_dir = Path(str(cfg.get("output-dir") or args.output_dir or ".")).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    run_stamp = os.environ.get("SLURM_JOB_ID", "manual")
    summary_path = output_dir / f"arrhenius_smoke_summary_{run_stamp}.json"
    summary = {
        "kind": "arrhenius_mini_regression_smoke",
        "scientific_claim": False,
        "engineering_scope": "mini import/data/cache/sparse/eval/train smoke only",
        "git_rev": _git_rev(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
        "repo": str(_repo_root()),
        "config": str(Path(args.config).resolve()) if args.config else "<built-in-clean-mini>",
        "output_dir": str(output_dir),
        "summary_path": str(summary_path),
        "precision": str(cfg.get("precision", "fp16")),
        "lidar_encoder": str(cfg.get("det-lidar-encoder", "pillar")),
        "sparse_conv_fp16": bool(cfg.get("det-sparse-conv-fp16", False)),
        "dataroot": str(cfg.get("nuscenes-dataroot", "")),
        "cache_dir": str(cfg.get("nuscenes-cache-dir", "")),
        "modes_requested": list(args.modes),
        "args": vars(args),
        "modes": [],
    }

    def write_summary() -> None:
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(_jsonable(summary), f, indent=2, sort_keys=True)

    write_summary()
    print("[smoke] summary", summary_path)
    print("[smoke] engineering_scope", summary["engineering_scope"])
    print("[smoke] git_rev", summary["git_rev"])
    print(
        "[smoke] precision",
        summary["precision"],
        "lidar_encoder",
        summary["lidar_encoder"],
        "sparse_conv_fp16",
        summary["sparse_conv_fp16"],
    )

    for mode in args.modes:
        record = {"mode": mode, "status": "started"}
        summary["modes"].append(record)
        write_summary()
        try:
            if mode == "import":
                record["result"] = import_sanity()
            elif mode == "spconv":
                record["result"] = spconv_smoke()
            elif mode == "sparse-lidar":
                record["result"] = sparse_lidar_smoke()
            elif mode == "data":
                report = data_preflight(cfg)
                record["result"] = report
                if args.require_data and not report["ok"]:
                    record["status"] = "failed"
                    write_summary()
                    raise SystemExit(3)
            elif mode == "eval":
                report = data_preflight(cfg)
                record["data_preflight"] = report
                if not report["ok"]:
                    if args.require_data:
                        record["status"] = "failed"
                        write_summary()
                        raise SystemExit(3)
                    print("[eval] SKIP data/cache unavailable")
                    record["status"] = "skipped"
                    write_summary()
                    continue
                record["result"] = eval_smoke(cfg)
            elif mode == "train":
                report = data_preflight(cfg)
                record["data_preflight"] = report
                if not report["ok"]:
                    if args.require_data:
                        record["status"] = "failed"
                        write_summary()
                        raise SystemExit(3)
                    print("[train] SKIP data/cache unavailable")
                    record["status"] = "skipped"
                    write_summary()
                    continue
                record["result"] = train_smoke(cfg, args.train_steps)
            elif mode == "dummy-train":
                record["result"] = dummy_train_smoke()
            else:
                raise ValueError(f"unknown mode {mode!r}")
            if record.get("status") == "started":
                record["status"] = "ok"
        except BaseException as exc:
            record["status"] = "failed"
            record["error_type"] = type(exc).__name__
            record["error"] = str(exc)
            write_summary()
            raise
        finally:
            write_summary()


if __name__ == "__main__":
    main()
