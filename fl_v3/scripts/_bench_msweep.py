"""Micro-benchmark the multi-sweep DELTA: loader (lidar load+compensate) + PointPillarsEncoder
(6-pass sort + scatter), single (n=1) vs multi (n=10). Camera/fusion/head are untouched by the lever,
so this isolates the entire 3→6 min/epoch slowdown. Login-GPU; no SLURM, no training."""
import sys, time
sys.path.insert(0, "fl_v3/src")
import numpy as np
import torch

from fl_v3.data.nuscenes import info_cache as IC, paths as P
from fl_v3.data.nuscenes.dataset import NuScenesMultimodalDataset
from fl_v3.models.fusion.collate import detection_collate_fn
from fl_v3.models.fusion.detector import DetectorConfig
from fl_v3.models.fusion.lidar_encoder import PointPillarsEncoder
from fl_v3.utils.runtime import enforce_determinism

enforce_determinism(strict=False, precision="bf16")
dev = torch.device("cuda")
CACHE = "./fl_outputs/nuscenes/info_cache_msweep10"
ms, _ = IC.load_cache(CACHE, "v1.0-mini", "mini_train")
toks = [i["sample_token"] for i in ms[:40]]
B = 4

def loader_time(n_sweeps):
    ds = NuScenesMultimodalDataset(ms, P.DATAROOT, sample_tokens=toks, n_sweeps=n_sweeps)
    # per-sample __getitem__ (the lidar portion dominates the delta) + collate of B
    t0 = time.perf_counter()
    nb = 0
    for s in range(0, len(ds) - B, B):
        batch = [ds[s + j] for j in range(B)]
        _ = detection_collate_fn(batch)
        nb += 1
    dt = (time.perf_counter() - t0) / max(nb, 1)
    npts = _["lidar_points"].shape[0]
    return dt * 1000.0, npts

def enc_time(n_sweeps, iters=30):
    ds = NuScenesMultimodalDataset(ms, P.DATAROOT, sample_tokens=toks, n_sweeps=n_sweeps)
    batch = detection_collate_fn([ds[j] for j in range(B)])
    pts = batch["lidar_points"].to(dev)
    enc = PointPillarsEncoder(out_channels=64, max_points=32, max_pillars=80000,
                              use_timestamp=(n_sweeps > 1)).to(dev)
    for _ in range(5):  # warmup
        enc(pts, B)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        enc(pts, B)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1000.0, pts.shape[0]

for n in (1, 10):
    lt, npts_l = loader_time(n)
    et, npts_e = enc_time(n)
    print(f"n_sweeps={n:2d}: loader/batch(B={B})={lt:7.1f} ms ({npts_l:>7d} pts) | "
          f"encoder.fwd={et:6.2f} ms ({npts_e:>7d} pts)")
