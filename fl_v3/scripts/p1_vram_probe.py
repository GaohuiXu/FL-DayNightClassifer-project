"""Full-detector VRAM + step-time probe (MCR P1 — the LiDAR-backbone OOM gate).

Builds the detector for the 4 configs {0.4m, 0.2m} × {no-backbone, +backbone}, runs fwd+loss+backward on a
real B=4 multi-sweep batch under bf16-AMP, and reports peak torch.cuda.max_memory_allocated() + per-step ms +
the encoder occupied-pillar count P (to catch 0.2m silent pillar-truncation at the max_pillars cap). This is
the go/no-go for the heavy 0.2m run — MUST run on an A100 (40GB); the login T4 (16GB) would OOM at 0.2m."""
import sys, time, json, copy
sys.path.insert(0, "fl_v3/src")
import torch

from fl_v3.data.nuscenes import info_cache as IC, paths as P
from fl_v3.data.nuscenes.dataset import NuScenesMultimodalDataset
from fl_v3.models.fusion.collate import detection_collate_fn
from fl_v3.training.tasks import get_task, _det_config_from_run
from fl_v3.utils.runtime import enforce_determinism

enforce_determinism(strict=False, precision="bf16")
dev = torch.device("cuda")
CACHE = "./fl_outputs/nuscenes/info_cache_msweep10"
ms, _ = IC.load_cache(CACHE, "v1.0-mini", "mini_train")
ds = NuScenesMultimodalDataset(ms, P.DATAROOT, sample_tokens=[i["sample_token"] for i in ms[:4]], n_sweeps=10)
batch = detection_collate_fn([ds[i] for i in range(4)])
batch = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in batch.items()}

BASE = {
    "det-camera-backbone": "swin_t", "det-swin-sdpa": True, "det-freeze-backbone": False,
    "det-lidar-sweeps": 10, "nuscenes-cache-dir": CACHE, "nuscenes-version": "v1.0-mini",
    "det-pretrained-backbone": True,
}
CONFIGS = [
    ("0.4m  no-bb", {"det-bev-voxel": 0.4, "det-max-pillars": 30000, "det-lidar-backbone": False}),
    ("0.4m +bb   ", {"det-bev-voxel": 0.4, "det-max-pillars": 30000, "det-lidar-backbone": True}),
    ("0.2m  no-bb", {"det-bev-voxel": 0.2, "det-max-pillars": 120000, "det-lidar-backbone": False}),
    ("0.2m +bb   ", {"det-bev-voxel": 0.2, "det-max-pillars": 120000, "det-lidar-backbone": True}),
    ("0.2m +bb+ck", {"det-bev-voxel": 0.2, "det-max-pillars": 120000, "det-lidar-backbone": True,
                     "det-lidar-backbone-checkpoint": True}),
]
task = get_task("nuscenes_detection")
print(f"{'config':14s} {'peakGB':>7s} {'ms/step':>8s} {'P(pillars)':>11s}  note")
for name, ov in CONFIGS:
    cfg = dict(BASE); cfg.update(ov)
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    try:
        model = task.build_model(cfg).to(dev).train()
        crit = task.build_criterion(cfg)
        # occupied-pillar count P (does 0.2m exceed the cap?)
        with torch.no_grad():
            pts = batch["lidar_points"]; from fl_v3.models.fusion.bev_grid import metric_to_grid, in_grid_mask, flat_index
            bevc = model.lidar_encoder.cfg
            col, row = metric_to_grid(pts[:, 1], pts[:, 2], bevc.x_min, bevc.y_min, bevc.vx, bevc.vy)
            keep = in_grid_mask(col, row, bevc.nx, bevc.ny) & (pts[:, 3] >= bevc.z_min) & (pts[:, 3] < bevc.z_max)
            key = pts[keep, 0] * (bevc.nx * bevc.ny) + flat_index(col[keep], row[keep], bevc.nx)
            Pn = int(torch.unique(key).numel())
        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
        def step():
            opt.zero_grad()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model(batch)
            out = {k: (v.float() if torch.is_tensor(v) else v) for k, v in out.items()}
            loss = crit(out, batch); loss.backward(); opt.step(); return float(loss)
        for _ in range(3): step()
        torch.cuda.synchronize(); t0 = time.perf_counter()
        for _ in range(5): step()
        torch.cuda.synchronize(); ms_step = (time.perf_counter() - t0) / 5 * 1000
        peak = torch.cuda.max_memory_allocated() / 1e9
        cap = cfg["det-max-pillars"]; warn = "  *** P > cap (truncating!)" if Pn > cap else ""
        print(f"{name:14s} {peak:7.1f} {ms_step:8.0f} {Pn:11d}{warn}")
        del model, crit, opt
    except RuntimeError as e:
        print(f"{name:14s} {'OOM' if 'out of memory' in str(e).lower() else 'ERR':>7s}  {str(e)[:60]}")
        torch.cuda.empty_cache()
