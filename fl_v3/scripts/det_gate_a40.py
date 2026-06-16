"""A40-pinned bit-determinism gate + wall-clock (T2, SPEC §0 + GATE).

The crown-jewel proof: same-seed two K-step central trainings → ``torch.equal`` on every
parameter, **on the production A40** (CPU / login-node Tesla T4 are a FALSE PASS — the
float-summation-order / topk-tie drift only surfaces cross-architecture). The job:

  1. asserts CUDA is present and ``get_device_name()`` contains ``A40`` — **errors LOUD
     (exit 2), never green, on no-CUDA or a non-A40 device**;
  2. asserts ``CUBLAS_WORKSPACE_CONFIG=:4096:8`` was set BEFORE the first CUDA context;
  3. runs two same-seed K-step trainings of the deterministic detector on a fixed
     synthetic batch (self-contained — no dataset / offline weights needed; the frozen
     backbone is random-but-seeded, which exercises the SAME deterministic op graph) and
     asserts ``torch.equal`` on every trained parameter;
  4. prints the **per-parameter SHA-256 weight checksum** (committed into
     ``collab/T2/SPEC.md`` as the evidence artifact; reproducible on any A40);
  5. measures wall-clock (forward + full train step) for BOTH the bring-up (ResNet-18)
     and the headline (frozen Swin-T, 6 cams, 256×704, real BEV grid) configs, + a
     projected per-round estimate for T3 planning.

Run via SLURM: ``sbatch fl_v3/scripts/run_det_gate_a40.sh``.
"""
from __future__ import annotations

import hashlib
import os
import sys
import time

import torch


def fatal(msg: str) -> None:
    print(f"[det_gate_a40] FATAL: {msg}", file=sys.stderr)
    sys.exit(2)


def assert_a40() -> str:
    if not torch.cuda.is_available():
        fatal("no CUDA available — the determinism gate is a FALSE PASS on CPU / login node. "
              "Run on an A40 via sbatch fl_v3/scripts/run_det_gate_a40.sh.")
    name = torch.cuda.get_device_name()
    if "A40" not in name:
        fatal(f"device {name!r} is not an A40 — the gate requires the production A40 "
              "(determinism that passes on a Tesla T4 can still drift on the A40 / ARM H200).")
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        fatal("CUBLAS_WORKSPACE_CONFIG != ':4096:8' (must be set BEFORE the first CUDA context). "
              "Export it in the SLURM script before any torch CUDA use.")
    return name


def synthetic_batch(seed: int, M: int = 6, n_pts: int = 20000, device="cuda") -> dict:
    """Fixed deterministic multimodal batch (T1 schema; self-contained, no dataset)."""
    g = torch.Generator().manual_seed(seed)
    imgs = (torch.rand(1, 6, 3, 900, 1600, generator=g) * 255).to(torch.uint8)
    K = torch.tensor([[800.0, 0, 800.0], [0, 800.0, 450.0], [0, 0, 1.0]])
    l2i = torch.eye(4).view(1, 4, 4).repeat(6, 1, 1); l2i[:, :3, :3] = K
    cam_int = K.view(1, 3, 3).repeat(6, 1, 1)
    xyz = (torch.rand(n_pts, 3, generator=g) * 2 - 1) * torch.tensor([45.0, 45.0, 3.0])
    # A DENSE cluster (~3k points in a ~1 m box) so several pillars EXCEED max_points —
    # this exercises the canonical-truncation path the Codex T2 finding flagged, so the
    # committed A40 checksum covers the over-cap regime.
    n_cl = 3000
    cluster = torch.tensor([10.0, -5.0, -0.5]) + (torch.rand(n_cl, 3, generator=g) * 2 - 1) * torch.tensor([0.5, 0.5, 0.3])
    xyz = torch.cat([xyz, cluster], dim=0)
    inten = torch.rand(xyz.shape[0], 1, generator=g)
    pts = torch.cat([torch.zeros(xyz.shape[0], 1), xyz, inten, torch.zeros(xyz.shape[0], 1)], dim=1)
    boxes = torch.zeros(M, 7); boxes[:, :2] = (torch.rand(M, 2, generator=g) * 2 - 1) * 30
    boxes[:, 2] = -1.0; boxes[:, 3:6] = torch.tensor([4.0, 2.0, 1.6]); boxes[:, 6] = torch.rand(M, generator=g) * 3.14
    b = {
        "images": imgs, "lidar2img": l2i.unsqueeze(0), "cam_intrinsics": cam_int.unsqueeze(0),
        "lidar_points": pts, "gt_boxes": [boxes], "gt_labels": [torch.zeros(M, dtype=torch.int64)],
        "gt_velocity": [torch.zeros(M, 2)],
    }
    from fl_v3.training.loop import _move_to_device
    return _move_to_device(b, torch.device(device))


def build(cfg, seed: int):
    from fl_v3.utils.runtime import enforce_determinism, seed_everything
    from fl_v3.models.fusion.detector import BEVFusionDetector

    enforce_determinism(strict=True); seed_everything(seed)
    return BEVFusionDetector(cfg).cuda()


def train_k(cfg, batch, K: int, seed: int):
    from fl_v3.utils.runtime import enforce_determinism, seed_everything
    from fl_v3.models.fusion.losses import CenterPointLoss

    enforce_determinism(strict=True); seed_everything(seed)
    model = build(cfg, seed)
    crit = CenterPointLoss(cfg=cfg.bev, n_classes=cfg.n_classes).cuda()
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    model.train()
    for _ in range(K):
        opt.zero_grad(); loss = crit(model(batch), batch); loss.backward(); opt.step()
    torch.cuda.synchronize()
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


def param_checksum(sd) -> str:
    h = hashlib.sha256()
    for k in sorted(sd):
        t = sd[k]
        h.update(k.encode()); h.update(str(tuple(t.shape)).encode()); h.update(str(t.dtype).encode())
        h.update(t.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def wall_clock(cfg, batch, label: str, iters: int = 10):
    from fl_v3.utils.runtime import enforce_determinism, seed_everything
    from fl_v3.models.fusion.losses import CenterPointLoss

    enforce_determinism(strict=True); seed_everything(0)
    model = build(cfg, 0); crit = CenterPointLoss(cfg=cfg.bev, n_classes=cfg.n_classes).cuda()
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    # warmup
    model.eval()
    with torch.no_grad():
        for _ in range(3):
            model(batch)
    torch.cuda.synchronize()
    t0 = time.time()
    with torch.no_grad():
        for _ in range(iters):
            model(batch)
    torch.cuda.synchronize()
    fwd = (time.time() - t0) / iters
    model.train()
    t0 = time.time()
    for _ in range(iters):
        opt.zero_grad(); loss = crit(model(batch), batch); loss.backward(); opt.step()
    torch.cuda.synchronize()
    step = (time.time() - t0) / iters
    return {"label": label, "fwd_s": fwd, "train_step_s": step}


def main():
    from fl_v3.models.fusion.detector import DetectorConfig

    name = assert_a40()
    print(f"[det_gate_a40] device = {name}")
    K = int(os.environ.get("DET_GATE_STEPS", "12"))

    # --- determinism gate on the bring-up (ResNet) config (fast, self-contained) ---
    cfg = DetectorConfig(camera_backbone="resnet18", pretrained_backbone=False,
                         neck_channels=64, context_channels=32, lidar_channels=32,
                         fusion_channels=64, bev_neck_channels=128, head_channels=32)
    batch = synthetic_batch(123)
    sd1 = train_k(cfg, batch, K, seed=7)
    sd2 = train_k(cfg, batch, K, seed=7)
    diverged = [k for k in sd1 if not torch.equal(sd1[k], sd2[k])]
    if diverged:
        fatal(f"BIT-IDENTITY BROKEN on A40: {len(diverged)} params diverged, e.g. {diverged[:5]}")
    checksum = param_checksum(sd1)
    print(f"[det_gate_a40] PASS: two same-seed {K}-step trainings byte-identical on the A40")
    print(f"[det_gate_a40] A40_WEIGHT_CHECKSUM = {checksum}")

    # --- wall-clock: bring-up (ResNet) AND headline (frozen Swin-T, 6 cams, 256×704) ---
    headline = DetectorConfig(camera_backbone="swin_t", pretrained_backbone=False,
                              freeze_camera_backbone=True)  # default channels = headline
    wc_resnet = wall_clock(cfg, batch, "bring-up resnet18")
    wc_swin = wall_clock(headline, synthetic_batch(123), "headline swin_t (frozen, 6cam, 256x704)")
    print("[det_gate_a40] WALL-CLOCK (A40):")
    for wc in (wc_resnet, wc_swin):
        # crude per-round projection: 25 clients × ~100 steps/local-epoch (T3 plans real)
        per_round = wc["train_step_s"] * 100 * 25
        print(f"  {wc['label']:48s} fwd={wc['fwd_s']*1000:7.1f}ms  "
              f"train_step={wc['train_step_s']*1000:7.1f}ms  "
              f"proj/round(25cl×100step)={per_round/60:.1f}min")


if __name__ == "__main__":
    main()
