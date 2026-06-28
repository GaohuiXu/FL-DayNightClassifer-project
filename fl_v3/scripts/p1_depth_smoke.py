"""Depth-sup CL smoke — real-data forward+backward through bb02d+depth-supervision.

Validates the ONE thing the unit tests can't (synthetic only): the real points / lidar2img path. Builds the
model in TRAIN mode (so the detector attaches depth_logits + projected GT), runs a few steps on real pooled
keyframes, and prints the per-step loss BREAKDOWN (hm / reg / depth / total) + depth-GT coverage + grad-norm,
with a finite/no-NaN gate. Mirrors train_one_epoch's bf16-AMP + fp32-head-upcast path exactly.

Run:  CACHE=<msweep10> sbatch fl_v3/scripts/run_p1_depth_smoke.sh
"""
from __future__ import annotations

import argparse
import json
import sys

sys.path.insert(0, "fl_v3/src")
import torch


def _scalar(s: str):
    if s in ("true", "false"):
        return s == "true"
    for c in (int, float):
        try:
            return c(s)
        except ValueError:
            pass
    return s


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="fl_v3/configs/p1_bb02d_depth.json")
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--ntok", type=int, default=32)
    ap.add_argument("--fp16", action="store_true",
                    help="fp16-AMP + GradScaler (Arrhenius precision de-risk) instead of bf16 — reports "
                         "GradScaler skips, the direct inf/NaN-overflow signal (e.g. the LSS splat).")
    ap.add_argument("overrides", nargs="*", default=[])
    a = ap.parse_args()

    from fl_v3.training.tasks import get_task, _aug_from_run
    from fl_v3.data.nuscenes.dataset import NuScenesMultimodalDataset, make_loader
    from fl_v3.data.nuscenes import paths as P
    from fl_v3.models.fusion.collate import detection_collate_fn
    from fl_v3.training.loop import _move_to_device
    from fl_v3.utils.runtime import enforce_determinism, seed_everything

    cfg = json.load(open(a.config))
    for ov in a.overrides:
        k, _, v = ov.partition("=")
        cfg[k] = _scalar(v)
    seed = int(cfg.get("seed", 42))
    seed_everything(seed)
    enforce_determinism(strict=False, precision=str(cfg.get("precision", "bf16")))
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    task = get_task("nuscenes_detection")
    part = task._partition(cfg)
    toks = sorted({t for ts in part["client_tokens"].values() for t in ts})[: a.ntok]
    info, _ = task._load_info(cfg, str(cfg["nuscenes-train-split"]))
    ds = NuScenesMultimodalDataset(info, P.get_dataroot(cfg), sample_tokens=toks,
                                   n_sweeps=int(cfg.get("det-lidar-sweeps", 1)), augment=_aug_from_run(cfg))
    loader = make_loader(ds, batch_size=int(cfg.get("batch-size", 4)), shuffle=True,
                         num_workers=2, seed=seed, collate_fn=detection_collate_fn)

    model = task.build_model(cfg).to(dev)
    model.train()
    crit = task.build_criterion(cfg)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
    amp_dtype = torch.float16 if a.fp16 else torch.bfloat16
    scaler = torch.amp.GradScaler("cuda", enabled=a.fp16)   # fp16 needs loss-scaling; bf16 ⇒ no-op passthrough
    print(f"[smoke] depth_sup={model.cfg.depth_supervision} depth_loss_weight={crit.depth_loss_weight} "
          f"device={dev} batch={int(cfg.get('batch-size',4))} sweeps={cfg.get('det-lidar-sweeps')} "
          f"ntok={len(toks)} amp={'fp16+GradScaler' if a.fp16 else 'bf16'}", flush=True)
    from itertools import islice, cycle as _cycle
    relaxed = not torch.backends.cudnn.deterministic
    ok, n_skip, i = True, 0, -1
    for i, batch in enumerate(islice(_cycle(loader), a.steps)):   # cycle ⇒ STEPS honored even if loader < STEPS batches
        batch = _move_to_device(batch, dev)
        opt.zero_grad()
        if relaxed and dev.type == "cuda":
            with torch.autocast("cuda", dtype=amp_dtype):
                out = model(batch)
            out = {k: (v.float() if torch.is_tensor(v) else v) for k, v in out.items()}
        else:
            out = model(batch)
        loss = crit(out, batch)
        scale_before = scaler.get_scale()
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        gn = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1e9)
        scaler.step(opt)
        scaler.update()
        stepped = scaler.get_scale() >= scale_before           # GradScaler drops the scale + SKIPS on inf/nan grads
        n_skip += int(not stepped)
        lt = crit.last_terms
        cov = "n/a"
        if "depth_target" in out:
            cov = f"{100.0 * float((out['depth_target'] >= 0).float().mean()):.2f}%"
        fin = bool(torch.isfinite(loss).item())
        depth_ok = (lt.get("depth_loss", 0.0) > 0.0) if model.cfg.depth_supervision else True
        ok = ok and fin and depth_ok
        print(f"[smoke] step {i}: total={lt['loss']:.4f} hm={lt['hm_loss']:.4f} reg={lt['reg_loss']:.4f} "
              f"depth={lt.get('depth_loss', 0.0):.4f} n_gt={lt['n_gt']} grad_norm={float(gn):.2f} "
              f"depth_cov={cov} finite={fin} scale={scaler.get_scale():.0f} skipped={not stepped}", flush=True)
    steps_run = i + 1
    if a.fp16 and n_skip >= steps_run:                          # ALL steps skipped ⇒ deadlock ⇒ fp16-unstable
        ok = False                                             # (a FEW early skips = normal GradScaler calibration)
    extra = (f" / GradScaler skipped {n_skip}/{steps_run} (early skips = calibration warmup; landing after = OK)"
             if a.fp16 else "")
    print(f"[smoke] {'PASS' if ok else 'FAIL'} — finite over {steps_run} real-data steps{extra}", flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
