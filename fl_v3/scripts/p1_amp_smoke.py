"""CL AMP smoke — real-data forward+backward through the detector under fp16+GradScaler.

Validates the real points / lidar2img / loss path end-to-end (what synthetic unit tests can't) and de-risks
the **fp16-AMP precision** that the GH200/Arrhenius sparse path needs (no bf16 there). Builds the model in
TRAIN mode, runs a few steps on real pooled keyframes, and prints the per-step loss BREAKDOWN (hm / reg /
total) + grad-norm, with a finite/no-NaN gate. ``--fp16`` switches to fp16 autocast + GradScaler and reports
the scale + calibration skips (the direct inf/NaN-overflow signal, e.g. the LSS splat). Mirrors
train_one_epoch's AMP + fp32-head-upcast path.

Run:  python fl_v3/scripts/p1_amp_smoke.py --precision fp16 nuscenes-cache-dir=/path/to/info_cache
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
    ap.add_argument("--config", default="fl_v3/configs/p1_bb02d.json")
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--ntok", type=int, default=32)
    ap.add_argument("--fp16", action="store_true",
                    help="compat alias for --precision fp16; reports "
                         "GradScaler skips, the direct inf/NaN-overflow signal (e.g. the LSS splat).")
    ap.add_argument("--precision", default=None, choices=["fp16", "fp32"],
                    help="Arrhenius precision policy: fp16 AMP+GradScaler or fp32 reference.")
    ap.add_argument("overrides", nargs="*", default=[])
    a = ap.parse_args()

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

    cfg = json.load(open(a.config))
    for ov in a.overrides:
        k, _, v = ov.partition("=")
        cfg[k] = _scalar(v)
    if a.fp16:
        cfg["precision"] = "fp16"
    precision = str(a.precision or cfg.get("precision", "fp16"))
    cfg["precision"] = precision
    seed = int(cfg.get("seed", 42))
    seed_everything(seed)
    enforce_determinism(strict=False, precision=precision)
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
    scaler = make_grad_scaler(dev, precision)
    print(f"[smoke] device={dev} batch={int(cfg.get('batch-size',4))} sweeps={cfg.get('det-lidar-sweeps')} "
          f"ntok={len(toks)} precision={precision} scaler={scaler.is_enabled()}", flush=True)
    from itertools import islice, cycle as _cycle
    ok, n_skip, i = True, 0, -1
    for i, batch in enumerate(islice(_cycle(loader), a.steps)):   # cycle ⇒ STEPS honored even if loader < STEPS batches
        batch = _move_to_device(batch, dev)
        opt.zero_grad()
        if precision == "fp16" and dev.type == "cuda":
            with precision_autocast_context(precision, dev):
                out = model(batch)
            out = _float_tensors(out)
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
        fin = bool(torch.isfinite(loss).item())
        ok = ok and fin
        print(f"[smoke] step {i}: total={lt['loss']:.4f} hm={lt['hm_loss']:.4f} reg={lt['reg_loss']:.4f} "
              f"n_gt={lt['n_gt']} grad_norm={float(gn):.2f} "
              f"finite={fin} scale={scaler.get_scale():.0f} skipped={not stepped}", flush=True)
    steps_run = i + 1
    if scaler.is_enabled() and n_skip >= steps_run:              # ALL steps skipped ⇒ deadlock ⇒ fp16-unstable
        ok = False                                             # (a FEW early skips = normal GradScaler calibration)
    extra = (f" / GradScaler skipped {n_skip}/{steps_run} (early skips = calibration warmup; landing after = OK)"
             if scaler.is_enabled() else "")
    print(f"[smoke] {'PASS' if ok else 'FAIL'} — finite over {steps_run} real-data steps{extra}", flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
