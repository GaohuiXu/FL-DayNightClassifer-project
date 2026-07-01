"""Bit-identity verifier for the speedup levers (lever1 mask-then-multiply, loss .item() fix, lever2
geometry cache). Builds the REAL detector + runs fwd+loss+bwd on a FIXED mini batch with a FIXED seed,
and prints checksums of the head outputs, the loss, and the trainable gradients. Run on the OLD code to
capture the reference, then on the NEW code — the checksums MUST match exactly (the refactors are
bit-identical or they are WRONG). Also isolates the view-transform + loss components directly.

Usage:
  source fl_v3/scripts/arrhenius_env.sh
  arrhenius_load_modules build
  arrhenius_activate_env
  PYTHONPATH=fl_v3/src python fl_v3/scripts/verify_levers.py <MINI_CACHE_DIR>
"""
from __future__ import annotations

import hashlib
import os
import sys

sys.path.insert(0, "fl_v3/src")
import torch

from fl_v3.training.tasks import get_task, trainable_state_dict
from fl_v3.utils.runtime import enforce_determinism, seed_everything


def ck(t):
    return hashlib.sha256(t.detach().cpu().contiguous().float().numpy().tobytes()).hexdigest()[:16]


def main():
    cache = (
        sys.argv[1] if len(sys.argv) > 1
        else os.environ.get("ARRHENIUS_NUSCENES_CACHE")
        or os.environ.get("NUSCENES_CACHE_DIR")
        or ""
    )
    if not cache:
        raise SystemExit(
            "verify_levers.py requires MINI_CACHE_DIR as argv[1] or ARRHENIUS_NUSCENES_CACHE/"
            "NUSCENES_CACHE_DIR; no historical host path is assumed."
        )
    cfg = {
        "task-type": "nuscenes_detection", "seed": 1234, "device": "cpu",
        "nuscenes-cache-dir": cache, "nuscenes-version": "v1.0-mini",
        "nuscenes-train-split": "mini_train", "nuscenes-val-split": "mini_val",
        "nuscenes-partition-mode": "iid", "nuscenes-num-clients": 2,
        "min-keyframes-per-client": 10, "num-clients": 2,
        "det-camera-backbone": "resnet18", "det-freeze-backbone": True,
        "det-pretrained-backbone": True, "batch-size": 2, "num-workers": 0,
    }
    precision = os.environ.get("PRECISION", "fp32")
    cfg["precision"] = precision
    seed_everything(int(cfg["seed"]))
    enforce_determinism(strict=True, precision=precision)
    print(f"[verify_levers] precision={precision}")
    device = torch.device("cpu")
    task = get_task("nuscenes_detection")
    model = task.build_model(cfg).to(device)
    criterion = task.build_criterion(cfg)

    # one fixed batch (eval loader, shuffle=False → deterministic first batch)
    loader = task.eval_loader(cfg)
    batch = next(iter(loader))
    from fl_v3.training.loop import _move_to_device
    batch = _move_to_device(batch, device)

    model.train()
    out = model(batch)
    loss = criterion(out, batch)
    loss.backward()

    print("=== verify_levers checksums (must match OLD vs NEW exactly) ===")
    print(f"loss            = {float(loss.item()):.10f}")
    for k in ("heatmap", "reg", "height", "dim", "rot", "vel"):
        if k in out:
            print(f"head.{k:<8}    = {ck(out[k])}")
    # gradient checksum over the trainable tensors (in contract order)
    h = hashlib.sha256()
    for name, p in model.named_parameters():
        if p.requires_grad and p.grad is not None:
            h.update(name.encode())
            h.update(p.grad.detach().cpu().contiguous().float().numpy().tobytes())
    print(f"grad_all        = {h.hexdigest()[:16]}")
    # also isolate the view-transform BEV output via return_intermediates
    model.eval()
    with torch.no_grad():
        out2 = model(batch, return_intermediates=True)
    print(f"camera_bev      = {ck(out2['_camera_bev'])}")
    print(f"fused_bev       = {ck(out2['_fused_bev'])}")
    print("=== end ===")


if __name__ == "__main__":
    main()
