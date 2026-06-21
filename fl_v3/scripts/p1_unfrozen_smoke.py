"""MCR Phase-1 login-node smoke: the UNFROZEN-Swin bf16 recipe trains clean.

Validates the new capability path end-to-end on the login-node T4 (real mini batch, real model):
unfreeze (det-freeze-backbone=false) + activation checkpointing on Swin + bf16 autocast + fp32 head
upcast + LR param groups (backbone @ lr*mult) + grad-clip. Asserts: no NaN/inf loss, the camera
backbone actually RECEIVES gradients (so unfreeze + checkpoint compose correctly), and the loss drops
on a fixed batch (the model learns). This is a SMOKE (pipeline/no-NaN), NOT a scientific result (Rule #3).

Run:  bash fl_v3/scripts/run_in_venv.sh python fl_v3/scripts/p1_unfrozen_smoke.py
"""
from __future__ import annotations

import sys

sys.path.insert(0, "fl_v3/src")
import torch

from fl_v3.utils.runtime import enforce_determinism, precision_state, seed_everything
from fl_v3.training.tasks import get_task
from fl_v3.training.loop import _move_to_device


def main() -> int:
    if not torch.cuda.is_available():
        print("[p1-smoke] no CUDA — bf16 autocast is a no-op; run on the login T4."); return 2
    enforce_determinism(strict=True, precision="bf16")   # science path: cudnn.deterministic=False → AMP on
    seed_everything(42)
    dev = torch.device("cuda")
    cfg = {
        "nuscenes-cache-dir": "./fl_outputs/nuscenes/info_cache", "nuscenes-version": "v1.0-mini",
        "nuscenes-train-split": "mini_train", "nuscenes-val-split": "mini_val",
        "nuscenes-partition-mode": "iid", "nuscenes-num-clients": 4, "seed": 42,
        "batch-size": 1, "num-workers": 0,
        # the Phase-1 headline lever: TRAIN the Swin backbone + activation-checkpoint it.
        "det-camera-backbone": "swin_t", "det-pretrained-backbone": True,
        "det-freeze-backbone": False, "det-activation-checkpoint": True,
        "precision": "bf16",
    }
    task = get_task("nuscenes_detection")
    cdata = task.client_data(0, cfg)
    seed_everything(42)
    try:
        model = task.build_model(cfg).to(dev)
    except Exception as e:
        print(f"[p1-smoke] SKIP: swin weights unavailable ({e})"); return 2
    crit = task.build_criterion(cfg)
    batch = _move_to_device(next(iter(cdata.trainloader)), dev)

    # (a) the unfreeze actually took: backbone params are trainable.
    bb = [p for n, p in model.named_parameters() if p.requires_grad and n.startswith("camera_backbone.")]
    rest = [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith("camera_backbone.")]
    assert bb, "camera backbone has NO trainable params — det-freeze-backbone=false did not take"
    print(f"[p1-smoke] trainable: {len(bb)} backbone tensors + {len(rest)} fusion/head tensors")
    print(f"[p1-smoke] precision_state.precision = {precision_state()['precision']} "
          f"(cudnn.deterministic={torch.backends.cudnn.deterministic}, activation_ckpt="
          f"{model.camera_backbone.activation_checkpoint})")

    base_lr = 3e-3
    opt = torch.optim.Adam([{"params": rest, "lr": base_lr}, {"params": bb, "lr": base_lr * 0.1}],
                           lr=base_lr, weight_decay=0.0)
    use_amp = (not torch.backends.cudnn.deterministic) and dev.type == "cuda"
    assert use_amp, "bf16 autocast did not engage under precision=bf16"
    model.train()
    losses, bb_grad_l1 = [], 0.0
    for step in range(25):
        opt.zero_grad()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(batch)
        out = {k: (v.float() if torch.is_tensor(v) else v) for k, v in out.items()}  # fp32 head upcast
        loss = crit(out, batch)
        assert torch.isfinite(loss), f"NON-FINITE loss at step {step}: {float(loss)}"
        loss.backward()
        bb_grad_l1 = sum((p.grad.abs().sum().item() if p.grad is not None else 0.0) for p in bb)
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 35.0)
        opt.step()
        losses.append(float(loss))
        if step in (0, 5, 12, 24):
            print(f"[p1-smoke] step {step:2d}: loss={float(loss):.4f}  backbone_grad_L1={bb_grad_l1:.3e}")

    assert bb_grad_l1 > 0.0, "backbone received ZERO gradient — unfreeze/activation-ckpt did not compose"
    assert losses[-1] < losses[0], f"loss did not drop on a fixed batch: {losses[0]:.3f}→{losses[-1]:.3f}"
    print(f"[p1-smoke] OK — unfrozen Swin + activation-ckpt + bf16 + grad-clip trains CLEAN "
          f"(loss {losses[0]:.3f}→{losses[-1]:.3f}, backbone learns).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
