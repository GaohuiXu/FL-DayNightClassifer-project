"""Centralized (single-node, full-data, NO FedAvg) training — D14 Phase-2 D baseline.

The matched-budget control for the FL reference: same model / data split / preprocessing / optimizer
/ numeric regime, trained on the POOLED data of exactly the FL clients (union of the log-group
partition's client tokens). At 1 local-epoch/round + full participation, R FL rounds ≈ R centralized
epochs of data exposure — so **centralized epochs == FL rounds** isolates the ONE remaining
difference: FedAvg averaging. (fails centrally too → the architecture defeats the attack; works
centrally but dies under FL → averaging dilution, the plan's Q2.)

Saves a self-contained ``final_model.pt`` + ``trainable_checksum.txt`` + ``provenance.json`` +
``train_curve.json`` (per-epoch loss), checkpointing after EACH epoch (resumable; any epoch is
post-hoc evaluable by t4_readiness_eval.py --diagnostic). Run via run_centralized_a40.sh.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, "fl_v3/src")
import torch


def _parse_scalar(s: str):
    low = s.lower()
    if low in ("true", "false"):
        return low == "true"
    for cast in (int, float):
        try:
            return cast(s)
        except ValueError:
            pass
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--epochs", type=int, required=True, help="matched budget: == FL rounds")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--tag", default="centralized_clean")
    ap.add_argument("--resume", action="store_true", help="resume from the latest epoch checkpoint")
    ap.add_argument("--max-steps", type=int, default=0,
                    help="cap steps/epoch (0=full epoch). For smoke validation ONLY — a capped run is "
                         "NOT a scientific baseline.")
    ap.add_argument("--compile", action="store_true", help="L8: torch.compile the frozen backbone")
    ap.add_argument("overrides", nargs="*", default=[])
    args = ap.parse_args()

    from fl_v3.training.tasks import get_task, trainable_state_dict
    from fl_v3.training.loop import train_one_epoch, _unpack_batch, _batch_size
    from fl_v3.engine.local_runner import numpy_state_checksum
    from fl_v3.data.nuscenes.dataset import NuScenesMultimodalDataset, make_loader
    from fl_v3.data.nuscenes import paths as P
    from fl_v3.models.fusion.collate import detection_collate_fn
    from fl_v3.utils.runtime import enforce_determinism, precision_state, seed_everything, truthy

    with open(args.config, encoding="utf-8") as f:
        cfg = json.load(f)
    for ov in args.overrides:
        k, _, v = ov.partition("=")
        cfg[k] = _parse_scalar(v)

    precision = str(cfg.get("precision", "bf16"))   # D16 science default = bf16 (centralized baseline)
    seed = int(cfg.get("seed", 42))
    # REVIEW-FIX (MED, det-review #6): the matched-budget control assumes 1 local epoch/round, so
    # R FL rounds == R centralized epochs of data exposure. If the FL reference ever uses >1 local
    # epoch, "epochs == rounds" would NO LONGER match the data budget — refuse rather than silently
    # produce a budget-mismatched (invalid) comparison.
    nle = int(cfg.get("num-local-epochs", 1))
    if nle != 1:
        raise SystemExit(
            f"[centralized] num-local-epochs={nle} != 1: the matched-budget comparison assumes 1 "
            "local epoch/round (R rounds == R epochs of exposure). Either set num-local-epochs=1 or "
            "extend this trainer to run num-local-epochs passes per 'round'. Refusing a mismatched run.")
    device = torch.device("cuda" if (str(cfg.get("device", "cuda")) == "cuda"
                                     and torch.cuda.is_available()) else "cpu")
    seed_everything(seed)
    enforce_determinism(strict=truthy(cfg.get("determinism-strict", True)), precision=precision)
    print(f"[centralized] precision={precision} device={device} precision_state={precision_state()}",
          flush=True)

    task = get_task("nuscenes_detection")
    # The POOLED training set = the union of the FL log-group partition's client tokens (so the
    # centralized model sees EXACTLY the data the FL clients collectively see — a matched comparison).
    part = task._partition(cfg)
    pooled_tokens = sorted({t for toks in part["client_tokens"].values() for t in toks})
    train_split = str(cfg["nuscenes-train-split"])
    info_list, _ = task._load_info(cfg, train_split)
    print(f"[centralized] pooled train set: {len(pooled_tokens)} keyframes "
          f"(union of {part['num_clients']} log-group clients)", flush=True)

    ds = NuScenesMultimodalDataset(info_list, P.get_dataroot(cfg), sample_tokens=pooled_tokens)

    def epoch_loader(epoch):
        # REVIEW-FIX (HIGH, det-review #1/2/4/5): build a FRESH loader each epoch seeded by (seed+epoch)
        # so the shuffle order is a pure function of (seed, epoch) — independent of where the run started.
        # The old "build once, reuse" path advanced a single private Generator across epochs, so --resume
        # (which rebuilds the loader fresh) replayed epoch-0's order at epoch K → resumed run != fresh run
        # → a different trainable_checksum. Per-epoch seeding makes resume byte-identical to a fresh run.
        return make_loader(ds, batch_size=int(cfg.get("batch-size", 16)), shuffle=True,
                           num_workers=int(cfg.get("num-workers", 4)), seed=seed + epoch,
                           collate_fn=detection_collate_fn)

    model = task.build_model(cfg).to(device)
    if args.compile:  # L8 (D15): compile the frozen backbone once (amortized over the long centralized run)
        model.camera_backbone = torch.compile(model.camera_backbone)
        print("[centralized] torch.compile(camera_backbone) enabled", flush=True)
    criterion = task.build_criterion(cfg)
    # MCR Phase 1 (D17): when the camera backbone is TRAINED (det-freeze-backbone=false), the pretrained
    # backbone wants a LOWER LR than the from-scratch fusion/head (else the high fusion LR wrecks the
    # ImageNet features). Split into LR param groups: backbone @ lr*mult, everything else @ lr. When the
    # backbone is FROZEN, it has no requires_grad params → `bb_params` is empty → a single flat Adam over
    # the trainable set, BYTE-IDENTICAL to the pre-MCR frozen baseline.
    base_lr = float(cfg.get("learning-rate", 3e-3))
    wd = float(cfg.get("weight-decay", 0.0))
    bb_mult = float(cfg.get("det-backbone-lr-mult", 0.1))
    bb_params, rest_params = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (bb_params if n.startswith("camera_backbone.") else rest_params).append(p)
    if bb_params:
        optimizer = torch.optim.Adam(
            [{"params": rest_params, "lr": base_lr},
             {"params": bb_params, "lr": base_lr * bb_mult}],
            lr=base_lr, weight_decay=wd)
        print(f"[centralized] TRAINED backbone: {len(bb_params)} backbone tensors @ lr={base_lr*bb_mult:.2e} "
              f"(mult={bb_mult}); {len(rest_params)} fusion/head tensors @ lr={base_lr:.2e}", flush=True)
    else:
        optimizer = torch.optim.Adam(rest_params, lr=base_lr, weight_decay=wd)  # frozen-backbone path (unchanged)
    grad_clip_norm = float(cfg.get("grad-clip-norm", 0.0))

    exp_dir = os.path.join(args.out_dir, args.tag)
    os.makedirs(exp_dir, exist_ok=True)
    ckpt_path = os.path.join(exp_dir, "final_model.pt")
    opt_path = os.path.join(exp_dir, "optimizer.pt")
    curve_path = os.path.join(exp_dir, "train_curve.json")
    curve = []
    start_epoch = 0
    if args.resume and os.path.isfile(ckpt_path) and os.path.isfile(opt_path):
        model.load_state_dict(torch.load(ckpt_path, map_location=device), strict=True)
        os_state = torch.load(opt_path, map_location=device)
        optimizer.load_state_dict(os_state["optimizer"])
        start_epoch = int(os_state["epoch"])
        if os.path.isfile(curve_path):
            curve = json.load(open(curve_path))
        print(f"[centralized] RESUMED from epoch {start_epoch}", flush=True)

    def save_ckpt(epoch):
        # torch.compile wraps the module → state_dict keys gain a "._orig_mod." segment. Strip it so the
        # saved checkpoint loads strict=True into a NON-compiled model (the readiness eval). No-op when
        # not compiled. (Trainable-only checksum below is unaffected — the frozen backbone isn't in it.)
        full = {k.replace("._orig_mod.", "."): v for k, v in model.state_dict().items()}
        torch.save(full, ckpt_path)
        torch.save({"optimizer": optimizer.state_dict(), "epoch": epoch}, opt_path)
        arrs = [v.detach().cpu().numpy() for v in trainable_state_dict(model).values()]
        chk = numpy_state_checksum(arrs)
        with open(os.path.join(exp_dir, "trainable_checksum.txt"), "w") as f:
            f.write(chk + "\n")
        prov = {
            "regime": "D14-centralized-baseline",
            "precision": precision, "numeric-mode": cfg.get("numeric-mode"),
            "epochs_done": epoch, "epochs_target": args.epochs,
            "pooled_keyframes": len(pooled_tokens), "n_log_group_clients": part["num_clients"],
            "nuscenes-version": cfg.get("nuscenes-version"),
            "nuscenes-train-split": train_split, "nuscenes-val-split": cfg.get("nuscenes-val-split"),
            "nuscenes-partition-mode": cfg.get("nuscenes-partition-mode"),
            "det-camera-backbone": cfg.get("det-camera-backbone"), "seed": seed,
            "batch-size": cfg.get("batch-size"), "learning-rate": cfg.get("learning-rate"),
            "num-local-epochs": nle,
            "FL_TRAINABLE_CHECKSUM": chk, "tag": args.tag,
            # Matched-budget caveat (det-review): vs FL this differs by (a) NO cross-client FedAvg
            # averaging AND (b) ONE Adam optimizer kept warm across epochs (FL rebuilds Adam per
            # client/round). A "works centrally, dies under FL" result implicates the FL regime
            # broadly (averaging + per-round optimizer reset), not averaging in isolation.
            "matched_budget_note": "epochs==rounds at 1 local-epoch/round; warm-Adam vs FL per-round reset",
        }
        json.dump(prov, open(os.path.join(exp_dir, "provenance.json"), "w"), indent=2, sort_keys=True)
        return chk

    print(f"===== centralized training: {args.epochs} epochs (matched to {args.epochs}-round FL) =====",
          flush=True)
    def _capped_epoch(loader, max_steps):
        """Smoke-only capped epoch (mirrors train_one_epoch but breaks early); NOT a scientific run."""
        model.train()
        total_loss, total_n, steps = 0.0, 0, 0
        for batch in loader:
            inputs, targets = _unpack_batch(batch, device)
            optimizer.zero_grad()
            loss = criterion(model(inputs), targets)
            loss.backward(); optimizer.step()
            bs = _batch_size(targets)
            total_loss += float(loss.item()) * bs; total_n += bs; steps += 1
            if steps >= max_steps:
                break
        return {"loss": total_loss / total_n if total_n else 0.0, "num_samples": float(total_n)}

    for epoch in range(start_epoch, args.epochs):
        t0 = time.perf_counter()
        seed_everything(seed + epoch)        # global-RNG per-epoch seed (any forward-side RNG)
        loader = epoch_loader(epoch)         # per-epoch loader seeded (seed+epoch) → resume-byte-identical
        m = (_capped_epoch(loader, args.max_steps) if args.max_steps > 0
             else train_one_epoch(model, loader, criterion, optimizer, device,
                                  grad_clip_norm=grad_clip_norm))
        dt = time.perf_counter() - t0
        chk = save_ckpt(epoch + 1)
        rec = {"epoch": epoch + 1, "train_loss": float(m["loss"]),
               "num_samples": int(m["num_samples"]), "seconds": dt, "checksum": chk}
        curve.append(rec)
        json.dump(curve, open(curve_path, "w"), indent=2)
        print(f"[centralized] epoch {epoch+1}/{args.epochs} loss={m['loss']:.5f} "
              f"n={int(m['num_samples'])} time={dt:.0f}s checksum={chk[:16]}", flush=True)

    print(f"[centralized] DONE — checkpoint {ckpt_path}", flush=True)
    print(f"[centralized] FL_TRAINABLE_CHECKSUM = {open(os.path.join(exp_dir,'trainable_checksum.txt')).read().strip()}",
          flush=True)


if __name__ == "__main__":
    main()
