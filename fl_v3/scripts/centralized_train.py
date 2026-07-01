"""Centralized (single-node, full-data, NO FedAvg) training — D14 Phase-2 D baseline.

The matched-budget control for the FL reference: same model / data split / preprocessing / optimizer
/ numeric regime, trained on the POOLED data of exactly the FL clients (union of the log-group
partition's client tokens). At 1 local-epoch/round + full participation, R FL rounds ≈ R centralized
epochs of data exposure — so **centralized epochs == FL rounds** isolates the ONE remaining
difference: FedAvg averaging. (fails centrally too → the architecture defeats the attack; works
centrally but dies under FL → averaging dilution, the plan's Q2.)

Saves a self-contained ``final_model.pt`` + ``trainable_checksum.txt`` + ``provenance.json`` +
``train_curve.json`` (per-epoch loss), checkpointing after EACH epoch (resumable; any epoch is
post-hoc evaluable by t4_readiness_eval.py --diagnostic). Run through the current Arrhenius Slurm wrapper.
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
    ap.add_argument("--compile", action="store_true", help="L8: torch.compile the camera backbone")
    ap.add_argument("--compile-bev", action="store_true",
                    help="MCR P2: also compile the static BEV stack (camera_neck+fusion+bev_neck+head); "
                         "forces drop_last so a partial batch can't trigger a recompile")
    ap.add_argument("--snapshot-epochs", default="",
                    help="comma-sep epochs at which to ALSO save a self-contained EMA snapshot "
                         "(<exp_dir>/ema_ep<N>/) for a post-hoc convergence-curve eval; the per-epoch "
                         "final_model.pt is overwritten, these are kept.")
    ap.add_argument("overrides", nargs="*", default=[])
    args = ap.parse_args()
    snap_epochs = {int(x) for x in args.snapshot_epochs.split(",") if x.strip()}

    from fl_v3.training.tasks import get_task, trainable_state_dict, _aug_from_run, _gtpaste_from_run
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

    precision = str(cfg.get("precision", "fp16"))   # Arrhenius sparse default: fp16 AMP + GradScaler
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
    # --- DDP (MCR P2): torchrun sets LOCAL_RANK/RANK/WORLD_SIZE → 4-GPU data-parallel. The single-GPU
    # path (no env) is UNCHANGED (rank0/world1, no sampler, no DDP-wrap, no all-reduce) → byte-identical. ---
    import torch.distributed as dist
    ddp = "LOCAL_RANK" in os.environ
    if ddp:
        local_rank = int(os.environ["LOCAL_RANK"]); rank = int(os.environ["RANK"]); world = int(os.environ["WORLD_SIZE"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
        device = torch.device("cuda", local_rank)
    else:
        local_rank = rank = 0; world = 1
        device = torch.device("cuda" if (str(cfg.get("device", "cuda")) == "cuda"
                                         and torch.cuda.is_available()) else "cpu")
    is_main = (rank == 0)

    def log(*a, **k):
        if is_main:
            print(*a, **k, flush=True)

    seed_everything(seed)
    enforce_determinism(strict=truthy(cfg.get("determinism-strict", True)), precision=precision)
    log(f"[centralized] precision={precision} device={device} ddp={ddp} world={world} "
        f"batch/gpu={cfg.get('batch-size')} global_batch={int(cfg.get('batch-size', 16)) * world} "
        f"precision_state={precision_state()}")

    task = get_task("nuscenes_detection")
    # The POOLED training set = the union of the FL log-group partition's client tokens (so the
    # centralized model sees EXACTLY the data the FL clients collectively see — a matched comparison).
    part = task._partition(cfg)
    pooled_tokens = sorted({t for toks in part["client_tokens"].values() for t in toks})
    train_split = str(cfg["nuscenes-train-split"])
    info_list, _ = task._load_info(cfg, train_split)
    log(f"[centralized] pooled train set: {len(pooled_tokens)} keyframes "
        f"(union of {part['num_clients']} log-group clients)")

    _aug = _aug_from_run(cfg)
    _gtp = _gtpaste_from_run(cfg)            # TRAIN-ONLY rare-class GT-paste (default None ⇒ byte-identical)
    ds = NuScenesMultimodalDataset(info_list, P.get_dataroot(cfg), sample_tokens=pooled_tokens,
                                   n_sweeps=int(cfg.get("det-lidar-sweeps", 1)), augment=_aug, gtpaste=_gtp)
    if _aug is not None:
        log(f"[centralized] BEV aug ON: {_aug}")
    if _gtp is not None:
        log(f"[centralized] GT-paste ON: db={_gtp['db_path']} counts={_gtp['counts']} "
            f"yaw_jitter={_gtp['yaw_jitter']} min_pts={_gtp['min_points']}")

    # MCR P1 (rare-class exposure lever; default-OFF ⇒ ds unchanged ⇒ byte-identical): class-balanced
    # repeat-factor resampling. Rare classes (trailer/bus/bicycle/moto/CV — each in ~18-21% of keyframes)
    # get oversampled so each epoch sees more distinct (re-augmented) rare-class scenes. Built ONCE
    # (epoch-invariant, seeded by the run seed ⇒ identical on every DDP rank) so the existing
    # DistributedSampler still owns the per-epoch reshuffle + 4-rank shard. Eval uses a SEPARATE loader.
    if truthy(cfg.get("det-cbgs", False)):
        from fl_v3.data.nuscenes.cbgs import dataset_inrange_classes, build_cbgs_indices, CBGSWrapper
        from fl_v3.data.nuscenes.class_map import NUM_CLASSES
        cbgs_thresh = float(cfg.get("det-cbgs-thresh", 0.5))
        cbgs_max = float(cfg.get("det-cbgs-max-repeat", 4.0))
        _idx, _stats = build_cbgs_indices(dataset_inrange_classes(ds), n_classes=NUM_CLASSES,
                                          thresh=cbgs_thresh, seed=seed, max_repeat=cbgs_max)
        ds = CBGSWrapper(ds, _idx)
        log(f"[centralized] CBGS ON: thresh={cbgs_thresh} max_repeat={cbgs_max} "
            f"{_stats['N']}→{_stats['expanded']} keyframes ({_stats['ratio']}x); "
            f"r_c={_stats['r_c']}")

    def epoch_loader(epoch):
        # REVIEW-FIX (HIGH, det-review #1/2/4/5): build a FRESH loader each epoch seeded by (seed+epoch)
        # so the shuffle order is a pure function of (seed, epoch) — independent of where the run started.
        # DDP (MCR P2): a DistributedSampler shards the epoch across ranks; seed it (seed+epoch) + set_epoch
        # so the global shuffle stays a pure function of (seed, epoch). drop_last when compiling the BEV
        # stack (no partial-batch recompile). Single-GPU: sampler=None → the original per-epoch shuffle.
        from torch.utils.data.distributed import DistributedSampler
        sampler = None
        if ddp:
            sampler = DistributedSampler(ds, num_replicas=world, rank=rank, shuffle=True,
                                         seed=seed + epoch, drop_last=args.compile_bev)
            sampler.set_epoch(epoch)
        return make_loader(ds, batch_size=int(cfg.get("batch-size", 16)), shuffle=(sampler is None),
                           sampler=sampler, num_workers=int(cfg.get("num-workers", 4)), seed=seed + epoch,
                           collate_fn=detection_collate_fn, drop_last=args.compile_bev)

    exp_dir = os.path.join(args.out_dir, args.tag)
    os.makedirs(exp_dir, exist_ok=True)
    ckpt_path = os.path.join(exp_dir, "final_model.pt")
    opt_path = os.path.join(exp_dir, "optimizer.pt")
    curve_path = os.path.join(exp_dir, "train_curve.json")

    model = task.build_model(cfg).to(device)
    criterion = task.build_criterion(cfg)

    # Resume loads the (prefix-stripped) checkpoint into the RAW model BEFORE compile/DDP-wrap so the keys
    # match (compile adds '._orig_mod.', DDP adds 'module.' — both absent from the saved ckpt); the optimizer
    # state is loaded after the optimizer is rebuilt. This also fixes a latent compile+resume key mismatch.
    curve, start_epoch, _os_state = [], 0, None
    if args.resume and os.path.isfile(ckpt_path) and os.path.isfile(opt_path):
        model.load_state_dict(torch.load(ckpt_path, map_location=device), strict=True)
        _os_state = torch.load(opt_path, map_location=device)
        start_epoch = int(_os_state["epoch"])
        if os.path.isfile(curve_path):
            curve = json.load(open(curve_path))
        log(f"[centralized] RESUMED from epoch {start_epoch}")

    # MCR Phase-1 Exp3 (all default-off ⇒ baseline byte-identical): build the EMA tracker from the BARE
    # model BEFORE compile/DDP — a clean deep copy whose .parameters() order matches the wrapped model's,
    # so the loop's per-step ema_model.update_parameters(model) stays aligned and the saved EMA weights
    # load strict=True into a bare eval model (no module./_orig_mod. prefixes). use_buffers=False keeps the
    # constant grid/frustum buffers at their (static) values.
    ema_decay = float(cfg.get("det-ema-decay", 0.0))
    ema_model = None
    if ema_decay and ema_decay > 0:
        from torch.optim.swa_utils import AveragedModel
        def _ema_avg(avg_p, new_p, _n, _d=ema_decay):
            return _d * avg_p + (1.0 - _d) * new_p
        ema_model = AveragedModel(model, avg_fn=_ema_avg, use_buffers=False)
        log(f"[centralized] EMA enabled (decay={ema_decay}) — saved to <tag>/ema/final_model.pt")

    if args.compile:                      # static-shape Swin subgraph (amortized over the run)
        model.camera_backbone = torch.compile(model.camera_backbone)
    if args.compile_bev:                  # MCR P2: the static BEV conv stack (camera_neck+fusion+bev_neck+head)
        model.camera_neck = torch.compile(model.camera_neck)
        model.fusion = torch.compile(model.fusion)
        model.bev_neck = torch.compile(model.bev_neck)
        model.head = torch.compile(model.head)
        if getattr(model, "lidar_backbone", None) is not None:   # MCR P1: static 256²/512² shapes → compiles clean
            model.lidar_backbone = torch.compile(model.lidar_backbone)
    log(f"[centralized] compile(backbone)={args.compile} compile(bev)={args.compile_bev}")

    # LR param groups (backbone @ lr*mult; fusion/head @ lr) over named params BEFORE DDP-wrap so the
    # 'camera_backbone.' prefix test still matches (DDP would prepend 'module.'). Frozen backbone →
    # bb_params empty → flat Adam, byte-identical to the pre-MCR baseline.
    base_lr = float(cfg.get("learning-rate", 3e-3))
    wd = float(cfg.get("weight-decay", 0.0))
    bb_mult = float(cfg.get("det-backbone-lr-mult", 0.1))
    bb_params, rest_params = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (bb_params if n.startswith("camera_backbone.") else rest_params).append(p)
    _fused = (device.type == "cuda" and precision == "fp16")
    # Exp3: Adam (default ⇒ baseline-identical) | AdamW (decoupled weight-decay — the correct WD vehicle,
    # still the Adam family per the reproduction-fidelity decision). Same args + fused path.
    opt_name = str(cfg.get("det-optimizer", "adam")).lower()
    OptCls = torch.optim.AdamW if opt_name == "adamw" else torch.optim.Adam
    if bb_params:
        optimizer = OptCls(
            [{"params": rest_params, "lr": base_lr},
             {"params": bb_params, "lr": base_lr * bb_mult}],
            lr=base_lr, weight_decay=wd, fused=_fused)
        log(f"[centralized] TRAINED backbone ({opt_name}): {len(bb_params)} backbone @ lr={base_lr*bb_mult:.2e} "
            f"(mult={bb_mult}); {len(rest_params)} fusion/head @ lr={base_lr:.2e} wd={wd} fused={_fused}")
    else:
        optimizer = OptCls(rest_params, lr=base_lr, weight_decay=wd, fused=_fused)  # frozen path
    if _os_state is not None:
        optimizer.load_state_dict(_os_state["optimizer"])
    grad_clip_norm = float(cfg.get("grad-clip-norm", 0.0))

    # Exp3: per-step LR schedule (default 'none' ⇒ flat LR, baseline-identical). 'onecycle' = warmup ramp
    # (pct_start) → cosine decay, peaking at the per-group base LRs; total_steps spans the WHOLE run so the
    # cosine tail lands at the final epoch. loop.train_one_epoch steps it once per optimizer step.
    sched = None
    sched_name = str(cfg.get("lr-schedule", "none")).lower()
    if sched_name == "onecycle":
        _spe = len(epoch_loader(start_epoch))
        steps_per_epoch = min(args.max_steps, _spe) if args.max_steps else _spe
        total_steps = max(1, (args.epochs - start_epoch) * steps_per_epoch)
        max_lrs = [base_lr, base_lr * bb_mult] if bb_params else [base_lr]
        warmup_frac = float(cfg.get("lr-warmup-frac", 0.15))
        sched = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=max_lrs, total_steps=total_steps, pct_start=warmup_frac,
            anneal_strategy="cos", div_factor=10.0, final_div_factor=100.0)
        log(f"[centralized] OneCycleLR: total_steps={total_steps} "
            f"(spe={steps_per_epoch}×{args.epochs-start_epoch}) pct_start={warmup_frac} max_lr={max_lrs}")
        if _os_state is not None and _os_state.get("scheduler") is not None:
            sched.load_state_dict(_os_state["scheduler"])

    if ddp:  # wrap AFTER compile + optimizer; GroupNorm/LayerNorm only (no BN) → no SyncBatchNorm needed.
        # static_graph=True: the model has a CONSTANT set of unused params every iteration —
        # GeneralizedLSSFPN discards lats[0]/lats[1] when out_level=2 (feat_stride=16), so
        # camera_neck.lateral.0/1 (+norms) never get grad. static_graph records the used set ONCE (more
        # efficient than find_unused_parameters=True, and the recommended combo with torch.compile).
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank,
            broadcast_buffers=True, gradient_as_bucket_view=True, static_graph=True)

    def save_ckpt(epoch):
        # RANK-0 ONLY under DDP (all ranks hold DDP-synced identical params → one writer; a barrier first so
        # no rank races ahead into the next epoch's sampler reseed). Unwrap DDP (.module) AND strip the
        # compile '._orig_mod.' + DDP 'module.' prefixes so the ckpt loads strict=True into a bare model
        # (the readiness eval). The trainable checksum is over VALUES → name-prefix-independent.
        if ddp:
            dist.barrier()
        if not is_main:
            return None
        core = model.module if ddp else model
        full = {k.replace("module.", "", 1).replace("._orig_mod.", "."): v for k, v in core.state_dict().items()}
        torch.save(full, ckpt_path)
        torch.save({"optimizer": optimizer.state_dict(), "epoch": epoch,
                    "scheduler": (sched.state_dict() if sched is not None else None)}, opt_path)
        arrs = [v.detach().cpu().numpy() for v in trainable_state_dict(core).values()]
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
            # Exp3 recipe knobs (recorded for the lever→mAP attribution table)
            "det-optimizer": opt_name, "weight-decay": wd, "lr-schedule": sched_name,
            "lr-warmup-frac": cfg.get("lr-warmup-frac") if sched_name != "none" else None,
            "det-ema-decay": ema_decay, "det-backbone-lr-mult": bb_mult,
            "FL_TRAINABLE_CHECKSUM": chk, "tag": args.tag,
            # Matched-budget caveat (det-review): vs FL this differs by (a) NO cross-client FedAvg
            # averaging AND (b) ONE Adam optimizer kept warm across epochs (FL rebuilds Adam per
            # client/round). A "works centrally, dies under FL" result implicates the FL regime
            # broadly (averaging + per-round optimizer reset), not averaging in isolation.
            "matched_budget_note": "epochs==rounds at 1 local-epoch/round; warm-Adam vs FL per-round reset",
        }
        json.dump(prov, open(os.path.join(exp_dir, "provenance.json"), "w"), indent=2, sort_keys=True)
        # Exp3: persist the EMA weights as a SELF-CONTAINED eval checkpoint in <exp_dir>/ema/ (own
        # checksum + provenance so t4_readiness_eval's checksum + precision-regime checks pass on it
        # directly). ema_model.module is the bare deep copy → state_dict keys are already prefix-free.
        if ema_model is not None:
            ema_dir = os.path.join(exp_dir, "ema")
            os.makedirs(ema_dir, exist_ok=True)
            ecore = ema_model.module
            efull = {k.replace("._orig_mod.", "."): v for k, v in ecore.state_dict().items()}
            torch.save(efull, os.path.join(ema_dir, "final_model.pt"))
            earrs = [v.detach().cpu().numpy() for v in trainable_state_dict(ecore).values()]
            echk = numpy_state_checksum(earrs)
            with open(os.path.join(ema_dir, "trainable_checksum.txt"), "w") as f:
                f.write(echk + "\n")
            eprov = dict(prov)
            eprov.update({"weights": "ema", "ema_decay": ema_decay, "FL_TRAINABLE_CHECKSUM": echk})
            json.dump(eprov, open(os.path.join(ema_dir, "provenance.json"), "w"), indent=2, sort_keys=True)
            # Milestone snapshot (convergence-curve diagnostic): a kept, self-contained EMA eval target.
            if epoch in snap_epochs:
                snap_dir = os.path.join(exp_dir, f"ema_ep{epoch}")
                os.makedirs(snap_dir, exist_ok=True)
                torch.save(efull, os.path.join(snap_dir, "final_model.pt"))
                with open(os.path.join(snap_dir, "trainable_checksum.txt"), "w") as f:
                    f.write(echk + "\n")
                json.dump(eprov, open(os.path.join(snap_dir, "provenance.json"), "w"), indent=2, sort_keys=True)
        return chk

    log(f"===== centralized training: {args.epochs} epochs (matched to {args.epochs}-round FL) =====")
    for epoch in range(start_epoch, args.epochs):
        t0 = time.perf_counter()
        seed_everything(seed + epoch)        # global-RNG per-epoch seed (any forward-side RNG)
        loader = epoch_loader(epoch)         # per-epoch loader seeded (seed+epoch) → resume-reproducible
        m = train_one_epoch(model, loader, criterion, optimizer, device,
                            grad_clip_norm=grad_clip_norm, scheduler=sched, ema_model=ema_model,
                            max_steps=args.max_steps, precision=precision)   # max_steps=0 → full epoch
        if ddp:  # global mean loss for the logged curve (each rank trained on 1/world of the epoch)
            tt = torch.tensor([m["loss"] * m["num_samples"], m["num_samples"]], device=device, dtype=torch.float64)
            dist.all_reduce(tt, op=dist.ReduceOp.SUM)
            m = {"loss": float(tt[0] / tt[1]) if float(tt[1]) > 0 else 0.0, "num_samples": float(tt[1])}
        dt = time.perf_counter() - t0
        chk = save_ckpt(epoch + 1)
        if is_main:
            rec = {"epoch": epoch + 1, "train_loss": float(m["loss"]),
                   "num_samples": int(m["num_samples"]), "seconds": dt, "checksum": chk}
            curve.append(rec)
            json.dump(curve, open(curve_path, "w"), indent=2)
            log(f"[centralized] epoch {epoch+1}/{args.epochs} loss={m['loss']:.5f} "
                f"n={int(m['num_samples'])} time={dt:.0f}s checksum={(chk or '')[:16]}")

    log(f"[centralized] DONE — checkpoint {ckpt_path}")
    if is_main:
        log(f"[centralized] FL_TRAINABLE_CHECKSUM = "
            f"{open(os.path.join(exp_dir,'trainable_checksum.txt')).read().strip()}")
    if ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
