"""MCR Phase-3 STEP 3 — cRT decoupling probe (the DECISIVE head-vs-representation test).

The investigation (read-only teardown + step-1 separability + step-2 gradient cosine) localized the
FL tail collapse to **tail-specific head UNDER-SHARPENING**: the BEV features retain tail signal
(sep-AUC 0.78-0.90) and the 25 clients agree in gradient direction (no conflict, %neg≈0), yet the
tail heatmap stays diffuse/low-confidence (TP≈FP, over-predicted 28-83×) → tail AP≈0 with high recall.
This probe runs the classic **classifier re-training (cRT; Kang et al. ICLR'20; CReFF/CCVR for FL)**:
freeze the FL-converged feature stack and re-train ONLY the detection head on **class-balanced** data
(CBGS repeat-factor sampling). It is the clean isolation experiment —

  - tail AP RECOVERS  ⇒ the gap is HEAD-CONFINED (calibration/sharpening); features are fine; the
                        deployable fix is a head-rebalancing step (federated cRT / CReFF).
  - tail AP STAYS LOW ⇒ the feature space IS tail-distorted (the CCVR/RUCR "bounded by representation"
                        caveat bites) ⇒ a representation-level fix (FedVLS during training, more rounds)
                        is required — and we learn this CHEAPLY, before spending a long FL run.

Either outcome is publishable and directs the next GPU-hour. This is a STANDALONE diagnostic that
CALLS the platform (mirrors centralized_train.py's pipeline + p3_grad_conflict.py's "no platform
mutation" norm); the FL library (loop/tasks/strategy) is untouched.

DESIGN (scientific cleanliness):
  * Server-side balanced retrain on the POOLED train set (union of the FL log-group clients' tokens) —
    exactly the data the 25 FL clients collectively saw. Answers "is it head-fixable?" unambiguously;
    the protocol-honest federated-cRT is a separate follow-up.
  * FREEZE = every module except the trainable prefix (default ``head``) gets requires_grad=False AND is
    pinned in eval() (the model is BN-free — Swin LayerNorm + GroupNorm — but eval() also disables Swin
    activation-checkpointing / any train-mode path, so the frozen feature extractor is DETERMINISTIC).
    We override each frozen child's .train() to a no-op so train_one_epoch's internal model.train()
    cannot wake them (the same "freeze survives model.train()" idiom CameraBackbone uses for itself).
  * Head-only Adam (train_one_epoch builds the optimizer over requires_grad params ⇒ head only) +
    OneCycle + optional EMA — the bb02d recipe, applied to the head.
  * Saved checkpoint is eval-compatible (final_model.pt = FULL state_dict + trainable_checksum.txt +
    provenance.json), so the SAME run_eval_ckpt_a100.sh / t4_readiness_eval --diagnostic that scored the
    0.247 FedAvg baseline scores this — an apples-to-apples per-class AP comparison. The checksum is
    computed with all params UN-frozen so it matches the eval's fresh-model recomputation (requires_grad
    based) and the eval's checksum gate passes.

Run:  CONFIG=fl_v3/configs/fl_bb02d_fedadam.json INIT=<fedavg round_15>/final_model.pt \
      CACHE=<msweep10> sbatch fl_v3/scripts/run_p3_crt_probe.sh
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--init-from", required=True,
                    help="FL global final_model.pt to start from (e.g. the FedAvg round_15 snapshot)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--tag", default="crt_probe")
    ap.add_argument("--epochs", type=int, default=2, help="head-retrain passes over the (CBGS-expanded) pool")
    ap.add_argument("--max-steps", type=int, default=0, help="cap steps/epoch (0=full). Smoke/cost-bound only.")
    ap.add_argument("--trainable-prefix", default="head",
                    help="comma-sep param-name prefixes to KEEP trainable; everything else is frozen. "
                         "'head' = full detection head (decisive run); 'head.heatmap' = purest classifier-only.")
    ap.add_argument("--snapshot-epochs", default="",
                    help="comma-sep epochs at which to ALSO save a kept self-contained snapshot dir "
                         "(<tag>/crt_ep<N>/ + <tag>/crt_ep<N>_ema/) for a per-epoch convergence eval.")
    ap.add_argument("--dry-run", action="store_true",
                    help="LOGIN SMOKE: build everything (model+init load+freeze+CBGS+optimizer+provenance) "
                         "but DO NOT train — validates the full wiring without a GPU forward, then exits.")
    ap.add_argument("overrides", nargs="*", default=[])
    args = ap.parse_args()
    snap_epochs = {int(x) for x in args.snapshot_epochs.split(",") if x.strip()}
    prefixes = tuple(p.strip() for p in args.trainable_prefix.split(",") if p.strip())
    if not prefixes:
        raise SystemExit("--trainable-prefix must name at least one module prefix to keep trainable")

    from fl_v3.training.tasks import get_task, trainable_state_dict, _aug_from_run, _gtpaste_from_run
    from fl_v3.training.loop import train_one_epoch
    from fl_v3.engine.local_runner import numpy_state_checksum
    from fl_v3.data.nuscenes.dataset import NuScenesMultimodalDataset, make_loader
    from fl_v3.data.nuscenes import paths as P
    from fl_v3.data.nuscenes.cbgs import dataset_inrange_classes, build_cbgs_indices, CBGSWrapper
    from fl_v3.data.nuscenes.class_map import NUM_CLASSES
    from fl_v3.models.fusion.collate import detection_collate_fn
    from fl_v3.utils.runtime import enforce_determinism, precision_state, seed_everything, truthy

    with open(args.config, encoding="utf-8") as f:
        cfg = json.load(f)
    for ov in args.overrides:
        k, _, v = ov.partition("=")
        cfg[k] = _parse_scalar(v)

    precision = str(cfg.get("precision", "bf16"))     # D16 science default
    seed = int(cfg.get("seed", 42))
    device = torch.device("cuda" if (str(cfg.get("device", "cuda")) == "cuda"
                                     and torch.cuda.is_available()) else "cpu")
    seed_everything(seed)
    # bf16-AMP science path (D16); strict=False (this is a bf16 diagnostic, not the offline byte-id dev tool).
    enforce_determinism(strict=False, precision=precision)
    print(f"[cRT] precision={precision} device={device} precision_state={precision_state()} "
          f"trainable_prefix={prefixes} init_from={args.init_from}", flush=True)

    task = get_task("nuscenes_detection")
    # POOLED train = union of the FL log-group partition's client tokens (the exact data the 25 FL clients
    # collectively saw) — so cRT operates on the same distribution FedAvg trained on, isolating the head.
    part = task._partition(cfg)
    pooled_tokens = sorted({t for toks in part["client_tokens"].values() for t in toks})
    train_split = str(cfg["nuscenes-train-split"])
    info_list, _ = task._load_info(cfg, train_split)
    print(f"[cRT] pooled train set: {len(pooled_tokens)} keyframes "
          f"(union of {part['num_clients']} log-group clients)", flush=True)

    _aug = _aug_from_run(cfg)            # TRAIN-time BEV aug (helps the head generalize; matches the recipe)
    _gtp = _gtpaste_from_run(cfg)        # default None ⇒ off
    ds = NuScenesMultimodalDataset(info_list, P.get_dataroot(cfg), sample_tokens=pooled_tokens,
                                   n_sweeps=int(cfg.get("det-lidar-sweeps", 1)), augment=_aug, gtpaste=_gtp)
    print(f"[cRT] aug={'on' if _aug is not None else 'off'} gtpaste={'on' if _gtp is not None else 'off'} "
          f"lidar_sweeps={int(cfg.get('det-lidar-sweeps', 1))}", flush=True)

    # CBGS (the cRT mechanism): class-balanced repeat-factor sampling so the rare tail classes get the
    # balanced gradient mass FL starved them of. DEFAULT ON for the probe (the whole point); cfg can tune.
    cbgs_stats = None
    if truthy(cfg.get("det-cbgs", True)):
        cbgs_thresh = float(cfg.get("det-cbgs-thresh", 0.5))
        cbgs_max = float(cfg.get("det-cbgs-max-repeat", 4.0))
        _idx, cbgs_stats = build_cbgs_indices(dataset_inrange_classes(ds), n_classes=NUM_CLASSES,
                                              thresh=cbgs_thresh, seed=seed, max_repeat=cbgs_max)
        ds = CBGSWrapper(ds, _idx)
        print(f"[cRT] CBGS ON: thresh={cbgs_thresh} max_repeat={cbgs_max} "
              f"{cbgs_stats['N']}→{cbgs_stats['expanded']} ({cbgs_stats['ratio']}x); r_c={cbgs_stats['r_c']}",
              flush=True)
    else:
        print("[cRT] CBGS OFF (balanced sampling disabled by config)", flush=True)

    def epoch_loader(epoch: int):
        # Fresh per-epoch loader seeded (seed+epoch): shuffle order a pure function of (seed, epoch).
        return make_loader(ds, batch_size=int(cfg.get("batch-size", 4)), shuffle=True,
                           num_workers=int(cfg.get("num-workers", 4)), seed=seed + epoch,
                           collate_fn=detection_collate_fn)

    # --- model + init-from-FL load (full state dict, strict) ---
    model = task.build_model(cfg).to(device)
    full = torch.load(args.init_from, map_location=device)
    model.load_state_dict(full, strict=True)
    criterion = task.build_criterion(cfg)
    n_total = sum(p.numel() for p in model.parameters())

    # --- FREEZE everything except the trainable prefix (per-PARAM requires_grad) ---
    for n, p in model.named_parameters():
        p.requires_grad_(n.startswith(prefixes))
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    train_names = sorted({n for n, p in model.named_parameters() if p.requires_grad})
    if n_train == 0:
        raise SystemExit(f"--trainable-prefix={prefixes} froze EVERYTHING (no param matched). Check the prefix.")
    # A top-level child stays TRAINABLE (kept in train mode) iff some trainable-prefix's first component
    # names it (it then contains the requires_grad params); every other child is frozen.
    train_tops = {pf.split(".")[0] for pf in prefixes}

    # EMA over the (head) params — the bb02d recipe used EMA to reach 0.5656, so the eval target matches that
    # regime. avg_fn on frozen params is a no-op (avg of identical values). Built from a CLEAN model, BEFORE the
    # no-op .train() monkeypatch below, so the deep-copied EMA model is unpatched.
    ema_decay = float(cfg.get("det-ema-decay", 0.999))
    ema_model = None
    if ema_decay and ema_decay > 0:
        from torch.optim.swa_utils import AveragedModel
        def _ema_avg(avg_p, new_p, _n, _d=ema_decay):
            return _d * avg_p + (1.0 - _d) * new_p
        ema_model = AveragedModel(model, avg_fn=_ema_avg, use_buffers=False)
        print(f"[cRT] EMA on (decay={ema_decay})", flush=True)

    # Pin frozen children in eval() AND make their .train() a no-op so train_one_epoch's model.train()
    # cannot wake them (deterministic frozen feature extractor — no activation-ckpt / train-mode path). Same
    # "freeze survives model.train()" idiom CameraBackbone uses for itself, generalized to every frozen child.
    frozen_children, train_children = [], []
    for name, child in model.named_children():
        if name in train_tops:
            train_children.append(name)
        else:
            child.eval()
            child.train = (lambda mode=True, _c=child: _c)  # frozen: ignore parent .train()
            frozen_children.append(name)
    print(f"[cRT] FROZEN children={frozen_children}", flush=True)
    print(f"[cRT] TRAINABLE children={train_children}  ({n_train:,}/{n_total:,} params = "
          f"{100.0*n_train/n_total:.2f}%, {len(train_names)} tensors)", flush=True)

    # --- head-only optimizer (over requires_grad params = the head; backbone frozen ⇒ flat single-group) ---
    base_lr = float(cfg.get("learning-rate", 1e-3))
    wd = float(cfg.get("weight-decay", 0.0))
    _fused = (device.type == "cuda" and not torch.backends.cudnn.deterministic)
    opt_name = str(cfg.get("det-optimizer", "adam")).lower()
    OptCls = torch.optim.AdamW if opt_name == "adamw" else torch.optim.Adam
    head_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = OptCls(head_params, lr=base_lr, weight_decay=wd, fused=_fused)
    grad_clip_norm = float(cfg.get("grad-clip-norm", 0.0))

    # OneCycle over the whole run (warmup→cosine), peaking at base_lr (head-only ⇒ single max_lr).
    sched = None
    sched_name = str(cfg.get("lr-schedule", "onecycle")).lower()
    if sched_name == "onecycle" and not args.dry_run:
        _spe = len(epoch_loader(0))
        steps_per_epoch = min(args.max_steps, _spe) if args.max_steps else _spe
        total_steps = max(1, args.epochs * steps_per_epoch)
        warmup_frac = float(cfg.get("lr-warmup-frac", 0.15))
        sched = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=base_lr, total_steps=total_steps, pct_start=warmup_frac,
            anneal_strategy="cos", div_factor=10.0, final_div_factor=100.0)
        print(f"[cRT] OneCycleLR: total_steps={total_steps} (spe={steps_per_epoch}×{args.epochs}) "
              f"pct_start={warmup_frac} max_lr={base_lr:.2e}", flush=True)

    exp_dir = os.path.join(args.out_dir, args.tag)
    os.makedirs(exp_dir, exist_ok=True)

    def _checksum_unfrozen(m) -> str:
        # Compute over the FULL bb02d trainable layout (all params requires_grad=True) so it MATCHES the
        # eval's fresh-model recomputation (trainable_state_dict is requires_grad-based) ⇒ checksum gate passes.
        saved = [(n, p.requires_grad) for n, p in m.named_parameters()]
        for _, p in m.named_parameters():
            p.requires_grad_(True)
        try:
            arrs = [v.detach().cpu().numpy() for v in trainable_state_dict(m).values()]
            chk = numpy_state_checksum(arrs)
        finally:
            for (n, rg), p in zip(saved, m.parameters()):
                p.requires_grad_(rg)
        return chk

    def _provenance(chk: str, epoch_done: int, weights: str) -> dict:
        return {
            "regime": "MCR-P3-cRT-probe",
            "scope": "diagnostic",
            "precision": precision, "seed": seed,
            "init_from": os.path.abspath(args.init_from),
            "trainable_prefix": list(prefixes), "trainable_children": train_children,
            "frozen_children": frozen_children,
            "trainable_params": n_train, "total_params": n_total,
            "weights": weights, "ema_decay": ema_decay if weights == "ema" else None,
            "epochs_done": epoch_done, "epochs_target": args.epochs,
            "pooled_keyframes": len(pooled_tokens), "n_log_group_clients": part["num_clients"],
            "det-lidar-sweeps": cfg.get("det-lidar-sweeps"), "det-cbgs": truthy(cfg.get("det-cbgs", True)),
            "cbgs": cbgs_stats,
            "learning-rate": base_lr, "det-optimizer": opt_name, "lr-schedule": sched_name,
            "grad-clip-norm": grad_clip_norm, "batch-size": cfg.get("batch-size"),
            "nuscenes-version": cfg.get("nuscenes-version"),
            "nuscenes-train-split": train_split, "nuscenes-val-split": cfg.get("nuscenes-val-split"),
            "nuscenes-partition-mode": cfg.get("nuscenes-partition-mode"),
            "det-camera-backbone": cfg.get("det-camera-backbone"),
            "FL_TRAINABLE_CHECKSUM": chk, "tag": args.tag,
        }

    def _save(dirpath: str, state_dict, m_for_chk, weights: str, epoch_done: int):
        os.makedirs(dirpath, exist_ok=True)
        full_sd = {k.replace("._orig_mod.", "."): v for k, v in state_dict.items()}
        fp = os.path.join(dirpath, "final_model.pt")
        torch.save(full_sd, fp)
        # verify-by-reload: a full quota / I-O error silently corrupts torch.save (ENOSPC — the call returns but
        # the bytes don't land; this exact failure cost us job 6781291). Reload immediately so a bad write fails
        # LOUDLY here instead of surfacing as a corrupt eval target hours later.
        try:
            _rt = torch.load(fp, map_location="cpu")
            if len(_rt) != len(full_sd):
                raise RuntimeError(f"reloaded {len(_rt)} keys != saved {len(full_sd)} (truncated write)")
        except Exception as e:
            raise RuntimeError(f"checkpoint {fp} failed verify-by-reload (disk full / I-O error?): {e}") from e
        chk = _checksum_unfrozen(m_for_chk)
        with open(os.path.join(dirpath, "trainable_checksum.txt"), "w") as f:
            f.write(chk + "\n")
        json.dump(_provenance(chk, epoch_done, weights),
                  open(os.path.join(dirpath, "provenance.json"), "w"), indent=2, sort_keys=True)
        return chk

    if args.dry_run:
        print("[cRT] DRY-RUN: wiring OK (model built, FL ckpt loaded, freeze applied, CBGS built, "
              "optimizer over head). Writing provenance to the raw dir and exiting WITHOUT training.",
              flush=True)
        chk = _save(exp_dir, model.state_dict(), model, "raw", 0)
        print(f"[cRT] DRY-RUN done — checksum={chk[:16]} dir={exp_dir}", flush=True)
        return

    # --- the cRT retrain loop (frozen stack stays eval via the no-op .train(); head trains) ---
    curve = []
    print(f"===== cRT head-retrain: {args.epochs} epoch(s) over the CBGS pool =====", flush=True)
    for epoch in range(args.epochs):
        t0 = time.perf_counter()
        seed_everything(seed + epoch)
        loader = epoch_loader(epoch)
        m = train_one_epoch(model, loader, criterion, optimizer, device,
                            grad_clip_norm=grad_clip_norm, scheduler=sched, ema_model=ema_model,
                            max_steps=args.max_steps)
        dt = time.perf_counter() - t0
        # always overwrite the rolling raw + ema checkpoints
        chk_raw = _save(exp_dir, model.state_dict(), model, "raw", epoch + 1)
        chk_ema = None
        if ema_model is not None:
            chk_ema = _save(os.path.join(exp_dir, "ema"), ema_model.module.state_dict(),
                            ema_model.module, "ema", epoch + 1)
        if (epoch + 1) in snap_epochs:   # kept per-epoch snapshot(s) for the convergence curve
            _save(os.path.join(exp_dir, f"crt_ep{epoch+1}"), model.state_dict(), model, "raw", epoch + 1)
            if ema_model is not None:
                _save(os.path.join(exp_dir, f"crt_ep{epoch+1}_ema"), ema_model.module.state_dict(),
                      ema_model.module, "ema", epoch + 1)
        rec = {"epoch": epoch + 1, "train_loss": float(m["loss"]), "num_samples": int(m["num_samples"]),
               "seconds": dt, "checksum_raw": chk_raw, "checksum_ema": chk_ema}
        curve.append(rec)
        json.dump(curve, open(os.path.join(exp_dir, "train_curve.json"), "w"), indent=2)
        print(f"[cRT] epoch {epoch+1}/{args.epochs} loss={m['loss']:.5f} n={int(m['num_samples'])} "
              f"time={dt:.0f}s raw={chk_raw[:12]} ema={(chk_ema or '')[:12]}", flush=True)

    print(f"[cRT] DONE — raw {os.path.join(exp_dir, 'final_model.pt')}"
          + (f" + ema {os.path.join(exp_dir, 'ema', 'final_model.pt')}" if ema_model is not None else ""),
          flush=True)
    print("[cRT] eval: CKPTS=\"<tag> <tag>/ema\" EXTRA=\"det-eval-limit=0\" "
          "bash fl_v3/scripts/run_eval_ckpt_a100.sh  (full val, same protocol as the 0.247 FedAvg baseline)",
          flush=True)


if __name__ == "__main__":
    main()
