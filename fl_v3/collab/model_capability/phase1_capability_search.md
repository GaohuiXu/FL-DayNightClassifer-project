# MCR Phase 1 — centralized capability search (bf16-AMP)

> Goal (D17): raise the centralized ceiling clearly above the **0.36 mAP** frozen-Swin baseline, in the
> D16 bf16-AMP regime, measuring official mAP/NDS + car recall + ASR-eligible count per lever. Headline
> lever: **train the camera backbone** (amends D1). Status: recipe built + validated; experiments in flight.

## Recipe infrastructure built (this session)

All additive + default-off ⇒ the frozen baseline + the FL/gate byte-identity path are unchanged
(40 affected tests pass; full suite green).

- **Backbone unfreeze** — `det-freeze-backbone=false` → `CameraBackbone(frozen=False)` trains all 28.3M
  Swin params (the `train()`-override eval-lock is gated on `self.frozen`, so unfrozen ⇒ normal train()).
  Verified NOT blocked by the 62-tensor `assert_trainable_layout` contract (that is called only in
  `test_fl_trainable_only`, never in production `build_model`; the layout/FL-wire ripple is a **Phase-3**
  concern, not centralized).
- **LR param groups** (`centralized_train.py`) — pretrained backbone @ `lr*det-backbone-lr-mult`
  (default 0.1), from-scratch fusion/head @ `lr`. Frozen path (no trainable backbone) → single flat Adam,
  byte-identical to before.
- **Activation checkpointing** (`camera_backbone.py`, `det-activation-checkpoint`) — per-Swin-stage
  `torch.utils.checkpoint(use_reentrant=False)`, only when trained + in train mode (frozen/eval ⇒
  byte-identical). The D16-envelope VRAM lever that lets bf16 backbone-training fit a useful batch.
- **Grad clip** (`grad-clip-norm`, default 0=off) + **scheduler/EMA hooks** in `train_one_epoch`
  (optional args, default None ⇒ no-op for FL/gate callers). Scheduler/EMA wiring in `centralized_train`
  is deferred to the schedule/EMA experiment (Exp3).

**Login-T4 smoke (real mini batch, real model) — PASS:** unfrozen Swin + activation-ckpt + bf16 autocast
+ fp32 head upcast + grad-clip + param groups train CLEAN — 169 backbone tensors trainable, backbone
receives gradients, loss 47.5→3.6 on a fixed batch, **no NaN** (`scripts/p1_unfrozen_smoke.py`).

## Experiment ladder (centralized, bf16-AMP, official mAP/NDS/car-recall/ASR-eligible per run)

| # | what | tier | status |
|---|---|---|---|
| ~~Exp0~~ | ~~frozen Swin bf16 baseline~~ | — | **ABANDONED** (orchestrator: don't need a weak baseline to compare against) |
| **Exp1** | **unfreeze backbone** + LR groups + activation-ckpt + grad-clip (the headline lever) | A100 | **BLOCKED on profiling** — no big run until per-component per-step profile + GPU-util verify + per-component optimization done (orchestrator) |
| Exp2 | + LiDAR multi-sweep (1→10) | A100fat | pending |
| Exp3 | + EMA + cosine+warmup schedule + longer (24–30ep) | A100fat | pending |
| Exp4 | + image/BEV-grid resolution bump (VRAM permitting) | A100fat | pending |
| — | combined best-recipe → the strong centralized recipe | A100fat | pending |

> **PAUSED 2026-06-21 (orchestrator):** the 12–30h/run cost is the concern. Both jobs cancelled before
> producing results. The recipe **code + config + smoke are committed and intact** (`f25d0b4`) — nothing
> is lost; only the long runs were halted. **Open discussion:** the compute strategy for Phase 1 (run
> budget per experiment, GPU tier, whether to use shorter/cheaper signal runs, epoch count, parallelism)
> before relaunching any heavy job.

Lever ranking (charter): unfreeze (headline) > LiDAR sweeps > resolution/BEV > schedule/EMA > fusion
redesign. SDPA rewrite of `ShiftedWindowAttention` is a Phase-2 SPEED enabler (not a mAP lever) — deferred
until the unfreeze mAP lift is proven (the unfrozen run is affordable now via activation-ckpt).

## Config

`configs/p1_unfrozen.json` = `t4_reference.json` + `det-freeze-backbone=false`, `det-backbone-lr-mult=0.1`,
`det-activation-checkpoint=true`, `grad-clip-norm=35.0`, `num-workers=8` — ONLY the backbone-training knobs
differ from the frozen baseline (clean lever isolation). `learning-rate=0.003` base kept (fusion/head LR
matches the frozen baseline; backbone @ 3e-4).

## Open / next

- Launch Exp1 on A100fat (resumable: centralized_train checkpoints per-epoch + --resume restores the
  param-group optimizer state).
- When Exp0/Exp1 land: official readiness eval (`t4_readiness_eval.py --diagnostic`) on each checkpoint →
  the lever→mAP table; decide whether the unfreeze clears the "clearly > 0.36 + high car recall + large
  ASR-eligible count" bar before stacking Exp2–Exp4.
