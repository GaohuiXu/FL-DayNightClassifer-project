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

## Directional result (job 6769898, 2026-06-21) — unfreezing WORKS

5-epoch unfrozen centralized run, locked optimized recipe (ckpt-off + SDPA + compile, bf16, A100):
- **mAP = 0.3398, NDS = 0.3323, car_recall = 0.91, eligible_N = 28,377** at **5 epochs** — verdict READY.
- Per-epoch loss STILL DROPPING (2.87 → 2.09 → 1.88 → 1.75 → **1.64**; ~0.10/epoch at ep5, not plateaued).
- vs the frozen baseline (0.36 at **15** epochs): unfrozen hits 0.34 in **5** epochs and is climbing; its
  **epoch-1 loss (2.87) is already below frozen's (3.21)** → converges faster per epoch → 15 ep should
  clearly beat 0.36. Model is attack-credible (recall 0.91, 28k eligible cars).
- Wall-clock: ~**11.6 min/epoch** (epoch-1 1018s incl. compile, then ~693s = 394 ms/step) — even the
  unfrozen+optimized model is FASTER per epoch than the frozen-UNoptimized baseline (~1009s).
- **PAUSED for a deeper optimization pass** (orchestrator: 11.6 min/epoch still too long for fast
  iteration → DDP + a code-level gap analysis before more big runs).

## Experiment ladder

| # | what | tier | status |
|---|---|---|---|
| ~~Exp0~~ | ~~frozen Swin bf16 baseline~~ | — | **ABANDONED** (orchestrator: don't need a weak baseline) |
| **Exp1** | unfreeze + LR-groups + grad-clip + ckpt-off + SDPA + compile, 5 ep | A100 | **DONE — mAP 0.34 @ 5ep, still climbing, READY** (job 6769898) |
| Exp1b | resume → 15 ep (matched-budget confirmation vs 0.36) | A100/DDP | pending deeper-optimization decision |
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
