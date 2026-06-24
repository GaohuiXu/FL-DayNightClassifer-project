# MCR Phase-1 — Capability summary (canonical; consolidates all phase-0/1/2 docs)

> The single, AUTHORITATIVE, human-readable record for the MCR capability push (D17) — read this first.
> Centralized BEVFusion-class detector on nuScenes, pure-PyTorch (no spconv, Rule #2), bf16-AMP (D16).
> **Current best: 0.5656 mAP / 0.5733 NDS** (val, official 10-class) — NOT locked (owner: keep pushing centralized).
> Replaces the per-step docs (profile/optcompare/gap/speedup/push) — see git history. (The agent's auto-loaded
> working copy, memory `project_mcr_progress.md`, mirrors this but is not in the repo; THIS file is the source of truth.)
>
> **TO THE NEXT SESSION:** this session's context grew VERY long, so any conclusion/estimate below may have drifted —
> treat the numbers as a log of what happened and the forward-looking items as SUGGESTIONS, not a plan. **Re-investigate
> independently: read the code, re-run the diagnostics, and form your own view before committing to a direction.**

## TL;DR

Started from a frozen-Swin 0.36-mAP baseline; reached **0.5656 mAP / 0.5733 NDS** (current best, NOT yet locked),
READY, car_recall 0.98, ~28.7k ASR-eligible objects — a credible, balanced, attack-grade centralized model. The gap
to BEVFusion-base (0.679 mAP) is ~0.11, of which only **~0.02–0.03 is structural** (the no-spconv constraint); the
rest (~0.06–0.10) is closeable but expensive (heavier trunk / GT-paste / CBGS).

**OWNER DECISION (2026-06-24): do NOT lock 0.5656 as the reference yet — keep working on CENTRALIZED capability to push
the model higher before Phase 3.** (The bar of ≥0.50 on BOTH mAP and NDS is already met on the centralized model;
the operative target is higher — TBD with the owner, plausibly headroom so the *FL* model stays ≥0.50 after dilution,
or a closer-to-SOTA mAP.) The gap-analysis "bank now, perception is instrumentation" recommendation is OVERRIDDEN
by the owner: a stronger centralized model is wanted for the USENIX submission. **Next levers below (Handoff §).**

## The lever-by-lever progression (the method + the mindset)

bf16-AMP, global-16 DDP, same pooled-25-client trainval, decode 0.01/500 (from step 2). Mindset: diagnose the
binding constraint empirically (per-class AP + TP errors + train-vs-val curve), pull the cheapest lever that
targets it, measure, re-diagnose.

| # | mAP | NDS | lever | implementation | diagnosis it addressed |
|---|---:|---:|---|---|---|
| 0 | 0.36 | — | frozen-Swin baseline | D1 frozen backbone, single-sweep, minimal recipe | — |
| 1 | 0.4042 | — | train camera backbone | unfreeze Swin-T (amends D1); 2 LR groups (bb@0.1×); grad-clip 35 | frozen backbone = headline undercapacity |
| 2 | 0.4357 | 0.413 | schedule+EMA+AdamW | OneCycle warmup+cosine; EMA 0.999; AdamW wd0.01; 15ep | bare recipe; gains hit rare classes (not overfit) |
| 3 | 0.4451 | 0.475 | multi-sweep 1→10 | ego-comp accumulation + dt channel (info_cache+loader+PFN 7→8d) | small mAP, **vel_err −46% → big NDS** (motion) |
| 4 | 0.4948 | 0.538 | BEV aug | GlobalRotScaleTrans+flip, consistent pts/boxes/vel/lidar2img | 40ep diagnostic → OVERFITTING; zero aug was the cap |
| 5 | 0.4994 | 0.540 | img-flip + per-class wt | h-flip (lidar2img row-update) + mean-1 heatmap weight | recipe route SATURATING (diminishing) |
| 6 | 0.5359 | 0.533 | dense 2D LiDAR backbone | SECOND/PillarNet-2D ~2.5M trunk (3-stage+FPN) pre-fusion + 0.2m voxel | 640-param PFN had ZERO pre-fusion RF (capacity) — trailer +0.052 vs +0.004 loss-wt |
| 7 | **0.5656** | 0.573 | 4-stage backbone @0.2m | +4th stride-2 stage (H/8) + FPN level | 0.2m regressed large objects; 4th stage RF → bus 0.32→0.53, keep ped 0.80 |

**Dead-ends (the mindset record):** longer-training-alone overfits (40ep<15ep pre-aug); per-class-weight-alone ≈0
(→ capacity, not loss-balance); global-64 DDP under-converges (→ global-16 faster+better); channels_last neutral;
depth-supervision closes ZERO vs BEVFusion (it uses UNSUPERVISED LSS); CBGS/GT-paste/heavier-trunk = GPU-weeks, not
pursued.

## The final model + best config/ckpt

Swin-T (TRAINED, SDPA) → GeneralizedLSSFPN → DepthLSSTransform (LSS) ⊕ [PointPillars PFN → **LidarBackbone2D
(4-stage SECOND/PillarNet-2D)**] → ConvFuser → SECOND-FPN → CenterPoint head. 10-sweep LiDAR, 0.2m voxel (512²),
GroupNorm throughout. **Config: `fl_v3/configs/p1_bb02d.json`** (det-lidar-backbone + stages=4 + 0.2m + max-pillars
120000 + ckpt; on the msweep10 cache). **Ckpt: `fl_outputs/nuscenes/experiments/cycle_04/p2_ddp/bb02d_r20/ema_ep15/`**
(0.5656/0.5733). Per-class@best: car .85, ped .80, cone .72, moto .68, barrier .65, bus .53, truck .48, bicycle .42,
constr .23, trailer .22. Other key configs kept: p1_unfrozen, p1_exp3, p1_msweep(_aug/_aug2), p1_bb04 (0.4m+bb).

## Throughput (Phase 2 — why training is affordable)

Single-GPU 793→311 ms/step (2.55×) via torch.compile(backbone) + SDPA Swin attention + compile-BEV + a loss
cache/batch fix (killed ~12k per-object HtoDs) + fused-AdamW. 4-GPU DDP: global-16 is the winner (~3.1×, NVLink
all-reduce ~free; global-64 loader-bound + under-converges). Multi-sweep loader: pin DataLoader workers to 1 BLAS
thread (OMP/OPENBLAS/MKL=1 + torch.set_num_threads(1)) — kills the 256-thread oversubscription (6→4.7 min/ep). The
profiler is `scripts/p1_profile_a100.py`; the VRAM probe (backbone × voxel) is `scripts/p1_vram_probe.py`
(0.2m+4stage+ckpt fits 40GB). Per-epoch: 0.2m+4stage ≈ ~10 min/epoch global-16.

## Determinism / precision (Phase 0)

ONE `precision={bf16,fp32}` knob (D16) replaced numeric-mode×determinism-level. bf16=science default;
fp32+strict=offline dev-regression tool (the static-AST ban over models/fusion/** + byte-identity). The LiDAR
backbone is AST-clean (Conv2d/GN/ReLU/interpolate) and downstream of the perm-invariant scatter ⇒ no new
determinism obligation. All capability knobs default-OFF ⇒ byte-identical baseline; 247-test suite + the new
backbone/aug tests green.

## SOTA-gap conclusion + USENIX framing (the bank rationale)

The ~0.11 gap to BEVFusion is NOT mostly structural: dense 2D pillars MATCH/BEAT sparse-conv at matched resolution
(PillarNet 0.599 > CenterPoint-SECOND 0.596; PillarNeXt 0.625) → no-spconv costs only ~0.02–0.03 mAP. The rest is
unspent budget (heavier trunk + GT-paste + CBGS), closeable but GPU-weeks for ZERO thesis credit (perception =
instrumentation; CLAUDE.md platform-first). **Framing (honest):** (1) NDS-forward (0.573 reads competitive); (2)
narrow, truthful portability cost (~0.02–0.05, cite PillarNet — do NOT claim the whole gap is structural); (3) the
trailer +0.052-from-architecture vs +0.004-from-loss-weight capacity ablation. Report ASR vs the clean model's OWN
per-class recall. (GT-paste later doubles as an attack artifact — a poisoned GT-database IS a backdoor vector;
future T5/T6.)

## Handoff → CONTINUE centralized (owner decision), THEN Phase 3

**owner decision (2026-06-24):** do NOT lock the current best — keep pushing CENTRALIZED to a higher bar before Phase 3.
Current best (NOT locked) = `bb02d_r20/ema_ep15` (0.5656/0.5733), config `p1_bb02d.json`. **Target: confirm with the owner**
(≥0.50 on both mAP+NDS is already met centrally; the operative goal is higher — plausibly closer-to-SOTA mAP, or
headroom so the FL model stays ≥0.50 after dilution).

**How to start (the actual instruction): re-investigate before committing to anything.** Read the model + loss + data
code (`models/fusion/`, `training/`), re-run the per-class + train-vs-val diagnostics on the current best, and form an
independent view of where the remaining gap is and which lever is worth it. The items below are **SUGGESTIONS from this
session's gap analysis — NOT a plan, and possibly inaccurate given the long context. Verify each against the code/data
before pursuing.**

- (suggestion) heavier LiDAR trunk (deeper/wider / ASPP) — our ~2.5M trunk may be well below the dense-LiDAR ceiling.
- (suggestion) GT-database copy-paste — absent in the repo; a known rare-class lever; note it could double as an attack
  artifact (a poisoned GT-DB is a backdoor vector) — possible T5/T6 relevance.
- (suggestion) CBGS / class-balanced resampling + longer training (the `sampler=` hook exists; high compute).
- (suggestion) richer image-space aug (we have flip only).
- (caution from this session) watch the ep15-peak / overfit pattern — eval snapshots, not just the final; longer-training
  ALONE overfits; per-class-weighting ALONE ≈ 0; depth-supervision appeared to close ~nothing vs BEVFusion. Re-verify.

Once the centralized model hits the owner's bar, lock the reference → **Phase 3** = FL recipe (FedAdam / ≥30 rounds) +
clean bf16 FL baseline, matched-budget (D17). **FL caveat (verify):** enabling the LiDAR backbone shifts the trainable
layout — likely needs a `lidar_backbone` entry in TRAINABLE_MODULE_SLICE_MAP (`tasks.py`) for FedAvg (centralized was
unaffected). Then **Phase 4** = re-baseline bindings (readiness / ASR subset / provenance) + T5 go/no-go.
