# MCR Phase-1 — Capability summary (canonical; consolidates all phase-0/1/2 docs)

> The single, AUTHORITATIVE, human-readable record for the MCR capability push (D17) — read this first.
> Centralized BEVFusion-class detector on nuScenes, pure-PyTorch (no spconv, Rule #2), bf16-AMP (D16).
> **★ LOCKED REFERENCE: bb02d = 0.5656 mAP / 0.5733 NDS** (val, official 10-class; ckpt `…/p2_ddp/bb02d_r20/ema_ep15`,
> config `configs/p1_bb02d.json`). Locked 2026-06-25 after a second session re-investigated independently and three
> further levers (CBGS, head-capacity, GT-paste) all came in BELOW it — 0.60 is confirmed out of reach in-tree.
> See **§ SESSION 2 (2026-06-25)** at the bottom for the negative ablations + the Phase-3 handoff. Replaces the
> per-step docs (profile/optcompare/gap/speedup/push) — see git history. (Memory `project_mcr_progress.md` mirrors
> this; THIS file is the source of truth.)

## TL;DR

Started from a frozen-Swin 0.36-mAP baseline; reached **0.5656 mAP / 0.5733 NDS** (LOCKED 2026-06-25),
READY, car_recall 0.98, ~28.7k ASR-eligible objects — a credible, balanced, attack-grade centralized model. The gap
to BEVFusion-base (0.679 mAP) is ~0.11, of which only **~0.02–0.03 is structural** (the no-spconv constraint).

**RESOLUTION (2026-06-25): bb02d 0.5656 is LOCKED.** The owner's 2026-06-24 "keep pushing" decision was honoured by a
second session that re-investigated independently and ran three more capability levers (target ~0.60) — **CBGS,
head-capacity, and GT-paste, ALL came in below bb02d** (§ SESSION 2). 0.60 is out of reach with in-tree levers
(the strong-LiDAR stack is past diminishing returns); the gap-analysis "perception is instrumentation, bank it"
call is vindicated. **Phase 3 (FL) is PAUSED pending owner planning of the FL recipe.**

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

---

## § SESSION 2 (2026-06-25) — independent re-investigation, three negative ablations, bb02d LOCKED

A fresh session (worktree `hungry-hofstadter-be5c1e`) re-investigated independently per the owner's "keep pushing to
~0.60" directive. It re-derived the diagnosis (per-class AP + train-split class frequencies + a free decode pre-flight)
and ran **three more capability levers**, each a clean A/B vs bb02d at global-16 / 15ep / snapshots@10,12,14 / decode
0.01-500. **All three came in below bb02d 0.5656 — 0.60 is out of reach in-tree.**

| lever (config) | best mAP | best NDS | vs bb02d | per-class verdict |
|---|---:|---:|---|---|
| **bb02d (LOCKED)** | **0.5656** | **0.5733** | — | the 4-stage-backbone reference |
| CBGS (`p1_cbgs.json`) | 0.5610 | 0.5668 | −0.005 | bicycle +0.065; trailer/CV overfit faster (peak slid ep15→ep10) |
| head-capacity 64→128 + depth1→2 (`p1_bb02h.json`) | 0.5559 | 0.5738 | −0.010 | bicycle +0.054; **large vehicles regress** (bus −0.060) — deeper class-agnostic head overfits the few instances |
| GT-paste (`p1_gtpaste.json`) | 0.5369 | 0.5078 | −0.029 | bus +0.020/trailer +0.005 but **moto −0.120 / bicycle −0.098 collapse** + **NDS −0.065** (dt=0 paste damages velocity) |

**The common cause:** bb02d already carries the strong-LiDAR stack (10-sweep + 4-stage 0.2 m dense backbone + aggressive
BEV aug + aggressive heatmap class-weights), so each further rare-class-exposure / capacity lever is **past diminishing
returns** and mostly trades one class for another (the same ceiling multi-sweep hit at +0.009). Re-confirmed dead-ends:
heatmap-class-weighting is maxed; longer-training-alone overfits; the reg-L1 was never class-weighted (a real finding,
but routing weights there + head capacity HURT large vehicles). **The remaining ~0.11 to BEVFusion is the
strong-LiDAR-fusion ceiling, not a missing lever.**

**GT-paste is fully implemented + tested and RETAINED — its durable value is the T5/T6 attack primitive.** New
`data/nuscenes/gt_database.py` + `gt_paste.py` (per-object yaw jitter + SAT collision + all-ragged-field extend,
seeded-numpy, default-OFF byte-identical), `scripts/build_gt_database.py` (+ SLURM launcher; built an 8k/class 10-sweep
DB), config `p1_gtpaste.json`, `tests/test_gt_paste.py` 9/9. The clean DB drives nothing now (capability no-go), but a
**poisoned GT-DB is the camera+LiDAR backdoor vector** the camera-only attack (D14, ~0 ASR) lacked — a supply-chain /
trigger-injection / label-flip surface for T5/T6. **Extending the threat model to a poisoned-DB vector is an
ORCHESTRATOR decision (a D2-amendment)** — this session built only the CLEAN capability version and flagged the rest.

**Determinism/precision unchanged** (bf16 relaxed; all new knobs — `det-cbgs*`, `det-reg-class-weights`,
`det-head-conv-layers`, `det-gt-paste*` — default-OFF ⇒ baseline byte-identical; data-side ⇒ AST-irrelevant; the PFN
content-sort absorbs pasted points). Test suites green throughout.

**STATUS: Phase 3 PAUSED (owner will plan the FL recipe separately).** When resumed, Phase 3 = FedAdam / ≥30 rounds /
clean bf16 FL baseline, matched-budget (D17), reference = bb02d. **Prerequisite (verified open):** `bb02d` enables BOTH
the lidar-backbone (+39 trainable tensors, 4-stage) AND would need any head/CBGS knobs OFF — so the FL
`TRAINABLE_MODULE_SLICE_MAP` (`tasks.py`) must be made config-conditional (it is asserted only in a TEST today, and the
centralized path doesn't call it, which is why this session deferred it). FedAvg's `requires_grad`-based update vector
already works regardless; the slice-map is needed for the per-module accounting / contract assertions.

---

## § CENTRALIZED MODEL ARCHITECTURE + BEVFusion COMPARISON (2026-06-25 — Phase-3 handoff reference)

> The locked reference `bb02d` in full. **bb02d = the step-6 "dense 2D LiDAR backbone (3-stage)" model PLUS one
> added 4th down-stage in that backbone** (and its 4th FPN level). That single addition is what took us from
> 0.5359 → 0.5656: the 4th stride-2 stage (H/8 at the 0.2 m / 512² grid) restores the large-object receptive
> field the 3-stage trunk lost at fine resolution — **bus 0.324 → 0.526, trailer 0.169 → 0.222**, while keeping
> the small-object gains (ped 0.80, cone 0.72). Everything else (camera branch, fusion, neck, head, recipe) is
> identical to step 6. Visuals: `capability_arc.html`, `model_structure.svg`, `module_details.html`,
> `bevfusion_comparison.html` (this dir).

### 1. Full model structure (the data-flow)

Two BEV branches on one shared 0.2 m grid (512²), fused, then detected:

```
6 cameras [6,3,900,1600] ─ preprocess(→256×704,norm) ─ Swin-T(trained,SDPA) ─ GeneralizedLSSFPN(→128, stride16)
                                                                                        │
                                                                 DepthLSSTransform (unsupervised depth, D=59 bins)
                                                                                        ↓  camera-BEV [B,80,512,512]
                                                                                   ConvFuser ──► SECOND-FPN ──► CenterPoint head ──► decode
                                                                                        ↑  lidar-BEV [B,128,512,512]   (256²@0.4m)        (NMS-free, top-500, score≥0.01)
LIDAR_TOP 10-sweep [P,6] ─ PointPillarsEncoder(PFN Linear8→64 + max) ─ LidarBackbone2D(4-stage 2D conv + FPN)
```

- **Total 33.2 M params**, all trainable (D17 — the camera backbone is trained, not frozen). Swin-T is 83 % of
  the model (27.5 M); the entire LiDAR path is 3.73 M; fusion+neck+head are ~1.4 M.
- **Geometry:** point-cloud range ±51.2 m (x,y) × [−5, 3] m (z); **0.2 m voxel → 512² fine grid** (camera-BEV and
  LiDAR-BEV both live here so the fuser can channel-concat); **head grid 256² at 0.4 m** (out_size_factor 2);
  10-sweep LiDAR (ego-motion-compensated + per-point dt), ≤120 k pillars.

### 2. Per-module detail

| # | module | params | in → out | role |
|---|---|---:|---|---|
| 1 | ImagePreprocessor | 0 | 6×[3,900,1600] → 6×[3,256,704] | resize + ImageNet normalize (no learnable params) |
| 2 | **CameraBackbone (Swin-T)** | **27.52 M** | [6,3,256,704] → 4 maps s{4,8,16,32} c{96,192,384,768} | trained Swin-Tiny, SDPA windowed attention (the capacity + the cost) |
| 3 | GeneralizedLSSFPN (camera neck) | 0.33 M | 4 maps → [6,128,16,44] | 1×1 laterals→128 + GN/ReLU, top-down add, one 3×3 smooth; emits the stride-16 level |
| 4 | DepthLSSTransform (view transform) | 0.17 M | [6,128,16,44] → camera-BEV [80,512,512] | depthnet → 59 depth-bin softmax (**UNSUPERVISED**) ⊗ 80-ch context, lift + splat to BEV |
| 5 | PointPillarsEncoder (PFN) | **640** | 10-sweep [P,6] → lidar-BEV [64,512,512] | single Linear(8→64)+GN+ReLU per point, max over ≤32 pts/pillar, index_copy scatter (perm-invariant, no spconv) |
| 6 | **LidarBackbone2D (4-stage)** | **3.73 M** | [64,512,512] → [128,512,512] | s1 64→128@512² · s2 128→256@256² · s3 256→256@128² · **s4 256→256@64²** + 4-level nearest-FPN; pure Conv2d/GN/ReLU. **s4 = bb02d's addition** |
| 7 | ConvFuser | 0.39 M | concat(80+128=208) → [128,512,512] | 2× (Conv2d 3×3 + GN + ReLU); plain channel-concat fusion, no gating |
| 8 | SecondFPNNeck (BEV neck) | 0.89 M | [128,512,512] → [256,256,256] | 2 stride-2 encoders + 2 decoders, concat → the head grid (256²/0.4 m) |
| 9 | CenterPointHead | 0.15 M | [256,256,256] → heatmap[10]+reg | shared 3×3 conv→64, then per-class heatmap (bias −2.19) + class-agnostic reg(2)/height(1)/dim(3)/rot(2)/vel(2) |
| — | decode | — | head dict → boxes | 3×3 maxpool local-max + stable-sort top-500, score≥0.01, **NMS-free** (deterministic) |

The LiDAR path is the deliberate Rule-#2 design: a 640-param PFN (no cluster-mean, for permutation-invariance) +
a 3.73 M **dense 2D** conv trunk — NOT a sparse 3D voxel net.

### 3. BEVFusion comparison — why we sit ~0.12 below it (BEVFusion is our REFERENCE, not the SOTA)

BEVFusion-base (Liu et al. 2022) ≈ **0.685 mAP / 0.714 NDS** (val). We are **0.5656 / 0.5733**. The branches are
deliberately the SAME where it's free (camera Swin-T + GeneralizedLSSFPN + unsupervised LSS + ConvFuser + SECOND-FPN
+ CenterPoint head); the gap is concentrated in the **LiDAR encoder** and what it enables:

| component | bb02d (ours) | BEVFusion-base | gap driver? |
|---|---|---|---|
| Camera backbone / neck / LSS | Swin-T / LSS-FPN / unsupervised LSS | same | no (identical) |
| Fusion / BEV neck / head | ConvFuser / SECOND-FPN / CenterPoint | same family | minor |
| **LiDAR voxel** | **0.2 m, dense 2D** | **0.075 m, sparse 3D** | **YES — the big one** |
| **LiDAR encoder** | PFN + 2D conv (3.73 M, z collapsed early) | VoxelNet + SECOND **sparse-3D-conv** | **YES** |
| Training recipe | 15–20 ep, no CBGS/GT-paste | 20 ep + CBGS + GT-paste | yes for them, **not transferable to us** |

**Honest gap decomposition (~0.12 mAP):**
1. **Voxel resolution (0.2 m vs 0.075 m) ≈ 0.04–0.07 — the single biggest piece, and it is COUPLED to Rule #2.**
   Finer voxels resolve small objects + box boundaries. We *cannot* cheaply go to 0.075 m: a **dense** 2D grid at
   0.075 m is 1365² (≈ 7× our 512² cells) → VRAM/compute blow-up. BEVFusion's **sparse** 3D conv only touches
   occupied voxels, so 0.075 m is affordable *for them*. The resolution gap exists because we forgo spconv.
2. **3D-sparse vs 2D-dense encoding ≈ 0.02–0.03 (the pure structural cost).** BEVFusion convolves the z-axis
   (height structure); our PFN collapses z into pillars then uses 2D conv. At *matched* resolution this is small —
   dense-2D PillarNet (0.599) ≈ sparse CenterPoint-SECOND (0.596) — so the paradigm itself isn't the main cost.
3. **Recipe (CBGS + GT-paste + longer schedule) ≈ 0.02–0.04 for BEVFusion, but NOT closeable for us.** We *ran*
   CBGS, head-capacity, and GT-paste this session (§ SESSION 2) — all net-negative, because our strong-LiDAR stack
   (10-sweep + 4-stage 0.2 m backbone + aggressive aug) is already past the point where those levers help.

**Framing for the writeup:** the residual to BEVFusion is dominated by **voxel resolution coupled to the no-spconv
constraint (Rule #2)** — a *deliberate, documented portability cost* (pure-PyTorch, ARM/H200-portable) — not a
training failure. The pure structural-encoding penalty is only ~0.02–0.03; bb02d 0.5656/0.5733 is a credible,
balanced, attack-grade reference within that envelope.
