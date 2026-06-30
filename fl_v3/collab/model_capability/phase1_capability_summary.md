# MCR Phase-1 — Capability summary (canonical; consolidates all phase-0/1/2 + the three later CL levers)

> The single, AUTHORITATIVE, human-readable record for the MCR centralized (CL) capability push (D17) — read this first.
> Centralized BEVFusion-class detector on nuScenes, pure-PyTorch (no spconv by default, Rule #2), bf16-AMP (D16).
> **★ STRONGEST CL REFERENCE SO FAR: bb02d = 0.5656 mAP / 0.5733 NDS** (val, official 10-class; ckpt `…/p2_ddp/bb02d_r20/ema_ep15`,
> config `configs/p1_bb02d.json`). Locked 2026-06-25 as the working reference after a second session re-investigated
> independently and three further levers (CBGS, head-capacity, GT-paste) came in below it; **three more levers were run
> later (depth-supervision, sparse-3D-voxel, + the structural-audit framing) and likewise did not improve on it under the
> recipes tried.** 0.5656 is the **current best on Alvis x86, not a final number** — on Arrhenius (GH200) the plan is to
> keep pushing the CL model, refine the spconv-based LiDAR branch, and speed it up (see § HANDOFF). See **§ SESSION 2
> (2026-06-25)** and **§ LATER CL CAPABILITY LEVERS (2026-06-28→30)** below for the negative ablations + the centralized
> handoff. Replaces the per-step docs (profile/optcompare/gap/speedup/push) — see git history. (Memory
> `project_mcr_progress.md` mirrors this; THIS file is the source of truth, and now also folds in the former
> structural-audit / depth-supervision / sparse-voxel per-lever docs — consolidated into the LATER-CL-LEVERS section here.)

## TL;DR

Started from a frozen-Swin 0.36-mAP baseline; reached **0.5656 mAP / 0.5733 NDS** (locked as the working reference
2026-06-25), READY, car_recall 0.98, ~28.7k ASR-eligible objects — a credible, balanced, attack-grade centralized model.
The gap to BEVFusion-base (≈0.68 mAP) is ~0.11; under the matched short-schedule recipe we tried, the pure structural
(no-spconv) share looks like only **~0.02–0.03**, with the rest sitting in voxel-resolution + the longer/CBGS recipe we
did not run. None of the levers tried this cycle moved past 0.5656 **under the recipes used** — every one of them is
recorded below as "did not help under the tried recipe", not as a closed door. On Arrhenius (GH200) the CL push is meant
to continue (finer voxels, a refined spconv LiDAR branch, longer/CBGS schedules, more compute).

**WORKING RESOLUTION (2026-06-25, extended 2026-06-30): bb02d 0.5656 is the locked working reference.** The owner's
2026-06-24 "keep pushing" decision was honoured across two further sessions: SESSION 2 ran CBGS, head-capacity, and
GT-paste (target ~0.60); the LATER CL LEVERS session ran the BEVDepth-style depth-supervision lever and the sparse-3D-voxel
(spconv) LiDAR encoder, framed by a literature structural audit. All of these came in below bb02d **under the matched ~15-epoch recipe**
(one of them — the voxel run — under an acknowledged batch-size confound). The reading we can defend is empirical:
**these levers did not improve on 0.5656 under the recipes tried on Alvis x86**, and the strong-LiDAR stack is past
diminishing returns *for those levers at that schedule* — not that ~0.60 is unreachable. This CL ledger is the shared
foundation for both the downstream FL reference and the attack/defense benchmark; the FL work itself is tracked
separately in `fl_baseline/phase3_fl_baseline.md`.

## The lever-by-lever progression (the method + the mindset)

bf16-AMP, global-16 DDP, same pooled trainval split, decode 0.01/500 (from step 2). Mindset: diagnose the
binding constraint empirically (per-class AP + TP errors + train-vs-val curve), pull the cheapest lever that
targets it, measure, re-diagnose.

| # | mAP | NDS | lever | implementation | diagnosis it addressed |
|---|---:|---:|---|---|---|
| 0 | 0.36 | — | frozen-Swin baseline | D1 frozen backbone, single-sweep, minimal recipe | — |
| 1 | 0.4042 | — | train camera backbone | unfreeze Swin-T (amends D1); 2 LR groups (bb@0.1×); grad-clip 35 | frozen backbone = headline undercapacity |
| 2 | 0.4357 | 0.413 | schedule+EMA+AdamW | OneCycle warmup+cosine; EMA 0.999; AdamW wd0.01; 15ep | bare recipe; gains hit rare classes (not overfit) |
| 3 | 0.4451 | 0.475 | multi-sweep 1→10 | ego-comp accumulation + dt channel (info_cache+loader+PFN 7→8d) | small mAP, **vel_err −46% → big NDS** (motion) |
| 4 | 0.4948 | 0.538 | BEV aug | GlobalRotScaleTrans+flip, consistent pts/boxes/vel/lidar2img | 40ep diagnostic → OVERFITTING; zero aug was the cap |
| 5 | 0.4994 | 0.540 | img-flip + per-class wt | h-flip (lidar2img row-update) + mean-1 heatmap weight | recipe route saturating (diminishing) |
| 6 | 0.5359 | 0.533 | dense 2D LiDAR backbone | SECOND/PillarNet-2D ~2.5M trunk (3-stage+FPN) pre-fusion + 0.2m voxel | 640-param PFN had ZERO pre-fusion RF (capacity) — trailer +0.052 vs +0.004 loss-wt |
| 7 | **0.5656** | 0.573 | 4-stage backbone @0.2m | +4th stride-2 stage (H/8) + FPN level | 0.2m regressed large objects; 4th stage RF → bus 0.32→0.53, keep ped 0.80 |

**Levers tried that did not improve on 0.5656 (the mindset record — each left open to revisit on Arrhenius/a different
recipe):** longer-training-alone overfits at this schedule (40ep<15ep pre-aug); per-class-weight-alone ≈0 here
(→ capacity, not loss-balance, was the binding constraint); global-64 DDP under-converges at this length (→ global-16
faster+better); channels_last neutral; CBGS / head-capacity / GT-paste / depth-supervision / sparse-3D-voxel all came in
below bb02d under the matched ~15-epoch recipe (see §§ SESSION 2, LATER CL CAPABILITY LEVERS). None of these is treated as
a closed door — they are negatives *under the recipe tried*.

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
all-reduce ~free; global-64 loader-bound + under-converges at this length). Multi-sweep loader: pin DataLoader workers to
1 BLAS thread (OMP/OPENBLAS/MKL=1 + torch.set_num_threads(1)) — kills the 256-thread oversubscription (6→4.7 min/ep). The
profiler is `scripts/p1_profile_a100.py`; the VRAM probe (backbone × voxel) is `scripts/p1_vram_probe.py`
(0.2m+4stage+ckpt fits 40GB). Per-epoch: 0.2m+4stage ≈ ~10 min/epoch global-16.

## Determinism / precision (Phase 0)

ONE `precision={bf16,fp32}` knob (D16) replaced numeric-mode×determinism-level. bf16=science default;
fp32+strict=offline dev-regression tool (the static-AST ban over models/fusion/** + byte-identity). The LiDAR
backbone is AST-clean (Conv2d/GN/ReLU/interpolate) and downstream of the perm-invariant scatter ⇒ no new
determinism obligation. All capability knobs default-OFF ⇒ byte-identical baseline; 247-test suite + the new
backbone/aug tests green. (The sparse-voxel encoder of the LATER-CL-LEVERS § *voxel* is the one path that pulls in spconv;
it is gated off by default so the shipped path stays AST-clean and spconv-free.)

## SOTA-gap conclusion + USENIX framing (the bank rationale)

The ~0.11 gap to BEVFusion does NOT look mostly structural under the recipe we ran: dense 2D pillars match/beat
sparse-conv at matched resolution (PillarNet 0.599 > CenterPoint-SECOND 0.596; PillarNeXt 0.625) → no-spconv looks like
only ~0.02–0.03 mAP, and the *voxel* spconv run (LATER CL LEVERS) gave no z-class advantage in a confounded A/B (so the
structural penalty estimate held up empirically there too). The rest is unspent recipe budget (finer voxels + heavier trunk +
GT-paste + CBGS + longer/depth-supervised schedules), closeable but GPU-weeks for low thesis credit on Alvis (perception
= instrumentation; CLAUDE.md platform-first) — and explicitly on the Arrhenius (GH200) to-do list, not abandoned.
**Framing (honest):** (1) NDS-forward (0.573 reads competitive); (2) narrow, truthful portability cost (~0.02–0.05, cite
PillarNet — do NOT claim the whole gap is structural); (3) the trailer +0.052-from-architecture vs +0.004-from-loss-weight
capacity ablation; (4) we relaxed Rule #2 and built the SOTA spconv encoder, and it did not beat the pure-PyTorch dense-2D
design under the run we did — so the portable design is not obviously leaving accuracy on the table. Report ASR vs the
clean model's OWN per-class recall. (GT-paste later doubles as an attack artifact — a poisoned GT-database IS a backdoor
vector; future T5/T6.)

## Handoff → CONTINUE centralized on Arrhenius

**owner decision (2026-06-24, still standing):** do NOT treat the current best as a final number — keep pushing CENTRALIZED
toward a higher bar where compute allows. Current best (locked as the working reference) = `bb02d_r20/ema_ep15`
(0.5656/0.5733), config `p1_bb02d.json`. **Arrhenius (GH200) CL to-do:** keep pushing the CL model, **refine the
spconv-based LiDAR branch** (the *voxel* encoder is fix-validated; re-run it matched-recipe at finer voxels rather than
the confounded global-32), and speed it up so longer/CBGS/depth-supervised schedules become affordable. 0.5656 is the
strongest CL reference we have on Alvis x86 *so far*.

**How to resume the CL push (the actual instruction): re-investigate before committing to anything.** Read the model +
loss + data code (`models/fusion/`, `training/`), re-run the per-class + train-vs-val diagnostics on the current best, and
form an independent view of where the remaining gap is and which lever is worth it. The items below are **levers already
tried this cycle (results below) plus suggestions — verify each against the code/data + the new compute envelope before
pursuing.**

- (tried, § *voxel*) **sparse-3D-voxel (spconv) LiDAR encoder** — fix-validated (0.2589→0.5046) but landed below pillar
  0.5656 in a global-32-vs-global-16 confounded run, no clear z-class advantage; **revisit matched-recipe + finer voxels
  on Arrhenius.**
- (tried, § *depth*) **BEVDepth-style LiDAR-supervised LSS depth** — net-negative at both w=1.0 and w=0.2 for this
  LiDAR-dominant fusion; refined the audit (unsupervised depth is a *feature* here, not a ceiling). A *soft/uncertainty-
  aware* depth loss + longer re-adaptation is the open variant if ever revisited.
- (tried, § SESSION 2) heavier LiDAR trunk / GT-paste / CBGS / head-capacity — all below 0.5656 at this schedule;
  CBGS especially is worth a clean re-test (CBGS *instead of* class-weights + schedule resized — see the structural audit).
- (suggestion) richer image-space aug (we have flip only).
- (caution from these sessions) watch the ep15-peak / overfit pattern — eval snapshots, not just the final; longer-training
  ALONE overfit at 15→40ep; per-class-weighting ALONE ≈ 0; depth-supervision hurt this fusion; the voxel A/B was
  confounded. Re-verify under the new compute envelope.

Once the CL model is at the owner's bar, the FL campaign continues in `fl_baseline/phase3_fl_baseline.md` (FL recipe +
clean bf16 FL baseline, matched-budget, D17). **CL-side caveat to flag for that handoff (verify):** enabling the LiDAR
backbone shifts the trainable layout — likely needs a `lidar_backbone` entry in TRAINABLE_MODULE_SLICE_MAP (`tasks.py`)
for the FL update path (the centralized path was unaffected).

---

## § SESSION 2 (2026-06-25) — independent re-investigation, three negative ablations, bb02d locked as working reference

A fresh session (worktree `hungry-hofstadter-be5c1e`) re-investigated independently per the owner's "keep pushing to
~0.60" directive. It re-derived the diagnosis (per-class AP + train-split class frequencies + a free decode pre-flight)
and ran **three more capability levers**, each a clean A/B vs bb02d at global-16 / 15ep / snapshots@10,12,14 / decode
0.01-500. **All three came in below bb02d 0.5656 under this recipe** — read as "did not help at this schedule", not as
"~0.60 is unreachable".

| lever (config) | best mAP | best NDS | vs bb02d | per-class verdict |
|---|---:|---:|---|---|
| **bb02d (working ref)** | **0.5656** | **0.5733** | — | the 4-stage-backbone reference |
| CBGS (`p1_cbgs.json`) | 0.5610 | 0.5668 | −0.005 | bicycle +0.065; trailer/CV overfit faster (peak slid ep15→ep10) |
| head-capacity 64→128 + depth1→2 (`p1_bb02h.json`) | 0.5559 | 0.5738 | −0.010 | bicycle +0.054; **large vehicles regress** (bus −0.060) — deeper class-agnostic head overfits the few instances |
| GT-paste (`p1_gtpaste.json`) | 0.5369 | 0.5078 | −0.029 | bus +0.020/trailer +0.005 but **moto −0.120 / bicycle −0.098 collapse** + **NDS −0.065** (dt=0 paste damages velocity) |

**The common cause (at this schedule):** bb02d already carries the strong-LiDAR stack (10-sweep + 4-stage 0.2 m dense
backbone + aggressive BEV aug + aggressive heatmap class-weights), so each further rare-class-exposure / capacity lever is
**past diminishing returns under the 15-epoch recipe** and mostly trades one class for another (the same plateau
multi-sweep hit at +0.009). Re-confirmed-at-this-schedule negatives: heatmap-class-weighting is maxed here;
longer-training-alone overfits at 15→40ep; the reg-L1 was never class-weighted (a real finding, but routing weights there +
head capacity HURT large vehicles). **The remaining ~0.11 to BEVFusion is, under this recipe, the strong-LiDAR-fusion
plateau rather than a single missing lever — but it is a current-recipe plateau, and the structural audit (§ *audit*
under LATER CL CAPABILITY LEVERS) argues a CBGS-correct + longer schedule has not actually been tried.**

**GT-paste is fully implemented + tested and RETAINED — its durable value is the T5/T6 attack primitive.** New
`data/nuscenes/gt_database.py` + `gt_paste.py` (per-object yaw jitter + SAT collision + all-ragged-field extend,
seeded-numpy, default-OFF byte-identical), `scripts/build_gt_database.py` (+ SLURM launcher; built an 8k/class 10-sweep
DB), config `p1_gtpaste.json`, `tests/test_gt_paste.py` 9/9. The clean DB drives nothing now (capability no-go at this
schedule), but a **poisoned GT-DB is the camera+LiDAR backdoor vector** the camera-only attack (D14, ~0 ASR) lacked — a
supply-chain / trigger-injection / label-flip surface for T5/T6. **Extending the threat model to a poisoned-DB vector is an
ORCHESTRATOR decision (a D2-amendment)** — this session built only the CLEAN capability version and flagged the rest.

**Determinism/precision unchanged** (bf16 relaxed; all new knobs — `det-cbgs*`, `det-reg-class-weights`,
`det-head-conv-layers`, `det-gt-paste*` — default-OFF ⇒ baseline byte-identical; data-side ⇒ AST-irrelevant; the PFN
content-sort absorbs pasted points). Test suites green throughout.

---

## § LATER CL CAPABILITY LEVERS (2026-06-28 → 2026-06-30) — structural audit + depth-supervision + sparse-voxel

A literature-grounded structural audit motivated two more CL levers; both were run as clean(-ish) A/Bs vs bb02d and
neither improved on it under the recipe tried. The audit itself is the framing; the two experiments (*depth*, *voxel*) are
the test. (The three former per-lever docs are consolidated here.)

### *audit*. Structural audit (vs the published literature) — why these levers were chosen

> The audit (consolidated here) is a theory-grounded read of **why** our CL recipe is what it is and whether each setting
> is reasonable vs the literature — read from the code/configs, not memory. CL recipe `configs/p1_bb02d.json`
> (→ 0.566 mAP); architecture `models/fusion/*.py`, loss `models/fusion/losses.py`, geometry `models/fusion/bev_grid.py`.

The audit's read: the CL 0.566 sits ~0.12 below SOTA BEVFusion (~0.68) for three *explainable* reasons under our recipe —
**unsupervised LSS depth**, **pillars-not-sparse-voxels**, and a **short / no-CBGS schedule**. Two looked fixable within
Rule #2 (depth supervision is free + pure-PyTorch; CBGS + longer schedule are config); one (pillars) is the accepted
no-spconv constraint. The structural significance for *this* doc: **our centralized model's own tail classes (trailer,
construction-vehicle, bus, bicycle) are weak relative to the literature**, and the audit explains why (architecture +
schedule + balancing all under-serve the tail) — which is the case for confirming the CL recipe is genuinely maximized
before relying on it downstream. (The separate FL tail-collapse / cRT investigation that originally prompted this audit is
tracked in `fl_baseline/phase3_fl_baseline.md`; its FL-specific numbers live there, not here.)

Per-setting verdicts vs the literature (✓ solid · ⚠ questionable · ✗ likely wrong-or-artifact under this recipe):

| setting | verdict | the literature note |
|---|---|---|
| GroupNorm not BatchNorm | ✓ | BN running stats break under non-IID FL (FedBN, Hsieh 2020) + small client batches; GN (Wu & He 2018) is correct |
| Unsupervised LSS depth | ✗ (audit's #1) | BEVDepth (Li 2022) makes explicit depth supervision *the* dominant camera-BEV lever (~+2–3 mAP); ours has **no depth loss** ⇒ camera→BEV splat geometrically unconstrained. **Tested in § *depth* — did not transfer to this LiDAR-dominant fusion.** |
| Pillars (2D) not sparse 3D voxels | ⚠ | the Rule #2 tax; pillars (Lang 2019) collapse z, costing tall/short discrimination (trailer/CV/bus). **Tested in § *voxel* — no z-payoff in the confounded run.** |
| 0.4 m head grid | ⚠ | reasonable (CenterPoint ~0.6–0.8 m effective); not a primary suspect |
| Missing IoU-aware confidence head | ⚠ | targets the exact TP≈FP confidence-compression symptom we measured (CenterPoint++/BEVFusion re-score by predicted IoU); **not yet tried** |
| No CBGS (and "CBGS hurts" finding) | ✗ | every strong nuScenes detector uses CBGS (Zhu 2019); our negative (`p1_cbgs.json` < bb02d) is against the grain ⇒ likely a config-interaction artifact (OneCycle sized for un-expanded length, and CBGS stacked *on top of* focal class-weights = double-balancing). **Re-test cleanly: CBGS instead of class-weights + schedule resized — flagged, not yet run.** |
| 15 epochs | ⚠ | CenterPoint/BEVFusion train 20ep + CBGS (~30 effective passes); we do ~15 no-CBGS ≈ half the tail exposure ⇒ even the CL model may be tail-under-trained |
| focal class-weights instead of resampling | ⚠ | reweights recognition loss on the same scarce positives rather than increasing exposure; weaker than CBGS by construction (the teardown saw +0.004 from a 1.68× trailer weight) |
| AdamW + OneCycle + EMA 0.9997 + clip 35 + bb×0.1 | ✓ | standard, no concerns |

**What the audit reframes:** two of the three "below-SOTA" reasons (depth, short/no-CBGS schedule) are recipe/architecture
choices that are *fixable in-tree*, and one (pillars) is the accepted constraint. The value-ordered to-do it produced —
(1) LiDAR-supervised LSS depth, (2) CBGS done correctly, (3) IoU-aware confidence head, (4) more epochs (the cheap control
under all three) — is the menu the later experiments and the Arrhenius handoff draw from. The document is analysis only;
nothing was launched from it directly. **References:** BEVFusion (Liu 2023, 2205.13542) · BEVDepth (Li 2022, 2206.10092)
· CenterPoint (Yin 2021, 2006.11275) · PointPillars (Lang 2019, 1812.05784) · CBGS (Zhu 2019, 1908.09492) · GroupNorm
(Wu & He 2018, 1803.08494) · FedBN (Li 2021, 2102.07623) · Non-IID BN (Hsieh 2020). Internal: the FL-side diagnostics
that originally motivated this audit (the investigation report, gradient teardown, cRT probe, and FL baseline result) are
consolidated in `fl_baseline/phase3_fl_baseline.md`.

### *depth*. BEVDepth-style depth-supervised LSS — net-negative for this LiDAR-dominant fusion (weight-independent)

> The audit's #1 lever (consolidated here): BEVDepth-style LiDAR depth supervision on the previously-UNsupervised LSS
> depthnet. Clean A/B vs bb02d (only depth-sup differs: global-64 batch, 15 epochs, EMA 0.9997, activation-checkpointing =
> value-identical memory fix). Config `p1_bb02d_depth.json` (`det-depth-supervision=true`). EMA ep15, full val,
> score 0.01 / maxobj 500.

**Result (w=1.0, job 6782278, 4×A100, bf16, 02:44): mAP 0.4003 vs bb02d 0.5656 (−0.165); NDS 0.3633 vs 0.5733 (−0.210).**

| class | bb02d | depth-sup w=1.0 | depth-sup w=0.2 | group |
|---|---:|---:|---:|---|
| car | 0.85 | 0.7619 | 0.76 | abundant |
| pedestrian | 0.80 | 0.7224 | 0.71 | camera |
| traffic_cone | 0.72 | 0.6687 | 0.67 | camera |
| barrier | 0.65 | 0.5665 | 0.56 | camera |
| motorcycle | 0.68 | 0.4212 | 0.40 | rare/vehicle |
| truck | 0.48 | 0.2913 | 0.30 | rare/vehicle |
| bicycle | 0.42 | 0.2560 | 0.28 | rare/vehicle |
| bus | 0.53 | 0.1570 | 0.17 | rare/vehicle |
| trailer | 0.22 | 0.0832 | 0.10 | rare/vehicle |
| construction_vehicle | 0.23 | 0.0751 | 0.10 | rare/vehicle |
| **mAP** | **0.5656** | **0.4003** | **0.4031** | |

- **First read was an over-weighting confound, then refuted.** At w=1.0 the depth-CE term was ~45% of the epoch-15 total
  loss (2.86) — nearly the whole detection loss — so detection trained on ~half the effective gradient. The per-class
  pattern (largest drops on the capacity-hungry rare/vehicle classes, smallest on car/ped/cone) matched
  detection-under-training from diverted capacity, and BEVDepth's per-pixel-normalized depth loss sits at only ~10–20% of
  its total vs our hard-bin CE at 45%. Training itself was healthy (loss monotone 7.16→2.86; depth CE dropped, so depth
  supervision DID learn; plumbing smoke PASS, fp16-safe).
- **w=0.2 re-test refuted the weight hypothesis (lever did not help, regardless of weight).** Jobs 6783049 (killed on a
  2×-slow node) → 6783051 (alvis3-13) → standalone eval 6783160 on the saved ema_ep15: **mAP 0.4031 / NDS 0.3618 —
  essentially identical to w=1.0**, per-class within ±0.02. A 5× weight change moved mAP by 0.003 ⇒ weight is not the knob.
- **Verified NOT a bug:** `data/nuscenes/augment.py` updates `lidar2img ← lidar2img·T⁻¹` (BEV aug pulls the camera
  projection back), so the depth GT (augmented points → original camera-frame depth) is aug-consistent; smoke confirmed
  73–79% projection coverage.
- **Mechanistic read (a real finding, left open):** the unsupervised LSS depthnet appears to have been learning
  detection-OPTIMAL feature placement, not metric depth. Forcing metric depth — even at a light 14%-of-loss weight — flips
  the camera-BEV to geometric and disrupts the LiDAR-dominant fusion that had adapted to the old camera-BEV; detection
  drops to ~0.40 and stays there independent of weight. This **refines the audit**: the "unsupervised depth is the #1
  ceiling" premise is from BEVDepth (camera-dominant); for our **strong-LiDAR fusion the unsupervised depth behaves as a
  feature here, and supervising it hurt under this recipe**. Depth-sup was not this model's lever under what we tried.
  **Open, lower-priority:** a *soft/uncertainty-aware* depth loss + longer re-adaptation might differ — not chased on the
  2 remaining Alvis days vs the clean 15-epoch matched negative, and a candidate for Arrhenius.

### *voxel*. Sparse-3D-voxel (spconv) LiDAR encoder — fix-validated, no z-payoff in a confounded run (paused, revisit matched)

> Consolidated here. **Owner decision 2026-06-30: pause this lever.** Pillar `bb02d`
> (0.5656 / 0.5733) stays the working reference; the spconv voxel encoder is fix-validated but showed no z-resolution
> payoff in the run we did — paused, not foreclosed (the clean matched-recipe re-run was offered and declined on compute,
> and is on the Arrhenius list).

**What was built.** `fl_v3/src/fl_v3/models/fusion/sparse_voxel_encoder.py` — a SECOND/CenterPoint-style sparse-3D-voxel
LiDAR encoder (spconv `SubMConv3d` keep-xy + `SparseConv3d` z-downsample → collapse-z → dense BEV). Drop-in for
`PointPillarsEncoder`, gated by `det-lidar-encoder=voxel` (default `pillar` ⇒ never built ⇒ spconv is **not** a runtime
dependency of the shipped path, so Rule #2's portability guarantee survives for the default build). Relaxes Rule #2
(spconv) **for this experiment only** — committed on worktree `stupefied-jennings-4b1992`, not v3. Motivation: the
structural audit (§ *audit*) flagged the pillar PFN's z-collapse as a possible binding constraint for tall/z-sensitive classes
(trailer/bus/CV/truck); the voxel encoder gives the LiDAR branch real z-resolution to test whether that constraint is binding.

**Two real bugs found and fixed (commit `4ac7392`; smoke `6786063` PASS).**
1. **EMA-deepcopy crash** (`353fadb`): storing the spconv *module* on `self` broke `AveragedModel` deepcopy
   (`cannot pickle 'module' object`). Fix: import `spconv.pytorch` locally in `forward`, never store it.
2. **GroupNorm-collapse + impoverished VFE** (the 0.26 bug, `4ac7392`): `_gn` with `groups==channels` (1 ch/group) on the
   no-spatial per-point/per-voxel tensors normalized each channel over its single value → 0, zeroing the LiDAR-BEV ⇒ the
   model ran **camera-only** (0.2589). Fix: `_gn` enforces ≥2 ch/group; mean-VFE replaced with a PFN-style per-point VFE
   (`[abs xyz, intensity, voxel-center-relative offsets] → Linear → GN → ReLU → masked max-pool`), matching the proven
   `PointPillarsEncoder`. Validated: LiDAR-BEV std 0→0.585, |diff| 0→0.128.

**Result (job 6786064, ep15 EMA, A100×4, bf16; spconv fp32 autocast-off).**

| | global batch | steps/ep | max_lr | mAP | NDS | car-recall |
|---|---|---|---|---|---|---|
| **pillar bb02d (working ref)** | 16 | 1758 | 0.003 | **0.5656** | **0.5733** | 0.970 |
| **voxel (this run)** | **32** | 879 | 0.003 | 0.5046 | 0.4764 | 0.970 |
| voxel (buggy, pre-fix) | 32 | 879 | 0.003 | 0.2589 | 0.288 | — |

Final train loss 0.917 (clean monotonic 3.41→0.917) — the fixed LiDAR branch genuinely contributes. Per-class AP
(voxel − pillar): car −0.056 · ped −0.071 · cone −0.031 · moto −0.081 · barrier −0.042 · **bus −0.156** · truck −0.058 ·
bicycle **+0.043** · **trailer −0.068** · CV −0.015.

**Why paused (verdict logic, conservative).**
- **Confounded A/B:** voxel ran at **global-32** (batch-8/gpu, chosen for speed after batch-16 OOM) vs the working ref's
  **global-16** — half the optimizer steps at the same peak LR. This project's own log records the effect: *"global-64
  under-converges → global-16 faster+better."* So 0.5046 is a **floor**, not a fair number.
- **No z-signal even so:** the drop is **uniform** — 9/10 classes down, including the z-*insensitive* ones
  (car/ped/cone/moto/barrier). That is the fingerprint of **under-convergence**, not an architecture effect; an
  architectural z-benefit would be *selective* (z-classes up). Instead trailer/bus/CV are all below pillar and bus is the
  single worst-hit class. No emerging z-payoff to justify paying for a matched-recipe re-run *on Alvis* — but a matched
  global-16 + finer-voxel re-run is exactly the Arrhenius follow-up.
- **Mechanism:** the pillar's z-collapse appears **already compensated** by the dense-2D-LiDAR backbone's receptive field
  (the 4-stage backbone is what lifted bus 0.32→0.53, trailer 0.169→0.222), so adding 3D z-resolution upstream may be
  redundant with it under this design.

**Thesis implication (positive framing of a negative-so-far result).** We **relaxed Rule #2** and built the SOTA
sparse-voxel (spconv) LiDAR encoder; under the run we did **it did not beat** the pure-PyTorch dense-2D-pillar design for
this camera+LiDAR fusion. The Rule-#2-compliant (portable, aarch64/H200-ready) default design therefore does **not** look
like it is leaving accuracy on the table — strengthening, not weakening, the platform's design choice — while the door
stays open to a matched-recipe + finer-voxel re-run on Arrhenius. **Caveat (for honest reporting):** the pause rests on a
**confounded** run (global-32 vs global-16); the claim we defend is the conservative one — *"sparse-voxel showed no
z-class advantage even when given an architectural z-resolution it lacked, under this confounded run"* — we do **not**
claim a precise matched delta, and a clean global-16 re-run was offered and declined on compute grounds.

---

## § CENTRALIZED MODEL ARCHITECTURE + BEVFusion COMPARISON (2026-06-25 — centralized handoff reference)

> The working reference `bb02d` in full. **bb02d = the step-6 "dense 2D LiDAR backbone (3-stage)" model PLUS one
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
a 3.73 M **dense 2D** conv trunk — NOT a sparse 3D voxel net. (The optional spconv voxel encoder of § *voxel* is a
gated, default-off experiment, not part of this shipped path.)

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
| Training recipe | 15–20 ep, no CBGS/GT-paste | 20 ep + CBGS + GT-paste | yes for them; not yet matched on our side |

**Honest gap decomposition (~0.12 mAP) under the recipe we ran:**
1. **Voxel resolution (0.2 m vs 0.075 m) ≈ 0.04–0.07 — the single biggest piece, and it is COUPLED to Rule #2.**
   Finer voxels resolve small objects + box boundaries. We did not cheaply go to 0.075 m on Alvis: a **dense** 2D grid at
   0.075 m is 1365² (≈ 7× our 512² cells) → VRAM/compute blow-up. BEVFusion's **sparse** 3D conv only touches
   occupied voxels, so 0.075 m is affordable *for them*. The resolution gap exists because we forgo spconv by default —
   and finer voxels (dense or via the gated spconv path) are on the Arrhenius compute list.
2. **3D-sparse vs 2D-dense encoding ≈ 0.02–0.03 (the pure structural cost) under this recipe.** BEVFusion convolves the
   z-axis (height structure); our PFN collapses z into pillars then uses 2D conv. At *matched* resolution this looks
   small — dense-2D PillarNet (0.599) ≈ sparse CenterPoint-SECOND (0.596) — so the paradigm itself isn't the main cost,
   and the *voxel* spconv run gave no z-class advantage in a confounded A/B, consistent with that ~0.02–0.03 estimate.
3. **Recipe (CBGS + GT-paste + longer/depth-supervised schedule) ≈ 0.02–0.04 for BEVFusion; not yet realized on our side.**
   We *ran* CBGS, head-capacity, GT-paste (§ SESSION 2) and depth-supervision (§ *depth*) — all net-negative under the
   matched 15-epoch recipe, because our strong-LiDAR stack is already past the point where those levers help *at this
   schedule*. The structural audit (§ *audit*) argues a **CBGS-correct + longer** schedule has not actually been tried,
   so this share is not foreclosed — it is unspent recipe budget for Arrhenius.

**Framing for the writeup:** under the recipe we ran, the residual to BEVFusion is dominated by **voxel resolution coupled
to the no-spconv default (Rule #2)** — a *deliberate, documented portability cost* (pure-PyTorch, ARM/H200-portable) —
rather than a training failure. The pure structural-encoding penalty looks like only ~0.02–0.03, and our own gated spconv
experiment did not overturn that; bb02d 0.5656/0.5733 is a credible, balanced, attack-grade reference within that
envelope, and the strongest CL number we have on Alvis x86 so far — explicitly a current best, with the CL push continuing
on Arrhenius (GH200).
