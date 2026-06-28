# MCR Phase-3 — structural analysis of the CL + FL recipe (architecture & training reasonableness)

> A theory-grounded audit of **why** our settings are what they are and whether each is reasonable vs the
> published literature — instead of trial-and-error ("gambling"). Everything in Part 1 is **read from the code
> / configs** (not memory): CL recipe `fl_v3/configs/p1_bb02d.json` (→ centralized 0.566 mAP), FL recipe
> `fl_v3/configs/fl_bb02d_fedadam.json` (→ FedAvg 0.247), architecture `fl_v3/src/fl_v3/models/fusion/*.py`,
> loss `models/fusion/losses.py`, geometry `models/fusion/bev_grid.py`. Authored 2026-06-28.

## TL;DR
The centralized **0.566 mAP is itself ~0.12 below SOTA BEVFusion (~0.68)** for three *explainable* reasons —
**unsupervised LSS depth**, **pillars-not-sparse-voxels**, and a **short / no-CBGS schedule**. Two are fixable
within Rule #2 (depth supervision is free + pure-PyTorch; CBGS + longer schedule are config); one (pillars) is an
accepted no-spconv constraint. **The FL tail collapse and the cRT "representation-limited" result are downstream
of a CL base whose tail was never strong to begin with.** Highest-value additions, in order: (1) LiDAR-supervised
LSS depth, (2) CBGS done correctly (re-test the suspicious negative), (3) an IoU-aware confidence head. More
epochs is the cheap control under all three.

---

## Part 1 — The settings inventory

### A. Architecture (CL and FL share the IDENTICAL model — 33.2 M params)

| Component | Setting (from code) |
|---|---|
| Topology | BEVFusion-class: camera + LiDAR → BEV concat-fusion → SECOND-FPN → CenterPoint head |
| Camera backbone | Swin-T, ImageNet-pretrained, **trained** (D17), SDPA attention; images 256×704 |
| Camera neck | GeneralizedLSS-FPN → 128 ch, output stride 16 |
| View transform | DepthLSS (lift-splat), **59 depth bins, 1–60 m, depth softmax UNSUPERVISED**, context 80 ch |
| LiDAR encoder | PointPillars, **0.2 m pillars**, 120 k max pillars, 32 pts/pillar, 64-ch PFN, **10 sweeps** |
| LiDAR backbone | Dense 2D conv, 4 stages, out 128 (the no-spconv capacity lever) |
| Fusion | ConvFuser (channel-concat camera-BEV ⊕ LiDAR-BEV) → 128 ch |
| BEV neck | SECOND-FPN → 256 ch, nearest-upsample |
| Head | CenterPoint: shared Conv-GN-ReLU → 64 ch; 10-class heatmap (focal-init bias −2.19) + L1 reg |
| Norm / activation | **GroupNorm + ReLU everywhere** (D6, deliberately BatchNorm-free); Swin internal LayerNorm+GELU |
| BEV grid | range ±51.2 m, z −5..3; **0.2 m fine grid (512²) → 0.4 m head grid (256²)** |
| Decode | NMS-free (3×3 max-pool peak + stable-sort top-200); no rotated NMS (determinism) |

### B. Training recipe — CL vs FL

| Axis | CL (p1_bb02d → 0.566) | FL (fl_bb02d_fedadam → 0.247) |
|---|---|---|
| Initialization | ImageNet Swin + random rest | same; FL global inits from same |
| Objective | Gaussian-focal (α2, γ4) heatmap + L1 reg (reg-w 0.25, vel code-w 0.2) | identical |
| Class imbalance | focal **class-weights** [car 1.0 … trailer/CV 2.5, bicycle 2.0]; **no CBGS** | same weights; no CBGS |
| Optimizer | AdamW, wd 0.01 | AdamW, wd 0.01 (client) |
| LR | 3e-3, **OneCycle** (15% warmup) | client 1e-3, **cosine-over-rounds** (2-round warmup, 0.05 floor) |
| Backbone LR | ×0.1 | ×0.1 |
| Server optimizer | — | **FedAdam η0.01 (bug) → FedAvg η1**; server-EMA 0.9 |
| Schedule length | **15 epochs** | **15 rounds × E=1** (= 15 data-passes, matched) |
| Batch | 16/GPU × 4 DDP = **64 global** | **4/client × 25 = 100** (big-batch average) |
| Grad clip | 35 | 35 |
| EMA | **0.9997** | server-EMA 0.9 |
| Augmentation | BEV rot ±45°, scale 0.9–1.1, translate 0.5 m, flip, img-flip 0.5 | identical |
| Precision | bf16-AMP | bf16-AMP |

---

## Part 2 — Is each setting reasonable? (verdicts vs the literature)

Legend: ✓ solid · ⚠ questionable · ✗ likely wrong-or-artifact.

### Architecture
- **✓ GroupNorm, not BatchNorm.** Correct and important. BatchNorm running stats break under non-IID FL
  (FedBN, Hsieh 2020) and under small per-client batches. GroupNorm (Wu & He 2018) is the right call —
  defensible without reservation.
- **✗ Unsupervised LSS depth — the single biggest architectural ceiling.** Every strong camera/fusion BEV
  detector supervises the depth distribution with projected-LiDAR depth: **BEVDepth (Li 2022)** showed explicit
  depth supervision is *the* dominant camera-BEV lever (~+2–3 mAP, qualitatively correct geometry); **BEVFusion
  (Liu 2023)** carries it. Our depthnet predicts a 59-bin softmax with **no depth loss at all** → the camera→BEV
  splat is geometrically unconstrained, the camera-BEV is blurry, and fusion almost certainly leans on LiDAR
  while the camera adds little. That disproportionately starves the camera-dependent classes (pedestrian,
  traffic-cone, bicycle, barrier — visually distinctive, weak LiDAR returns). LiDAR is already in the batch;
  depth GT is free. **Highest-value missing component.** (`models/fusion/view_transform.py`,
  `models/fusion/losses.py` has no depth term.)
- **⚠ Pillars (2D) instead of sparse 3D voxels.** The Rule #2 (no-spconv) tax — a *deliberate* ceiling, not a
  mistake, but real. Sparse-voxel encoders (VoxelNet / CenterPoint-voxel, 0.075 m) keep z-resolution; pillars
  (Lang 2019) collapse z into one bin, costing orientation and tall/short discrimination (trailer, construction-
  vehicle, bus — our worst classes). The 0.2 m pillars + 4-stage dense backbone is a sound compromise within the
  constraint; just don't expect pillar features to match SOTA on the large/awkward tail.
- **⚠ 0.4 m head grid.** Reasonable (CenterPoint uses ~0.6–0.8 m effective), possibly even slightly fine. Not a
  primary suspect.
- **⚠ Missing IoU-aware confidence.** Directly relevant to our own finding: we diagnosed the tail failure as
  *confidence compression / TP≈FP* (`phase3_investigation_report.md` step 1). The standard fix in CenterPoint++ /
  BEVFusion is an **IoU-rectification head** that re-scores each box by predicted localization quality, decoupling
  "is it there" from "how confident." We don't have one. This targets the exact symptom we measured — more
  directly than cRT did.

### Training
- **✗ No CBGS — and our "CBGS hurts" finding contradicts universal nuScenes practice.** *Every* strong nuScenes
  detector (CenterPoint, BEVFusion, TransFusion) uses CBGS / class-balanced resampling (**Zhu 2019**); it is the
  standard answer to the long tail and reliably *helps* rare classes. We dropped it because an ablation
  (`p1_cbgs.json`) scored below bb02d — a negative so against the grain it should be treated as a **likely
  config-interaction artifact**. Two plausible, fixable culprits: (1) CBGS expands the epoch ~1.5×, but the
  **OneCycle schedule was sized for the un-expanded length** → LR mis-annealed; (2) CBGS **stacked on top of the
  focal class-weights** = double-balancing → over-tilts to the tail and drops head classes (net mAP down) —
  exactly the tail↔head reshuffle cRT showed. **Re-test cleanly: CBGS *instead of* class-weights, schedule
  resized to the expanded data**, before accepting "CBGS doesn't work here."
- **⚠ 15 epochs is short for nuScenes.** CenterPoint/BEVFusion train **20 epochs with CBGS** (≈ 20 × ~1.5 ≈ ~30
  effective passes). We do 15 passes, no CBGS ≈ **half the standard tail exposure**. The gradient teardown already
  showed the cosine annealed LR→0 before convergence — so even the CL model is plausibly tail-under-trained, and
  the FL tail collapse sits on top of an already tail-weak CL base.
- **⚠ focal class-weights instead of resampling.** Non-standard. The teardown found +0.004 from a 1.68× trailer
  weight — near-useless, because it reweights *recognition* loss on the same scarce positives rather than
  increasing *exposure*. Weaker than CBGS by construction.
- **✓ AdamW + OneCycle + EMA 0.9997 + grad-clip 35 + backbone ×0.1.** Standard and reasonable on the CL side;
  no concerns.
- **⚠ FL: E=1, R=15, big-batch averaging.** Beyond the (fixed) FedAdam bug, E=1 full-participation FedAvg ≈ one
  100-sample big-batch step/round; a 15-round cosine that hits ~0 early is under-converged by construction
  (teardown). The FL client LR (1e-3) is also 3× below the CL peak (3e-3) — reasonable for stability, but it
  compounds the under-training.

### Data imbalance (the through-line)
nuScenes is severely long-tailed (car ~44% of instances → bicycle/CV/motorcycle ~1%). The field's standard
treatment is **CBGS + depth-supervised fusion + 20-epoch schedules**. We use **none of the three**: no CBGS
(ablated out, suspiciously), no depth supervision (architectural), 15-epoch schedule. The tail is under-served at
*every* stage — architecture, schedule, and balancing — and the FL non-IID partition is the final straw, not the
root cause.

---

## Part 3 — What this reframes

The honest structural read: **the CL 0.566 is itself ~0.12 below SOTA BEVFusion (~0.68)** for three explainable
reasons — unsupervised depth, pillars-not-voxels, and a short/no-CBGS schedule. Two are fixable within Rule #2
(depth supervision is pure-PyTorch and free; CBGS + longer schedule are config), one is an accepted constraint
(pillars). **The FL tail collapse — and even the cRT "representation-limited" result — are downstream of a CL
base whose tail was never strong to begin with.** cRT froze tail-weak features and, unsurprisingly, couldn't
conjure tail signal the CL recipe never built in.

That is the structural case for validating the **CL recipe is actually maximized before any more FL-side levers**,
because a tail-healthier CL model is the shared foundation for *both* the FL reference and the attack/defense
benchmark. Components to add/restore, in value order:

1. **LiDAR-supervised LSS depth** — biggest lever, free (depth GT = projected LiDAR), pure-PyTorch, in-tree.
2. **CBGS done correctly** — instead of focal class-weights, with the schedule resized; re-test the suspicious
   negative controlling for the double-balance + schedule confounds.
3. **IoU-aware confidence head** — targets the exact TP≈FP confidence-compression symptom we measured.
4. **More epochs** (e.g. 20) — the cheap baseline control underneath all three.

These are CL-side recipe/architecture changes (code, not just config) and would each need their own validation;
none are launched here — this document is analysis only.

---

## References (key)
BEVFusion — Liu et al. 2023 (2205.13542) · BEVDepth — Li et al. 2022 (2206.10092) · CenterPoint — Yin et al. 2021
(2006.11275) · PointPillars — Lang et al. 2019 (1812.05784) · CBGS (class-balanced grouping & sampling) — Zhu et
al. 2019 (1908.09492) · GroupNorm — Wu & He 2018 (1803.08494) · FedBN — Li et al. 2021 (2102.07623) · Non-IID
BatchNorm — Hsieh et al. 2020. Internal: `phase3_investigation_report.md`, `phase3_gradient_teardown.md`,
`phase3_crt_probe_result.md`, `phase3_fl_baseline_result.md`.
