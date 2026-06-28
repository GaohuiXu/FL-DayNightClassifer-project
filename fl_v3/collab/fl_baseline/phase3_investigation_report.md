# MCR Phase-3 — Why FL can't train a comparable BEV detector: investigation report

> Investigation (no platform code changes) into the FedAvg-0.247-vs-centralized-0.5656 gap, per the owner's
> directive to understand the MECHANISM (gradients/modules), not sweep hyperparameters. Two evidence streams:
> a multi-agent literature+theory workflow (wf_69cb3c68) and a read-only gradient/module teardown of our own
> snapshots ([phase3_gradient_teardown.md]). They CONVERGE. This may be paper-worthy (see §5).

## 0. TL;DR

The under-training is **structural, not a hyperparameter**: under our geographic (by-log) non-IID, the shared
**feature extractor learns fine (recall is high everywhere) but the per-class CenterPoint heatmap HEAD becomes
mis-calibrated for the tail** — clients lacking a class supply only background supervision to that class's
channel, FedAvg averages mostly-suppressed channels → tail "detected but ~0 AP". FedAdam η=0.01 was a separate,
already-fixed under-stepping bug (dense detector gradients → adaptive step collapses to O(η)). The literature
says this head/classifier-drift is THE non-IID failure mode — but that **classifier-only fixes are bounded by
representation quality** (the feature space is also distorted), so a full fix is likely two-part.

## 1. Q1 — Does prior FL-BEV/3D-detection-on-nuScenes work exist? (VERDICT: essentially no credible baseline)

Adversarially confirmed. There is **no published, credible, reproducible federated camera+LiDAR BEV/3D detector
on nuScenes under the official 10-class mAP/NDS protocol with an honest FedAvg-vs-centralized table.**
- **AutoFed** (MobiCom 2023, arXiv:2302.08646) — the most credible peer-reviewed federated *multimodal AD*
  detector — is deliberately **camera-less (LiDAR+radar), self-collected data, single-class BEV AP.** Not a
  comparator; its design targets the same missing-class non-IID we diagnose.
- **FedM2former** (Processes/MDPI 2025, 71.2 mAP) and **BEV-FePNet/DP-DeceFL** (Neurocomputing 2024, ~71.6 mAP)
  nominally match (federated multimodal nuScenes) but report mAP **above centralized BEVFusion SOTA (~68–70)** —
  internally implausible, FL recipe undisclosed, paywalled. Red flags, not baselines.
- The "FedAvg ≈ centralized on nuScenes (56→60)" result is **2D camera detection** (arXiv:2509.01868), a
  small-model/mild-partition artifact — not evidence about a 33M-param 3D detector.
- **Honest comparators (different domain) reproduce our signature:** Fed3D (indoor RGB-D 3D det, arXiv:2604.15795)
  −14–15 mAP@0.25 + **−22 AR@0.5** under non-IID; FedPylot (2D YOLO, geographic non-IID, arXiv:2406.03611) gap
  widens with long-tail (10-class 47.8 → 23-class 28.3) and finds **large server momentum β=0.9 destabilizes
  non-IID detection**; FedDrive (AD segmentation) — the dominant lever is normalization-stat handling.

⇒ **Our honest FedAvg-0.247 / centralized-0.5656 with per-class retention is, as far as the literature shows,
the first credible such comparison.** Novelty is real (§5).

## 2. Q2/Q4 — WHY FL under-trains: ranked mechanistic hypotheses (our data + literature)

### H1 (STRONGEST) — Per-class heatmap HEAD drift + tail confidence-dilution
- **Mechanism.** Non-IID aligns the backbone but biases the final per-class head (CCVR NeurIPS'21; Partial-VR
  CVPR'23; FedImpro ICLR'24). A log with **zero trucks → only background gradient on the truck heatmap channel
  → local SGD suppresses it** (FedRS; FedVLS). Averaging 25 mostly-suppressed tail channels → globally
  under-confident channel that fires spatially (recall) but with calibrated-down peaks (~0 AP).
- **Our-data support (already measured):** backbone barely moves (rel Δ 0.0003); the per-class heatmap head
  updates UNIFORMLY across classes (so the tail head isn't "not training") yet tail AP≈0 with recall 0.42–0.84;
  per-client norm-CoV ~0.08 (magnitude-homogeneous → the conflict is in DIRECTION, not magnitude).
- **Decisive measurements (not yet run):** (a) **per-module cross-client cosine** — predict backbone/fusion
  positive, heatmap head ≈0/negative; (b) **per-class sign-agreement** on heatmap channels — predict tail ≈50%
  (random) = signal cancellation; (c) **cRT weight-norm probe** — tail head-channel norms collapse vs centralized.
- **Candidate fixes:** post-hoc per-class score calibration (**NorCal** NeurIPS'21 — the only detection-AP,
  sigmoid-head, post-hoc method; FedLC n^{-1/4} margin); server-side balanced **head retraining (cRT/CReFF)**;
  training-time **vacant-class KD (FedVLS)** anchoring missing-class channels to the global; local class-balanced
  focal weighting.
- **CRITICAL caveat (adversarial verification):** head-only fixes are **bounded by representation quality** —
  CCVR's own limitation + RUCR (IEEE TIFS'24) show the non-IID **feature space is also distorted**. So a head
  intervention quantifies how much is calibration vs representation; a full fix is likely two-part.

### H2 — Big-batch / schedule under-convergence (a real but secondary lever)
- E=1 full-participation FedAvg ≈ ONE large-batch step/round (the repo's own "global-64 under-converges"). Our
  **cosine LR annealed to ~0 by round 15 before convergence** (per-client update-norm 36→2.2 tracks the 20× LR
  decay; all modules decelerate 16–20×). So R=15 is under-trained *by construction*. ⇒ more rounds with a
  stretched schedule WILL keep improving the head classes — but won't fix the tail *direction* conflict (H1).

### H3 — FedAdam mis-allocation (explained + discarded as the primary cause)
- Reddi et al. (ICLR'21) validated FedAdam ONLY on classification/LM; its gains live in the **sparse-gradient**
  regime (rare-word embeddings: small v_t → η/τ boosts rare coords). A **dense** BEV/CenterPoint detector is the
  opposite — dense gradients → large v_t → per-coordinate step collapses to O(η); η=0.01 froze the global (our
  1-client proof). Our teardown also shows FedAdam's √v̂ normalization **over-moves the near-frozen backbone,
  under-moves the head** — exactly wrong for a detector. ⇒ for a dense detector, well-tuned **FedAvg / FedAvgM
  (small β~0.1)** is expected to match or beat FedAdam (matches 0.247 ≫ 0.057). If keeping adaptivity: η~1.0,
  τ 1e-2…1e-1, or **FedYogi** (resists v_t blow-up under heterogeneity).

## 3. Q3 — FedAdam vs FedAvg for 3D detection (the gradient-density story)
FedAdam's benefit is conditional on **sparse** gradients; our detector's gradients are **dense** ⇒ adaptivity
gives no rare-coordinate boost and its normalization mis-scales the multi-module step. The right experiment is
NOT "sweep FedAdam η" but **measure the gradient density (mean √v̂ vs τ) + per-module conflict**, which tells us
the operating regime directly. Expectation: FedAvg/FedAvgM is the correct server optimizer here; FedAdam is a
controlled ablation, not the baseline.

## 4. Recommended experiment ladder (cheapest → most decisive; ALL need owner OK — diagnostics, read-only/minimal)

1. **NorCal post-hoc re-scoring (FREE, no training, ~minutes).** Apply a per-class score offset ∝ global class
   frequency to the EXISTING round-15 model's predictions and re-eval. If tail AP jumps with the model
   untouched → quantifies the *pure-calibration* fraction of the gap. Cheapest possible, model-frozen.
2. **Per-client gradient analysis (read-only, ~1 FL round ≈ 12 min on 4×A100).** From a global snapshot, run
   each of the 25 clients' 1 local epoch (reusing `train_local`), capture the 25 update vectors, compute
   per-module pairwise cosine + per-class heatmap sign-agreement + per-module conflict ratio. Confirms H1's
   "head-drift + tail sign-cancellation" and delivers the owner's #2/#3 gradient analysis. (The platform's
   gsm telemetry computes the global-flattened cosine but NOT per-module/per-class — this analysis is richer.)
3. **cRT decoupling probe (the DECISIVE intervention; freeze backbone+neck, balanced-retrain ONLY the head, a
   few epochs, fast).** If tail AP recovers → head-fixable (calibration); if it stays low → feature space is
   distorted (H1 caveat) → a representation-level fix is needed. Either outcome is publishable and directs the fix.

## 4b. EXPERIMENT RESULTS

### Step 1 — per-class TP-vs-FP score separability (FREE, on saved round-15 predictions)
For each class, matched the saved `det_eval/results.json` predictions to GT (≤2 m, greedy by score) and measured
the within-class score-separability AUC = P(TP score > FP score):

| class | n_pred | recall@2m | TP score md | FP score md | sep-AUC |
|---|---:|---:|---:|---:|---:|
| car | 672k | 0.97 | 0.54 | 0.07 | 0.97 |
| pedestrian | 904k | 0.93 | 0.47 | 0.07 | 0.96 |
| truck | 255k | 0.86 | 0.20 | 0.07 | 0.86 |
| bus | 83k | 0.73 | 0.17 | 0.07 | 0.86 |
| trailer | 49k | 0.44 | 0.09 | 0.07 | 0.78 |
| construction_vehicle | 48k | 0.52 | 0.11 | 0.06 | 0.82 |
| motorcycle | 119k | 0.81 | 0.19 | 0.06 | 0.90 |
| bicycle | 151k | 0.78 | 0.22 | 0.06 | 0.89 |

**Findings (refine the hypothesis):** (a) sep-AUC is HIGH for every class (0.78–0.97) ⇒ NOT a rank-inversion;
**empirically confirms per-class score calibration (NorCal/temperature) is a no-op** for nuScenes AP
(rank-invariant). (b) The tail failure is **confidence COMPRESSION + over-prediction**: tail TP scores are low
and nearly equal to FP scores (trailer 0.09 vs 0.07; truck 0.20 vs 0.07) vs head TP≫FP (car 0.54 vs 0.07), and
the tail is over-predicted 28–83× ⇒ tail TPs are correct but buried, low-confidence, in a flood of near-equal
FPs ⇒ precision (hence AP) collapses despite high recall. (c) The decent sep-AUC means **the BEV features retain
tail signal (backbone OK)** ⇒ **head retraining on balanced data (cRT, step 3) is promising** and the earlier
"bounded by representation" caveat is likely mild here. ⇒ the fix must SHARPEN/raise tail confidence (cRT /
FedVLS vacant-class KD / local class-balanced focal), not re-score post-hoc.

### Step 2 — per-client gradient conflict (job 6778494, round_8 global, 25 clients × full local epoch)
Per-module cross-client cosine (mean | %neg): lidar_encoder 0.671|0%, fusion 0.287|0%, head 0.232|0%,
bev_neck 0.191|0%, lidar_backbone 0.142|0%, view_transform 0.199|0%, camera_neck 0.088|0%, camera_backbone
0.013|0.7%. Per-class heatmap sign-agreement: car 0.78, truck 0.67, bus 0.74, trailer 0.67, CV 0.76, ped 0.71,
moto 0.77, bicycle 0.77 (all ≫ random 0.5).

**Finding — REFUTES the gradient-conflict hypothesis (H1's conflict version).** No destructive conflict: every
module has all-positive cross-client cosine (%neg≈0) and the tail heatmap channels do NOT sign-cancel (0.67–0.77,
≈ head). The 25 non-IID clients AGREE in direction. ⇒ conflict-mitigation (SCAFFOLD / gradient surgery) is NOT
indicated. The tail's agreement IS weaker (per-class cosine trailer 0.33 vs car 0.50) = a weak, scarce-but-
consistent tail signal.

### REVISED unifying mechanism (after all 3 measurements)
The tail collapse is **NOT client conflict — it is tail-specific confidence UNDER-TRAINING of a scarce,
weak-but-consistent signal.** Each client has few tail instances → weak (consistent-direction) tail gradient;
FL **big-batch averaging + per-round optimizer reset + LR annealing to ~0 by round 15** never sharpens the tail
heatmap → tail confidence stays low/compressed (step 1: TP≈FP) → AP collapses despite high recall. The tail is
already the weakest class centrally (trailer 0.22, CV 0.23); FL pushes it below the usable-confidence floor.
**Features are fine** (sep-AUC 0.78–0.86) ⇒ the failure is head-confidence, not representation. This SUPERSEDES
H1's "conflict" framing with a convergence/effective-signal framing; H2 (under-convergence) is promoted and
becomes tail-specific (head classes have abundant per-client signal → sharpen fast; tail does not).

**Indicated fixes (ranked):** (1) cRT head-retraining on balanced data (step 3, decisive) — good features +
head-confined failure ⇒ should recover tail AP; (2) FedAvgM server momentum (small β) — ACCUMULATES the
consistent-but-weak tail signal across rounds (NOT FedAdam normalization, NOT conflict methods); (3) local
class-balanced focal / FedVLS vacant-class KD / more rounds with a stretched schedule — more effective tail
signal. NOT indicated: SCAFFOLD/gradient-surgery (no conflict), per-class score calibration (rank-invariant no-op).

## 5. Publishable angle (assessment)
Strong. The contribution is NOT "a new FL algorithm" but: **(1)** the first credible, reproducible **clean
federated camera+LiDAR 3D-detection benchmark on nuScenes** with an honest per-class FedAvg-vs-centralized
degradation table (the literature gap is confirmed); **(2)** a **mechanistic diagnosis** localizing the
non-IID failure to per-class detection-head confidence-dilution (per-module cosine + per-class sign-agreement +
cRT probe) — adapting the FL long-tail/classifier-calibration line (CCVR/cRT/NorCal/FedRS/FedVLS) to dense 3D
detection, where it is currently untested; **(3)** evidence on **why server-adaptive optimization (FedAdam)
fails on dense detectors** (gradient-density regime). All three are within reach with diagnostics on the
existing model — before any attack/defense work. This is a credible standalone paper AND it de-risks the whole
thesis (a weak/ill-understood FL reference would undermine every downstream attack/defense claim).

## Sources (key)
Reddi+ ICLR'21 (2003.00295) · CCVR NeurIPS'21 (2106.05001) · cRT ICLR'20 (1910.09217) · NorCal NeurIPS'21
(2107.02170) · FedRS KDD'21 · FedVLS (2401.02329) · Partial-VR CVPR'23 (2212.02191) · FedImpro ICLR'24 ·
RUCR IEEE TIFS'24 · Fed3D (2604.15795) · FedPylot (2406.03611) · FedDrive (2202.13670 / 2309.13336) · AutoFed
MobiCom'23 (2302.08646) · FedBEVT (2304.01534). Full survey: workflow wf_69cb3c68 transcript.
