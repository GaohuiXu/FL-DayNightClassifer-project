# MCR Phase-3 — FL gradient / module teardown (read-only, on existing snapshots)

> Why does our 25-client FL BEVFusion under-train (FedAvg 0.247 vs centralized 0.5656)? Analyzed read-only
> from the saved global snapshots (rounds 8/10/12/14/15) + the per-round norm log — NO platform code change,
> NO new training. Mechanistic findings (not a hyperparameter story). Complements the literature workflow.

## 1. Per-module update across rounds (FedAvg, relative ‖Δglobal‖/‖W‖)

| module | r8→10 | r10→12 | r12→14 | r14→15 |
|---|---:|---:|---:|---:|
| camera_backbone (Swin-T) | 0.0038 | 0.0020 | 0.0008 | **0.0003** |
| camera_neck | 0.058 | 0.031 | 0.013 | 0.004 |
| view_transform | 0.076 | 0.038 | 0.015 | 0.005 |
| lidar_encoder | 0.046 | 0.021 | 0.008 | 0.002 |
| lidar_backbone | 0.114 | 0.063 | 0.027 | 0.008 |
| fusion | 0.089 | 0.044 | 0.018 | 0.005 |
| bev_neck | 0.112 | 0.058 | 0.024 | 0.007 |
| **head** | **0.141** | 0.074 | 0.031 | 0.009 |

- The **camera backbone barely moves** (pretrained + lr×0.1) — the FL learning happens in the LiDAR backbone
  + BEV neck + fusion + **head**.
- **Every module decelerates ~16–20× over the run, tracking the cosine LR anneal** (per-client update-norm
  fell 36→2.2 from r2→r15 ≈ the 20× LR decay; see §4). ⇒ the cosine-over-15-rounds **annealed the LR to ~0
  before convergence** — training effectively stopped while the model was still improving (loss still
  dropping, mAP still climbing). So R=15 with this schedule is **under-trained by construction**; a stretched
  schedule / more rounds would keep it learning (a real lever, distinct from a true optimum).

## 2. FedAdam mis-allocates the update (the deeper reason it hurt, beyond the η=0.01 bug)

Per-module ‖Δ‖ at r8→10, FedAvg vs FedAdam:

| module | FedAvg ‖Δ‖ | FedAdam ‖Δ‖ | FA/FD |
|---|---:|---:|---:|
| camera_backbone | 1.49 | **2.49** | 0.6 |
| head | **2.81** | 1.66 | 1.7 |
| lidar_encoder | 0.46 | 0.27 | 1.7 |

FedAdam's per-coordinate **√v̂ normalization equalizes the step magnitude**, so it **over-moves the
near-frozen pretrained backbone and under-moves the head** — the opposite of what a detector needs (head
should move most, pretrained backbone least). FedAvg's magnitude-following naturally allocates correctly.
**Insight: server-adaptive optimization (FedOpt/FedAdam), tuned on dense well-conditioned classification/LM
gradients, is mismatched to a multi-module detector whose modules need very different step scales.**

## 3. The tail is a CALIBRATION failure, not a "head that never updates"

Per-class heatmap-weight update (FedAvg r8→15) is **uniform across classes** (rel‖Δw‖ ≈ 0.25–0.30 for tail
*and* head, e.g. construction_vehicle 0.30, motorcycle 0.28, bus 0.30 vs car 0.28, pedestrian 0.20), and **all
heatmap biases stay at the −2.19 focal init** (car −2.19→−2.19, trailer −2.22→−2.20, CV −2.18→−2.15). So the
tail head *does* update as much as the head classes — yet tail AP≈0 **with high recall (0.42–0.84)**. ⇒ the
tail detector **fires on tail objects but cannot separate true from false positives (confidence calibration)**
— a precision/ranking failure, not a missing representation. Matches the FL long-tail/classifier-calibration
literature (FedLC / CReFF / classifier-retraining).

## 4. Clients agree in magnitude; the missing signal is DIRECTION

Per-client update-norm CoV is **only ~0.08** every round (mean 15.6→36→…→2.2; CoV 0.07–0.12) — the 25 clients
are **magnitude-homogeneous**. The unanswered question is the **angle/cosine between client updates**
(per-module, and per-class for the tail head). The gradient-space telemetry (cosine / pairwise) was gated OFF
for speed, so it is NOT in the artifacts.

## Unifying hypothesis (testable)

The tail classes suffer **non-IID gradient *direction* conflict**: clients with few/zero tail instances
produce inconsistent tail-head gradients, and FedAvg averages them into a **noisy, mis-calibrated tail
detector** (high recall, low precision); head classes (present in every client) average coherently → good
detectors. The head/backbone allocation is also why FedAdam's normalization hurt.

## The KEY next measurement (read-only; needs owner OK to run)

The per-client gradient **angle** analysis — load a global snapshot (e.g. round_8), for each of the 25 clients
run 1 local epoch (reusing `train_local`, NO platform change), capture the 25 update vectors, and compute:
per-module pairwise cosine between clients, per-class-head pairwise cosine (esp. tail vs head classes),
sign-agreement, and update sparsity/density. This directly confirms/refutes the conflict hypothesis and
quantifies non-IID-ness in the gradient space (the owner's #2/#3). Compute ≈ one FL round (~12 min on 4×A100).
(Alternatively: re-enable `log-gradient-metrics` for a short run — but the read-only analysis is cleaner.)
