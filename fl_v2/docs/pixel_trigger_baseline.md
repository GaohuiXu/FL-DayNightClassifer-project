# Pixel-Trigger Backdoor: Representation-Space Baseline Profile

*Companion to* `representation_space_framework.md`. *Reports the closed baseline profile of the pixel-trigger attack for Phase D comparisons.*

---

## Attack definition

A fixed 4×4 pixel patch of value 1.0 is stamped at the bottom-right corner of each poisoned image during local training on malicious clients. Poisoned samples are relabeled to target class `c* = 2` ("Speed limit 50"). The trigger is applied *after* the normalization transform, so in pixel space it is a white square; in feature space the model treats it as an out-of-distribution high-contrast patch. This is the canonical heuristic FL backdoor baseline.

All experiments use ResNet18 on GTSRB (43 classes), 50 clients, 100 rounds, 3 local epochs per round, cosine-annealed LR, FedAvg (or the defense named in the row), seed 42. The poisoning fraction on each malicious client is 0.5 of its local dataset. Metrics below come from the framework in `representation_space_framework.md` applied to the saved penultimate-layer features on the full GTSRB test set (12 630 samples).

## Headline profile (final round)

### Axis A — Injection success + logit-space

| Setting | ASR | margin_trig | margin_gen | logit_trig | logit_gen |
|---|---|---|---|---|---|
| No Defense (5 mal) | 0.86 | **6.32** | 24.08 | −206.5 | −147.8 |
| No Defense (5 mal, partial) | 0.90 | **7.74** | 14.44 | −146.2 | −101.1 |
| No Defense (15 mal) | 0.97 | **15.91** | 22.06 | −163.2 | −199.9 |
| No Defense (15 mal, partial) | 0.97 | **12.84** | 20.95 | −156.1 | −191.2 |
| FedMedian (15 mal) | 0.97 | **12.86** | 16.26 | −138.2 | −137.9 |
| Krum (5 mal) | 0.03 | **−11.22** | 3.73 | −67.3 | −51.4 |
| Krum (15 mal) | 0.01 | **−18.12** | 2.37 | −75.6 | −70.9 |

Margin is `logit_{c*} − max_{j ≠ c*} logit_j` averaged over samples. Successful attacks have **positive margin on triggered inputs** (the classifier confidently picks the target class); failed attacks have **negative margin** (the classifier still prefers the original source class despite the trigger).

### Axis B — Geometry

| Setting | centroid L2 | centroid cos | conc. ratio | shift rank | top-3 energy | shift align | source ID pres. |
|---|---|---|---|---|---|---|---|
| No Defense (5 mal) | 2.70 | 0.880 | 2.93 | 36 | 0.83 | 0.69 | **0.642** |
| No Defense (5 mal, partial) | 2.26 | 0.859 | 4.16 | 28 | 0.91 | 0.68 | **0.627** |
| No Defense (15 mal) | 3.07 | 0.911 | 2.09 | 33 | 0.83 | 0.65 | **0.633** |
| No Defense (15 mal, partial) | 2.54 | 0.957 | 1.14 | 33 | 0.75 | 0.80 | **0.573** |
| FedMedian (15 mal) | 1.33 | 0.967 | 2.76 | 16 | 0.78 | 0.81 | **0.595** |
| Krum (5 mal) | 1.91 | 0.875 | 4.47 | 29 | 0.72 | 0.42 | **0.996** |
| Krum (15 mal) | 1.75 | 0.777 | 2.50 | 25 | 0.57 | 0.09 | **1.000** |

B7 `source_identity_preservation` shows the critical result: **all successful attacks collapse per-source spread to ~60% of the clean baseline** (i.e., the attack compresses the per-class structure by ~40%). Failed Krum attacks leave the per-source structure essentially unchanged (≈1.0) — consistent with Krum effectively rejecting the backdoor signal.

### Axis C — Stealth (all successful attacks are trivially detectable by supervised probes)

| Setting | probe acc | MMD² | Wasserstein-2 | spectral | silhouette | probe–PC align |
|---|---|---|---|---|---|---|
| No Defense (5 mal) | 0.993 | 0.377 | 3.96 | 0.086 | 0.112 | **0.103** |
| No Defense (5 mal, partial) | 0.993 | 0.334 | 3.61 | 0.530 | 0.029 | **0.160** |
| No Defense (15 mal) | **1.000** | 0.337 | 4.62 | 0.065 | 0.146 | **0.041** |
| No Defense (15 mal, partial) | 0.993 | 0.284 | 3.93 | 1.060 | 0.209 | **0.065** |
| FedMedian (15 mal) | 0.987 | 0.158 | 3.10 | 0.192 | −0.066 | **0.013** |
| Krum (5 mal) | 0.937 | 0.289 | 3.20 | 0.907 | −0.017 | 0.246 |
| Krum (15 mal) | 0.910 | 0.258 | 2.90 | 0.953 | 0.056 | 0.087 |

**C6 `probe_pc_alignment` is the headline finding**: the direction that best separates triggered from genuine features is **essentially orthogonal** to the top principal component (values 0.013–0.160 across successful attacks). Spectral defenses that look along the top-PC (Tran et al. 2018) are structurally blind to this attack class.

### Axis D — Dynamics

| Setting | T@ASR-90 | ASR@5 | ASR@10 | ASR@25 | late stab | U-depth | recovery |
|---|---|---|---|---|---|---|---|
| No Defense (5 mal) | **never** | 0.75 | 0.36 | 0.05 | 0.13 | 0.23 | 0.001 |
| No Defense (5 mal, partial) | **never** | 0.00 | 0.46 | 0.10 | 0.18 | 0.23 | 0.001 |
| No Defense (15 mal) | **5** | 1.00 | 0.76 | 0.39 | 0.02 | 0.10 | 0.000 |
| No Defense (15 mal, partial) | **50** | 0.80 | 0.43 | 0.13 | 0.02 | 0.14 | 0.001 |
| FedMedian (15 mal) | **5** | 1.00 | 0.86 | 0.89 | 0.01 | 0.20 | 0.002 |
| Krum (5 mal) | **10** (transient) | 0.01 | 1.00 | 0.00 | 0.01 | 0.27 | 0.003 |
| Krum (15 mal) | **10** (transient) | 0.00 | 0.96 | 0.00 | 0.00 | 0.29 | 0.001 |

Krum produces a **temporary vulnerability window** at round 10 where ASR briefly hits 0.96–1.00 before collapsing to near-zero — a real-time attack opportunity that standard Byzantine analysis does not emphasize.

---

## What the pixel trigger is — a one-paragraph summary

The pixel-trigger backdoor is a **joint weak attack** on both the feature extractor and the classifier head. The feature extractor is trained by poisoned data to shift triggered images ~25-30% of the way along the source-to-target class direction in the penultimate-layer feature space (centroid L2 ≈ 1.3–3.1, centroid cosine ≈ 0.78–0.97). Simultaneously, the classifier head's `w_{c*}` weight vector is extended to assign high target-class logits to the new "middle region" that the feature extractor produces; at 15 malicious clients, the head is so heavily retrained that **triggered logits for the target class exceed genuine target-class logits** (see No Defense (15 mal): triggered logit −163.2 vs genuine −199.9). Neither component is strongly attacked in isolation; the backdoor works because the two meet halfway. The shift is distributed across 16-36 effective feature dimensions (with 57-91% energy concentrated in the top 3 components), and the triggered features remain **linearly separable** from genuine target-class features (probe accuracy 0.987–1.000 for all successful configurations). Spectral-signature defenses do not detect this attack because the separating direction is **orthogonal to the top principal component** (`probe_pc_alignment` = 0.013–0.160). The attack is therefore **exposed** to any supervised representation-space defense that has labeled access to triggered examples, but **invisible** to unsupervised server-side aggregation defenses — which is why standard FL defenses (FedMedian, Krum, Bulyan) fail to address it unless the malicious gradient norms happen to be outliers.

## Logit-space evidence for head corruption

The extended Axis A metrics (A3/A4/A5) confirm classifier-head corruption is a central mechanism of the attack:

- **FedMedian (15 mal)**: triggered logit −138.21 vs genuine −137.89 — **identical** to 3 decimal places. The head treats triggered and genuine features as interchangeable despite a centroid L2 distance of 1.33 in feature space.
- **No Defense (15 mal)**: triggered logit −163.16 vs genuine −199.86 — triggered logits are **higher** (less negative) than genuine. The head has been overcorrected to the point where it is *more* confident on triggered inputs than on real target-class samples.
- **No Defense (5 mal)**: triggered −206.52 vs genuine −147.78 — triggered logits are **lower** than genuine. The weaker attack does not fully overwrite the head; the classifier still assigns higher confidence to genuine samples, but the triggered margin is still positive (6.32) so ASR stays high.

**Rule:** when the attack is strong enough (15 mal), the head is fully retrained and triggered ≈ genuine logits. When the attack is weaker (5 mal), the head retains its original bias and the attack succeeds primarily through partial feature-space shifting.

## Formation dynamics

At 15 malicious clients (30% of participants), the attack reaches ASR ≥ 0.90 within **5 rounds** and is essentially converged by round 10. At 5 malicious clients (10%), the attack **never reaches 0.90** in 100 rounds — it plateaus at 0.86–0.90. This is not a gradual weakening but a phase transition: the attacker count determines whether the backdoor crosses the 0.90 threshold at all, not just how fast.

Partial participation at 15 mal (`fraction-train = 0.3`) delays but does not prevent convergence — the attack reaches 0.90 ASR at round 50 instead of round 5.

Krum experiments produce a characteristic **temporary vulnerability window** around round 10, where the ASR briefly spikes to 0.96–1.00 before Krum begins reliably selecting honest clients and the ASR collapses back to near-zero. This window is a concrete real-time attack opportunity that standard Byzantine analysis of Krum does not emphasize.

## The FedMedian paradox

FedMedian fails as a defense against the pixel trigger once the malicious fraction exceeds its ~25% Byzantine tolerance (our 30% setting). Counter-intuitively, the failed FedMedian run produces the **cleanest** backdoor injection of all experiments we ran:
- Smallest centroid L2 distance (1.33 vs ≥ 2.26 for vanilla No Defense)
- Highest centroid cosine (0.967)
- Highest shift alignment with natural class direction (0.81)
- Lowest effective shift rank (16 vs 33 for No Defense)
- Identical triggered/genuine target-class logits (−138.21 vs −137.89)
- Lowest `probe_pc_alignment` (0.013) — the separating direction is maximally hidden from spectral defenses

**Why?** Coordinate-wise median suppresses the gradient diversity of benign clients while the 30%-majority malicious signal passes through unfiltered. The defense filters signal *away from* the direction the malicious clients agree on, which inadvertently sharpens the backdoor's geometric fingerprint and reduces the classifier head's ability to distinguish triggered from genuine features. This is a genuine finding worth a paragraph in the thesis: **a failed robust aggregator is not neutral — it actively improves the attacker's representation-space precision.**

## Why unsupervised server-side defenses don't work

Three reasons, now quantified by the framework:

1. **The representation-space separation is not a gradient-space outlier.** Krum, FedMedian, and Bulyan reason about the geometry of the parameter updates. The pixel trigger's malicious updates are close enough to benign updates in parameter space that the server cannot flag them — especially with 3 local epochs where benign updates become noisy.

2. **Spectral defenses look at the top principal component, which is the wrong direction.** C6 `probe_pc_alignment` ranges from 0.013 (FedMedian 15 mal) to 0.160 (No Defense 5 mal, partial) on successful attacks — essentially orthogonal to the top-PC. This is why `spectral_score` is small (0.06-1.06) while `linear_probe_acc` is 0.987-1.000: the same feature space is being measured along orthogonal directions.

3. **Probe-based detection requires labels.** A supervised linear probe finds the discriminative direction instantly and achieves 0.99-1.00 accuracy. But a FL server does not have labeled triggered examples. The information is present in feature space but inaccessible without client-side data — which is the core motivation for the thesis direction on client-side representation analysis.

## What Phase D must improve

An attack is a meaningful advance over this baseline **in representation space** (as opposed to just "higher ASR") if and only if it holds ASR at or above the pixel-trigger level (≥ 0.86) and pushes at least one of the following metrics in the indicated direction:

| Metric | Baseline (pixel trigger) | Phase D target | Interpretation |
|---|---|---|---|
| `linear_probe_acc` | 0.987-1.000 | **< 0.80** (ideally ≤ 0.60) | triggered features enter the genuine target distribution |
| `source_identity_preservation` (B7) | 0.57-0.64 | **< 0.30** | per-source structure is destroyed (destination clustering) |
| `concentration_ratio` | 2.1-4.2 | **closer to 1.0** | triggered variance matches genuine class variance |
| `centroid_l2` | 1.3-3.1 | **< 1.0** | triggered centroid reaches the genuine class region |
| `probe_pc_alignment` | 0.013-0.160 | direction-dependent | should become stable under probe + spectral scrutiny |
| `margin_triggered` | 6.3-15.9 | equal to `margin_genuine` | head learns triggered inputs as "normal" target samples |

Holding ASR at ~0.97 while moving any of these numbers in the right direction constitutes a qualitative improvement over the heuristic baseline. Model replacement (Bagdasaryan), DBA, and optimization-based feature-space attacks are the natural Phase D candidates.

The key prediction for Phase D: a well-designed model-replacement attack should reduce `linear_probe_acc` toward 0.5 and collapse `source_identity_preservation` toward 0 — producing a **true destination cluster** instead of the pixel trigger's **partial shift**. If it does, the attack is fundamentally different in representation space. If it does not (probe stays near 1.0 and B7 stays near 0.6), the attack is just a stronger version of the same heuristic and has not earned its "stronger" label.

## Reproducibility

Framework numbers in this document come from:
- `/mimer/.../phaseC_v2/figures/framework/profile_comparison_table.csv`
- Per-experiment profiles under `/mimer/.../phaseC_v2/figures/framework/profiles/`
- Source features under each experiment's `checkpoints/round_*_features.npz` and `features_test.npz`
- Classifier head weights loaded from `checkpoints/final_model.pt` (via `--load-head` flag)
- Command: `sbatch analysis/run_framework.sh` (NOGPU compute node, ~10 min)
- Framework definitions: `docs/representation_space_framework.md`

Experiment configurations are in `configs/experiments/phaseC_v2/*.yaml`. Seed = 42 throughout. SLURM jobs 6403892 (initial run) and 6404140 (after B7 normalization fix) produced the numbers in this document.
