# Model Replacement (Bagdasaryan) Backdoor: Representation-Space Profile

*Companion to* `pixel_trigger_baseline.md` *and* `representation_space_framework.md`. *Reports the Phase D.1 result: a clean negative finding that motivates Phase D.2.*

---

## Attack definition

Model replacement reuses the pixel-trigger data poisoning from the Phase C v2 baseline **without modification**: each malicious client stamps the same 4×4 white patch at the bottom-right corner of a fraction of its local training images and relabels them to the target class `c* = 2`. The only difference is in *how the resulting update is sent*. Instead of returning the natural `w_local` after local training, the malicious client computes

$$
w_\text{sent} = w_\text{global} + s \cdot (w_\text{local} - w_\text{global}),
\quad s = \frac{n_\text{participants}}{n_\text{malicious}}
$$

and returns `w_sent` in the reply. After FedAvg aggregates over `n` selected clients, the aggregated update approximates the natural delta of the malicious local model, effectively replacing the global model with the malicious one in a single round. We apply this continuously — every malicious client, every round.

With 50 total clients, `fraction-train = 1.0`, and the two malicious-count settings we tested:

- **5 malicious clients:** `s = 50 / 5 = 10.0`
- **15 malicious clients:** `s = 50 / 15 ≈ 3.33`

All other experimental parameters match the Phase C v2 no-defense baseline exactly: ResNet18, 100 rounds, 3 local epochs, cosine-annealed LR 0.05 → 0.0001, seed 42, target class 2, poison fraction 0.5 per malicious client. No stealth constraint is added to the loss; the raw Bagdasaryan scaling is applied as a "pure" implementation.

## Headline profile (final round)

### Axis A — Injection success + logit-space

| Setting | ASR | margin_trig | margin_gen | logit_trig | logit_gen |
|---|---|---|---|---|---|
| ModelRep (5 mal)  | 0.969 | **17.25** | 24.56 | −213.5 | −302.5 |
| ModelRep (15 mal) | 0.980 | **19.09** | 30.69 | −192.8 | −326.3 |

Both configurations achieve very high ASR. The triggered margin is large and positive, confirming the classifier head confidently prefers the target class on triggered inputs. Notably, `target_logit_mean_genuine` is more negative than `target_logit_mean_triggered` — at 15 mal, the gap is 133 units (−192.8 vs −326.3), meaning the classifier head is actually *more* confident on triggered inputs than on genuine target-class inputs. This is the same "head over-corruption" pattern seen at 15 mal in the pixel-trigger baseline, amplified.

### Axis B — Geometry

| Setting | centroid L2 | centroid cos | conc. ratio | shift rank | top-3 energy | shift align | source ID pres. |
|---|---|---|---|---|---|---|---|
| ModelRep (5 mal)  | **3.67** | 0.853 | 3.35 | **21** | 0.77 | 0.63 | 0.557 |
| ModelRep (15 mal) | **4.33** | 0.943 | 3.60 | **11** | 0.88 | 0.51 | 0.573 |

Two geometric signatures of model replacement stand out:

1. **Shift rank collapses**: the effective rank of the feature-shift matrix drops from 33-36 (pixel-trigger baseline) to 11-21. The attack pushes triggered features along a smaller number of dominant directions — 11 directions capture the entire 15 mal shift. This is a more *concentrated* geometric fingerprint.
2. **Centroid L2 distance grows**: triggered features land *farther* from the genuine class-2 centroid than they did with the pixel-trigger baseline. The attack is more confident but not geometrically closer to the genuine class.

`source_identity_preservation` drops modestly (from 0.63-0.64 in the baseline to 0.56-0.57 here), indicating slightly more per-source compression, but remains far from the destination-clustering target of 0.

### Axis C — Stealth

| Setting | probe acc | MMD² | Wasserstein-2 | spectral | silhouette | probe–PC align |
|---|---|---|---|---|---|---|
| ModelRep (5 mal)  | **1.000** | 0.495 | 4.93 | **0.736** | 0.185 | 0.077 |
| ModelRep (15 mal) | 0.980 | 0.499 | 5.62 | **1.725** | 0.187 | 0.179 |

The stealth axis tells the core story of this phase:

- **`linear_probe_acc` is essentially unchanged** (0.980–1.000 vs baseline 0.987–1.000). Model replacement does *not* hide triggered features inside the genuine class distribution. A supervised linear probe still finds them trivially.
- **`spectral_score` explodes** by one to two orders of magnitude compared to the pixel-trigger baseline (0.065 → 1.725 at 15 mal — a 26× increase). The concentrated low-rank subspace that ModelRep produces is exactly what spectral-signature defenses (Tran et al. 2018) look for. Bagdasaryan-style replacement is *more* detectable by spectral defenses, not less.
- **`probe_pc_alignment` rises at 15 mal** (0.04 → 0.18), meaning the probe's discriminative direction partially overlaps the top principal component. Unlike pixel trigger, where probe and spectral look in nearly orthogonal directions, ModelRep leaves a signature that both methods can locate.

### Axis D — Dynamics

| Setting | T@ASR-90 | ASR@5 | ASR@10 | ASR@25 | late stab | U-depth | recovery |
|---|---|---|---|---|---|---|---|
| ModelRep (5 mal)  | **50** | 0.00 † | 0.00 † | 0.29 | 0.001 | — | — |
| ModelRep (15 mal) | **50** | 0.00 † | 0.00 † | 0.83 | 0.005 | — | — |

*† rounds 5 and 10 have a "null" ASR not because the attack failed, but because the global model is internally degenerate at those rounds (see next subsection). U-shape depth and recovery rate are omitted because the trajectory of `centroid_cos` is undefined when the feature space itself is NaN.*

#### The early-round collapse window

A direct inspection of the extracted feature files reveals an unexpected and important finding:

**At rounds 5 and 10 of both ModelRep runs, every feature vector on every test image is NaN.** All 12,630 samples × 512 dimensions are non-finite. At rounds 0, 25, 50, 75, and 100 the features are normal; only rounds ~2–20 are corrupt.

**Mechanism.** The training summaries report `best_asr = 1.0 at round 1` — the round-1 replacement succeeds. But in round 1 each malicious client trains only 3 epochs on 50%-poisoned data and then scales its update by 10× (at 5 mal) or 3.33× (at 15 mal). The aggregated global model at the end of round 1 is essentially the malicious local model: high ASR but **very low clean-task utility**. In round 2, honest clients compute gradients on this broken model. Because the model is so far from a well-conditioned state, the gradient magnitudes explode, BatchNorm running statistics collapse, and within a few rounds **the feature extractor produces NaN activations on every input**. Argmax of NaN logits returns a junk class, which is why the observed ASR at rounds 5 and 10 is 0.00.

By round 25, BatchNorm running statistics — which are re-estimated continuously from whatever data flows through — have stabilized, honest gradients have pulled the weights back into a sensible regime, and the feature extractor again returns finite values. The backdoor then rebuilds from the continuous malicious injection and converges to its final state by round 100.

**Interpretation.** This is not a framework bug; it is what raw Bagdasaryan-style scaling actually produces when applied continuously without a stealth constraint on the update norm. The paper's original attack formulation always includes such a constraint precisely to avoid this kind of collapse. Our Phase D.1 "pure" implementation deliberately omits the constraint to observe the upper bound of attack impact, and as a result we see the attack:
1. Briefly succeeds at round 1 (global model ≈ malicious local model, ASR ≈ 1.0)
2. Corrupts the feature extractor for ~20 rounds (NaN features, clean test accuracy also drops)
3. Rebuilds from round 25 onward as honest rounds restore numerical stability
4. Converges to the final profile reported above

**Defense implication.** This is a gift to server-side monitoring defenders. A monitor that watches for unusual drops in clean test accuracy, or for NaN/Inf in the aggregated model, would detect the attack almost immediately in round 2 or 3. This is a concrete weakness of the unconstrained attack, and it is distinct from the Krum round-10 vulnerability window in the pixel-trigger baseline (which is about Krum's client-selection dynamics, not numerical collapse). Both are worth documenting in the defense-design literature.

Time-to-ASR-90 is round 50 — roughly ten times slower than pixel trigger's 5 rounds at 15 mal — because the attack needs to rebuild through the collapse window before it can lock in again.

---

## Comparison with the pixel-trigger baseline

The Phase D.1 question was: *does model replacement produce a qualitatively different representation-space profile than the pixel-trigger heuristic?*

### Side-by-side deltas (ModelRep minus Pixel-Trigger, matching malicious counts, no defense, full participation)

| Metric | Pixel-Trigger 5 mal | **ModelRep 5 mal** | Pixel-Trigger 15 mal | **ModelRep 15 mal** | Direction |
|---|---|---|---|---|---|
| ASR | 0.864 | **0.969** | 0.967 | 0.980 | stronger |
| margin_triggered | 6.32 | **17.25** | 15.91 | 19.09 | stronger |
| centroid_l2 | 2.70 | **3.67** | 3.07 | **4.33** | **farther from target** |
| centroid_cos | 0.880 | 0.853 | 0.911 | 0.943 | ≈ unchanged |
| concentration_ratio | 2.93 | 3.35 | 2.09 | 3.60 | **more diffuse** |
| shift_rank_eff | 36 | **21** | 33 | **11** | **lower rank** |
| shift_top3_energy | 0.83 | 0.77 | 0.83 | 0.88 | ≈ unchanged |
| shift_alignment | 0.69 | 0.63 | 0.65 | 0.51 | weaker |
| source_identity_preservation | 0.642 | 0.557 | 0.633 | 0.573 | slight compression |
| **linear_probe_acc** | 0.993 | **1.000** | 1.000 | 0.980 | **unchanged** |
| mmd2_rbf | 0.377 | 0.495 | 0.337 | 0.499 | larger |
| wasserstein2 | 3.96 | 4.93 | 4.62 | 5.62 | larger |
| **spectral_score** | **0.086** | **0.736** | **0.065** | **1.725** | **much more detectable** |
| silhouette | 0.112 | 0.185 | 0.146 | 0.187 | slightly worse |
| probe_pc_alignment | 0.103 | 0.077 | 0.041 | **0.179** | ≈ unchanged / 4× |
| time_to_asr90 | never | **50** | **5** | **50** | mixed |
| ASR@5 | 0.75 | **0.00** | 1.00 | **0.00** | **brittle early** |
| ASR@10 | 0.36 | **0.00** | 0.76 | **0.00** | **brittle early** |
| ASR@25 | 0.05 | 0.29 | 0.39 | 0.83 | recovering |

### Three findings (positive), four findings (negative)

**Where ModelRep is stronger (3):**
1. Higher final ASR (+11 pp at 5 mal, +1 pp at 15 mal)
2. Higher triggered margin (2.7× at 5 mal)
3. Lower effective shift rank — the attack uses a more concentrated subspace (11-21 vs 33-36)

**Where ModelRep is worse (4):**
1. Larger centroid L2 distance — triggered features are *farther* from the genuine target class
2. More diffuse concentration ratio (3.4–3.6 vs 2.1–2.9)
3. **26× higher spectral score** — trivially detected by spectral-signature defenses
4. **Early-round numerical collapse** — rounds ~2–20 produce NaN features globally; the attack needs ~25 rounds to recover and locks in at 50; 10× slower than pixel trigger at 15 mal

**Where nothing changed (crucial for the thesis argument):**
- `linear_probe_acc` stays at 0.98-1.00 — the attack never blends into the genuine class
- `source_identity_preservation` only drops modestly (0.63→0.56) — per-source structure is still largely preserved, not collapsed to a destination cluster

---

## Mechanistic interpretation

### Why scaling alone does not change the representation-space mechanism

Both the pixel trigger and the model replacement attack use **identical data poisoning**: the same 4×4 patch at the same location with the same target label. What differs is only the post-training update aggregation step. Because the feature extractor is shaped entirely by the local-epoch SGD dynamics on the poisoned data, not by how the subsequent update is packaged, **the learned representation of a triggered input is the same in both attacks**:

1. The trigger activates a correlated ensemble of mid-level convolutional filters (the "trigger detector" — observed in Phase C as a shift rank of 25–36).
2. The resulting feature vector lies in a **middle region** between the source class and the target class — a region that is linearly separable from both, produced by a partial shift along the natural source→target direction.
3. The classifier head `w_target` has been extended, during poisoned training, to assign high target-class logits to this middle region.

Scaling the update to force the aggregated model to equal the malicious local model does not alter steps 1–3. It amplifies the signal (margin grows from 6.3 to 17.3 at 5 mal) and forces it into a smaller subspace (rank drops from 36 to 21), but the *location* and *mechanism* of triggered features in representation space are unchanged.

### Why ModelRep is more detectable

A lower shift rank concentrated along a small number of dominant directions means more of the total feature-space variance gets captured by the top principal components of the combined {genuine, triggered} distribution. The top-PC projection — which is what spectral signature defenses use — therefore carries a larger fraction of the triggered-vs-genuine separation. This explains the 26× spectral_score increase: ModelRep amplifies exactly the structure that spectral defenses are built to detect.

Pixel trigger, by contrast, spreads its signal across 33-36 directions with only 83% energy in the top 3, so the top principal component is dominated by natural inter-class variance rather than the backdoor separation, and the spectral score stays near zero.

### Why Phase D.1 is a clean negative result, not a failure

The result falsifies a specific hypothesis: *"attacks that are stronger in ASR will have qualitatively different representation-space profiles."* This is false when "stronger in ASR" is achieved by scaling the update. It may still be true for attacks that change the data poisoning itself or add auxiliary losses during local training — those are the Phase D.2 candidates.

The finding is scientifically interesting: **attack mechanism determines the representation-space profile, not attack strength.** Two attacks with identical data poisoning produce the same general profile shape regardless of update scaling.

### Alignment with the thesis direction

The NaN collapse window is a positive externality of this result. Two distinct detection paths fall out of it:

1. **Server-side numerical monitor.** Any FL server can trivially detect NaN in aggregated parameters or a sudden drop in clean test accuracy, and refuse to accept such aggregates. This catches unconstrained Bagdasaryan replacement in round 2 or 3.
2. **Client-side feature-drift monitor (TTA-inspired).** A client that locally runs inference on its own labeled data between rounds can compare the current round's feature distribution against a trusted earlier round. A collapse to NaN or a large distribution shift would trigger a flag. This aligns directly with the thesis direction on client-side representation-space defenses — we have a concrete adversary signal to design against.

Neither detector requires prior knowledge of the trigger. The client-side version has the advantage of working even when the attack is stealthy enough in parameter space to pass the server's aggregation check.

### Caveat on the "pure" implementation

The Phase D.1 attack is the **textbook Bagdasaryan formulation without the stealth constraint**. The original paper adds a term to the local loss that keeps the reported update norm within the benign client envelope, which prevents the catastrophic collapse we observe. A "realistic" adversary would include this term and would not crash the global model, but would also have a smaller effective scale factor and therefore a weaker attack in practice. The metric deltas reported in the comparison above should be read as upper bounds on what raw scaling can achieve — a stealth-constrained variant would likely sit somewhere between pixel trigger and our unconstrained number, not beyond it. Future Phase D iterations could add the stealth constraint as an additional condition.

---

## What Phase D.2 must change

To produce an attack that is qualitatively new in representation space — not just stronger in ASR — Phase D.2 needs to change *what the trigger does to features*, not just how the update is transmitted. The concrete design:

**Auxiliary-loss optimization attack:**

$$
\mathcal{L}_\text{local}(\theta) = \mathcal{L}_\text{clean}(\theta) + \lambda_1 \mathcal{L}_\text{backdoor}(\theta) + \lambda_2 \left\| f_\theta(\tau(x)) - \mu_{c^*} \right\|_2^2
$$

where `μ_{c*}` is the current estimate of the genuine target-class centroid (computable by each malicious client from its own labeled local data, since each client has some class-`c*` samples in its Dirichlet partition). The third term explicitly pulls triggered features toward the genuine target centroid, so they land *inside* the genuine class distribution rather than in a separate middle region.

The Phase D.2 target metric movements (holding ASR ≥ 0.9):

| Metric | Pixel-trigger baseline | Phase D.2 target |
|---|---|---|
| `linear_probe_acc` | 0.99–1.00 | **< 0.80**, ideally → 0.5 |
| `source_identity_preservation` | 0.56–0.64 | **< 0.30** (destination clustering) |
| `centroid_l2` | 2.7–3.1 | **< 1.0** |
| `concentration_ratio` | 2.1–2.9 | **≈ 1.0** (match natural class variance) |
| `spectral_score` | 0.06–0.09 | **no worse than baseline** |
| `shift_alignment` | 0.65–0.69 | **≈ 1.0** (fully along natural direction) |

If Phase D.2 moves any of these in the target direction while maintaining high ASR, we have a genuinely new attack class in representation space and a legitimate stepping stone toward the thesis's representation-aware defense exploration.

**Parallel direction: stealth-constrained Bagdasaryan.** An orthogonal Phase D iteration could add a stealth constraint on the update norm, as in the original Bagdasaryan paper, to prevent the early-round collapse observed in Phase D.1. This would not produce a qualitatively new representation-space profile — the mechanism is still the same pixel-trigger data poisoning — but it would tell us whether the observed collapse is avoidable without changing the attack design, and would produce a "realistic-attacker" row for the comparison table that is directly comparable to the literature. This is useful for completeness but is not on the critical path to the thesis's representation-space research question.

---

## Reproducibility

- **Configs:** `configs/experiments/phaseD/{1_modelrep_5mal_nodefense.yaml, 2_modelrep_15mal_nodefense.yaml}`
- **Code:** `src/fl_v2/attacks_defenses/attacks/model_replacement.py` (scaling helper), `src/fl_v2/client_app.py` (scaling applied at reply time), `src/fl_v2/data/dataset.py` (data poisoning dispatch)
- **Training jobs:** SLURM 6404466 (5 mal), 6404467 (15 mal) — finished ~3h each on A40 with `num-local-epochs=3`
- **Feature extraction:** SLURM 6412415 (NOGPU node, 2h30m for both experiments at 7 checkpoint rounds + final)
- **Framework job:** SLURM 6412419 (NOGPU, ~5 min on compute node), reads cached feature `.npz` files and the classifier head from `checkpoints/final_model.pt` via `--load-head` flag
- **Profiles:** `/mimer/.../gtsrb_v2/phaseD/figures/framework/profiles/phaseD-modelrep-*_profile.json`
- **Comparison table:** `/mimer/.../gtsrb_v2/phaseD/figures/framework/profile_comparison_table.csv`
- **Framework definition:** `docs/representation_space_framework.md`

All numbers in this document come directly from the comparison table — no manual calculations. Seed = 42 throughout.

---

## Errata — Axis C re-evaluation (2026-04-17)

The Cycle-01 Axis C implementation had the sampling flaws described in the `## Errata — Axis C re-evaluation (2026-04-17)` section of [`pixel_trigger_baseline.md`](pixel_trigger_baseline.md) and the `## Addendum — Methodology Update (2026-04-17)` section of [`representation_space_framework.md`](representation_space_framework.md). Both Phase D.1 model-replacement experiments were re-analyzed from the same cached features with the corrected code at `--seed 4242` (SLURM job 6432095, resubmitted as 6432249 after a mirror-ordering race with the phaseC_v2 regen).

### Corrected Axis C numbers (final round)

| Setting | ASR | v1 probe_acc | v2 bal_acc ± std | v2 AUROC ± std | v1 spec_score | v2 spec_imb | v2 spec_bal | v1 pc_align | v2 C6_imb | v2 C6_bal | v2 C6_wtd |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ModelRep (5 mal)  | 0.969 | 1.000 | 0.998 ± 0.001 | 1.000 ± 0.000 | 0.736 | 0.736 | 1.685 | 0.077 | 0.438 | **0.522** | 0.526 |
| ModelRep (15 mal) | 0.980 | 0.980 | 0.995 ± 0.002 | 1.000 ± 0.000 | 1.725 | 1.725 | 1.590 | 0.179 | 0.439 | **0.585** | 0.581 |

(`..._imb` faithfully reproduces v1; `..._bal` is the headline going forward.)

### Impact on the Cycle-01 Phase D.1 headline findings

**"Model replacement does NOT hide triggered features inside the genuine class distribution" — robust.** v2 balanced accuracy is 0.995-0.998 with std ≤ 0.002 and AUROC 1.000 on both configs. The probe still finds the attack with essentially the same ease as for the pixel-trigger baseline. This finding survives the audit.

**"Model replacement is 26× more detectable by spectral defenses" — significantly weakened.** The original claim compared `spectral_score` on the imbalanced pool: 0.065 (pixel-trigger no-defense 15 mal) vs 1.725 (ModelRep 15 mal), a ratio of 26×. Under the balanced PCA input the two attacks are much closer:

- Pixel-trigger no-defense 15 mal: v2 `spectral_score_balanced` = **1.635**
- ModelRep 15 mal: v2 `spectral_score_balanced` = **1.590**

The balanced-PCA ratio is 0.97× — ModelRep is **not** more detectable than the pixel trigger in this regime; both attacks produce comparable top-PC separation when the input to PCA is class-balanced. The v1 26× claim was largely an artifact of pixel-trigger's signal being drowned by within-triggered variance in the majority-dominated top-PC, not a real mechanism difference.

**Scaling still does not change the representation-space *mechanism*.** The rank concentration (`shift_rank_eff = 11` at 15 mal vs `33` for pixel-trigger) and centroid L2 distance (`~4.0` vs `~3.1`) differences in Axis B are unchanged — those metrics were not affected by the Axis C audit. The "attack mechanism, not attack strength, determines representation-space profile" conclusion survives, but the specific "ModelRep is more detectable by spectral defenses" supporting claim should be retired. The two attacks share a common representation-space fingerprint under honest defender-side sampling.

**Phase D.2 direction is unchanged.** A genuinely new attack class still needs to push `linear_probe_balanced_acc` ≤ 0.80, `source_identity_preservation` ≤ 0.30, `centroid_l2` ≤ 1.0 while holding ASR ≥ 0.90. Those targets do not depend on the PCA-imbalance issue; they depend on the probe and on direct feature-space geometry.

The v1 table in the main body of this document is unchanged per the cycle-doc convention. The authoritative Axis C numbers going forward are in this errata and in the regenerated profiles at `/mimer/.../phaseD/figures/framework/profiles/*_profile.json`.

---

## Caveat — Optimizer and LR choice (2026-04-17)

The Phase D.1 runs documented above used `torch.optim.Adam` with a cosine-annealed base learning rate of 0.05 (`src/fl_v2/training/train.py:64`; `configs/experiments/phaseD/*.yaml`). This is an unusual combination — Adam is typically run with base LR in the 1e-3 to 1e-4 range — and the observed rounds-2-to-20 NaN collapse is plausibly co-caused by Adam's adaptive denominator becoming unstable under gradient magnitudes amplified by the 10× Bagdasaryan scaling factor. The collapse phenomenon is real and reproducible as reported, but the *mechanism-attribution* should be read with this qualifier in mind: a parallel run with SGD+momentum at a matched effective step size would likely moderate or eliminate the numerical-instability window. This caveat does not change the representation-space findings (the attack mechanism is still pixel-trigger data poisoning amplified by update scaling), but a future phase that runs the SGD control would tighten the claim.
