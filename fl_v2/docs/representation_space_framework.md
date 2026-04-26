# Representation-Space Analysis Framework for Backdoor Attacks

## Motivation

Most federated learning backdoor papers report only the Attack Success Rate (ASR) and visualize a t-SNE plot of final features. This gives a coarse, subjective view: "the attack works" or "it doesn't." It is insufficient for rigorously comparing different attack mechanisms.

Two attacks can achieve identical ASR while producing very different representation-space behavior:
- One may place triggered features in a tight cluster far from the genuine target-class distribution (easy to detect).
- Another may place them diffusely inside the genuine distribution (hard to detect).
- One may inject the backdoor in 5 rounds; another may take 50.
- One may exploit a single direction in feature space; another may distribute the signal across many dimensions.

Without a principled framework, these differences are invisible, and claims like "attack A is stronger than attack B" become unanchored.

This document defines a **4-axis, ~15-metric profile** for characterizing a backdoor attack's manifestation in the penultimate-layer feature space. It operates offline on saved features (no retraining required), uses only standard linear algebra and well-known statistical distances, and produces a compact profile per attack that can be compared on a radar chart or a table.

---

## Notation

- `f : X → ℝ^d` : the model's feature extractor (penultimate layer). For ResNet18 on GTSRB, `d = 512`.
- `x` : a clean input.
- `τ(x)` : the triggered version of `x` (e.g., pixel patch applied).
- `y(x)` : the true label of `x`.
- `c*` : the target class (in our experiments, class 2).
- `D_clean = {x | y(x) = c*}` : clean samples from the target class.
- `D_trig = {τ(x) | y(x) ≠ c*}` : triggered samples from non-target classes.
- `μ_C`, `Σ_C` : empirical mean and covariance of `f(D_clean)`.
- `μ_T`, `Σ_T` : empirical mean and covariance of `f(D_trig)`.
- `N_C`, `N_T` : sample counts in each group.

---

## Axis A — Injection Success

**Question:** *Did the attack achieve its objective (misclassification as `c*`)?*

### A1. Attack Success Rate (ASR)

$$
\text{ASR} = \frac{1}{N_T} \sum_{x \in D_{\text{trig}}} \mathbb{1}[\arg\max f_{\text{clf}}(\tau(x)) = c^*]
$$

where `f_clf` is the full model (feature extractor + linear head).

**Range:** [0, 1]. **Good attack:** close to 1.

### A2. Target-class margin (optional)

Mean logit margin `logit_{c*}(τ(x)) − max_{c ≠ c*} logit_c(τ(x))` over `D_trig`. Positive margin means the target class is preferred; larger margin means stronger preference.

**Interpretation:** ASR alone is binary at the sample level; margin gives a continuous view of "how confidently" the backdoor fires.

### A3 / A4. Target-logit mean on triggered vs genuine features

Using the trained classifier head `(W, b)`, compute `logit_{c*}(f) = W[c*] \cdot f + b[c*]` for both feature groups:

$$
\text{A3} = \text{mean}_{x \in D_{\text{trig}}}\; W[c^*] \cdot f(\tau(x)) + b[c^*]
$$
$$
\text{A4} = \text{mean}_{x \in D_{\text{clean}}}\; W[c^*] \cdot f(x) + b[c^*]
$$

**Interpretation.** A3 and A4 probe whether the classifier head treats triggered inputs *as if they were* genuine target-class samples, independent of where the features geometrically sit. When `A3 ≈ A4` despite a large `centroid_l2` (Axis B1), the evidence strongly favors **classifier-head corruption** — the head has been trained to assign the same (or comparable) target-class logit to a new, geometrically-distinct region of feature space.

### A5. Margin of triggered samples

$$
\text{A5} = \text{mean}_{x \in D_{\text{trig}}}\; \left( \text{logit}_{c^*}(f(\tau(x))) - \max_{j \neq c^*} \text{logit}_j(f(\tau(x))) \right)
$$

Continuous counterpart to ASR. Positive mean margin means triggered features cross the `c*` decision boundary; larger margin = the crossing is deep, not marginal.

*Metrics A3-A5 require the trained classifier head. They are computed only when `--load-head` is enabled in the CLI, which loads `final_model.pt` once per experiment and extracts `fc.weight`/`fc.bias`.*

---

## Axis B — Injection Geometry

**Question:** *Where in feature space did the attack place the triggered representations?*

This is the axis that distinguishes a well-designed attack (places features in a specific, controlled region) from a heuristic one (triggered features are scattered).

### B1. Centroid distance

$$
d_{\text{cen}} = \|\mu_T - \mu_C\|_2
$$

L2 distance between the triggered centroid and the genuine target-class centroid.

**Range:** [0, ∞). **Good attack:** small (attack reaches the target region).

### B2. Centroid cosine

$$
\cos_{\text{cen}} = \frac{\langle \mu_T, \mu_C \rangle}{\|\mu_T\|_2 \|\mu_C\|_2}
$$

Angular alignment. Complements B1 because L2 is sensitive to scale.

**Range:** [−1, 1]. **Good attack:** close to 1.

### B3. Concentration (compactness)

$$
\text{conc}_T = \frac{1}{d} \text{tr}(\Sigma_T) = \frac{1}{d} \sum_{k=1}^{d} \text{Var}(f_k(D_{\text{trig}}))
$$

Average per-dimension variance of triggered features. Low value means a tight cluster.

**Range:** [0, ∞). **Good (designed) attack:** small. **Heuristic attack:** often large.

### B4. Concentration ratio

$$
\text{conc}_{\text{ratio}} = \frac{\text{tr}(\Sigma_T)}{\text{tr}(\Sigma_C)}
$$

Triggered concentration normalized by the genuine target class concentration.

**Range:** [0, ∞). **Interpretation:**
- `< 1` : triggered features are *tighter* than the natural class (suspicious / detectable).
- `≈ 1` : triggered features are as spread as the natural class (stealthy).
- `> 1` : triggered features are more spread (weak / diffuse attack).

### B5. Shift direction rank

Let the per-sample shift vector be `δ(x) = f(τ(x)) − f(x)` for `x ∈ D_trig`. Stack into a matrix `Δ ∈ ℝ^{N_T × d}`. Compute the SVD `Δ = UΣV^T` and count singular values above a threshold (e.g., 1% of the largest):

$$
\text{rank}_{\text{eff}} = \big|\{i : \sigma_i(\Delta) > 0.01 \cdot \sigma_{\max}(\Delta)\}\big|
$$

**Range:** [1, d]. **Interpretation:**
- `rank_eff = 1` : the attack pushes all samples along a single direction in feature space (canonical "trigger detector" hypothesis).
- `rank_eff ≫ 1` : the attack uses a distributed representation.

### B6. Shift alignment with natural class direction

For each non-target class `c`, the "natural" inter-class direction is `μ_C − μ_c` where `μ_c` is the centroid of the source class. Let `v = mean shift vector across all triggered samples, restricted to source class c`. The alignment is:

$$
\text{align}_c = \frac{\langle v_c, \mu_C - \mu_c \rangle}{\|v_c\|_2 \|\mu_C - \mu_c\|_2}
$$

Average over all source classes weighted by sample count.

**Range:** [−1, 1]. **Interpretation:**
- High positive: the attack mimics the natural "source → target" direction (stealthy).
- Near zero: the attack creates a novel, orthogonal direction (detectable anomaly).

### B7. Source identity preservation

For each source class `s ≠ c*`, compute two per-class centroids — one in triggered features and one in clean features — from the same underlying samples:

$$
\mu_T^{(s)} = \text{mean}\{f(\tau(x)) : y(x) = s\}, \quad
\mu_C^{(s)} = \text{mean}\{f(x) : y(x) = s\}
$$

Measure how well the triggered representation preserves the *natural* per-class centroid spread by comparing the mean pairwise distances:

$$
\text{sip} = \frac{\frac{1}{\binom{|S|}{2}} \sum_{s < s'} \|\mu_T^{(s)} - \mu_T^{(s')}\|_2}{\frac{1}{\binom{|S|}{2}} \sum_{s < s'} \|\mu_C^{(s)} - \mu_C^{(s')}\|_2}
$$

The clean per-class centroid spread is the *reference scale* — it is the natural inter-class structure that the feature extractor has learned for non-target classes.

**Range:** [0, ∞). **Interpretation:**
- **≈ 1**: triggered features preserve the same per-source centroid structure as clean features. The attack is a **uniform shift** that moves each source class by roughly the same vector in feature space, so their relative positions are unchanged. Source-class identity is preserved through the transformation; the model still "knows" which source class produced the triggered sample. This is characteristic of a **weak / heuristic attack** (e.g., the pixel trigger).
- **→ 0**: triggered features collapse toward a single point; per-source centroids all coincide. The attack produces a **destination cluster** — triggered samples lose source identity and become interchangeable in feature space. This is characteristic of a **designed / optimization-based attack** that explicitly targets a specific feature-space region.
- **> 1**: triggered per-source centroids are *more* spread than their clean counterparts. Unusual; indicates the attack disrupts the feature extractor beyond the natural class structure.

The normalization against the clean per-class spread gives this metric a natural reference point: 1.0 is "no restructuring", and lower values correspond to stronger structural manipulation. The metric is independent of the feature extractor's absolute scale, which makes it comparable across models and training regimes.

---

## Axis C — Injection Stealth

**Question:** *Can a defender distinguish triggered representations from genuine target-class representations?*

> **Important clarification.** "Stealth" in this framework refers specifically to **representation-space indistinguishability**: can a defender with access to features tell triggered samples apart from genuine target-class samples? This is **not** the same as "the backdoor is hidden from the model's classifier head." A backdoor can have high ASR (the classifier head outputs the target class) while simultaneously having features that are trivially separable from genuine target-class features (high linear probe accuracy). Such an attack is **stealthy to the classifier-head decision** but **exposed to any supervised representation-space defense**. The framework's stealth metrics measure the latter.

### C1. Linear probe accuracy

Train a binary logistic regression classifier on the task
`{genuine class-c* features} ⟶ 0` vs `{triggered features} ⟶ 1`
with a balanced 80/20 train/test split. Report test accuracy.

**Range:** [0.5, 1]. **Interpretation:**
- `0.5` : triggered features are linearly indistinguishable from genuine features (maximum stealth).
- `1.0` : trivially separable by a linear classifier (attack is an anomaly).

### C2. Maximum Mean Discrepancy (MMD²)

With an RBF kernel `k(x, y) = exp(−γ‖x − y‖²)`:

$$
\text{MMD}^2 = \frac{1}{N_C^2} \sum_{i,j} k(c_i, c_j) + \frac{1}{N_T^2} \sum_{i,j} k(t_i, t_j) - \frac{2}{N_C N_T} \sum_{i,j} k(c_i, t_j)
$$

where `c_i ∈ f(D_clean)`, `t_i ∈ f(D_trig)`. We use the median heuristic for `γ`: `γ = 1 / (2 σ²)` where `σ` is the median pairwise distance.

**Range:** [0, ∞). **Interpretation:** `0` = identical distributions; larger = separable.

### C3. Wasserstein-2 distance (approximation)

Empirical Sinkhorn approximation of the 2-Wasserstein distance between the two feature distributions, on random 500-sample subsets:

$$
W_2(P_T, P_C) \approx \min_{\pi \in \Pi(P_T, P_C)} \left( \sum_{i,j} \pi_{ij} \|t_i - c_j\|_2^2 \right)^{1/2}
$$

solved via entropic regularization (Sinkhorn iterations).

**Range:** [0, ∞). **Interpretation:** same as MMD, but metric-space interpretation.

### C4. Spectral signature score

Project the concatenated `{genuine, triggered}` features onto the top principal component. Compute the normalized mean difference:

$$
s_{\text{spec}} = \frac{|\bar{p}_T - \bar{p}_C|}{\hat{\sigma}_p}
$$

where `p` is the top-PC projection and `σ̂_p` is its pooled standard deviation.

**Range:** [0, ∞). **Interpretation:** following Tran et al. 2018, high `s_spec` means triggered samples are outliers along the top-variance direction — the signature that spectral defenses detect.

### C5. Silhouette score

Silhouette coefficient of the two-cluster partition `{genuine, triggered}`.

**Range:** [−1, 1]. **Interpretation:**
- `≈ 1` : tight, well-separated clusters (attack is a clear anomaly in representation space).
- `≈ 0` : fully mixed distributions (attack blends into the target-class region).
- `< 0` : average pairwise distance *within* the triggered cluster exceeds average distance to the genuine cluster — the triggered group does not form a coherent cluster and overlaps heavily with genuine features (one interpretation of representation-space stealth; another is that the triggered features do not cluster among themselves at all).

Note: Silhouette is based on Euclidean distances between all points and does *not* search for a discriminative direction. A successful attack can have silhouette near 0 while still being trivially separable by a supervised linear probe (C1) because the separating direction may not align with the dominant variance axes that silhouette implicitly weighs.

### C6. Probe direction vs top-PC alignment

Let `w_{probe}` be the weight vector of the trained linear probe from C1 (the direction that best separates triggered from genuine features). Let `v_1` be the top principal component of the concatenated `{genuine, triggered}` features. Define:

$$
\text{pc-align} = \left| \frac{\langle w_{probe}, v_1 \rangle}{\|w_{probe}\|_2 \|v_1\|_2} \right|
$$

**Range:** [0, 1]. **Interpretation:**
- **Near 1**: the probe uses the same direction PCA finds — the spectral signature (C4) and the probe agree. Spectral-based defenses (Tran et al. 2018) would detect the attack.
- **Near 0**: the probe uses a direction **orthogonal** to the top principal component. The separating signal is hidden below the top variance; spectral defenses miss it; only supervised methods find it.

This metric is what quantitatively resolves the apparent paradox between high linear probe accuracy (C1) and low spectral signature (C4) that we observe on the pixel-trigger baseline: if `pc-align ≈ 0`, both measurements are self-consistent and reflect the same underlying structure — a separating direction that carries little raw variance.

---

## Axis D — Injection Dynamics

**Question:** *How does the attack form over federated training rounds?*

These metrics require the full trajectory `{ASR(r), μ_T(r), μ_C(r)}` across rounds `r ∈ {0, 5, 10, 25, 50, 75, 100}` (or finer granularity).

### D1. Time-to-ASR-90

$$
T_{90} = \min \{ r : \text{ASR}(r) \geq 0.90 \}
$$

Or `∞` if never reached. **Interpretation:** small = fast injection.

### D2. Early-round ASR snapshots

`ASR(5), ASR(10), ASR(25)`. **Interpretation:** high early ASR means the attack converges fast.

### D3. Late-round ASR stability

$$
\text{stab}_{\text{ASR}} = \text{std}\{\text{ASR}(r) : r \in [50, 100]\}
$$

**Interpretation:** low = stable (the backdoor is "locked in"); high = oscillating (clean learning is still competing with the attack).

### D4. Competition depth (U-shape depth)

Empirically we observed a U-shape in `cos_cen(r)`: the cosine similarity drops during early class formation (clean learning pulls features apart), then recovers as the backdoor settles.

$$
\text{depth}_{\text{U}} = \max_r \cos_{\text{cen}}(r) - \min_r \cos_{\text{cen}}(r)
$$

**Interpretation:** deep U means the attack had to "fight hard" to recover. Shallow U means the backdoor was never threatened by clean learning.

### D5. Recovery rate

Slope of `cos_cen(r)` from its minimum round to the final round:

$$
\text{recov} = \frac{\cos_{\text{cen}}(r_{\text{final}}) - \min_r \cos_{\text{cen}}(r)}{r_{\text{final}} - r_{\min}}
$$

**Interpretation:** fast recovery = resilient attack.

---

## Interpretation: What does a "good" (i.e., designed, powerful) attack look like?

| Axis | Metric | Good-attack value |
|------|--------|-------------------|
| A | ASR | high (→ 1) |
| A | A3/A4 target-logit gap | small (triggered logit ≈ genuine logit) |
| A | A5 triggered margin | large positive |
| B | Centroid distance | low |
| B | Centroid cosine | high (→ 1) |
| B | Concentration | low (tight) OR ≈ genuine class (stealthy) |
| B | Shift rank | low (1-3), if designed; variable if heuristic |
| B | Shift alignment | high (exploits natural direction) |
| B | Source identity preservation (B7) | **low** (destination cluster, not per-source nudge) |
| C | Linear probe accuracy | near 0.5 (indistinguishable) |
| C | MMD² | small |
| C | Wasserstein-2 | small |
| C | Spectral score | small |
| C | Silhouette | ≤ 0 (mixed) |
| C | Probe–PC alignment (C6) | any value is informative; near 1 means spectral defenses would also catch it |
| D | Time-to-ASR-90 | small |
| D | ASR stability | small |
| D | U-shape depth | small (attack not threatened) |
| D | Recovery rate | high |

Note the **tension between B and C**: a "concentrated" attack (B3 low) is usually *easier to detect* (C high) if the compact cluster sits outside the genuine target-class distribution. A truly strong attack balances the two — the triggered cluster should be **inside** the genuine class distribution, which simultaneously gives low C1/C2/C3 (stealth) and arbitrary B3 (concentration).

---

## Comparison methodology

### Radar chart (visual summary)

Select 6–8 normalized metrics across all four axes, rescale each to [0, 1] with 1 = "better attack" (invert stealth metrics so that low detection = 1). Plot as a radar chart. Two attacks appearing as different polygon shapes give immediate visual comparison.

### Profile table (for thesis)

Tabulate all ~15 metrics × all attacks in one CSV. For each row (metric), highlight the best value.

### Statistical significance

When comparing two attacks quantitatively, use bootstrap confidence intervals for each metric (resample `D_trig` 1000 times). Report `mean ± 95% CI`.

---

## Applicability to Phase D and beyond

This framework is designed to be **attack-agnostic** — it requires only:
1. Access to the trained model's feature extractor (any model architecture with a `forward_features` method).
2. Clean and triggered versions of the same test samples.
3. Optional: checkpoints at multiple rounds for Axis D.

It therefore applies directly to:
- Pixel-trigger backdoors (current baseline)
- Model replacement attacks (Phase D)
- Distributed Backdoor Attacks (Phase D)
- Optimization-based feature-space attacks (Phase D+)
- Future attacks on multimodal architectures (thesis direction)

The framework's value is not in any single metric but in the **profile shape**: two attacks with equal ASR but different geometry, stealth, and dynamics are clearly different objects, and this framework makes those differences measurable.

---

## Limitations

1. **Feature-space only.** The framework measures representation-space behavior; it does not assess input-space properties (trigger visibility, perceptual similarity to clean inputs).
2. **Global centroids.** Many metrics summarize `D_trig` by its centroid/covariance. Multimodal or clustered triggered distributions are only partially captured by global centroids; the **B7 source-identity-preservation** metric addresses this by explicitly measuring per-source-class centroid spread.
3. **Linear probe is a lower bound on detectability.** A nonlinear detector might separate features that a linear probe cannot. We use linear probe for efficiency and interpretability.
4. **Finite-sample effects.** MMD and Wasserstein estimates are noisy for small `N_T` (e.g., at round 0 when the model is random). Report confidence intervals when sample size is small.
5. **Correlated with ASR.** Several metrics (especially A1 and parts of B) are correlated with ASR by construction. The value of the framework is in the metrics that are *not* correlated with ASR — those reveal structure that ASR alone cannot see.
6. **Probe-based defenses require labels.** The linear probe (C1) and the probe-PC alignment (C6) require labeled clean and triggered features to train. In practice a defender does not have labeled triggered data at test time. These metrics should therefore be read as **upper bounds on the stealth of the attack against a hypothetical defender who knows the trigger** — useful for comparing attacks, but not directly usable as a defense mechanism without further design.

---

## Corrected Interpretation: The Joint Weak Attack Pattern

An initial reading of the framework metrics on the pixel-trigger baseline tempted us to a simpler hypothesis: *successful backdoor attacks place triggered features inside the genuine target-class cluster, producing low linear-probe accuracy (representation-space stealth) as a byproduct of attack success.* The data on the pixel-trigger baseline **refutes** that hypothesis. Across all successful configurations (ASR ≥ 0.86), the linear-probe accuracy is 0.99–1.00 — triggered and genuine features are trivially separable in 512-dim. The corrected interpretation is the subject of this section.

### What the data actually shows

Three observations must be reconciled:
1. **ASR is high** (0.86–0.97): the model's classifier head labels triggered inputs as the target class with high confidence.
2. **Linear probe accuracy is high** (≈ 1.0): a supervised binary classifier can distinguish triggered features from genuine target-class features.
3. **Centroid cosine is high** (0.88–0.97) but centroid L2 distance is non-trivial (1.3–3.1): the triggered centroid points in roughly the same direction as the target-class centroid but at a substantial linear distance.

Observations 1 and 2 jointly imply that the triggered features do **not** lie inside the genuine target-class distribution. Observation 3 implies that they do not lie on the source side either. They occupy a **third region** of feature space — geometrically distinct from both, yet angularly aligned with the target direction.

### Mechanistic interpretation

The pixel-trigger backdoor is a **joint weak attack on two components**:

1. **Feature extractor partial shift.** The poisoned training data teaches the convolutional backbone to respond to the 4×4 trigger patch. Triggered features move toward the target class, but the shift is incomplete: the triggered centroid is ~25-30% of the way along the `source → target` direction and stops. The shift is distributed across 25-35 effective dimensions (B5) with 75-90% energy in the top 3 components — not a single "trigger detector" but a correlated ensemble of mid-level filter responses.

2. **Classifier head decision-region extension.** Simultaneously, the linear classifier head `w_2` is trained on poisoned examples where inputs from *other* classes are labeled as the target. The head accommodates this by extending its class-`c*` decision polytope to cover the new, geometrically distinct "middle region" that the feature extractor's partial shift produces. This is not memorization of specific samples — the extension generalizes because the shift is systematic (driven by the same trigger patch).

Neither modification, **individually**, would succeed. A head-only attack cannot label a region it was never trained on; a feature-only attack without head corruption would not move features far enough to reach the existing `c*` region. The backdoor works because the two components meet **halfway**, exchanging responsibility: the feature extractor provides 25-30% of the movement, and the head provides the rest by redrawing the decision boundary.

### Consequences for defense design

The joint weak attack pattern has direct implications for defense design:

1. **Spectral defenses (Tran et al. 2018) fail on this attack class.** The separating direction between triggered and genuine features is *not* the top variance direction (C6 metric ≈ 0 on the baseline). Spectral signatures look in the wrong place.
2. **Supervised probes succeed but require labeled triggered data.** The probe accuracy of 1.00 is a theoretical ceiling, not a deployable defense. A defender without trigger knowledge cannot train such a probe directly.
3. **Server-side unsupervised aggregation defenses are fundamentally blind to this attack if the malicious updates look like normal gradient noise.** Krum/FedMedian only help when the poisoned updates are geometric outliers in parameter space; the pixel trigger can be tuned so that per-round updates remain within the benign envelope.
4. **Client-side defenses with access to (unlabeled) local data** are the natural path forward. A client running local test-time adaptation or activation inspection over its own labeled data can detect the split between genuine and triggered representations that the linear probe finds. This is the basis of the thesis direction.

### What Phase D attacks must change

For an attack to be meaningfully "stronger" than the pixel-trigger baseline **in representation space**, it must lower one or more of the following:

- `linear_probe_acc`: target **< 0.8**, ideally ≤ 0.6. This requires pushing triggered features *into* the genuine class distribution, not a separate middle region.
- `concentration_ratio`: target **closer to 1.0** (matching natural class variance). The current baseline is 2–4×.
- `source_identity_preservation` (B7): target **low** (destination clustering), not the current high-value pattern that reveals per-source identity.
- `centroid_l2`: target **small** (within 1–2 units), not the current 2–3+ units.
- `probe_pc_alignment` (C6): either near 0 (probe still finds it — baseline-like) or, ideally, the whole separating direction disappears as the attack becomes indistinguishable even to supervised methods.

An attack that holds ASR constant while moving any of these metrics in the right direction is a genuine advance over the pixel-trigger heuristic, not just an ASR improvement. This is the bar that Phase D (model replacement, DBA, optimization-based attacks) needs to clear.

---

## Addendum — Methodology Update (2026-04-17)

After Cycle 01 closed, a methodology audit surfaced five sampling and reproducibility issues in the Axis C implementation that affect how the stealth metrics should be read. The underlying mathematical definitions in the main body of this document are unchanged. Only the evaluation protocol was corrected. This section records the change and points at the errata sections in the closed profile docs for the corresponding numerical revisions.

### 1. Axis C implementation replaced in place

The original `compute_axis_c` balanced the linear probe by subsampling `n_min = min(len(feat_T), len(feat_C)) = 750`, which discarded ~94% of the available triggered features (~11,130 of 11,880 on the GTSRB test set) and produced a 300-sample test set whose resolution was ≈0.33 pp — below the reported precision of the "1.000" headline values. The C6 `probe_pc_alignment` metric then compared the probe direction (trained on balanced 750/750) against a top-PC computed on the imbalanced full pool, an asymmetric comparison that gave the impression "spectral defenses miss what the probe finds" with higher confidence than the data justified.

The function now:
- Trains the linear probe on **all** features via `Pipeline(StandardScaler, LogisticRegression(class_weight='balanced'))` and reports **5-fold stratified CV balanced accuracy and AUROC** — two metrics that do not inflate under 94%/6% class imbalance. Raw accuracy is intentionally **not** reported; at this imbalance, a constant-negative predictor scores 94% accuracy, making the raw number uninterpretable.
- Emits `linear_probe_balanced_acc_mean`, `linear_probe_balanced_acc_std`, `linear_probe_auroc_mean`, `linear_probe_auroc_std`.

### 2. Three variants for C4 and C6

Because the scientific question — "does a spectral defender see the same direction the probe finds?" — is sensitive to whose data you give each side, we now report three variants of C6 and two of C4:

- **`..._imbalanced`**: PCA on the full 12,630-sample pool. Reproduces the Cycle-01 numbers for continuity.
- **`..._balanced`** (headline going forward): PCA on a stratified 750/750 subsample that matches the probe's effective sampling regime. This is the defender-realistic number.
- **`..._weighted`** (C6 only): PCA on all 12,630 points with per-sample weight `w_i = 1/n_{class(i)}` so each class contributes equally to the covariance. Answers "what would a rebalanced spectral defender find?".

All three variants use the same final-fit probe direction, projected into the same standardized feature space as the PCA input. Sign ambiguity is handled by `abs(⟨w, v⟩)`.

### 3. Class-ratio dependency of C2, C3, C5 (documented, not changed)

MMD² (C2), Wasserstein-2 (C3), and silhouette (C5) are mathematically correct per their definitions but are evaluated on the natural GTSRB test-set class ratio (~1:16 for genuine target vs triggered non-target). Absolute values are **not directly comparable to studies with different class distributions**. Within-study rankings across our Cycle-01 experiments remain valid because every run uses the same test set. A reader comparing our numbers to another paper's should re-derive with the other paper's class ratio before drawing conclusions.

### 4. Independent RNG streams (reproducibility)

The original implementation shared a single `np.random.RandomState(seed)` across C1/C2/C3/C5 subsampling, meaning any change to C1's subsample implicitly shifted C2/C3/C5's subsamples. The new implementation uses `RandomState(seed + k)` per metric (C2 → seed+1, C3 → seed+2, C5 → seed+3) so that the metrics are decoupled. The `_median_heuristic_gamma` and `_sinkhorn_w2` helpers now accept a `seed` parameter instead of hard-coding 42.

### 5. Analysis seed vs training seed

Training uses seed 42 throughout the pipeline. The analysis runners (`analysis/run_framework.sh`, `analysis/run_phaseD_framework.sh`) now pass `--seed 4242` to `framework_metrics` so that PCA, probe, and subsampling randomness is decorrelated from the training-data partition seed. The default CLI seed in `framework_metrics.main()` remains 42 for historical reproducibility; the runners explicitly override.

### Where the new numbers live

- The framework JSON profiles at `/mimer/.../phaseC_v2/figures/framework/profiles/*_profile.json` and `.../phaseD/figures/framework/profiles/*_profile.json` are regenerated in place with the new field names (`linear_probe_balanced_acc_mean`, `probe_pc_alignment_{imbalanced,balanced,weighted}`, `spectral_score_{imbalanced,balanced}`).
- The side-by-side errata tables are in [`pixel_trigger_baseline.md`](pixel_trigger_baseline.md) and [`model_replacement_profile.md`](model_replacement_profile.md) under `## Errata — Axis C re-evaluation (2026-04-17)`.
- The v1 tables in the main body of those closed profile docs are **unchanged** per the cycle-doc convention that closed documents are never edited in place.

