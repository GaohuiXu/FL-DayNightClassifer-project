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

---

## Axis C — Injection Stealth

**Question:** *Can a defender distinguish triggered representations from genuine target-class representations?*

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
- `≈ 1` : tight, well-separated clusters (attack is a clear anomaly).
- `≈ 0` : fully mixed (attack is stealthy).
- `< 0` : triggered samples are closer to genuine cluster on average than to each other — the attack has successfully blended in.

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
| B | Centroid distance | low |
| B | Centroid cosine | high (→ 1) |
| B | Concentration | low (tight) OR ≈ genuine class (stealthy) |
| B | Shift rank | low (1-3), if designed; variable if heuristic |
| B | Shift alignment | high (exploits natural direction) |
| C | Linear probe accuracy | near 0.5 (indistinguishable) |
| C | MMD² | small |
| C | Wasserstein-2 | small |
| C | Spectral score | small |
| C | Silhouette | ≤ 0 (mixed) |
| D | Time-to-ASR-90 | small |
| D | ASR stability | small |
| D | U-shape depth | small (attack not threatened) |
| D | Recovery rate | high |

Note the **tension between B and C**: a "concentrated" attack (B) is usually *easier to detect* (C). A truly strong attack balances the two — tight enough to reliably fire on triggers, diffuse enough to blend into the genuine class distribution.

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
2. **Global centroids.** Many metrics summarize `D_trig` by its centroid/covariance. Multimodal or clustered triggered distributions are not fully captured. Future extension: per-source-class breakdowns.
3. **Linear probe is a lower bound on detectability.** A nonlinear detector might separate features that a linear probe cannot. We use linear probe for efficiency and interpretability.
4. **Finite-sample effects.** MMD and Wasserstein estimates are noisy for small `N_T` (e.g., at round 0 when the model is random). Report confidence intervals when sample size is small.
5. **Correlated with ASR.** Several metrics (especially A1 and parts of B) are correlated with ASR by construction. The value of the framework is in the metrics that are *not* correlated with ASR — those reveal structure that ASR alone cannot see.
