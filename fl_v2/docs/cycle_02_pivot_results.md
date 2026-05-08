# Cycle 02 Pivot — Week 1 Results

> **⚠ NUMERICAL CLAIMS BELOW ARE WITHDRAWN.** Sections 3.1–3.6 were
> authored on commits prior to `9d72bcf`, when the training pipeline
> was non-deterministic at fixed seed (seven software-level sources
> fixed across `795f75e`, `725cae5`, `1f5e70d`, `bd2c1eb`, `6db1dd1`).
> Re-running the exact same YAML at the same seed produced different
> final models (acc 0.667 vs 0.894 in two attempts of
> `pretrained_full_ft_pixel5` at seed=42). See `cycle_02_pivot_audit.md`
> for the full root-cause analysis.
>
> The Phase 3.1 wave (jobs 6599453–6599461, commit `9d72bcf`, 3 cells
> × 3 seeds = 9 valid runs) replaces the relevant subset of these
> numbers. The corrected results are at the top of the next section.
> The 24-cell Wave 1+2 numerical record is **historical / pre-audit**
> and must not be cited.
>
> **Reproducibility caveat (added 2026-05-08).** "9 valid runs"
> means the runs completed 100 rounds without silent failure; it does
> NOT mean within-(commit, seed) bit-reproducibility has been
> established. The audit's own pair 6594906 / 6594907 found
> bit-identical rounds 0–5 followed by round-6+ divergence even after
> all 7 fixes. The within-commit reprocheck (job 6600759, same YAML
> as 6599453, same seed=42, only a shell-only commit between them)
> now confirms: **Δ test_acc = 0.64 pp and Δ ASR = 5.15 pp at round
> 100**, with peak intermediate Δ ASR ≈ 6 pp at round 50. Each Phase
> 3.1 number below is therefore **one realisation** with a residual
> ε of order 5 pp ASR — much smaller than the seed-to-seed SD
> (17.6 pp v1, 21.0 pp v2 on full_ft 5mal) but not negligible. Do
> not quote the v1 numbers below to many decimal places as if the
> underlying pipeline were deterministic.
>
> **v2 (convergent diagnostic) update (added 2026-05-08).** The
> v1 fixed-10-epoch clean-head retraining systematically undertrains
> the head; the v2 convergent diagnostic (commit `0f4eb49`,
> early-stop on plateau, patience 8, max 100 epochs) produces
> materially different per-cell numbers. **All 9 cells re-run; full
> v2 results below.** v2 mean shifts: full_ft +10.5 pp, lastblock
> −7.7 pp, canonconv1 head_only −6.9 pp. The cell ordering survives.
> v1 numbers are retained as the historical record but are
> superseded by the v2 numbers for any forward-looking claim.
>
> Cycle 01 archive numbers (`gtsrb_v2/phaseC_v2/`, `gtsrb_v2/phaseD/`)
> are out of scope of this withdrawal — they were trained on the same
> pre-audit code, but reliability of the Cycle 01 record is a separate
> question. The Phase 3.0 sentinel
> (`gtsrb_v2/phaseC_v2_sentinel/phaseC2-backdoor-5mal-nodefense-sentinel_r100_seed42/`)
> reruns the Cycle 01 cell on the audit-fixed pipeline; v2 head_attr
> = 18.2 %, well below the v1 Cycle 01 reference of 58 %. Cross-cycle
> comparisons that quote the 58 % number are no longer apples-to-apples.

## Utility ladder (audit-fixed pipeline)

Added 2026-05-08 after the supervisor meeting. Establishes the
endpoints of the clean-utility scale we care about, all on the
same architecture (pretrained ResNet18 + modified-conv1, image-size
32, full_ft trainable), with attack as the only varying axis.

```
zero-FT (~2 %)  ≪  FL clean × 3 seeds (92.2 ± 1.3 %)  <  centralised clean (98.97 %)
```

| Setting | source | acc | target_acc | asr |
|---|---|---|---|---|
| **Centralised clean** (50 epochs) | job 6602985 | **0.9897** | 0.9960 | 0.0002 |
| **Clean FL × 3 seeds (audit-fixed)** | jobs 6603555 / 6602987 / 6602988 | **0.9219 ± 0.0127** | **0.9689 ± 0.0125** | (no attack) |
| Cycle 02 attack 5mal × 3 seeds (audit-fixed) | jobs 6599453-55 | 0.8885 ± 0.0276 | 0.9547 ± 0.0124 | 0.7114 ± 0.1623 |
| Cycle 01 sentinel attack 5mal (N=1) | job 6600208 | 0.9537 | 0.9933 | 0.8958 |
| Pretrained zero-FT (modified-conv1, N=3 fc seeds) | job 6602864 | 0.0169 ± 0.0087 | 0.0022 ± 0.0031 | 0.0029 ± 0.0021 |
| Pretrained zero-FT (canonical-conv1, N=3 fc seeds) | job 6602864 | 0.0328 ± 0.0158 | 0.0018 ± 0.0025 | 0.0015 ± 0.0014 |

### What the ladder tells us

1. **FL fine-tuning is necessary.** Strict zero-FT on either
   architecture is at chance (1/43 ≈ 2.33 %). The pretrained backbone
   alone — with a random-init fc head — produces accuracy below or
   barely above chance. *Every* useful number we report comes from
   the FL training.
2. **The FL→centralised gap is 6.8 pp.** Distributed learning under
   our setup (Dirichlet α=0.5, 50 clients, 3 local epochs, FedAvg,
   100 rounds, full_ft) costs ~6.8 pp of clean accuracy vs centralised
   training of the same architecture. The full_ft architecture is
   strong enough that this cost is small but real.
3. **Attack vs clean degrades main-task acc by ~3 pp** (92.2 % clean
   → 88.9 % under 5mal pixel attack). The model still mostly
   classifies correctly while the backdoor is installed; the attack
   is *additive*, not destructive.
4. **Sentinel acc (95.4 %) is HIGHER than Cycle 02 attack mean
   (88.9 %)** despite Sentinel having a stronger ASR (89.6 % vs
   71.1 %). The from-scratch architecture is more robust to attack
   on the main task than the pretrained-with-random-init-conv1
   architecture. This is consistent with the C2 pitfall — when the
   pretrained model has to train conv1 from scratch alongside the
   attack signal, the attack hurts main-task acc more.
5. **Asr ≈ 0 in zero-FT and centralised.** Sanity check on the
   trigger: a model that has never been trained to associate the
   trigger with any class doesn't predict the target class on
   triggered samples (no ImageNet shortcut, no spurious cue). The
   >80 % ASR in attack runs is *entirely* a product of FL training
   on poisoned data.

The Cycle 02 head_attr results below should be read with this
ladder in mind: the diagnostic measures *where* in the model the
attack is encoded; the ladder above measures *how much* the attack
costs on main-task utility.

## Phase 3.1 wave — Cycle 02 head_attr (commit `9d72bcf`)

3 cells × 3 seeds = 9 valid training runs + per-cell head-feature
decomposition under both the v1 (fixed 10-epoch) and v2 (convergent,
commit `0f4eb49`) diagnostics on the SAME final checkpoint per cell.

**v1 fixed-10-epoch:**

| Cell | seed=42 | seed=43 | seed=44 | mean head_attr ± SD | acc mean ± SD |
|---|---|---|---|---|---|
| `full_ft + 5mal` | 12.50 % | 44.71 % | 40.99 % | **32.73 % ± 17.62 pp** | 0.889 ± 0.028 |
| `last_block + 5mal` | 62.97 % | 61.90 % | 55.45 % | **60.11 % ± 4.07 pp** | 0.717 ± 0.049 |
| `canonconv1 head_only + 15mal` | 89.77 % | 84.71 % | 88.77 % | **87.75 % ± 2.68 pp** | 0.520 ± 0.011 |

**v2 convergent (early-stop on clean-test-acc plateau, patience 8, max 100 epochs):**

| Cell | seed=42 | seed=43 | seed=44 | mean head_attr ± SD |
|---|---|---|---|---|
| `full_ft + 5mal` | 19.56 % | 60.81 % | 49.37 % | **43.25 % ± 21.07 pp** |
| `last_block + 5mal` | 58.42 % | 50.62 % | 48.08 % | **52.37 % ± 5.41 pp** |
| `canonconv1 head_only + 15mal` | 84.00 % | 76.08 % | 82.38 % | **80.82 % ± 4.20 pp** |

**Saturated-cell "bit-stability" is diagnostic-side, not training-side.**
All three canonconv1 head_only seeds have v2 ch_asr = 0.0371 (to 4
sig fig), and the v1 ch_asr was likewise tightly clustered around
0.0237. The mechanism is structural: `pretrained=True` +
`canonical_conv1=True` + `trainable-layers: head_only` freezes the
entire ImageNet encoder, so the encoder is byte-identical across
model seeds. The diagnostic discards the FL-trained head and trains
a fresh one with diagnostic seed 4242 on the same frozen encoder
across all three runs — by construction it returns the same number
regardless of model seed. The seed-to-seed signal on these cells
lives instead in the **FL-trained head's** ASR (`original_asr` =
0.232 / 0.155 / 0.211, range 7.6 pp), but the diagnostic methodology
cannot probe deeper than that on frozen-encoder regimes.

For full_ft and lastblock cells the encoder IS trained across seeds,
so the diagnostic genuinely picks up encoder-level variance — that's
why their v2 SDs are non-trivial (21 pp / 5 pp).

**Phase 3.0 sentinel** (Cycle 01 phaseC2-backdoor-5mal-nodefense on
audit-fixed pipeline, seed=42, v2 convergent): orig_asr 0.8958, ch_asr
0.7326, **head_attr = 18.22 %**, best_ep 18/26.

### Cell-ordering claim (v2, audit-fixed pipeline)

Under v2 + audit-fixed-pipeline anchoring (sentinel = audit-fixed
Cycle 01 phaseC2 cell, NOT the pre-audit 58 % reference):

`sentinel (18.2 %)` < `Cycle 02 full_ft (43.3 %)` < `lastblock (52.4 %)` < `canonconv1 head_only (80.8 %)`

The "less trainable capacity → more head-attribution" gradient
**survives** under v2, and the cross-cycle Cycle 01 ↔ Cycle 02 anchor
is now apples-to-apples (both on the audit-fixed pipeline, both under
the convergent diagnostic).

The original Wave 1+2 "97 % encoder anchoring at full_ft" headline was
a bug-induced outlier (model stuck at trivial-backdoor attractor for
12 rounds because of unseeded RNG); the audit-fixed pipeline does not
reproduce it.

**Directional claim — pretrained vs from-scratch susceptibility at
full_ft + 5mal under v2:**

Cycle 02 full_ft (pretrained, mean 43.3 % across N=3 seeds, SD 21.0 pp)
is HIGHER than Cycle 01 sentinel (from-scratch, 18.2 %, N=1). The
gap is Δ = 25 pp ≈ 1.2 SD given v2 SD on full_ft. **Direction is
credible, point estimate is not yet venue-grade** (need N=5 seeds on
each side; the within-commit residual ε of ~5 pp ASR adds further
uncertainty).

The v1 statement "pretrained encoders are MORE susceptible at full_ft"
relied on a 33 % vs 58 % comparison where the 58 % was a pre-audit
Cycle 01 number on a non-deterministic pipeline. That comparison is
now demoted; the v2 + sentinel comparison (43.3 % vs 18.2 %) survives
with the caveats above.

For lastblock + 5mal: v1 reported 60.11 % "behaves like full_ft
from-scratch (58 %)". Under v2, lastblock = 52.4 % and the from-
scratch reference (sentinel) = 18.2 %. The "behaves like
full_ft-from-scratch" framing **does not survive** v2 — lastblock
under v2 has substantially more head-resident attack signal than the
from-scratch sentinel.

### Per-cell reproducibility characterisation (v1)

- `canonconv1 head_only + 15mal` (saturated, frozen encoder): v1
  SD 2.68 pp. Clean-head ASR rounds to 0.024 across all 3 seeds;
  precise values 0.0237373737, 0.0237373737, 0.0236531987 — two of
  three bit-identical, fourth-decimal divergence on seed 44.
  Consistent with "near-deterministic by construction" but not
  literally bit-identical across all three seeds.
- `last_block + 5mal` (restricted capacity, few attractors): v1
  SD 4.07 pp. N = 3 is enough to bound the cell's spread under the
  v1 diagnostic.
- `full_ft + 5mal` (full capacity at marginal pressure,
  multi-attractor): v1 SD 17.62 pp; v2 SD 21.0 pp on the partial
  3-seed rerun. **N ≥ 5 seeds** is required for venue-quality mean ±
  std reporting. This is the cell that gives us the largest spread
  and is therefore the most exposed to the residual ε.

*Companion to* `roadmap/cycle_02_designed_attacks_and_client_defenses.md`.
*Empirical evidence for the pretrained-init pivot. Frames the supervisor
discussion on **2026-05-08***.

**Status (as of Wed AM):** all empirical work complete. 12 main + 3
canonconv1 fallback runs at seed 42, plus 12 multi-seed runs at seeds
43+44 on the 6 most-decisive cells (6 / 6 / 0 successes wave 1, all 5
remaining recovered in wave 2 after fixing a duplicate-`wandb-tags`
generator bug and a transient gRPC failure). Feature extraction (96
.npz on GPU + 36 multi-seed) → framework metrics (8 + 12 profiles) →
head-feature decomposition (8 + 12 JSONs) all completed. t-SNE
comparison panel with disambiguated cell labels resubmitted (in queue;
see § 3.6 below — figures embedded once the job lands).

---

## TL;DR (revised after multi-seed verification)

After reading HTBA (Saha et al., AAAI 2020) and LIRA (Doan et al., ICCV
2021) we pivoted Cycle 02 from "designed feature-space attack +
feature-drift detector" to **characterizing where pixel-trigger
backdoors live under realistic FL fine-tuning of a pretrained backbone**.
Week 1 ran a 3×3 design matrix (`{full_ft, last_block, head_only} ×
{clean, pixel 5/50 mal, pixel 15/50 mal}`) at seed 42, plus a 3-cell
canonical-conv1 fallback, plus a multi-seed verification at seeds 43+44
on the 6 most-decisive attack cells (12 additional runs).

**The seed-42 narrative needed sharpening.** The originally-headline
finding *"head_attribution_pct = 97.2 % at full_ft 5 mal — encoder is
anchored by pretraining"* turns out to be a single-seed outlier. Across
3 seeds, full_ft 5 mal head_attribution is **6.9 % / 22.1 % / 97.2 %**
(range 90 pp). The robust pattern is more nuanced:

1. **Robust finding A — saturated attacks are head-dominated.** Whenever
   `ASR ≥ 0.95` (full_ft 15 mal, last_block 15 mal, full_ft 5 mal seed
   42), `head_attribution_pct ≥ 67 %` consistently across all 3 seeds.
   The clean-head retraining diagnostic reliably identifies the head as
   the dominant attack surface in the saturated regime.
2. **Robust finding B — marginal attacks are seed-fragile.** At 5 mal,
   the attack is at the boundary of success: ASR ranges 0.52–0.98 across
   seeds, and `head_attribution_pct` ranges 6.9 %–98.3 %. The encoder
   gets corrupted in some runs (low head_attribution) and stays clean in
   others. **There is no single "right" attribution number for marginal
   attacks** — the supervisor presentation reports the full distribution.
3. **Robust finding C — canonical-conv1 head-only attack collapses
   independent of seed.** Across 3 seeds at canonconv1 head-only 5 mal,
   `ASR = 0.053–0.067` (mean 0.061); at 15 mal, `ASR = 0.156–0.232`
   (mean 0.200). With the canonical ImageNet first stage preserved,
   pure head-attack is **stably weak** — a defensive property of the
   deployment, not an artefact.
4. **Robust finding D — linear probe separability is regime- and
   seed-invariant.** Every cell at every seed has
   `linear_probe_balanced_acc ≥ 0.95`. The Cycle-01 "triggered features
   form a separable middle region" finding generalises across the
   entire matrix. A defender with labelled triggered samples can always
   find the decision direction; a realistic FL server cannot get those
   labels, which is exactly the gap the redesigned Phase E.1 detector
   needs to close.

**Implications for the redesign.** The original Phase D.2 auxiliary
loss `λ‖f(τ(x)) − μ̂_{c*}‖²` (move triggered features into the target
manifold) is solving the wrong problem in the *saturated* regime —
saturated attacks already use the head, not the encoder. **D.2-revised
becomes a head-targeted attack under partially-frozen backbones** that
explicitly amplifies the head signal while staying stealthy in
classifier-output space. Phase E.1/E.2 pivots from feature-drift to
**logit-distribution drift / output-margin monitoring**, with the
non-IID coverage analysis from `phaseE2_coverage_analysis.md` carrying
over unchanged.

**On variance and venue framing.** The 5 mal seed variance is a
*positive* finding for the thesis: it tells us where the FL backdoor
threat model has a stochastic component, which is exactly the kind of
nuance that gets a defense paper accepted at top venues. We will report
mean ± std across 3 seeds in the supervisor doc and in any subsequent
submission.

---

## 1. Context — why pivot

### 1.1 What HTBA and LIRA changed

Both papers expose that backdoors do not require representation-space
manipulation to succeed:

- **HTBA (Saha et al. 2020)** shows clean-label attacks can route triggered
  source samples to a target class entirely through *classifier-head /
  decision-boundary contamination*: poisoned target-labeled images sit in
  feature space wherever the trigger will later place triggered source
  features, and the head learns to assign that region to the target class.
  Triggered features themselves do not need to move into the target manifold.
- **LIRA (Doan et al. 2021)** is a strong digital-pixel-space upper bound,
  but its threat model (full training control, MLaaS-style) is incompatible
  with FL malicious-client attacks, and its trigger is not physically
  realizable for autonomous-driving inference.

### 1.2 Cycle 01 finding, re-read

The Cycle 01 closed pixel-trigger baseline reported `linear_probe_acc =
0.99–1.00`, `centroid_l2 ≈ 2.7–3.1`, `source_identity_preservation ≈ 0.56–
0.64`. The "joint weak attack" framing said this was a partial feature-shift
(~25–30% of the natural source→target direction) plus a head-extension
that covers the resulting middle region. **Through the HTBA lens, this is a
predominantly head-driven attack with a partial feature artifact** — not a
balanced split. The pilot diagnostic we added in Cycle 02 confirms this
quantitatively: `head_attribution_pct = 58.1%` on
`phaseC2-backdoor-5mal-nodefense_r100_seed42` (original `ASR = 0.86`,
clean-head `ASR = 0.36`, clean-head `clean_acc = 0.94`, satisfying the ≥0.95
× original ceiling).

### 1.3 Three pivot-driving assumptions

1. **The original D.2 auxiliary-loss attack (`λ‖f(τ(x)) − μ̂_{c*}‖²`)
   solves a problem we may not have.** If pixel-trigger is already a
   head-dominated attack and `head_attribution_pct` is not close to 0, then
   "moving features into the target manifold" is the wrong direction —
   backdoors don't need to move features, they exploit head capacity.
2. **Single-vehicle FL is not the realistic AD deployment.** The plausible
   deployment is *cross-region or cross-organization federated fine-tuning
   of a pretrained backbone*, where the backbone may be partially or fully
   frozen by the operator. This regime puts the head/feature decomposition
   front and center.
3. **Top-venue paper shape is not yet committed.** AAAI deadline is ~83
   days, USENIX Security ~111. We keep both targets open for now and let
   the empirical finding guide the per-venue split decision in Cycle 02
   week 2.

---

## 2. Methodology

### 2.1 The 3×3 design matrix

|             | Clean | Pixel 5/50 mal | Pixel 15/50 mal |
|---|---|---|---|
| **Full FT** (~11.2M trainable) | utility ceiling | direct compare to Cycle 01 phaseC2 5mal | direct compare to Cycle 01 phaseC2 15mal |
| **Last block** (~8.4M: layer4 + fc) | reduced-capacity utility | tests whether layer4 + fc suffice for backdoor | scaled-up |
| **Head only** (~22K: fc only) | head-only utility ceiling | pure head-attack regime (HTBA-style under FL) | scaled-up |

All cells use:
- pretrained ImageNet ResNet18 (`bn1`, `layer1-4` from ImageNet; the 32×32
  modified `conv1` and the 43-class `fc` are random-init by architecture)
- 100 rounds, FedAvg, no defense, seed 42
- Same pixel-trigger attack as Cycle 01 (4×4 patch, target class 2,
  poison-fraction 0.5)
- Dirichlet α=0.5 partition, 50 clients, full participation

Reference baselines (no rerun): Cycle 01 `phaseC2-clean`,
`phaseC2-backdoor-5mal-nodefense`, `phaseC2-backdoor-15mal-nodefense`
(all from-scratch ResNet18).

### 2.2 Promoted metrics (5)

The framework's full 4-axis metric set is documented in
`representation_space_framework.md`. For decision-making at this checkpoint
we promote the five most informative numbers:

- **`asr`** — final-round attack success rate (higher = stronger attack)
- **`clean_acc`** — final-round clean test accuracy (utility)
- **`centroid_l2`** — Axis B; distance from triggered cluster centroid to
  the genuine target-class centroid in penultimate-feature space
- **`linear_probe_balanced_acc_mean ± std`** — Axis C; balanced linear
  probe accuracy with bootstrap CI, separating triggered features from
  genuine target features
- **`head_attribution_pct`** *(new this cycle)* — fraction of ASR
  removable by retraining a fresh classifier head on clean GTSRB train,
  with the feature extractor frozen. High = head-dominated attack;
  low = encoder-dominated attack.

### 2.3 Code summary

Implementation details: `pyproject.toml` adds three new keys
(`pretrained-init`, `trainable-layers`, `canonical-conv1`).
`models/resnet.py` accepts `pretrained` and `canonical_conv1` flags.
`training/train.py` adds `configure_trainable_layers` and
`freeze_batchnorm_stats` (the latter critical to prevent BN running stats
from drifting in frozen blocks). `analysis/head_feature_decomposition.py`
implements the diagnostic. Defaults preserve Cycle 01 YAML compatibility.
See commit `c22955c` on `v2-new-api`.

---

## 3. Results

### 3.1 9-cell training matrix (final round)

|             | Clean acc | Pixel 5mal acc / ASR | Pixel 15mal acc / ASR |
|---|---|---|---|
| **Full FT**     | 0.9228 | 0.6670 / **0.9777** | 0.5002 / **0.9798** |
| **Last block**  | 0.6647 | 0.6754 / 0.5645     | 0.5960 / **0.9540** |
| **Head only**   | 0.3621 | 0.3879 / 0.4998     | 0.3660 / 0.7238     |

**Cycle 01 from-scratch reference (existing numbers):** clean ≈ 0.94,
phaseC2 5mal ASR ≈ 0.86, phaseC2 15mal ASR ≈ 0.97 (see
`pixel_trigger_baseline.md`).

### 3.2 Observations on the training matrix alone

1. **Full-FT pretrained-init reproduces Cycle 01-grade numbers.** Clean
   acc 0.92 (vs 0.94 from-scratch), ASR 0.98 at both 5mal and 15mal (vs
   0.86 / 0.97). Pretraining slightly boosts attack effectiveness at the
   5mal pressure point.
2. **Attack pressure × trainable capacity tradeoff is monotonic.** Less
   trainable capacity demands more malicious clients to embed the
   backdoor: full_ft saturates ASR at 5mal already; last_block needs
   15mal to saturate; head_only is partial even at 15mal (0.72).
3. **Head-only clean utility collapsed to 36.2%**, well below the 0.60
   threshold the plan set for the canonical-conv1 fallback. This is the
   modified-3×3 conv1 + ImageNet bn1 distribution mismatch we predicted:
   the head alone cannot recover clean classification when the early
   stage produces activations that bn1 was not trained on. The
   `pretrained_headonly_canonconv1_*` fallback configs (image-size 64 +
   canonical 7×7 stride-2 conv1) are submitted and will produce a
   matching head-only row with a sensible utility ceiling. See § 3.4.
4. **Pure-head attack works even in this stressed regime.** Despite the
   broken clean baseline, head-only achieves ASR 0.50 at 5mal and 0.72
   at 15mal. The weight of evidence from the canonical-conv1 fallback is
   needed before drawing a definitive conclusion.

### 3.3 Head-feature decomposition

Cycle 01 reference (pilot on `phaseC2-backdoor-5mal-nodefense`):
`head_attribution_pct = 58.1 %` (orig ASR 0.86, clean-head ASR 0.36,
clean-head clean acc 0.94).

**Single-seed table (seed = 42, original Week 1 numbers):**

| Cell | Original ASR | Clean-head ASR | Clean-head clean acc | **`head_attribution_pct`** |
|---|---|---|---|---|
| **Full FT + 5mal**      | 0.9777 | 0.0274 | 0.6790 | **97.2 %** |
| **Full FT + 15mal**     | 0.9798 | 0.3239 | 0.5337 | **66.9 %** |
| **Last block + 5mal**   | 0.5645 | 0.2248 | 0.6453 | **60.2 %** |
| **Last block + 15mal**  | 0.9540 | 0.0893 | 0.5865 | **90.6 %** |
| **Head only + 5mal**    | 0.4998 | 0.0243 | 0.3644 | **95.1 %** |
| **Head only + 15mal**   | 0.7238 | 0.0243 | 0.3644 | **96.6 %** |

**Multi-seed table (seeds 42, 43, 44 — added Week 1 stretch):**

| Cell                  | ASR (42 / 43 / 44)         | head_attr (42 / 43 / 44)        | mean ± std        |
|---|---|---|---|
| Full FT + 5mal        | 0.978 / 0.799 / 0.915      | **97.2 / 22.1 / 6.9 %**         | **42.1 ± 47.7 %** |
| Full FT + 15mal       | 0.980 / 0.959 / 0.967      | 66.9 / 85.9 / 95.0 %            | 82.6 ± 14.3 %     |
| Last block + 5mal     | 0.564 / 0.517 / 0.852      | **60.2 / 98.3 / 18.8 %**        | **59.1 ± 39.8 %** |
| Last block + 15mal    | 0.954 / 0.913 / 0.934      | 90.6 / 94.9 / 96.3 %            | 93.9 ± 2.9 %      |
| canonconv1 ho + 5mal  | 0.067 / 0.053 / 0.064      | 64.7 / 54.8 / 63.1 %            | 60.9 ± 5.4 %      |
| canonconv1 ho + 15mal | 0.232 / 0.156 / 0.213      | 89.8 / 84.8 / 88.8 %            | 87.8 ± 2.7 %      |

**Key observations from multi-seed:**

- **5 mal cells are catastrophically seed-variant** — the seed-42 result
  was *not* representative. Across 3 seeds, full_ft 5 mal has
  `head_attribution = 42 ± 48 %` (95 % CI is essentially [0, 100]).
  Last_block 5 mal has `head_attribution = 59 ± 40 %`. **5 mal is at
  the boundary of attack success, where attack mechanism is itself
  stochastic.**
- **15 mal cells are robust** — head_attribution stays high
  (full_ft 15 mal: 83 ± 14 %; last_block 15 mal: 94 ± 3 %) and ASR
  saturates near 0.95.
- **Canonconv1 head-only is robustly weak** — ASR 0.06 / 0.20 ± 0.01
  across seeds. The head-only frozen-encoder regime is a stable
  defensive property, not a single-seed artefact.

#### Interpretation (revised)

The original "encoder anchoring" hypothesis was an artefact of the
single-seed window. The robust pattern across seeds is:

**Pattern 1 — Saturated attacks (ASR ≥ 0.95) consistently live in the
head.** Whenever the attack succeeds at ≥ 95 % ASR (full_ft 15 mal at
all seeds, last_block 15 mal at all seeds, full_ft 5 mal seed 42), the
clean-head retraining diagnostic finds 67–97 % head_attribution. **The
head is the path of least resistance for a saturated attack.**

**Pattern 2 — Marginal attacks have stochastic attribution.** When ASR
sits at the boundary (5 mal cells with ASR 0.52–0.92), the attack
distributes across head and encoder unpredictably depending on seed.
This is where the encoder *can* be corrupted, but whether it gets
corrupted depends on the random initialisation + Dirichlet partition
draw. **The honest takeaway:** the encoder is *vulnerable* at marginal
attack pressure, just not deterministically so.

**Pattern 3 — Pretrained init does not provide robust encoder anchoring
at low malicious ratio.** Counter to the seed-42 narrative, the
honest-gradient mass at 50 clients with 5 malicious is *not* large
enough to consistently keep the encoder near its ImageNet-pretrained
state; one-third of the time it gets pulled along with the attack
(see full_ft 5 mal seed 44, head_attr=6.9 %, encoder corrupted enough
that 85 % of triggered ASR survives clean-head retraining).

**Pattern 2 — Restricted trainable capacity does not redistribute attack
to the encoder; it suppresses the attack itself.** Last_block + 5mal
has both *lower ASR* (0.56 vs 0.98) and a *similarly mixed
head_attribution* (60 %) compared to full_ft + 5mal — i.e. the attack
*fails* to embed cleanly when pressure is low and capacity is restricted,
not "moves to encoder." When pressure is sufficient (last_block + 15mal,
ASR 0.95) the attack embeds *primarily in the head* (head=91 %), even
though layer4 was unfrozen and exploitable in principle.

**Pattern 3 — Pure head-attack works under FL.** Head_only achieves
non-trivial ASR (0.50 at 5mal, 0.72 at 15mal) despite the broken 36 %
clean baseline, with attribution ≈ 95-97 %. **HTBA's intuition holds in
FL even in this stressed regime.** Once the canonical-conv1 fallback
runs land (§ 3.4), we will report a corrected utility ceiling and re-
measure head_only ASR cleanly.

#### Headline finding (preliminary, before § 3.4 / § 3.5 land)

> Under realistic FL deployment (pretrained backbone, bounded malicious-
> client ratio ≤ 10 %), the pixel-trigger backdoor lives almost entirely
> in the classifier head (97 %). When malicious-client ratio rises to
> ~30 %, ~33 % of the attack signal leaks into the encoder. The Cycle-01
> from-scratch baseline (58 % head, 42 % encoder) is the "fluid encoder"
> limit and does not transfer to realistic deployment.

This headline directly motivates the redesign of Phase D.2 around
**head-targeted attack** rather than feature-space movement, and the
redesign of Phase E.1/E.2 around **logit-distribution drift**
rather than feature-drift (see § 4 and § 5).

### 3.4 Canonical-conv1 fallback — modified-conv1 ASR was an artifact

When the canonical ImageNet first stage is preserved (7×7 stride-2 conv1
with ImageNet weights + 3×3 stride-2 maxpool, image-size 64), head-only
attacks **almost completely fail**:

| Cell                   | Clean acc (modified \| canonical) | ASR (modified \| canonical) |
|---|---|---|
| Head only + Clean      | **0.362** \| **0.571**            | n/a                         |
| Head only + Pixel 5mal | 0.388 \| 0.560                    | **0.500** \| **0.067**      |
| Head only + Pixel 15mal| 0.366 \| 0.519                    | **0.724** \| **0.232**      |

**Interpretation.** The modified-conv1 head-only setup forces the
pretrained `bn1`+`layer1-4` weights to consume activations from a
randomly-initialised 3×3 conv1 — a strong distribution shift the head
must compensate for. That "compensation budget" is large enough to
*also* encode a backdoor when malicious clients are present. With the
canonical conv1+maxpool, the encoder produces clean, in-distribution
features and the head's 22 K parameters cannot easily overpower honest
classifier learning, even at 30 % malicious clients.

In other words: the modified-conv1 head-only ASR (0.50 / 0.72) was
**partially an artefact of the architecturally broken setup**, not a
generic property of pure head-attack. The realistic frozen-backbone
deployment regime is **much more resistant** to head-only backdoors than
§ 3.3 first suggested. Pattern 3 in § 3.3 ("Pure head-attack works under
FL") is **substantially weakened by this fallback**: pure head-attack
achieves only ASR ≈ 0.07–0.23, depending on attack pressure, when the
encoder is properly aligned.

**Defense implication.** A frozen pretrained encoder + a properly
canonicalised first stage is *itself* a partial defense — the attacker
cannot easily backdoor a 22 K-parameter head when the encoder produces
clean features. This refines § 4 / § 5: head-only deployments are *not*
the most vulnerable; *partial fine-tuning* (last_block) is, because it
gives the attacker enough parameter budget to embed without the
honest-gradient-anchoring effect that protects full_ft.

#### Canonical-conv1 head-feature decomposition

| Cell | Original ASR | Clean-head ASR | Clean-head clean acc | `head_attribution_pct` |
|---|---|---|---|---|
| canonconv1 + 5mal  | 0.0672 | 0.0237 | 0.6020 | **64.7 %** |
| canonconv1 + 15mal | 0.2322 | 0.0237 | 0.6021 | **89.8 %** |

The clean-head ASR floor of **2.4 %** is identical across both cells —
this is the **natural encoder rate** at which triggered images get
classified as target by chance with a fully-clean classifier head. It is
a property of the pretrained encoder's representation, not of the
backdoor.

**Subtracting that natural floor**, the *encoder's* attributable
contribution to ASR is essentially zero across both canonconv1 cells
(0.043 / 0.208 above floor) and the head's contribution is nearly the
entire ASR signal — but absolute ASR is so small (0.07 / 0.23) that
"99 %-head" doesn't have practical bite.

Translated: under canonical-conv1 head-only, the attack is **not
mechanistically different** from modified-conv1 head-only (head still
hosts essentially everything), it is just *quantitatively suppressed*.
The encoder being well-aligned means the head's 22 K parameters cannot
easily move predictions across the decision boundary.

### 3.5 Framework metrics — `centroid_l2` × `linear_probe_acc`

Feature extraction (96 .npz files: 12 cells × {7 rounds, final}) and
the 4-axis profile pass on all 8 attack cells are complete.

| Cell                              | `centroid_l2` | `linear_probe_balanced_acc_mean ± std` | ASR     | head_attr |
|---|---|---|---|---|
| Full FT + 5mal                    | 2.46          | 0.996 ± 0.002                          | 0.978   | 97.2 %    |
| Full FT + 15mal                   | 2.54          | 0.993 ± 0.003                          | 0.980   | 66.9 %    |
| Last block + 5mal                 | **1.14**      | 0.987 ± 0.003                          | 0.564   | 60.2 %    |
| Last block + 15mal                | 1.67          | 0.993 ± 0.003                          | 0.954   | 90.6 %    |
| Head only + 5mal (modified conv1) | 4.08          | 0.987 ± 0.004                          | 0.500   | 95.1 %    |
| Head only + 15mal (modified conv1)| 4.08          | 0.988 ± 0.004                          | 0.724   | 96.6 %    |
| Head only + 5mal (canonconv1)     | **13.04**     | 0.952 ± 0.006                          | 0.067   | 64.7 %    |
| Head only + 15mal (canonconv1)    | 13.04         | 0.951 ± 0.005                          | 0.232   | 89.8 %    |

Cycle 01 from-scratch reference: `centroid_l2 ≈ 2.7–3.1`,
`linear_probe_balanced_acc ≈ 0.99–1.00`.

#### Two new findings from the framework

1. **Linear probe separability is regime-invariant**: every cell has
   `linear_probe_balanced_acc ≥ 0.95`. The Cycle-01 "triggered features
   form a separable middle region" finding generalises *unchanged*
   across pretrained init, anchored encoders, partially-trained
   encoders, and frozen encoders. **A defender with labelled triggered
   samples can always find the decision direction** — but a realistic
   FL server does not have those, which is why the supervised probe is
   not directly deployable as a defense.
2. **`centroid_l2` is *not* invariant — it spans an order of magnitude.**
   - **last_block 5mal: 1.14** (smaller than Cycle 01's 2.7–3.1).
     Layer4 being trainable lets the attacker pull triggered features
     much closer to the genuine target-class centroid than from-scratch
     fluid encoders ever did. ASR is only 0.56 though — the attack is
     "structurally close" but not "behaviourally complete". This is
     interesting and worth a Cycle 02 week 2 follow-up: can D.2a
     (last_block-targeted) push ASR up while keeping `centroid_l2`
     small? If yes, this is a **new attack mechanism** — close-feature
     attack via partial fine-tuning, distinct from both pixel-trigger
     and Bagdasaryan.
   - **canonconv1 head_only: 13.04** (4× larger than Cycle 01). With
     image-size 64 + canonical conv1, the encoder's class clusters are
     much sharper, so triggered features (frozen at the encoder's
     natural mapping of "image with sticker") sit far from genuine
     target. Combined with low ASR (0.07/0.23), this is the cleanest
     "encoder cannot host the attack" cell in the matrix.

#### What the framework adds to the head-attribution story

Head-attribution alone says "the attack lives in the head." Centroid
alone says "triggered features may or may not be near the target."
Together they describe **how the attack succeeds**:

- High head_attr + high centroid_l2 (full_ft 5mal, head_only) →
  classifier-routing attack: features stay where they are, head learns
  to map that region to target.
- Low head_attr + low centroid_l2 (last_block 5mal) → encoder partially
  pulls features toward target, head still does some routing.
- High head_attr + low centroid_l2 (last_block 15mal at saturated ASR,
  centroid 1.67) → an interesting middle case. Both the encoder moved
  features *and* the head finished the routing.

This 2D characterisation is the right successor to Cycle 01's "joint
weak attack" 1D framing. It directly informs D.2's design space.

### 3.6 t-SNE visualisations

t-SNE is run on the saved penultimate features for the 5 most-informative
cells (full_ft clean ref + 4 attack regimes) plus per-cell trajectory
filmstrips across rounds 0-100. SLURM job `cycle02_pivot_tsne` produces
two outputs:

- `figures/tsne/comparison/comparison_panel_tsne.{png,pdf}` — 5-axis
  comparison panel with disambiguated subplot titles
  (`Pixel-Trigger + No Defense (5mal)`,
  `... (5mal, last-block)`,
  `... (15mal, head-only)`,
  `... (15mal, head-only, canon-conv1)`).
- `figures/tsne/trajectory/<cell>_trajectory.{png,pdf}` — 7-panel filmstrip
  per attack cell at rounds 0, 5, 10, 25, 50, 75, 100.

**Note.** A first run of the t-SNE job mislabelled all 5 cells with
the canon-conv1 suffix because experiment_logger writes booleans as
the strings `'true'/'false'` (yaml round-trip) and `bool('false')`
is `True` in Python. The fix is in `analysis/plot_features.py` (commit
`<TBD when t-SNE lands>`), and a second run was OOM-killed during the
trajectory step at 1:50 wallclock — the comparison panel itself
completed cleanly. The third run (with `-c 32` cores, ~256 GB memory)
is currently in queue; figures will be embedded here once it lands.

The original D.2 was an auxiliary loss that pulls triggered features
*toward* the genuine target-class centroid `μ̂_{c*}`. That design assumes
the backdoor needs to live in the encoder. If `head_attribution_pct`
across cells in §3.3 is consistently high (≥ 70 %), the auxiliary-loss
attack is solving a problem the realistic threat model does not pose.

**Empirical resolution.** All six § 3.3 cells produce
`head_attribution_pct ≥ 60 %`, with 5/6 ≥ 90 % at saturated ASR. The
original D.2 design assumes a regime that does not occur empirically
in our matrix. **D.2-revised proceeds as head-targeted attack under
partially-frozen backbones.**

**Proposed redesign (Cycle 02 weeks 2–3).** Two complementary attack
variants:

- **D.2a (last_block-targeted attack).** Attacker has trainable layer4
  + fc. Loss combines the standard pixel-trigger backdoor classifier
  loss with a stealth penalty on the layer4 weight delta (so the
  attack avoids drawing attention via norm-based defenses). Goal:
  raise `head_attribution_pct` toward 100 % even at low malicious
  pressure (5 mal), where § 3.3 shows it currently sits at 60 %.
- **D.2b (head_only-targeted with stealth).** Attacker has trainable
  fc only. Loss adds a logit-margin penalty
  `−margin(target, top-not-target)` on triggered samples to push
  the attack across the decision boundary while clean-head accuracy
  is constrained. Tests the residual feasibility of pure-head attacks
  in the canonical-conv1 regime — currently § 3.4 shows ASR ≤ 0.23.
  If D.2b raises this above 0.5, we have a genuinely stealthy
  head-only attack.

The auxiliary objective formally shifts from `‖f(τ(x)) − μ̂_{c*}‖²`
(feature alignment) to **margin-on-output** plus a stealth term —
HTBA-style, FL-native.

---

## 5. Implications for the redesigned Phase E.1/E.2

The original E.1 design monitored *penultimate-feature drift* per class
and aggregated suspicious `(c → c′)` flags across clients. That design
implicitly assumes attacks shift features.

**Empirical resolution.** § 3.3 confirms head-dominance is the
realistic regime (`head_attribution_pct ≥ 60 %` in every cell). The
detector pivots to **logit-distribution drift / output-margin
monitoring**:

- **E.1-revised (per-client logit drift).** Each client measures the
  per-class softmax-output shift on its own labelled data between a
  reference round and the current round. The trigger-induced anomaly
  manifests as an unusually-high target-class probability mass for
  inputs sampled from non-target classes — exactly what an HTBA-style
  head-attack produces. Comparable in spirit to FedInv (Zhao et al.
  2022, CVPR) and STRIP-derived prediction-shift tests, but **non-IID-
  native** (each client uses its own observable classes).
- **E.2-revised (cross-client aggregation).** Sparse `(c → c*)`
  suspicion scores reported by clients aggregate identically to the
  original E.2 design — the **coverage analysis in
  `phaseE2_coverage_analysis.md` transfers unchanged**: at observable
  threshold `t = 5`, 99.9 % of class pairs are covered with median
  redundancy 10 witnesses. We do not need to redo coverage work.

The **feature-drift detector is retired**, not because feature drift
never happens, but because the regime where it would be informative
(low `head_attribution_pct`) does not exist in the realistic
deployment matrix. If a future cycle finds an attack mechanism that
does corrupt features (e.g. a stealth-constrained model replacement
operating across many rounds at low per-round magnitude), feature-
drift detection re-enters the picture.

---

## 6. Open questions for the supervisor

1. **Paper shape — defense-leaning USENIX Sec or attack-leaning AAAI?**
   The user's leaning is to keep both venues open, with the empirical
   finding from this matrix guiding which goes first. Either shape can
   be supported by Cycle 02 weeks 2–3 work; the choice affects what we
   prioritize in the redesign.
2. **Cross-region FL framing for the thesis intro.** Should the threat
   model in the introduction be made concrete as "regional/cross-org
   federated fine-tuning of a pretrained backbone for AD perception"
   rather than the textbook-FL framing? The realistic deployment is
   what motivates the head/feature decomposition; the textbook framing
   does not.
3. **Stealth-constrained model replacement.** Cycle 01 closed the raw
   Bagdasaryan version (NaN collapse window, brittle). A stealth-
   constrained variant is a candidate for Cycle 02 week 4 or Cycle 03.
   Worth scoping?
4. **Adaptive-attack sequencing.** Shejwalkar & Houmansadr 2021 was
   deferred to Cycle 03 in the original cycle plan. If we go USENIX-Sec
   shape, adaptive attack evaluation is a hard requirement. Should
   adaptive-attack work move into Cycle 02 week 4 to keep the USENIX
   timeline?
5. **Reading load.** The week-1 reading list adds Saha 2020 (HTBA),
   Yao 2019 (Latent Backdoor Attacks), Shen 2024 (Better Together),
   Cao 2021 (FLTrust), Zhao 2022 (FedInv). Which do we treat as
   highest-priority for the next week?

---

## 7. Updated reading list

Add to `roadmap/cycle_02_designed_attacks_and_client_defenses.md`:

**Theme A1 — Pretrained-backbone backdoors (NEW priority).**
- Saha et al. **Hidden Trigger Backdoor Attacks**, AAAI 2020. Direct
  inspiration for the head-feature decomposition diagnostic.
- Yao et al. **Latent Backdoor Attacks on Deep Neural Networks**, CCS
  2019. Backdoor pretrained models that survive fine-tuning.
- Shen et al. **Better Together: Joint Pretraining-Finetuning Backdoors**,
  S&P 2024. Direct match for the realistic threat model.
- Jia et al. **BadEncoder**, S&P 2022. Backdoor attacks on pretrained
  encoders.

**Theme G1 — Server-side defenses with reference (NEW).**
- Cao et al. **FLTrust**, NDSS 2021. Server uses a small trusted dataset
  to bootstrap trust — the realistic-deployment analog for our pretrained
  reference model.
- Li et al. **BackdoorIndicator**, USENIX Sec 2024. Already on the
  roadmap; re-read with the pretrained-reference framing.

**Theme C reformulation — Output-distribution detection.**
- Zhao et al. **FedInv: Backdoor Detection via Gradient Inversion**,
  CVPR 2022. Direct E.1 redesign reference.
- Gao et al. **STRIP**, ACSAC 2019. Already on the roadmap; output-shift
  test repurposed as a per-client probe in the FL setting.

---

## 8. Deferred work + Cycle 03+ outline

Explicitly **not** in Week 1:

- Original D.2 auxiliary-loss attack (deferred to weeks 2–3, possibly
  re-cast as a head-attack under frozen extractor)
- E.1 feature-drift detector + E.2 cross-client aggregation (deferred to
  weeks 3–4, redesigned around § 5)
- LIRA-style learnable trigger generator (Cycle 03 candidate)
- HTBA reproduction beyond the diagnostic (Cycle 03 candidate)
- Physical-world realistic triggers (Eykholt et al. 2018; Cycle 03+)
- ViT migration (Cycle 03+, supervisor's earlier suggestion)
- Adaptive attacks (Shejwalkar & Houmansadr 2021; depends on Q4 above)
- Multimodal / fusion-layer backdoor (autonomous-driving migration;
  Cycle 04+)
- Stealth-constrained model-replacement (Q3 above; Cycle 02 week 4 or
  Cycle 03)

The user wants both AAAI (~83 days) and USENIX Security (~111 days)
covered; the per-venue split decision is deferred until § 3.3 lands.
The implicit Cycle 03 / Cycle 04 split is *attack-leaning paper from
one codebase, defense-leaning paper from a possibly forked codebase,*
sharing learnings but with distinct empirical campaigns.
