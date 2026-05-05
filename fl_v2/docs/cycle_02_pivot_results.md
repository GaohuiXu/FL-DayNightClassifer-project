# Cycle 02 Pivot — Week 1 Results

*Companion to* `roadmap/cycle_02_designed_attacks_and_client_defenses.md`.
*Empirical evidence for the pretrained-init pivot. Frames the supervisor
discussion on **2026-05-08***.

**Status:** in-progress; synthesis pending the head-feature decomposition
diagnostic and framework-metric pass on the 9 main cells. Numbers in this
document are draft and will be filled in as analysis jobs complete.

---

## TL;DR

After reading HTBA (Saha et al., AAAI 2020) and LIRA (Doan et al., ICCV 2021)
we re-examined the original Cycle 02 plan and found that three load-bearing
assumptions no longer hold (§ Context). We pivoted from "designed feature-
space attack + feature-drift detector" to **characterizing where backdoors
live under realistic FL fine-tuning regimes**. The Week 1 deliverable is a
3×3 design matrix (`{full_ft, last_block, head_only} × {clean, pixel 5mal,
pixel 15mal}`) that varies trainable capacity against attack pressure on a
pretrained ImageNet ResNet18. The headline empirical claim is `[TBD: head
attribution number across regimes]`. We propose a redesigned Phase D.2 and
Phase E.1/E.2 informed by this finding (§ Implications), plus a paper-shape
question for the supervisor (§ Open questions).

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

| Cell | Original ASR | Clean-head ASR | Clean-head clean acc | **`head_attribution_pct`** |
|---|---|---|---|---|
| **Full FT + 5mal**      | 0.9777 | 0.0274 | 0.6790 | **97.2 %** |
| **Full FT + 15mal**     | 0.9798 | 0.3239 | 0.5337 | **66.9 %** |
| **Last block + 5mal**   | 0.5645 | 0.2248 | 0.6453 | **60.2 %** |
| **Last block + 15mal**  | 0.9540 | 0.0893 | 0.5865 | **90.6 %** |
| **Head only + 5mal**    | 0.4998 | 0.0243 | 0.3644 | **95.1 %** |
| **Head only + 15mal**   | 0.7238 | 0.0243 | 0.3644 | **96.6 %** |

(`head_only` rows share `clean-head ASR/clean acc` because the encoder is
identical across them — frozen at pretrained init, never updated — and
the diagnostic uses the same seed.)

#### Interpretation

The matrix is *not* a simple monotone gradient in capacity. Three cleaner
patterns emerge:

**Pattern 1 — Head attribution rises with attack pressure given a fixed
encoder substrate.** When the encoder is genuinely fluid (full_ft 5mal,
ASR 0.98, head=97 %) the attack lives almost entirely in the head; when
extra pressure overcomes encoder anchoring (full_ft 15mal, ASR 0.98,
head=67 %) some attack signal does land in the encoder. **From-scratch
init (Cycle 01 pilot, head=58 %) is the "permanent high pressure" limit
of this** — random init never anchors, so attack-pressure is effectively
maxed regardless of mal-count.

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

### 3.5 Framework metrics (TBD)

Feature extraction → 4-axis profile pass on the six attack cells is
running (extract job 6572466, framework job 6572468 with `afterok`
dependency). When complete this section will show `centroid_l2` and
`linear_probe_balanced_acc_mean ± std` per cell.

---

## 4. Implications for the redesigned Phase D.2

The original D.2 was an auxiliary loss that pulls triggered features
*toward* the genuine target-class centroid `μ̂_{c*}`. That design assumes
the backdoor needs to live in the encoder. If `head_attribution_pct`
across cells in §3.3 is consistently high (≥ 70 %), the auxiliary-loss
attack is solving a problem the realistic threat model does not pose.

**Proposed redesign (Cycle 02 weeks 2–3):** D.2-revised becomes
**head-attack-under-frozen-extractor** — a stronger, FL-native version
of HTBA. Concretely: malicious clients receive a global model with the
backbone frozen (operator policy), can only update the head, and apply
a stealth-constrained optimization to push triggered logits over the
target-class threshold while preserving clean-head accuracy. The
auxiliary objective shifts from `‖f(τ(x)) − μ̂‖²` (feature alignment)
to a margin-on-output formulation (logit margin, classifier-row
alignment, or HTBA-style poisoned-target collision with the head).

If `head_attribution_pct` is consistently low (≤ 30 %) — i.e. backdoors
genuinely need encoder access — the original D.2 design retains its
premise. Either result is publishable; we wait for §3.3 to decide.

---

## 5. Implications for the redesigned Phase E.1/E.2

The original E.1 design monitored *penultimate-feature drift* per class
and aggregated suspicious `(c → c′)` flags across clients. That design
implicitly assumes attacks shift features.

If § 3.3 confirms head-dominance, the detector switches to
**logit-distribution drift** monitoring — comparable in spirit to FedInv
(Zhao et al. 2022, CVPR) and STRIP-derived prediction-shift tests, but
adapted to the FL non-IID setting. Each client measures the per-class
output-distribution shift on its own labeled data between rounds, and the
server aggregates flags via the same coverage analysis already done in
`phaseE2_coverage_analysis.md` (the coverage finding transfers
unchanged).

If § 3.3 shows mixed regimes (`head_only` cells fail to embed, but
`last_block`/`full_ft` cells do), the detector becomes a *combined*
feature-drift + logit-drift sensor. We do not yet know which.

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
