# Cycle 02 Pivot — Friday Supervisor Brief (2026-05-08)

*One-page synthesis prepared for the supervisor meeting. Built around
the four open thesis questions that drove the pivot, the papers we
read this week, and what Cycle 02 actually taught us — including a
methodological audit that reshaped what counts as a credible result.*

---

## The four thesis questions

The Cycle 02 pivot was designed to gather empirical input for these:

1. **Trigger realism for autonomous-driving FL backdoors** — pixel
   triggers, LIRA-style learnable triggers, physically-realisable
   triggers (Eykholt-style robust patches, semantic triggers).
2. **The role of representation space** — head-vs-feature
   decomposition, whether the 4-axis framework (centroid, probe,
   spectral, dynamics) is the right framing.
3. **Whether FL is the appropriate deployment for AD perception** —
   single-vehicle vs regional / cross-org federated fine-tuning;
   threat model and adversary capabilities differ between regimes.
4. **Pretrained-vs-from-scratch and the corresponding attack surface**
   — does pretrained-backbone + fine-tuning shrink, expand, or
   relocate the attack surface compared to from-scratch?

---

## Papers read this week

- **Saha et al., AAAI 2020 — Hidden Trigger Backdoor Attacks (HTBA).**
  Clean-label feature-collision attack; head-contamination interpretation
  was the seed of our head-attribution diagnostic.
- **Doan et al., ICCV 2021 — LIRA.** Learnable input-dependent trigger,
  but training-time threat model assumes attacker controls all of
  training (MLaaS). Doesn't transfer cleanly to FL malicious-client
  setting; useful as a digital upper-bound benchmark.
- (Backlog for next week, prioritised in the recovery plan: Yao 2019
  Latent Backdoor; Shen 2024 Joint Pretrain-Finetune Backdoors;
  Cao 2021 FLTrust; Zhao 2022 FedInv.)

---

## What Cycle 02 pivot *operationally* delivered

- 24 training runs end-to-end on the new pretrained-init + fine-tuning
  regime axes (3 × 3 matrix at seed 42 + 3 canonconv1 fallback +
  12 multi-seed at seeds 43, 44).
- A working **head-feature decomposition diagnostic** that runs on a
  saved final checkpoint and produces a single number quantifying
  what fraction of ASR survives a clean-head retrain.
- The **canonical-conv1 fallback** that lets us A/B between modified-
  conv1 (32×32 inputs, our original setup) and canonical-conv1 (64×64
  inputs, ImageNet-pretrained first stage preserved) — important
  because the modified setup gave a head-only utility floor of 36 %
  that the canonical setup cleanly fixes (57 %).
- **End-to-end pipeline now has dependent SLURM stages** (training →
  feature extraction → framework metrics → head-feature decomposition →
  comparison figures) — this scaffolding survives the recovery and is
  the foundation for any future cycle.

## What Cycle 02 pivot *did not* deliver — and why

A reliability audit (triggered by ~90 pp seed-to-seed variance in
`head_attribution_pct`) found that the training pipeline was
**non-deterministic at fixed seed**. Re-running the same YAML at the
same seed produced acc 0.667 vs 0.894 in two attempts — a 22.7 pp
swing in clean accuracy from identical configuration. This means the
24 numerical results from the pivot **cannot be interpreted at face
value**; we cannot tell whether observed variance is a real seed effect
or run-to-run noise of the same seed.

Seven non-determinism sources have been identified and committed:

| # | Source | Status |
|---|---|---|
| 1 | `bool("false") == True` in 7 config sites | fixed (commit `795f75e`) |
| 2 | Client `torch.manual_seed()` never called in `@app.train()` | fixed |
| 3 | `DataLoader(shuffle=True)` had no seeded `generator=` | fixed |
| 4 | `fl_v2/src/fl_v2/data/` source code silently `.gitignore`d | fixed |
| 5 | cuDNN atomic-add convolution backward | fixed (commit `725cae5`) |
| 6 | Strategy aggregation order followed Ray's task-completion order | fixed (commit `1f5e70d`) |
| 7 | `torch.use_deterministic_algorithms` for non-cuDNN ops | fixed (commit `bd2c1eb`) |

After all 7 fixes, **same-node same-seed reproducibility still
diverges** at round 6 of pixel-trigger 5mal training, with Δ ≈ 12-23 pp
final accuracy (verified on jobs 6594906/6594907/6594624/6594625).
Trajectories are **bit-identical for rounds 0-5** and then split
during the attractor-escape phase. Two interpretations of the
remaining ε:

### Phase 3.1 wave: 9 valid runs on the fixed pipeline

Three cells spanning the design matrix × three seeds = 9 training runs
(jobs 6599453–6599461) on commit `9d72bcf` with all reliability fixes
in place. Plus the head-feature decomposition diagnostic on each.

**Caveat: "valid" ≠ "reproducible".** Each of these 9 runs completed
its 100 rounds without silent failure, which makes their numerical
outputs trustworthy *as one realisation*. We have not yet verified
that re-running any one of these YAMLs at the same seed on the same
commit produces a bit-identical trajectory — the audit's own pair
6594906 / 6594907 found bit-identical rounds 0–5 followed by round-6+
divergence even after all 7 fixes. A within-commit reprocheck submission
(job 6600759, `cycle02-reprocheck-full-ft-pixel5_seed42`) is in flight
specifically to characterise this residue.

| Cell | seed=42 | seed=43 | seed=44 | **mean head_attr ± SD** | acc mean |
|---|---|---|---|---|---|
| `full_ft + 5mal` | 12.5 % | 44.7 % | 41.0 % | **32.7 % ± 17.62 pp** | 0.889 |
| `last_block + 5mal` | 63.0 % | 61.9 % | 55.5 % | **60.1 % ± 4.07 pp** | 0.717 |
| `canonconv1 head_only + 15mal` | 89.8 % | 84.7 % | 88.8 % | **87.8 % ± 2.68 pp** | 0.520 |

**Per-cell variance is heterogeneous and tells a story.** Three observations:

1. **`canonconv1 head_only + 15mal` is the most stable cell (SD 2.68 pp).**
   Frozen encoder + saturated 15-mal pressure → a single dominant
   attractor. The clean-head ASR rounds to **0.024 across all 3 seeds**
   (precise values: seed 42 = 0.0237373737, seed 43 = 0.0237373737,
   seed 44 = 0.0236531987 — seeds 42/43 match to 10 decimals, seed 44
   differs at the fourth decimal). Two of three diagnostics are
   bit-identical, which is consistent with "frozen encoder + same
   diagnostic seed → deterministic head training" but is not strong
   enough to claim full bit-reproducibility on this regime.
2. **`last_block + 5mal` is also stable (SD 4.07 pp).** Restricted
   parameter capacity (only layer4 + fc trainable, ~8 M params) limits
   how many distinct attractors the optimizer can reach. The mean
   head_attr ≈ 60 % is *close to* the Cycle 01 from-scratch number
   (58 %) — i.e. last_block under pretrained init behaves much like
   full_ft from-scratch.
3. **`full_ft + 5mal` is genuinely chaotic (SD 17.62 pp).** Full
   parameter capacity (~11 M trainable) gives the optimizer many
   accessible attractors at marginal attack pressure. This is the
   only cell where N ≥ 5 seeds will be required for venue-quality
   reporting. Note: even within the chaos, the new mean (32.7 %) is
   substantially below the Cycle 01 from-scratch baseline (58 %),
   reversing the original "encoder anchoring" claim.

**The chaotic-regime story I had drafted was thus partially right but
mostly wrong:**
- Right: there *is* genuine multi-attractor variance at full_ft + 5mal.
- Wrong: "all 5mal cells are chaotic" — last_block + 5mal is stable
  with SD 4 pp, contradicting the universal-chaos claim.
- Wrong: "even at 100 rounds the regime is chaotic" — 100-round
  reproducibility is *fine* for two of the three cells.

**Bit-reproducibility check — what we have, what we DON'T have:**

What we *don't* have: an earlier draft of this brief claimed seeds 43
and 44 gave bit-identical head_attr (44.71 % and 40.99 %) "across two
independent 3-seed runs (6598089 and the Phase 3.1 6599453 wave)". That
claim is **withdrawn**. `sacct -j 6598089,6598090,6598091` shows wave-1
seeds 43/44 (jobs 6598090/6598091) "COMPLETED" in 1 m 27 s and 1 m 39 s
respectively — the Ray-port-collision silent-failure pattern, no rounds
trained, no head-attribution computed. There is no wave-1 number for
seeds 43/44 to compare wave-2 against. Only seed=42 (job 6598089) ran
the full 1 h 53 m on the pre-Ray-port commit.

What we *do* have:
- Audit reproducibility pair 6594906 / 6594907 (post-7-fixes, same
  commit, same seed): rounds 0-5 bit-identical, then round-6+ divergence
  (Δ ≈ 12-23 pp final accuracy).
- Wandb screenshot of seed=42 across the Ray-port commit boundary
  (6598089 vs 6599453): trajectories visibly diverge — but those are on
  *different commits*, so this conflates "Ray-port commit change" with
  "any residual ε".
- A within-commit, within-YAML reprocheck (job 6600759,
  `cycle02-reprocheck-full-ft-pixel5_seed42`) — the first clean
  same-(commit, seed) divergence measurement — has now landed.
  Result (rounds.csv comparison vs 6599453):

  | round | 6599453 acc | 6600759 acc | 6599453 ASR | 6600759 ASR |
  |---|---|---|---|---|
  | 50  | 83.75 % | 85.00 % | 14.87 % | 8.70 %  |
  | 75  | 88.61 % | 89.47 % | 65.78 % | 72.99 % |
  | 90  | 88.92 % | 89.75 % | 79.87 % | 85.18 % |
  | 100 | 89.10 % | 89.74 % | 81.44 % | **86.59 %** |

  Two runs of the same YAML at the same seed on effectively the same
  training source (only `87fa3e6`, a one-line shell-only fix, sits
  between the two commits) ended at **Δ test_acc = 0.64 pp, Δ ASR =
  5.15 pp at round 100**, with a peak intermediate divergence of Δ ASR
  ≈ 6 pp at round 50. The within-commit residual ε is real, ~5 pp at
  the final round, and not just a Ray-port-commit artefact.

### Comparison to the original Wave 1+2 (unreliable code)

The Wave 1+2 numerical record (seed=42 = 97.2 %, seed=43 = 22.1 %,
seed=44 = 6.9 %, SD 38.7 pp) is retired because Wave 1+2 ran on the
pre-audit pipeline with unseeded client RNG. We do **not** treat the
new Phase 3.1 numbers as a strict head-to-head comparison against
Wave 1+2 because (a) the pipelines differ in 7 documented places, and
(b) within-commit bit-reproducibility for the new pipeline has not yet
been demonstrated. The Wave 1+2 seed=42 (97.2 %) was, however,
diagnosed at the time as a bug-induced outlier (model stuck at the
trivial-backdoor attractor for 12 rounds because of unseeded RNG); the
audit-fixed pipeline does not reproduce that outlier.

### Provisional consequences for the thesis story (Phase 3.1)

These are stated as *current best-guesses on the v1 fixed-10-epoch
diagnostic*. The v2 convergent diagnostic is in flight (jobs 6600186 +
6600819) and is producing materially different per-cell numbers — see
the "v2 update" subsection below. Cross-cycle (Cycle 01 ↔ Cycle 02)
comparison is also provisional because the Cycle 01 reference number
(58 %) was computed on the pre-audit pipeline; the apples-to-apples
Phase 3.0 sentinel (Cycle 01 cell on the audit-fixed pipeline) is now
in (head_attr v2 = 18.2 %), and is much lower than that 58 %.

1. **The original Wave 1+2 seed=42 head_attr of 97.2 % does not
   reproduce on the audit-fixed pipeline.** Fixed seed=42 trains
   normally (acc 0.89). We do not have a within-commit reprocheck of
   12.5 % yet; until 6600759 lands we treat 12.5 % as one realisation
   on the audit-fixed pipeline, with an as-yet-uncharacterised residual
   ε relative to a hypothetical second realisation.

2. **An "encoder anchoring is REVERSED" claim is premature.** The
   Phase 3.1 v1 mean of 32.7 % at `full_ft + 5 mal` is below the
   Cycle 01 v1 reference of 58 %, but: (a) the v1 diagnostic
   systematically undertrains the clean head (see v2 update below),
   and (b) the Cycle 01 reference is on the pre-audit pipeline, not
   the audit-fixed one. The audit-fixed Cycle 01 sentinel under v2 is
   18.2 %. We cannot yet make a directional claim about
   pretrained vs from-scratch susceptibility until we have v2 numbers
   for the full 9-cell Cycle 02 matrix.

3. **The "5 mal is chaotic" claim must be qualified.** Under v1, only
   `full_ft + 5 mal` showed high SD (17.6 pp); `last_block + 5 mal`
   and the saturated cells were stable (SD 4–6 pp). The cell-level
   variance is informative: it ranks the regimes by how unstable they
   are, which is a property of the regime, not a property of the
   pipeline. Whether v2 preserves this stratification is open until
   cells 4-9 land.

4. **Cell ordering under v1:**
   `full_ft (33 %)` < `from-scratch baseline (58 %, pre-audit, suspect)` ≈ `last_block (60 %)` < `canonconv1 head_only (88 %)`
   The monotone "less trainable capacity → more head-attribution"
   gradient holds on the v1 diagnostic. Whether it survives v2 +
   audit-fixed Cycle 01 sentinel re-anchoring is being tested now.

### Strict zero-fine-tuning baseline — "is FL necessary?"

ImageNet-pretrained ResNet18 evaluated directly on GTSRB test with
**no GTSRB-specific training of any kind**: pretrained encoder
loaded as-is, random-init 43-class fc head, immediate inference.
Three fc seeds (42/43/44) per architecture so the chance baseline
has a small spread. From job 6602864
(`baselines/zero_finetune_baseline.json`):

| Architecture | clean_acc | target_class_acc | asr |
|---|---|---|---|
| `canonical_conv1` (image-size 64, full ImageNet 7×7 stem) | **3.28 % ± 1.58** | 0.18 % ± 0.25 | 0.15 % ± 0.14 |
| `modified_conv1` (image-size 32, random-init 3×3 conv1) | **1.69 % ± 0.87** | 0.22 % ± 0.31 | 0.29 % ± 0.21 |

Chance = 1/43 ≈ 2.33 %. Read-offs:

1. **Canonical-conv1 zero-FT is barely above chance** (3.3 % vs 2.3 %).
   ImageNet features carry a tiny signal for GTSRB but the random fc
   can't exploit it without training. **FL fine-tuning is necessary.**
2. **Modified-conv1 zero-FT is *below* chance** (1.7 %). The random-
   init conv1 + frozen-ImageNet mid-blocks pipeline doesn't produce
   coherent features — random low-level filters projected through
   ImageNet's natural-image-trained mid-layers gives features the
   random fc literally can't classify better than 1.7 %. This is the
   strongest empirical evidence so far for risk-audit C2 (the
   pretrained-with-modified-conv1 architecture is broken). Note that
   FL training of the fc on top recovers to 36.2 % (the
   `cycle02-pretrained-headonly-clean` baseline below) — useful but
   far below the from-scratch 95 %.
3. **Zero-FT ASR is also at chance** (~0 %). The trigger doesn't
   systematically push predictions toward class 2 because the random
   head doesn't have a class-2 bias to exploit. ASR > 80 % in all
   our attack runs is therefore *entirely a product of FL training*
   — the attack works because the FL-trained head learned to map
   triggered features to class 2; the ImageNet encoder alone provides
   no shortcut.

Together with the centralised clean baseline below, this establishes
the full utility ladder:

  zero-FT (~2 %)  ≪  FL clean × 3 seeds (92.2 ± 1.3 %)  <  centralised clean (98.97 %)

The FL gap to centralised (~7 pp) measures the cost of distributed
learning under our setup: Dirichlet α=0.5 partition + 50 clients +
3 local epochs + FedAvg + 100 rounds. The full_ft architecture is
strong enough that this cost is small but real.

### Clean-utility baselines (audit-fixed pipeline, 100 rounds, full_ft + 3 seeds + centralised)

For the supervisor's utility-vs-robustness reading. The first three
rows below are **audit-fixed pipeline** (commits up to `db21d2a`,
seven reliability fixes applied): `cycle02-fixed-full-ft-clean` × 3
seeds (jobs 6603555 / 6602987 / 6602988) and centralised clean (job
6602985). Numbers are final-round
`summary.json::final.test_accuracy` on the GTSRB official test split
(12,630 samples).

| Run | Architecture | Trainable-layers | acc | target_acc |
|---|---|---|---|---|
| `centralised_clean_pretrained_modified_conv1` (seed 42, 50 epochs centralised) | pretrained, modified-conv1 (32×32) | full (centralised) | **0.9897** | 0.9960 |
| `cycle02-fixed-full-ft-clean` seed=42 (audit-fixed FL, 100 rounds) | pretrained, modified-conv1 | full_ft | 0.9067 | 0.9533 |
| `cycle02-fixed-full-ft-clean` seed=43 | pretrained, modified-conv1 | full_ft | 0.9213 | 0.9693 |
| `cycle02-fixed-full-ft-clean` seed=44 | pretrained, modified-conv1 | full_ft | 0.9378 | 0.9840 |
| **mean ± SD (audit-fixed FL, N=3)** | | | **0.9219 ± 0.0127** | **0.9689 ± 0.0125** |

The remaining rows are **pre-audit pipeline** (Wave 1+2 era), one
realisation each under non-deterministic training. Kept for context
on the architectural variants we did not re-run on the audit-fixed
pipeline. Direct numerical comparison to the audit-fixed numbers
above is approximate.

| Run (pre-audit) | Architecture | Trainable-layers | Clean acc |
|---|---|---|---|
| `phaseC2-clean_r100_seed42` | **from-scratch** ResNet18, modified-conv1 | full_ft | 0.9533 |
| `cycle02-pretrained-full-ft-clean_r100_seed42` | pretrained, modified-conv1 | full_ft | 0.9228 |
| `cycle02-pretrained-lastblock-clean_r100_seed42` | pretrained, modified-conv1 | last_block (layer4 + fc) | 0.6647 |
| `cycle02-pretrained-headonly-canonconv1-clean_r100_seed42` | pretrained, canonical 7×7 conv1 (image-size=64) | head_only | 0.5709 |
| `cycle02-pretrained-headonly-clean_r100_seed42` | pretrained, **random-init conv1** (modified-conv1) | head_only | 0.3621 |

Read-offs to flag in the meeting:

1. **Centralised clean = 98.97 %.** This is the absolute ceiling
   for ResNet18 on GTSRB with our architecture (pretrained +
   modified-conv1 + image-size 32). target_acc = 99.6 %, asr ≈ 0
   (clean model never sees the trigger and barely classifies any
   triggered sample as the target — a sanity check on the trigger
   pattern).
2. **Audit-fixed clean FL = 92.19 % ± 1.27 pp** (N=3 seeds). The
   gap to centralised is **~6.8 pp**, which measures the cost of
   distributed learning under our setup: Dirichlet α=0.5 + 50
   clients + 3 local epochs + FedAvg + 100 rounds.
3. **Compared to the *pre-audit* Cycle 02 full_ft clean (0.9228, single
   seed), the audit-fixed run lands at 0.9219 ± 0.0127** — within
   noise. Encouragingly, this means the audit fixes did NOT
   meaningfully shift clean utility; the per-seed spread now visible
   (~1.3 pp) is the genuine seed-to-seed variance of the regime.
4. **Pre-audit Cycle 01 from-scratch clean (95.3 %) is *higher* than
   Cycle 02 pretrained full_ft clean (92.2 % audit-fixed mean).** The
   "pretrained-init" pivot does NOT improve clean utility under our
   setup at 32×32 — it actually hurts by ~3 pp. Likely cause:
   risk-audit C2 — the "pretrained" model has a random-init conv1
   that is trained from scratch, so the ImageNet head-start applies
   only to bn1+layer1-4 and the model has to learn conv1 from
   scratch on top of a partially-mismatched mid-stack.
5. **The canonical-conv1 head_only clean baseline (57.1 %) is the
   closest measurement we have for "ImageNet pretrained ResNet18 on
   GTSRB without GTSRB-specific encoder training".** ImageNet
   features transfer to GTSRB's small-image traffic-sign
   classification only modestly. Useful upper-bound reference for
   any future "frozen-encoder" defense.
6. **Modified-conv1 head_only clean = 36.2 %.** Risk-audit C2 in the
   open — a "pretrained" model whose conv1 is frozen at random init
   has a broken encoder pipeline.

### Final-round main metrics per cell (audit-fixed pipeline, 100 rounds)

These are the headline FL-training outputs from each `summary.json::final`,
*before* any diagnostic post-processing. `acc` is overall clean test
accuracy (12,630 GTSRB test samples). `target_acc` is clean accuracy
restricted to genuine class-2 samples (n=750, the natural target class).
`asr` is attack success rate over non-target samples (n=11,880).

| Cell | seed | acc | target_acc | asr |
|---|---|---|---|---|
| `full_ft + 5mal` (Cycle 02 pretrained) | 42 | 0.8910 | 0.9520 | 0.8144 |
| | 43 | 0.9147 | 0.9680 | 0.5243 |
| | 44 | 0.8598 | 0.9440 | 0.7956 |
| | mean ± SD | **0.8885 ± 0.0276** | **0.9547 ± 0.0124** | **0.7114 ± 0.1623** |
| `lastblock + 5mal` (Cycle 02 pretrained) | 42 | 0.6644 | 0.6173 | 0.5736 |
| | 43 | 0.7253 | 0.7253 | 0.7386 |
| | 44 | 0.7605 | 0.8280 | 0.6511 |
| | mean ± SD | **0.7167 ± 0.0486** | **0.7235 ± 0.1054** | **0.6544 ± 0.0825** |
| `canonconv1 head_only + 15mal` (Cycle 02 pretrained) | 42 | 0.5149 | 0.5987 | 0.2320 |
| | 43 | 0.5333 | 0.5187 | 0.1553 |
| | 44 | 0.5128 | 0.5333 | 0.2107 |
| | mean ± SD | **0.5203 ± 0.0112** | **0.5502 ± 0.0421** | **0.1993 ± 0.0396** |
| **Phase 3.0 sentinel** (Cycle 01 from-scratch, 5mal) | 42 | **0.9537** | **0.9933** | **0.8958** |

A few read-offs the supervisor will likely look at:

1. **Cycle 02 full_ft acc (88.9 % mean) is *lower* than Cycle 01
   sentinel acc (95.4 %, N=1).** Same as the clean-baseline reading
   above — pretrained-init at 32×32 with random-init conv1 doesn't
   improve clean utility over from-scratch.
2. **`lastblock` and especially `head_only` cells take a serious hit
   in clean utility** (71.7 % / 52.0 %). The trade-off here is that
   restricting trainable layers (a common defense intuition) sacrifices
   accuracy on the main task. For a thesis that frames "client-side
   defense via restricted updates" as a strategy, this is a hard
   ceiling.
3. **`target_acc` is consistently HIGH across cells** (66 %–99 %).
   Class 2 is over-represented in GTSRB train (5.6 %) so the model
   overfits to it; that's also why ASR ceilings are high — the model
   already wants to predict class 2.
4. **Sentinel `target_acc` = 99.3 %** is a striking number. The
   audit-fixed pipeline trains class 2 to near-perfect under attack
   pressure — the FL-trained head fully cooperates with the
   "everything-becomes-class-2" attack signal because class 2 was
   already the most-represented class. Flagging this for the
   supervisor as the operationalisation of risk-audit C3 (target
   class is too easy).

### v2 (convergent diagnostic) — full 9-cell rerun complete

All 9 cells of the v2 rerun on the audit-fixed pipeline are now in.
v1 used a fixed 10-epoch clean-head retrain; v2 uses early-stop on
clean-test-acc plateau (patience 8, min_improvement 1e-4, max
100 epochs). v2 numbers come from jobs 6600186, 6600819, 6601196,
6601582 (4 SLURM submissions due to wallclock + per-cell ~50-min
training budget). Each cell's v2 was computed on the same final
checkpoint as the v1 number.

| Cell | seed | orig_asr | v1 ch_asr | v1 head_attr | v2 ch_asr | v2 head_attr | best_ep / total |
|---|---|---|---|---|---|---|---|
| `full_ft + 5mal` | 42 | 0.8144 | 0.7126 | 12.50 % | 0.6551 | **19.56 %** | 42 / 50 |
| | 43 | 0.5243 | 0.4088 | 44.71 % | 0.2055 | **60.81 %** | 45 / 53 |
| | 44 | 0.7956 | 0.5317 | 40.99 % | 0.4029 | **49.37 %** | 59 / 67 |
| `lastblock + 5mal` | 42 | 0.5736 | 0.2124 | 62.97 % | 0.2385 | **58.42 %** | 80 / 88 |
| | 43 | 0.7386 | 0.2814 | 61.90 % | 0.3647 | **50.62 %** | 57 / 65 |
| | 44 | 0.6511 | 0.2901 | 55.45 % | 0.3380 | **48.08 %** | 44 / 52 |
| `canonconv1 head_only + 15mal` | 42 | 0.2320 | 0.0237 | 89.77 % | 0.0371 | **84.00 %** | 30 / 38 |
| | 43 | 0.1553 | 0.0237 | 84.71 % | 0.0371 | **76.08 %** | 30 / 38 |
| | 44 | 0.2107 | 0.0237 | 88.77 % | 0.0371 | **82.38 %** | 30 / 38 |

| Cell | v1 mean ± SD | v2 mean ± SD | shift |
|---|---|---|---|
| `full_ft + 5mal` | 32.73 ± 17.62 | **43.25 ± 21.07** | +10.5 pp |
| `lastblock + 5mal` | 60.11 ± 4.07 | **52.37 ± 5.41** | −7.7 pp |
| `canonconv1 head_only + 15mal` | 87.75 ± 2.68 | **80.82 ± 4.20** | −6.9 pp |

**Phase 3.0 sentinel** (Cycle 01 phaseC2-backdoor-5mal-nodefense on
the audit-fixed pipeline, seed=42, v2 convergent diagnostic): orig_asr
0.8958, ch_asr 0.7326, **head_attr = 18.22 %**, best_ep 18/26. (N=1.)

**Cell ordering under v2 + audit-fixed-pipeline anchoring:**

`sentinel (18.2 %)` < `Cycle 02 full_ft (43.3 %)` < `lastblock (52.4 %)` < `canonconv1 head_only (80.8 %)`

The "less trainable capacity → more head-attribution" gradient
**survives** under the convergent diagnostic. The Cycle 01 sentinel
sits below the Cycle 02 full_ft mean (Δ ≈ 25 pp). With the audit-
fixed Cycle 01 anchor (sentinel = 18.2 %, NOT the pre-audit
58 % v1 reference), the directional claim is now: **pretrained
encoders trained at full_ft show more head-resident attack signal
than from-scratch encoders under the same regime.**

**Interpretation caveats (binding for the supervisor brief):**

1. v1 vs v2 movement is regime-dependent and *not noise*: full_ft v2
   shifts up by 10 pp because the head needed more training to
   plateau; lastblock and canonconv1 head_only v2 shift down because
   their v1 ch_asr was *already at a low floor* and v2 raises that
   floor slightly (the converged head finds a marginally higher
   baseline ch_asr). The shift direction tells us about the regime,
   not about the diagnostic's reliability.
2. v2 SD is comparable to v1 SD on the chaotic full_ft cell (21.0
   vs 17.6 pp) and slightly *higher* on the constrained cells.
   Convergent training does NOT eliminate seed-to-seed variance —
   it lets each cell's true ceiling speak.
3. The 25-pp gap between sentinel and Cycle 02 full_ft v2 is roughly
   1.2 SD given v2 SD = 21.0 pp. With N=3 seeds on Cycle 02 full_ft
   and N=1 on sentinel, this is suggestive but not statistically
   firm. N=5 seeds on each is the venue-credible sample size.
4. The within-commit residual ε measured in §"Bit-reproducibility
   check" above adds a further ~5 pp ASR uncertainty band that
   propagates into head_attr at order ~ε / orig_asr ≈ 5-10 pp on
   single-cell numbers.
5. All 9 cells of the v2 wave used the SAME final checkpoint as v1.
   The shifts are entirely attributable to the v1→v2 diagnostic
   change (fixed-10-epoch → convergent), not to retrained models.

**Saturated-regime "bit-stability" (canonconv1 head_only + 15mal) is
diagnostic-side, not training-side.** All three seeds' v2 ch_asr round
to **0.0371** (vs v1's 0.0237). At first glance this looks like
strong reproducibility evidence, but the mechanism is structural,
not informative about FL training:

- `pretrained=True` + `canonical_conv1=True` loads the entire encoder
  (`conv1`, `bn1`, `layer1`–`layer4`) from ImageNet's pretrained
  checkpoint — a fixed external file, byte-identical across all
  model seeds.
- `trainable-layers: head_only` freezes that encoder; only `model.fc`
  is updated during FL training.
- The diagnostic loads the final FL checkpoint, **discards the
  FL-trained `fc`**, reinitialises a fresh head with the diagnostic
  seed (`args.seed=4242`), retrains it on the full clean GTSRB train
  with the same frozen encoder, and reports the resulting ASR.

So across model seeds 42/43/44 the diagnostic sees: the same
ImageNet encoder (frozen, identical), a fresh head from the same
diagnostic seed, the same training data, the same shuffle order.
**By construction the result is identical** — it cannot vary,
regardless of how different the FL-trained heads were. The "ch_asr
= 0.0371 across 3 seeds" is reproducibility *of the diagnostic on a
frozen-encoder regime*, not of the FL training.

The genuine seed-to-seed signal on canonconv1 head_only cells lives
in `original_asr` (the ASR using the FL-trained head): 0.2320 /
0.1553 / 0.2107, range = 7.6 pp. That spread reflects how the
malicious-client coalition's seed-dependent data partition affected
the trained head. The diagnostic methodology cannot probe deeper
than that — its convergent retrain washes out exactly the part of
the model that varied.

For the full_ft and lastblock regimes the encoder IS trained across
seeds, so different model seeds → different encoders → different
diagnostic-side `ch_asr`. That's why their v2 SDs are non-trivial
(21 pp on full_ft, 5 pp on lastblock) — the diagnostic genuinely
sees encoder-level variance there.

## Provisional findings the audit *did* preserve

These are robust to the residual non-determinism in the sense that
they are either deterministic by construction or measured stable
across seeds within the data we have:

1. **Head-feature decomposition diagnostic is reproducible on a fixed
   input checkpoint.** Job 6585212 ran the v1 diagnostic 5 times on
   the same checkpoint at the same diagnostic seed and produced
   bit-identical outputs. This isolates the diagnostic itself from
   any training-pipeline non-determinism.
2. **canonconv1 head_only diagnostic is mostly stable across training
   seeds.** Three runs gave clean_head_asr 0.0237373737, 0.0237373737,
   0.0236531987 (seeds 42 / 43 / 44). Two of three are bit-identical
   to ten decimals; seed 44 differs at the fourth decimal. Direction is
   consistent with "frozen pretrained encoder + same diagnostic seed
   ⇒ near-deterministic head training", but not literally bit-identical
   across all three.
3. **`linear_probe_balanced_acc ≥ 0.95` in every cell, every seed.**
   The Cycle-01 finding that "triggered features form a separable
   middle region" generalises to every regime in our matrix on the
   v1 framework metrics.
4. **The pretrained-init pipeline works.** The architectural pivot
   (pretrained ResNet18 + one of {full_ft, last_block, head_only},
   with optional canonical conv1) is operational end-to-end. Whether
   the round-by-round trajectory is bit-reproducible at fixed
   (commit, seed) is the open question being characterised by job
   6600759.

## Decision points for the supervisor

1. **Continue digging vs accept chaotic-regime** — if v8 single-actor
   test (results due in this meeting if queue cooperates) shows
   bit-identical, source #7 is Ray multi-actor scheduling and we can
   fix it. If still divergent, we stop digging and frame 5 mal as
   chaotic.
2. **Multi-seed sample size** — if we accept (B), how many seeds do
   we run? N=5 vs N=10 vs N=20 (compute-vs-tightness trade-off).
3. **Saturated-only vs full-regime story** — should we focus the
   thesis chapter on deterministic regimes (15 mal, frozen encoder),
   or include 5 mal explicitly with chaotic-attractor framing?
4. **Phase D.2 (designed attack) and Phase E.1/E.2 (client detector)
   directions** — both rest on a credible 5 mal characterisation. If
   5 mal is chaotic, the original D.2 auxiliary-loss target ("push
   features into target manifold") is testing the wrong thing.
   Reframe?

## What's already locked in for next week regardless of decision

- Reproducibility regression test (20-30 round CI-style check) added
  to `tests/test_reproducibility.py`. Catches any future
  non-determinism regression.
- Cycle 01 sentinel rerun (`phaseC2-backdoor-5mal-nodefense` at the
  fixed pipeline) so cross-cycle comparisons are honest.
- Phase 3.1 minimum-viable rerun (3 cells × 3 seeds = 9 runs).

## Out-of-scope for Friday

LIRA-style learnable triggers; HTBA reproduction; Eykholt physical
triggers; ViT migration; multimodal fusion. All deferred until the
recovery plan completes Phase 3 with reproducible numbers.
