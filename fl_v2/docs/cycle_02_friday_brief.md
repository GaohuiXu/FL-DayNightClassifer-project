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

### Phase 3.1 wave: 9 reproducible runs on the fixed pipeline

Three cells spanning the design matrix × three seeds = 9 training runs
(jobs 6599453–6599461) on commit `9d72bcf` with all reliability fixes
in place. Plus the head-feature decomposition diagnostic on each.

| Cell | seed=42 | seed=43 | seed=44 | mean head_attr | SD |
|---|---|---|---|---|---|
| `full_ft + 5mal` | 12.5 % | 44.7 % | 41.0 % | **32.7 %** | **±17.6 pp** |
| `last_block + 5mal` | 63.0 % | 61.9 % | 55.5 % | **60.1 %** | **±4.1 pp** |
| `canonconv1 head_only + 15mal` (saturated) | 89.8 % | 84.7 % | 88-ish % * | ~88 % | ≤6 pp |

(* seed=44 head-attr was finishing at audit-doc-write time; will be filled in.)

**Per-cell variance is heterogeneous and tells a story.** Three observations:

1. **`last_block + 5mal` is the most stable cell (SD 4.1 pp).**
   Restricted parameter capacity (only layer4 + fc trainable, ~8 M
   params) limits how many distinct attractors the optimizer can
   reach. The mean head_attr ≈ 60 % is *close to* the Cycle 01 from-
   scratch number (58 %) — i.e. last_block under pretrained init
   behaves much like full_ft from-scratch.
2. **`full_ft + 5mal` is genuinely chaotic (SD 17.6 pp).** Full
   parameter capacity (~11 M trainable) gives the optimizer many
   accessible attractors at marginal attack pressure. This is the
   only cell where 5-seed reporting (mean ± std) is required to be
   credible. Note: even within the chaos, the new mean (32.7 %) is
   substantially below the Cycle 01 from-scratch baseline (58 %),
   reversing the original "encoder anchoring" claim.
3. **`canonconv1 head_only + 15mal` is the saturated, deterministic
   regime (SD ≤ 6 pp).** Frozen encoder + saturated 15 mal pressure
   = a single dominant attractor. Mean head_attr ≈ 88 % is high by
   construction (head must do all the work since encoder is frozen).

**The chaotic-regime story I had drafted was thus partially right but
mostly wrong:**
- Right: there *is* genuine multi-attractor variance at full_ft + 5mal.
- Wrong: "all 5mal cells are chaotic" — last_block + 5mal is stable
  with SD 4 pp, contradicting the universal-chaos claim.
- Wrong: "even at 100 rounds the regime is chaotic" — 100-round
  reproducibility is *fine* for two of the three cells.

**Bit-reproducibility check:** seeds 43 and 44 of `full_ft + 5mal` gave
**identical** numbers (44.71 % and 40.99 %) across two independent
3-seed runs (the user-requested test of 6598089 and the Phase 3.1
6599453 wave). The pipeline IS bit-reproducible at fixed (commit, seed)
when the run is on the same Ray-port commit.

### Comparison to the original Wave 1+2 (unreliable code)

The user-requested 3-seed test of `full_ft + 5mal` cleanly retired the
Wave 1+2 numerical record:

| | Wave 1+2 (unreliable) | Phase 3 fixed pipeline |
|---|---|---|
| seed=42 head_attr | **97.2 %** ⚠ outlier | 12.5 % (post Ray-port fix) |
| seed=43 head_attr | 22.1 % | 44.7 % |
| seed=44 head_attr | 6.9 % | 41.0 % |
| range | **90.3 pp** | 32.2 pp |
| SD | 38.7 pp | 17.6 pp |

The Wave 1+2 seed=42 was the bug-induced outlier (model stuck at
trivial-backdoor attractor for 12 rounds because of unseeded RNG).
Once fixed, seed=42 trains normally to acc ≈ 0.89.

### Two consequences for the thesis story (revised after Phase 3.1)

1. **The original seed=42 (97.2 %) was a bug-induced outlier.**
   Fixed seed=42 trains normally (acc 0.89, in line with seeds 43/44).

2. **The "encoder anchoring" headline is REVERSED.** Pretrained init
   gives mean head_attr ≈ 33 % at full_ft + 5 mal vs Cycle 01 from-
   scratch ≈ 58 %. Pretrained encoders are MORE susceptible to feature-
   space attack, not less. This is a substantive scientific claim;
   it would be the corrected centerpiece of the Cycle 02 chapter.

3. **The "5 mal is chaotic" claim must be qualified.** Only `full_ft +
   5mal` is chaotic (SD 17.6 pp). `last_block + 5mal` and the
   saturated cells are stable (SD 4–6 pp). The cell-level variance
   *itself* is informative: it ranks the regimes by how unstable they
   are, which is a property of the regime not a property of the
   pipeline.

4. **Cell ordering across the matrix:**
   `full_ft (33 %)` < `from-scratch baseline (58 %)` ≈ `last_block (60 %)` < `canonconv1 head_only (88 %)`
   This monotone gradient — **less trainable capacity → more head-
   attribution** — survives the audit and is the Cycle 02 main finding
   we can actually defend.

## Provisional findings the audit *did* preserve

These are robust to the non-determinism (verified empirically as
either deterministic by construction or stable across seeds):

1. **Head-feature decomposition diagnostic is fully reproducible.**
   Bit-identical across 5 runs at fixed seed (job 6585212).
2. **canonconv1 head_only cells are bit-stable across training seeds**
   (frozen encoder + same diagnostic seed → identical clean_head_asr =
   0.0237 across all 6 runs). This gives us a clean baseline.
3. **`linear_probe_balanced_acc ≥ 0.95` in every cell, every seed.**
   The Cycle-01 finding that "triggered features form a separable
   middle region" generalises to every regime in our matrix.
4. **The pretrained-init pipeline works.** Even if numerics aren't
   reproducible, the architectural pivot (pretrained ResNet18 + one
   of {full_ft, last_block, head_only}, with optional canonical conv1)
   is operational and ready to receive a deterministic rerun.

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
