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

- **(A) Software bug we haven't caught yet** — most likely candidate
  is Ray's multi-actor task-scheduling order, which our strategy-level
  sort doesn't fully linearise. Currently testing single-actor Ray
  configuration (job 6595430) to isolate this.
- **(B) Genuine chaotic-attractor regime** — 5 mal pixel-trigger has
  multiple basins (trivial-backdoor attractor at ASR 1.0 / acc 6 %, vs
  normal-training attractor at acc ~90 %). Tiny floating-point ε at
  the bifurcation point flips one run into one basin and the other
  into the other. This is a *property of the dynamical system*, not a
  bug.

If (A): we keep fixing until same-seed → bit-identical. If (B): we
**accept the chaotic regime as a finding**, switch from "single-seed
point estimate" to "mean ± std across N ≥ 5 seeds" for 5 mal cells,
and report saturated cells (15 mal, frozen-encoder canonconv1) as
the deterministic baseline.

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
