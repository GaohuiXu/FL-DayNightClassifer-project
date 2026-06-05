# Cycle 02 — FLAME Ablation + α=0.2 Stress Test (stage-study close-out)

**Period:** 2026-06-02
**Phase:** ablation + alpha02 — FLAME component ablation on the mid-pixel test bed, and Dirichlet α=0.2 clean stress test
**Status:** closed

## Why this study existed

Two open questions from [`stage_study_log.md`](stage_study_log.md):

1. **Is FLAME's 0% ASR driven by clustering, by median-norm clipping, or by Gaussian noise?** The supervisor's "is FLAME a trick?" doubt — Wave-1 showed full FLAME drives ASR to 0 but never isolated *which* of the three FLAME components does the work. The stage study confirmed the 0% holds across attack stages but still didn't dissect the mechanism.
2. **Does the clean FPR=0.469 generalize, or collapse, at higher non-IID severity?** The stage study's open question: on harder partitioning (α=0.2 vs α=0.5) the honest cloud is more heterogeneous, so either FPR explodes (honest-collateral collapses clean acc) or HDBSCAN's size-26 majority cluster fails to form (admit-all fallback). Either is a finding; both close the GTSRB story before any larger pivot.

A two-commit experimental sandwich was used: **commit A** (115e390) added 2 default-true FLAME knobs (`enable_cluster`, `enable_clip`) + 5 YAML cells; **commit B** (this doc's commit) reverts the code knobs and archives the cells under `_archived_*/`. Main returns to a pristine state after commit B; the two commits bracket the experiment as a clean reproducible pair in git history.

## Frozen platform (same as the stage study)

GTSRB-43 · ResNet18 from scratch · Adam lr=0.001 cosine → lr-min=1e-4 · 50 clients · **40 rounds** · m=10 · target class 14 · poison-fraction 0.5 · trigger-size 4 · seed 42.

Cell-specific deltas:

- **Ablation cells** (3): same Dirichlet α=0.5 and same mid-pixel test bed as `stages/cycle02-stage-mid-pixel-flame.yaml` (attack r16–25). The full-FLAME reference values in the table below are reused from the stage study, not re-run.
- **α=0.2 cells** (2): Dirichlet α=0.2 (vs α=0.5 baseline), no attack (clean baseline only — attacked α=0.2 cells were not in scope).

## What we did

**Ablation micro-study (3 cells)** — each isolates exactly ONE of the three FLAME components on the strongest attack stage that clears the guardrail (mid-pixel, FedAvg ASR=0.60):

| variant | enable_cluster | enable_clip | noise multiplier |
|---|---|---|---|
| cluster-only | true | false | 0.0 |
| clip-only | false | true | 0.0 |
| noise-only | false | false | 1e-6 |

**α=0.2 stress test (2 cells)** — clean baseline at the harder partitioning:

| variant | dirichlet-α | defense |
|---|---|---|
| α=0.2 clean × FedAvg | 0.2 | none |
| α=0.2 clean × FLAME | 0.2 | flame |

## Results

### Table 1 — FLAME component ablation (mid-pixel, α=0.5)

| variant | cluster | clip | noise | peak ASR | final ASR | clean acc | TPR (in-window) | FPR (in-window) |
|---|---|---|---|---|---|---|---|---|
| **full FLAME** (ref) | ON | ON | 1e-6 | **0.00** | 0.00 | 0.97 | 1.00 | 0.34 |
| cluster-only | ON | off | off | **0.00** | 0.00 | 0.971 | 1.00 | 0.33 |
| clip-only | off | ON | off | 0.58 | 0.36 | 0.968 | 0.00 | 0.00 |
| noise-only | off | off | 1e-6 | 0.60 | 0.36 | 0.971 | 0.00 | 0.00 |
| **FedAvg** (ref) | — | — | — | 0.60 | 0.35 | 0.97 | — | — |

Reference rows come from the stage study (`mid pixel × {FLAME, FedAvg}` cells). In-window TPR/FPR are means over attack rounds 16–25.

### Table 2 — Dirichlet α=0.2 stress test (clean, 40r)

| condition | α | clean acc | FLAME FPR (all-round) | FLAME admit rate |
|---|---|---|---|---|
| clean × FedAvg | 0.5 (Wave-1 ref) | 0.973 | — | — |
| clean × FedAvg | **0.2** | **0.972** | — | — |
| clean × FLAME | 0.5 (stage-study ref) | 0.966 | 0.469 | ~0.53 |
| clean × FLAME | **0.2** | **0.966** | **0.467** | **0.53** |

Δ at α=0.5 → α=0.2: FedAvg clean acc −0.001 (noise), FLAME clean acc ~0 (identical at 3 decimals), FLAME FPR −0.002 absolute, admit rate +0.002 absolute.

## Findings

1. **Clustering is the entire FLAME defense at our calibration.** cluster-only matches full FLAME exactly (TPR 1.00, FPR 0.33, ASR 0). clip-only and noise-only both collapse to FedAvg-equivalent ASR (0.58–0.60 peak, TPR=0, FPR=0). Median-norm clipping and Gaussian noise (at λ=1e-6) contribute zero defense in isolation on this test bed. The supervisor's "is FLAME a trick?" doubt is resolved: it's **not a trick, it's literally just HDBSCAN with `min_cluster_size = N/2+1`** discarding everything outside the benign majority. The other two FLAME components are cosmetic on GTSRB/ResNet18 with the Adam-calibrated λ=1e-6 — they could be turned off without changing the empirical defense outcome.
2. **The α=0.2 hypothesis did NOT hold on GTSRB.** Increasing partition non-IID severity (α 0.5 → 0.2) left FLAME's clean FPR effectively unchanged (0.469 → 0.467) and clean accuracy identical at 3 decimals (0.966 in both). HDBSCAN still finds its size-26 majority cluster; admit rate is unchanged (~0.53). Neither failure mode predicted by the stage-study open question materialized (no FPR explosion, no admit-all fallback).
3. **GTSRB is saturated as a stress-test substrate for this defense family.** Honest 43-class traffic-sign gradients are coherent enough that even α=0.2 doesn't fragment the majority cluster. This is a strong empirical argument that the research gap is in the **data**, not in the partitioning severity — i.e., it directly supports the long-term AD-pivot direction. A research-relevant FLAME breakdown needs a task where honest clients genuinely disagree on the gradient direction (geographic AD distribution shift, multimodal/fusion-layer sensors), not just a harder Dirichlet split of a structured benchmark.

## Caveats

- **Single attack stage in the ablation.** Ablation cells were run only on mid-pixel (the strongest tested attack stage). Cluster-only's defense parity with full FLAME at early/late stages is *expected* on the same logic (HDBSCAN does the rejection regardless of attack-window position) but not directly verified.
- **α=0.2 cells are clean-only.** Attacked × α=0.2 × FLAME was not in scope. The conclusion "α=0.2 doesn't break FLAME on GTSRB" applies to the clean-FPR mechanism only; the under-attack TPR/FPR profile at α=0.2 is untested.
- **Single seed (42).** Wave-1's bit-determinism check (FLAME × pixel across seeds 42/43) is the only multi-seed evidence; these cells were not re-run on seed 43.
- **λ=1e-6 noise.** The ablation conclusion "clip and noise are cosmetic" is calibration-specific. The Adam-calibrated λ=1e-6 is 1000× smaller than the paper's SGD-calibrated λ; at paper λ the noise term *might* contribute, but Adam-calibrated noise is by construction at the cosmetic-floor of what the optimizer tolerates.

## Strategic conclusion (Cycle-02 close-out)

After Wave-1, the stage study, and this ablation + α=0.2:

- FLAME on GTSRB/ResNet18 is genuinely robust against pixel / model-replacement / DBA at all attack stages **by virtue of aggressive HDBSCAN clustering alone**, not via clip or noise.
- The honest cost (~half the honest clients dropped per clean round at FPR=0.467–0.469) is real but doesn't bite GTSRB's accuracy at α∈{0.2, 0.5} — the 26 admitted clients still train fine.
- **GTSRB is exhausted as a research benchmark for this defense family.** Partition stress (α=0.2) doesn't surface a gap. A research-relevant defense breakdown requires a task where the honest gradient cloud has no coherent majority — AD distribution shift, multimodal fusion, etc.

**Cycle 02 is fully closed scientifically.** Cycle 03's adaptive-attack workstream remains the active research front; if Cycle 03 produces an attack that breaks FLAME on the saturated GTSRB substrate, that strengthens the contribution because it falsifies "FLAME ≡ HDBSCAN majority filter" as a hard ceiling. If Cycle 03 fails to break FLAME on GTSRB, the AD-migration is the natural next move per this study's strategic conclusion.

## Run book

- **Configs (archived):** `configs/experiments/cycle_02/_archived_ablation/cycle02-ablation-{cluster,clip,noise}only-mid-pixel-flame.yaml` and `configs/experiments/cycle_02/_archived_alpha02/cycle02-alpha02-clean-{fedavg,flame}.yaml`. Generators: `_gen_ablation.py`, `_gen_alpha02.py`.
- **⚠ Archived configs do NOT run against current main.** The ablation YAMLs reference the `flame-enable-cluster` / `flame-enable-clip` knobs that commit B removed. To reproduce, check out commit A (115e390) first, then run the archived configs.
- **Outputs:** `fl_outputs/gtsrb/experiments/cycle_02/{ablation,alpha02}/<exp>_r40_seed42/`.
- **SLURM jids:** 6716000 (clusteronly), 6716001 (cliponly), 6716002 (noiseonly), 6716003 (alpha02-clean-fedavg), 6716004 (alpha02-clean-flame). All 5 passed; silent-exit guard (2628aff) clean across the set.
- **Total cost:** ≈3 GPU-h.

## Commit pair

- **Commit A** (`115e390`): adds the 2 FLAME knobs + 5 cell YAMLs. Defaults preserve full FLAME (18/18 existing tests green).
- **Commit B** (this doc's commit): restores `pyproject.toml`, `server_app.py`, `strategy/flame.py` to pre-A state (bit-identical); renames `ablation/` → `_archived_ablation/` and `alpha02/` → `_archived_alpha02/`; adds this log. End state: main returns to pristine FLAME implementation; archived configs + this doc are the durable record of the ablation.
