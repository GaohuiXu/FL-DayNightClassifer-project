# Cycle 02 — Stage Study Log (post-Wave-1 follow-up)

**Period:** 2026-05-29
**Phase:** stages — backdoor injection at early / mid / late FL stages, relative to convergence
**Status:** closed

## Why this study existed

Two questions from the supervisor after Wave-1:

1. **Does FLAME's 0% ASR reflect a real defense or aggressive over-filtering?** Wave-1 logs hinted FLAME admitted only ~26–28/50 clients per round even when attacked.
2. **Wave-1's attack window 10–35 spans only the post-convergence regime** (the model converges by ~r12). Are backdoor dynamics different when injected during rapid learning vs deep plateau? Splitting the run into stages defined *relative to convergence* is more informative than a fixed window inside a 60-round horizon.

A third design point also surfaced: the defender must not know when the attacker starts/stops. We **verified** this in code: the strategy's `aggregate_train` runs every round; only client-side poisoning is window-gated. The defense is genuinely always-on, so clean-baseline cells (no attack + defense) quantify FLAME's "cost at idle."

## Frozen platform (same as Wave-1 except total rounds)

GTSRB-43 · ResNet18 from scratch · Adam lr=0.001 cosine → lr-min=1e-4 · 50 clients · Dirichlet α=0.5 · **40 rounds** · base poison regime · m=10 · target class 14 · poison-fraction 0.5 · trigger-size 4 · seed 42. Cosine LR kept (option b: late = mature model + decayed LR, the realistic coupling).

**Stage windows (fixed 10-round attack window each, defined relative to R_c≈12):**

- early: attack r1–10 (injection during rapid learning)
- mid: attack r16–25 (just-converged)
- late: attack r28–37 (deep plateau)

## What we did

14 cells × 40 rounds:

- 2 clean baselines (no attack): `clean × {FedAvg, FLAME}` — FLAME's idle defense cost.
- 12 attack cells: {pixel, dba} × {FedAvg, FLAME} × {early, mid, late}.

Per stage, the × FedAvg cell is the **scientific guardrail**: confirms the attack is viable at this stage (control ASR > 0.3). Only then is the matching × FLAME 0% interpretable as defense, not as "attack never landed."

## Results

| stage / attack | FedAvg (control) peak / final / acc | FLAME peak / final / acc | FLAME TPR | FLAME FPR (in-window) |
|---|---|---|---|---|
| **clean** (no attack) | — / — / **0.973** | — / — / **0.966** | — | **0.469** ← baseline |
| early pixel | 0.51 / 0.23 / 0.97 | 0.00 / 0.00 / 0.97 | 0.85 | 0.39 |
| early dba | 0.42 / 0.09 / 0.97 | 0.00 / 0.00 / 0.97 | 0.84 | 0.38 |
| mid pixel | **0.60** / 0.35 / 0.97 | 0.00 / 0.00 / 0.97 | 1.00 | 0.34 |
| mid dba | 0.33 / 0.13 / 0.97 | 0.00 / 0.00 / 0.97 | 1.00 | 0.34 |
| late pixel | 0.03 / 0.01 / 0.97 ⚠ | 0.00 / 0.00 / 0.97 ⚠ | 1.00 | 0.33 |
| late dba | 0.05 / 0.01 / 0.97 ⚠ | 0.00 / 0.00 / 0.97 ⚠ | 1.00 | 0.33 |

⚠ control fails the guardrail (ASR < 0.3 without defense) → FLAME row not a defense test.

## Findings

1. **Attack viability is strongly stage-dependent.** Peak ASR (no defense): early 0.42–0.51 → **mid 0.33–0.60 (sweet spot)** → late 0.03–0.05 (near-total failure). Late = mature model + decayed cosine LR; 10 rounds of small-LR poisoning can't move it.
2. **Early backdoors don't last.** 30 rounds of subsequent clean training wash early peaks (0.42–0.51) down to 0.09–0.23.
3. **Late cells are not a valid defense test** — control ASR < 0.3 means the attack never landed. FLAME's 0% there says nothing.
4. **Where the attack is real (early + mid), FLAME holds at 0% ASR.** Detection improves with maturity: TPR = 0.85 at early (rapid-learning gradients are diverse; some malicious slip through), TPR = 1.00 at mid/late. FLAME still gets 0% at early because the early backdoor is too weak for the few admitted malicious to establish it.
5. **FLAME's honest-client collateral is *worse* on clean rounds than under attack.** Clean FPR = **0.469** (~23/50 dropped) vs in-attack FPR = **0.33–0.39** (~14/40 dropped — lower in both absolute count and rate). The malicious outlier cluster gives FLAME a clear boundary; without it the heterogeneous honest cloud has no clean structure to separate against. **The defense costs the most when there's nothing to defend.** This directly quantifies the supervisor's doubt — FLAME's 0% isn't a trick; it's bought with aggressive honest-filtering (~half every clean round).

## Caveats

- **LR confound (option b kept):** late-stage failure conflates "rigid model" with "decayed LR (~1e-4)." Honest framing: *late-stage (mature model + decayed LR) attacks fail.*
- **Window length:** fixed 10-round windows give lower absolute ASR than Wave-1's 25-round window (mid-pixel 0.60 here vs 0.82 Wave-1). Price paid for fair stage comparison.
- **Single seed.** Wave-1 verified FLAME × pixel is bit-deterministic across seeds 42/43.

## Outstanding question this study OPENED

**FLAME's clean FPR=0.469** means it drops ~half the honest clients per round at idle. On GTSRB (43 classes, α=0.5, easy features) this costs only 0.66 pp clean accuracy because 26 admitted clients still train fine. On harder data — higher non-IID (α=0.2) or a harder dataset (more classes, complex features) — the honest cloud is more heterogeneous, so FLAME's clustering would have less coherent structure to separate against. Either:

- FPR rises further (honest-collateral collapses clean acc), or
- HDBSCAN fails to form any size-26+ cluster → admit-all fallback (no defense, no collateral).

Either is a finding. This is the natural follow-up — a small α=0.2 stress test on GTSRB closes the GTSRB story before any larger pivot.

## Run book

- Configs: `configs/experiments/cycle_02/stages/cycle02-stage-{clean,early/mid/late}-{pixel,dba}-{fedavg,flame}.yaml` (generated by `_gen_stages.py`).
- Outputs: `fl_outputs/gtsrb/experiments/cycle_02/stages/<exp>_r40_seed42/`.
- SLURM jids: 6701766, 6701775, 6701894–6701905, 6701942 (1 resubmit after a silent-exit race, caught by the new run_alvis.sh guard committed at 2628aff).
