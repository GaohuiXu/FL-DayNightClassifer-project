# Cycle 03 — Implementation Plan (WS workflow)

> **Purpose.** This is the durable, execution-level companion to the roadmap doc
> `../roadmap/cycle_03_stronger_adaptive_attacks.md`. The roadmap doc says *why* and *what*;
> this file says *how* — exact files, line numbers, knobs, guardrails, and the
> phase-by-phase workflow a fresh session can follow without any session-scoped state.
>
> Authoritative Wave-1 baseline results: `./wave1_log.md`. Project context: `../../CLAUDE.md`.
> Repo root: `/mimer/NOBACKUP/groups/naiss2024-22-991/gaohui/thesis_workspace/fl_weather_project/`.
> Branch: `v2-new-api`.

## Context

Cycle 02 Wave-1 showed **FLAME drives ASR to 0.000 against all 3 static attacks** (pixel,
model_replacement, DBA) on the audit-fixed codebase. Until we have an attack FLAME does NOT
shut down, there is no meaningful target for a new defense — any "improvement" over FLAME at
the 0.000 floor is busywork. So Cycle 03 builds a stronger attacker by reproducing adaptive
attacks from 2023-2025 literature. The cos2mean separation, NormClip-no-op, and
DBA-evades-FoolsGold observations from Wave-1 are recorded as intuition for later, **not** as
the contribution. The verdict for each attack turns on **ASR** (does it break FLAME?), not on
gradient-space metrics.

## Frozen platform (identical to Wave-1)

GTSRB-43 · ResNet18 from scratch · Adam lr=0.001 cosine → lr-min=0.0001 · 50 clients ·
Dirichlet α=0.5 · 60 rounds · attack window 10–35 · base poison regime · m=10 malicious
(20%) · target class 14 (Stop) · poison-fraction 0.5 · trigger-size 4 · trigger-value 1.0 ·
trigger-position bottom-right · seed 42 (43 for the Day-0 determinism check).

## Threat-model addendum

Only one axis changes from Wave-1: **attacker knowledge of the defense**. Cycle-03 attackers
are **defense-aware static-adaptive** — they know the exact aggregator (FLAME), its
hyper-parameters (HDBSCAN `min_cluster_size = ⌈N/2⌉+1`, median-norm clipping, λ-scaled
noise), and design the attack to evade *that specific recipe*. Still no real-time probing;
still `m < n/2`; still no inter-client communication.

---

## WS-A — Day 0: Logging hardening + determinism baseline (half day)

Three batched additions land before any new attack code.

1. **Per-class ASR breakdown.** In `src/fl_v2/training/server_eval.py::server_evaluate`
   (lines 41–89), track `asr_by_source_class[c]` for c ∈ {0..42} \ {target}. Add to the
   returned metrics dict. Splits global ASR into "trained-on" (base sources `{1,2,5,12,13}`)
   and "unseen" averages — Wave-1's global ASR averages over heterogeneous sources and
   under-reports worst-case effectiveness.
2. **Exact trigger-attributable ASR.** Add a third forward pass at eval: non-target test
   samples WITHOUT trigger, count how many are classified as target → `clean_floor_to_target`.
   Returned in metrics; derived `tasr = asr − clean_floor_to_target`. Floor was ≤ 0.07% in
   Wave-1, but the exact value makes the claim defensible.
3. **MultiKrum NormTracking refactor.** `CapturedMultiKrum` in
   `src/fl_v2/strategy/krum_wrappers.py` (lines 50–66) does not inherit `NormTrackingFedAvg`,
   so the MultiKrum column of the gradient-space matrix is empty (Wave-1 Table 2). Refactor
   via **composition**: keep `NormTrackingFedAvg._compute_and_log_norms` body for logging,
   delegate the selection step to Flower's `MultiKrum.aggregate_train`. (Direct multiple
   inheritance `NormTrackingMultiKrum(NormTrackingFedAvg, MultiKrum)` hits an MRO conflict —
   confirmed by exploration; use composition.)

**Plus one determinism baseline.** Re-run `cycle02-pixel-base-flame` with `seed: 43` (≤30
min). Expected ASR = 0.000. If yes → single-seed adaptive cells are interpretable against
the deterministic baseline. If no (ASR > 0.01) → escalate the 3 headline cells to 3 seeds
(+6 GPU-h).

**Null-config regression.** Existing Wave-1 cells loaded under the new schema must populate
new metric keys via default-0/NaN — no breakage of `summary.json` consumers.

**Critical files.**
- `src/fl_v2/training/server_eval.py` (lines 41–89).
- `src/fl_v2/utils/experiment_logger.py` (lines 136–149 — schema extension).
- `src/fl_v2/strategy/krum_wrappers.py` (composition refactor).
- New YAML: `configs/experiments/cycle_03/phase0/cycle02-pixel-base-flame-seed43.yaml`.

---

## WS-B — Day 1: small-LR adaptive attack

**Hypothesis.** BackdoorIndicator (Li et al. USENIX-Sec'24): reducing the malicious learning
rate nudges updates into the benign manifold while still backdooring. Reported FLAME-defended
BSR ~83% on CIFAR-10/ResNet18 at LR=0.01 (SGD). We test the Adam-analog.

**Implementation (smallest surface).**
- New YAML key `malicious-learning-rate` (default `""` → no override, reproduces Wave-1).
- In `src/fl_v2/client_app.py` where `lr` is set per round (≈ lines 331–340): when
  `is_malicious && in_attack_window && malicious-learning-rate != ""`, use the override.
- No new module; existing pixel-trigger / DBA data poisoning unchanged.

**Scientific guardrail (CRITICAL — the "no meaningless evaluations" gate).** Small-LR can
break the attack rather than evade the defense. **Required control:** small-LR × FedAvg MUST
run before the FLAME row is interpretable.
- small-LR × FedAvg ASR > 0.3 → attack real, FLAME row interpretable.
- small-LR × FedAvg ASR < 0.1 → attack broken by the LR itself; FLAME column meaningless.
  Sweep `malicious-learning-rate` ∈ {1e-4, 5e-5, 1e-5} × optional `malicious-local-epochs`
  ∈ {3, 5} until a working point is found.

**Critical files.** `client_app.py`, `pyproject.toml`, new YAMLs under
`configs/experiments/cycle_03/phase1_smalllr/`.

---

## WS-C — Day 2–3: Layer-Poisoning (LP) attack

**Hypothesis.** Zhuang et al. ICLR'24: restricting backdoor gradients to "backdoor-critical"
layers (top-k by gradient-magnitude contrast between clean and poisoned batches) shrinks the
L2 / direction signature in unmasked layers — enough to evade FLAME at ~89% reported BSR on
CIFAR-10/ResNet18 (closest comparator to our setup).

**Implementation.**
- New module `src/fl_v2/attacks_defenses/attacks/layer_poisoning.py`:
  - `select_critical_layers(named_params, clean_loader, poisoned_loader, k) → list[str]`:
    top-k layers by `‖∇L_poison − ∇L_clean‖`.
  - `make_lp_grad_mask(named_params, selected_layer_names) → dict[id, bool]`.
- Hook into `src/fl_v2/training/train.py` at the SAME point as Neurotoxin (line 141, after
  `loss.backward()` before `optimizer.step()`): zero gradients OUTSIDE selected layers
  (inverse polarity of Neurotoxin's coord-zeroing).
- YAML knobs: `attack-type: layer_poisoning`, `lp-num-layers` (default 3),
  `lp-layer-selection` (`gradient_magnitude` default; `manual` = comma-separated list for
  reproducibility tests).
- Critical-layer selection runs once per malicious client at attack-start round, cached for
  the run (deterministic).

**Scientific guardrail (CRITICAL).** Log selected layer names on the first attack round; they
must include classifier-head-adjacent layers (the backdoor moves samples to a specific output
class, so head + last conv block should rank high). If the heuristic picks unrelated layers,
it's a config bug, not a defense success.

**Scientific guardrail (CRITICAL).** Same control as small-LR: LP × FedAvg MUST show ASR > 0.3
before the FLAME row is interpretable. If LP breaks the attack, sweep `lp-num-layers`
∈ {2, 3, 5, 8}.

**Critical files.** New `attacks_defenses/attacks/layer_poisoning.py`, `client_app.py`
(`is_malicious + lp_mask` through `train_local`), `training/train.py` (line 141), `pyproject.toml`.
New test `tests/test_layer_poisoning.py` (`select_critical_layers` ranking + mask shape).

---

## WS-D — Day 4–7: A3FL — design TBD pending user paper review

**Hypothesis.** A3FL (Zhang et al. NeurIPS'23): bilevel optimization learns a trigger that
survives global-model unlearning. Reported high BSR against 12 defenses incl. FLAME on
CIFAR-10.

**Status — design deferred.** User is reading the A3FL paper before finalizing. **Open
question: malicious-client coordination.** A3FL's paper formulation implicitly shares trigger
state across malicious clients via the global model between rounds; our threat model forbids
inter-client comm. Three candidate adaptations:
1. Independent per-client trigger from a shared deterministic init (closest to no-coord; may
   give a weaker attack; easiest to make deterministic).
2. Single shared trigger seeded off run-seed (no per-client adaptation; weakest).
3. Paper-faithful sync-via-global-model (arguably Kerckhoffs-consistent since the global model
   is broadcast anyway; stretches "no inter-client comm").

**Re-engage the user to pick one before WS-D starts.**

**Scaffolding that can land before the decision** (no science impact):
- Skeleton `src/fl_v2/attacks_defenses/attacks/a3fl.py` with `learnable_trigger_init(rng_seed,
  trigger_shape)` and `optimize_trigger_inner_loop(model, trigger, loader, n_steps, rng)`.
  All RNG via `derive_seed` (`src/fl_v2/utils/runtime.py:43–54`).
- YAML knobs: `attack-type: a3fl`, `a3fl-inner-steps`, `a3fl-trigger-lr`,
  `a3fl-coordination-mode` (per the decision).

**Scientific guardrail (CRITICAL).** A3FL is the most expensive cell. Validate inner-loop
convergence on a 20-round smoke (attack window 5–15) before the 60-round matrix.

---

## WS-E — Day 8–9: 3DFed (conditional)

**Triggered only if WS-B/C/D all leave FLAME standing** (all of small-LR/LP/A3FL × FLAME ASR
< 0.5). 3DFed (Li et al. S&P'23): closed-loop feedback with decoy models. Heaviest
implementation; **determinism is the central risk** — cross-round per-malicious-client state
must not depend on Ray actor scheduling (same constraint that rejected the cached-global-proxy
Neurotoxin variant in Cycle 02 WS3). If triggered, write a focused 3DFed addendum to this
file before starting.

---

## WS-F — Day 10: Final matrix + Cycle-03 log

Run any missing cells. For each adaptive attack, log the headline FLAME-defended ASR and the
descriptive gradient-space metrics (Cohen's d on L2 norm, cos2mean, pairwise_cos over
attack-window rounds 10–35) alongside the Wave-1 static analogs. Write the Cycle-03 results
log to `./cycle03_log.md`. The log makes one primary claim per attack: "FLAME-defended ASR =
X; Cycle 04 implication: …".

---

## Cycle-03 matrix (single-seed except the Day-0 verification)

3 adaptive attacks × 4 defenses = **12 new cells**. Wave-1 static rows stay as reference.

```
attack             | FedAvg | MultiKrum | FoolsGold | FLAME
-------------------+--------+-----------+-----------+----------
small-LR-pixel     |  ctrl  |    ✓      |    ✓      |  HEAD
small-LR-dba       |  ctrl  |    ✓      |    ✓      |   ✓
LP-pixel           |  ctrl  |    ✓      |    ✓      |  HEAD
LP-dba             |  ctrl  |    ✓      |    ✓      |   ✓
A3FL               |  ctrl  |    ✓      |    ✓      |  HEAD
```

`ctrl` = mandatory guardrail (attack isn't broken by its own constraint). `HEAD` = headline
cell whose ASR decides the Cycle-04 go/no-go. Plus 1 determinism-baseline cell on Day 0
(pixel × FLAME × seed-43). Net-new compute ≈ 15 cells × ~1 h ≈ 15 GPU-h.

## Descriptive gradient-space metrics (evidence, not the claim)

For each adaptive attack's HEAD cell, log Cohen's d on L2 norm, cos2mean, pairwise_cos.
(Topk_energy dropped — Wave-1 showed it has no discriminative power.) Record alongside the
Wave-1 static analogs (pixel cos2mean d=−1.34, dba d=−1.48 under FedAvg) so the Cycle-04
designer can reason about WHICH mechanism each adaptive attack exploits. **Interpret only for
cells where ASR > 0.1.**

## Verdict criterion (ASR-based, the cycle's binary output)

For each HEAD cell:
- **ASR > 0.5** → attack breaks FLAME; this is the target Cycle 04 needs. The gradient-space
  d table is descriptive input to the Cycle-04 design, not the gate.
- **0.1 ≤ ASR ≤ 0.5** → partial breakage; the attack class needs strengthening (longer
  window, more inner-loop steps) before Cycle 04 can use it.
- **ASR < 0.1** → FLAME holds. If ALL three reproduced attacks fall here, escalate to 3DFed
  or pivot the threat model.

## Critical files (summary)

**Modified.** `training/server_eval.py` (WS-A), `utils/experiment_logger.py` (WS-A),
`strategy/krum_wrappers.py` (WS-A), `client_app.py` (WS-B/C/D gating),
`training/train.py` (WS-C grad-mask, WS-D trigger hook), `pyproject.toml` (new knobs).

**New.** `attacks_defenses/attacks/layer_poisoning.py` (WS-C),
`attacks_defenses/attacks/a3fl.py` (WS-D skeleton), `configs/experiments/cycle_03/<phase>/*.yaml`,
`docs/cycle_03/cycle03_log.md` (WS-F), `tests/test_layer_poisoning.py`, `tests/test_a3fl.py`.

## Verification

- **Null-config regression.** small-LR with no override = pixel baseline (bit-identical);
  LP with `lp-num-layers: -1` (no masking) = pixel baseline; A3FL with `a3fl-inner-steps: 0`
  = fixed fallback trigger (no-op). Each diffed against the Wave-1 result.
- **Determinism baseline (Day 0).** pixel × FLAME × seed-43 → ASR 0.000 within float
  tolerance; else escalate headline cells to 3 seeds.
- **Unit tests.** `select_critical_layers` (synthetic-batch ranking), `make_lp_grad_mask`
  (param shapes), `a3fl.learnable_trigger_init` (deterministic given seed).
- **Per-cell scientific gates.** `ctrl` cell ASR > 0.3 before the FLAME cell; LP layer-name
  sanity; A3FL inner-loop convergence.

## Decision points pending

1. **A3FL coordination mechanism (WS-D)** — user reads paper, decides before Day 4.
2. **3DFed go/no-go (WS-E)** — auto-triggered if WS-B/C/D leave FLAME standing.
3. **Multi-seed escalation** — if Day-0 seed-43 ≠ 0.000, escalate 3 headline cells to 3 seeds.
4. **NormClip** — dropped per Wave-1; reinstate as a tight-clip (`clip-norm: 10`) ablation
   only if a reviewer demands it.

---

## Suggested starting point for a fresh session

1. Read this file, `./wave1_log.md`, and `../../CLAUDE.md`.
2. Start **WS-A**: read the 3 target files, propose the minimal diff, apply, run
   `tests/test_ws5_defenses.py` + `tests/test_dba_trigger.py` for regression, commit.
3. Submit the seed-43 verification YAML; confirm ASR ≈ 0.000.
4. Proceed to WS-B (small-LR), running the `ctrl` cell first per the guardrail.
