# Cycle 03 — Stronger Adaptive Backdoor Attacks for FL

**Dates:** 2026-05-28 → TBD
**Status:** active
**Prerequisite:** Cycle 02 Wave-1 closed — FLAME drives ASR to 0.000 across all 3 static
attacks on the audit-fixed codebase (see `cycle_02_gradient_space_mechanism_study.md` and
the authoritative results log `../cycle_02/wave1_log.md`).
**Headline result (filled in when closed):** —

> This document is the permanent record for Cycle 03. The session working-plan that
> produced it carried the same content at finer implementation granularity (per-file diffs,
> exact line numbers); the durable references are this document and the Wave-1 log.

---

## 1. Why this cycle exists

Cycle 02 Wave-1 produced a blunt result: **FLAME drives ASR to 0.000 against all three
implemented static attacks** (pixel, model_replacement, DBA) on our threat model. The FLAME
implementation was audited 10/10 clean, so the result is genuine, not an artifact.

This has one decisive consequence for the thesis trajectory:

> **Until we have an attack that FLAME does NOT shut down, there is no meaningful target for
> a new defense.** Any "improvement" over FLAME on attacks FLAME already handles is
> statistically indistinguishable from FLAME at the 0.000 floor — it would be busywork.

So Cycle 03 is fundamentally about **building a stronger attacker**, not about validating a
defense or any specific signal noticed in Wave-1 (the cos2mean separation, NormClip being a
no-op, DBA bypassing FoolsGold — all recorded in the Wave-1 log as intuition for later, not
as the contribution). The cycle is in **step 1 (reproduce) of the project's
reproduce → understand → build-intuition → design pipeline**, applied to adaptive attacks.

## 2. Research question

**Does any adaptive backdoor attack from the 2023-2025 literature break FLAME's 0.000
baseline on our threat model?**

- If yes (≥1 attack drives FLAME-defended ASR > 0.5) → Cycle 04 designs a new general
  defense, informed by what the breaking attack reveals about FLAME's failure mode.
- If no (all reproduced attacks leave FLAME < 0.1) → either we picked the wrong attacks
  (escalate: 3DFed, real-time probing) or FLAME is genuinely robust on this threat model
  and Cycle 04 changes the threat model (malicious-majority, multimodal, edge regime).

The verdict turns on **ASR**, not on gradient-space metrics. Gradient-space Cohen's d
values are logged as descriptive input for the Cycle-04 designer, not as the gate.

## 3. Threat model (delta from Cycle 02)

Frozen platform is **identical to Wave-1**: GTSRB-43 · ResNet18 from scratch · Adam
lr=0.001 cosine → 0.0001 · 50 clients · Dirichlet α=0.5 · 60 rounds · attack window 10–35 ·
base poison regime · m=10 malicious (20%) · target class 14 (Stop) · poison-fraction 0.5 ·
trigger-size 4 · seed 42 (43 for the determinism check).

**The only axis that changes is attacker knowledge of the defense.** Wave-1 attackers were
defense-aware but *static* — the attack did not adapt to a specific aggregator. Cycle-03
attackers are **defense-aware static-adaptive**: they know the exact aggregator (FLAME),
its hyper-parameters (HDBSCAN `min_cluster_size = ⌈N/2⌉+1`, median-norm clipping, λ-scaled
noise), and design the attack to evade *that specific recipe*. Still no real-time probing of
the live defender; still `m < n/2`; still no inter-client communication.

## 4. Attacks and defenses studied

**Attacks** (reproduced from literature, ordered by implementation cost):

- **small-LR** (BackdoorIndicator, Li et al. USENIX-Sec'24) — reduce the malicious learning
  rate to nudge updates into the benign manifold while still backdooring. Reported FLAME-
  defended BSR ~83% on CIFAR-10. Smallest code surface (one YAML knob).
- **LP / Backdoor-Critical-Layers** (Zhuang et al. ICLR'24) — restrict backdoor gradients
  to top-k layers by clean-vs-poisoned gradient-magnitude contrast, shrinking the L2 /
  direction signature in unmasked layers. Reported ~89% BSR on CIFAR-10/ResNet18 — the
  closest comparator to our setup.
- **A3FL** (Zhang et al. NeurIPS'23) — bilevel optimization learns a trigger that survives
  global-model unlearning. Reported high BSR against 12 defenses incl. FLAME.
  *Coordination-mechanism design deferred pending paper review (see §7).*
- **3DFed** (Li et al. S&P'23) — closed-loop feedback with decoy models. **Conditional**:
  only if the three above leave FLAME standing. Heaviest implementation; determinism is the
  central risk (cross-round per-client state must not depend on Ray actor scheduling).

**Defenses** — the 4-defense gradient-space matrix: `FedAvg`, `multikrum`, `foolsgold`,
`flame`. **NormClip is dropped** — Wave-1 showed `clip-norm=100` is a no-op on this threat
model (too loose to bite honest gradients); reinstate only as a tight-clip ablation if a
reviewer demands it.

## 5. Phases

Workstreams, ordered for cheap-decisive-first (≈10 working days at Wave-1 cadence):

- **WS-A — Day 0: logging hardening + determinism baseline. ✅ DONE.** Three batched
  additions: per-class ASR breakdown (`asr_src<c>`), exact trigger-attributable ASR
  (`backdoor_attribute_asr = asr − clean_floor_to_target`, computed by reusing the clean
  forward pass — no extra model call), and a MultiKrum → NormTracking composition refactor
  (`NormTrackingMultiKrum`, fills the empty MultiKrum column). Seed-43 re-run of pixel ×
  FLAME **passed** (peak ASR 0.0081% / final 0.0008%) → the 0.000 baseline is seed-robust,
  not a seed-42 artifact; single-seed adaptive cells are interpretable.
- **WS-B — Day 1: small-LR attack.** One YAML knob (`malicious-learning-rate`). No new
  module.
- **WS-C — Day 2–3: LP attack.** New module `attacks_defenses/attacks/layer_poisoning.py`;
  gradient mask hooked into `train.py` (same point as Neurotoxin, inverse polarity).
- **WS-D — Day 4–7: A3FL.** New module `attacks_defenses/attacks/a3fl.py` with bilevel
  trigger optimization. Coordination mechanism TBD (§7).
- **WS-E — Day 8–9: 3DFed (conditional).** Only if WS-B/C/D leave FLAME standing.
- **WS-F — Day 10: final matrix + Cycle-03 log.** Extended outcome table + descriptive
  gradient-space d's; verdict per attack; written to `../cycle_03/cycle03_log.md`.

**Matrix:** 3 adaptive attacks × 4 defenses = 12 new cells, single-seed (Wave-1 static rows
stay as reference). Net-new compute ≈ 15 GPU-h.

**Scientific guardrails (the "no meaningless evaluations" gates):**
- Every adaptive attack MUST first pass a control cell (× FedAvg ASR > 0.3) before its FLAME
  row is interpretable — otherwise the attack is broken by its own constraint, not by the
  defense, and the FLAME number says nothing.
- LP: log selected layer names on the first attack round; they must include classifier-head-
  adjacent layers, else it's a config bug.
- A3FL: validate inner-loop convergence on a 20-round smoke before the 60-round matrix.
- Gradient-space d's interpreted only for cells where ASR > 0.1.

## 6. Success criteria / deliverables

- The extended outcome table (Wave-1 static rows + Cycle-03 adaptive rows).
- Per-cell descriptive gradient-space Cohen's d (L2 norm, cos2mean, pairwise_cos) — input
  for Cycle-04 design, not the gate.
- A one-line verdict per reproduced attack: did it break FLAME (ASR > 0.5), partially
  (0.1–0.5), or not (< 0.1)?
- A **go/no-go for Cycle 04**: a real defense-design target exists, or escalate
  attacks / change the threat model.

## 7. Reading list

Active reading for this cycle (cards under `../active_reading/` as written):

- **BackdoorIndicator** (Li et al. USENIX-Sec 2024) — small-LR stealthy poisoning; gives
  the concrete ~83% FLAME-defended BSR prediction WS-B tests.
- **Backdoor-Critical Layers / LP** (Zhuang et al. ICLR 2024) — layer-selection heuristic;
  implementation reference for WS-C.
- **A3FL** (Zhang et al. NeurIPS 2023) — bilevel adversarial-trigger optimization;
  implementation reference for WS-D. *Read before finalizing the coordination mechanism
  decision (independent per-client trigger vs single shared trigger vs paper-faithful
  sync-via-global-model — our threat model forbids inter-client comm, so the paper's
  implicit coordination needs an explicit adaptation choice).*
- **3DFed** (Li et al. S&P 2023) — closed-loop feedback; read only if WS-E triggers.

Background (already covered in Cycle 02): FLAME, FoolsGold, DBA, Neurotoxin.

## 8. Decision points pending

1. **A3FL coordination mechanism (WS-D)** — user reading the A3FL paper; decision required
   before Day 4.
2. **3DFed go/no-go (WS-E)** — auto-triggered if WS-B/C/D leave FLAME standing.
3. **Multi-seed escalation** — ✅ RESOLVED: the Day-0 seed-43 baseline passed (≪ 0.01), so
   single-seed headline cells are interpretable; no 3-seed escalation needed.
