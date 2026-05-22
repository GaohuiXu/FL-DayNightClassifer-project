# Cycle 02 — Gradient-Space Backdoor Mechanism Study

**Dates:** 2026-05-13 → TBD
**Status:** active
**Prerequisite:** the audit-fixed, bit-deterministic codebase (post 7-fix audit + Pass 2/3a).
**Headline result (filled in when closed):** —

> This document **replaces** the retired Cycle-02 plan ("Designed Attacks & Client-Side
> Defenses"). That plan and the Cycle-01 representation-space findings it built on were
> produced on the pre-audit codebase and are no longer trusted as quantitative ground
> truth. Cycle 02 is redesigned from the literature up.

---

## 1. Why this cycle exists

The thesis goal is fixed: **propose a new *general* backdoor defense for federated
learning, targeting USENIX-SECURITY**. "General" means the defense must not be tuned to one
trigger type or one attacker assumption. Such a defense cannot be designed without first
understanding, on a trustworthy codebase, how modern attacks and defenses actually behave.

A literature pass (BadNets, Krum, DBA, Neurotoxin, FoolsGold, FLAME, STRIP) shows the field
fights almost entirely in **one space — the per-round client update vector**. Every defense
is a low-dimensional projection of that vector (Krum: a distance scalar; FoolsGold: pairwise
cosine; FLAME: cosine cluster + norm). Every modern attack wins by keeping *that* projection
benign while backdooring in the directions the projection discards (DBA keeps Euclidean
distance small; Neurotoxin keeps high-activity-coordinate energy benign).

## 2. Research question

**Does gradient/update space contain a "law" — a signature that modern backdoor attacks
cannot remove while still backdooring?**

- If yes → the next cycle designs the new defense around that law.
- If no → the next cycle moves to other spaces (frequency, representation).

Both outcomes are publishable cycle results. This cycle is mechanism-understanding and
threat-model design, not a finished defense.

## 3. Threat model (primary deliverable)

Encoded in YAML config; full rationale in `docs/cycle_02/` (threat-model note).

- **Platform:** GTSRB 43-class, ResNet18 (ImageNet-init), 50 clients, Dirichlet α=0.5,
  `fraction-train=1.0`, `partition-seed` fixed and decoupled from the model seed.
- **Attacker goal:** targeted all-to-one backdoor; target = **class 14 (Stop)** (safety-
  critical, visually distinct — not the most-common class 2, which inflates ASR).
- **Attacker control:** `m=10` of `n=50` malicious clients; hard bound `m < n/2`
  (malicious-majority is provably unsolvable by aggregation — out of scope).
- **Attacker knowledge:** Kerckhoffs — knows the architecture and that a robust aggregator
  runs; **static** this cycle (no real-time probing). Adaptive attacker = paper push.
- **Poisoned-data regime:** base (common-class sources) vs edge (rare-tail sources) — the
  key 2-level factor; edge is the primary scientific interest.
- **Durability:** attacks fire only inside an attack window; runs continue past it so
  post-attack ASR decay is observed (the curve is baked in; the timing grid is deferred).

## 4. Attacks and defenses studied

**Attacks** — `pixel` (trivial baseline), `model_replacement` (loud baseline), **`dba`**
(distributed), **`neurotoxin`** (constrained, durable).

**Defenses** — the gradient-space landscape, six mechanism families: `FedAvg` (no defense),
`norm_clipped`, `fed_median`, `multikrum`, **`foolsgold`** (similarity), **`flame`**
(clustering + dynamic clip + noise).

SIG / patched / shadowed triggers and frequency-/representation-space defenses (FreqFed,
CrowdGuard) are explicitly deferred to later cycles.

## 5. Phases

1. **Re-organization + instrumentation.** Repo organized under `cycle_02/`; per-client
   per-round gradient-space metrics logged (update L2 norm, cosine-to-mean, pairwise
   cosine, top-k coordinate-energy split).
2. **Re-baseline + first signal (gated).** Job 1: clean FL, 100 rounds — confirm
   convergence + determinism. Job 2: pixel attack vs FedAvg, 100 rounds — confirm the
   metrics visibly separate malicious from honest updates. Reviewed before any batch.
3. **Attack build-out.** DBA and Neurotoxin attacks; FoolsGold and FLAME defenses.
4. **Evasion matrix.** 4 attacks × 4 gradient-space metrics, on FedAvg raw-signature runs.
5. **Outcome table.** 4 attacks × 6 defenses: ASR, trigger-attributable ASR, durability.
6. **Client-side prototype.** A rough STRIP-style client-side validation probe with
   federated consensus — feasibility and per-round cost only.

## 6. Success criteria / deliverables

- A written, rationalized threat model.
- The filled evasion matrix — a metric that separates *every* attack is a candidate law; a
  matrix with no clean column is the "gradient-space is too hard" conclusion.
- The attack × defense outcome table.
- The client-side prototype's feasibility + wallclock cost.
- A **go/no-go** for the next cycle (exploit a law vs. move to another space).

## 7. Reading list

Read (active-reading cards under `docs/active_reading/`):
- **BadNets** (Gu et al. 2017) — canonical backdoor threat model.
- **How to Backdoor FL** (Bagdasaryan et al. 2020) — model replacement / scaling.
- **Krum** (Blanchard et al. 2017) — Byzantine-robust aggregation baseline.
- **DBA** (Xie et al. 2020) — distributed backdoor; defeats distance/similarity defenses.
- **Neurotoxin** (Zhang et al. 2022) — constrained, durable; defeats norm clipping.
- **FoolsGold** (Fung et al. 2020) — similarity-based defense.
- **FLAME** (Nguyen et al. 2022) — SOTA: clustering + dynamic clip + noise.
- **STRIP** (Gao et al. 2019) — inference-time / client-side perturbation defense.

Next cycle (queued, not read yet): FreqFed (NDSS 2024), CrowdGuard (NDSS 2023),
3DFed (S&P 2023), FLDetector (KDD 2022), MESAS (CCS 2023).
