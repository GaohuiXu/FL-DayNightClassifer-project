# T<N> — SPEC: <task title>

> Written by the **build (Claude) session** before/while implementing. It is the contract the Codex
> review checks against. Copy this template to `fl_v3/collab/T<N>/SPEC.md` and fill every section.

## 1. Scientific intent (one paragraph)
What this task must accomplish and why it matters to the Cycle-04 plan. Link the plan section.

## 2. Scope
- **In scope:** …
- **Out of scope / deferred:** …
- **Files created/changed:** `fl_v3/…` (list)

## 3. Invariants (must hold; Codex checks each)
- **Bit-determinism:** any new RNG via the determinism harness (`derive_seed` analog); same-seed →
  bit-identical. Banned ops: atomic scatter, `grid_sample` backward, non-stable sort/topk, flash-attn.
- **Null-config:** the clean/`poison_rate=0` path reproduces the baseline bit-for-bit (where applicable).
- **Oracle parity (carry-over only):** given the same input vectors, the reimplemented logic matches
  the `fl_v2` oracle's decision on a saved fixture. (Implementation equivalence ONLY — not scientific
  validity on AD.)
- **Threat-model / metric knobs honored:** (list the relevant ones, e.g. `ρ`, `m_r` vs `f_r`,
  `τ_clean`, `d_clean`, `δ`).

## 4. Reference (ground truth for the review)
- Paper section(s): …
- Reference implementation (file/URL, Apache-2.0 or note): …
- The exact equations / definitions being reproduced: …

## 5. Scientific failure modes to check (point Codex here)
- e.g. units/calibration transplanted from a different optimizer/scale (the FLAME λ class of bug)
- e.g. coordinate-frame / yaw / class-mapping error
- e.g. ASR eligibility/denominator computed wrong
- …

## 6. GATE (objective pass criteria — copy from the plan's T<N> gate)
- [ ] …
- [ ] …

## 7. Self-review — what I'm least sure about
The 2–3 things most likely wrong; the soft spots Codex should attack hardest.
