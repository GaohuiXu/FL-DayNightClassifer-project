# Research Cycle Index

A chronological record of research planning cycles for this project. Each cycle is a self-contained document describing its hypothesis, phases, success criteria, and paper reading list. Closed cycles carry a headline finding; active cycles show their status.

The cycle is the unit of planning: typically 2–3 weeks of work, one well-defined scientific question, and a clearly-scoped set of phases. When a cycle closes we write its headline finding here and start a new cycle document for the next question. Old cycles are never overwritten — they become the research trajectory of the project.

## Cycles

| Cycle | Dates | Theme | Status | Headline result |
|---|---|---|---|---|
| [01](cycle_01_platform_and_representation_baseline.md) | 2026-03-25 → 2026-04-15 | Platform establishment + representation-space baseline | **closed** | Pixel trigger is a "joint weak attack" (middle-region cluster, linearly separable from genuine target). Bagdasaryan replacement amplifies but does not change the representation-space mechanism. The probe direction is nearly orthogonal to top-PC, so unsupervised spectral defenses miss what a supervised probe trivially catches. |
| [02](cycle_02_designed_attacks_and_client_defenses.md) | 2026-04-15 → TBD | Designed attacks (optimization-based feature-space) + non-IID-aware client-side detection | **active** | — |
| 03 | TBD | Adaptive attacks + ViT migration (planned) | — | — |

## Reading order for a new collaborator

If you're reading this for the first time and want to understand the project state in one session:

1. **`../../CLAUDE.md`** — project identity, platform status, current scientific findings, near-term priorities (short, high signal)
2. **`../representation_space_framework.md`** — the 4-axis analytical framework (methodology reference)
3. **`../pixel_trigger_baseline.md`** — Cycle 01 closed baseline profile (pixel trigger)
4. **`../model_replacement_profile.md`** — Cycle 01 closed profile (Bagdasaryan model replacement, negative result)
5. **[cycle_02_designed_attacks_and_client_defenses.md](cycle_02_designed_attacks_and_client_defenses.md)** — what we're working on now

Steps 3 and 4 are the empirical ground truth that Cycle 02 builds on.

## Document conventions

- **Cycle files** are named `cycle_NN_short_descriptive_label.md` where `NN` is the two-digit cycle number and the label is 3–5 words describing the theme. Filenames should tell you what the cycle did without opening the file.
- **Status** is one of `planned`, `active`, or `closed`.
- **Headline result** is filled in when a cycle closes and must be a single sentence that captures the most important scientific finding from that cycle.
- **Paper reading lists** are grouped by theme (not prioritized), with a one-sentence rationale per paper explaining why it matters to us.
- **Success criteria** are quantitative thresholds against the representation-space framework metrics wherever possible.
- **Closed-cycle documents are never edited** except to fix factual errors — they are the research trajectory and rewriting them would lose the historical context.
