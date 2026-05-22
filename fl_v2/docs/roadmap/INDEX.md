# Research Cycle Index

A chronological record of research planning cycles for this project. Each cycle is a self-contained document describing its hypothesis, phases, success criteria, and paper reading list. Closed cycles carry a headline finding; active cycles show their status.

The cycle is the unit of planning: typically 2–3 weeks of work, one well-defined scientific question, and a clearly-scoped set of phases. When a cycle closes we write its headline finding here and start a new cycle document for the next question. Old cycles are never overwritten — they become the research trajectory of the project.

## Cycles

| Cycle | Dates | Theme | Status | Headline result |
|---|---|---|---|---|
| [01](cycle_01_platform_and_representation_baseline.md) | 2026-03-25 → 2026-04-15 | Platform establishment + representation-space baseline | **closed** | Pixel trigger is a "joint weak attack" (middle-region cluster, linearly separable from genuine target). Bagdasaryan replacement amplifies but does not change the representation-space mechanism. The probe direction is nearly orthogonal to top-PC, so unsupervised spectral defenses miss what a supervised probe trivially catches. |
| ~~02-pivot~~ | 2026-04-15 → 2026-05-12 | Designed attacks + representation-space framework (**retired** — ran on pre-audit codebase; archived under `cycle_02_pivot/`) | **retired** | Findings not trusted; superseded. |
| [02](cycle_02_gradient_space_mechanism_study.md) | 2026-05-13 → TBD | Gradient-space backdoor mechanism study — modern attacks/defenses on the audit-fixed codebase, threat-model design, client-side prototype | **active** | — |
| 03 | TBD | Defense design (informed by Cycle-02 go/no-go) — gradient-space law or other-space pivot | — | — |

## Reading order for a new collaborator

If you're reading this for the first time and want to understand the project state in one session:

1. **`../../CLAUDE.md`** — project identity, platform status, near-term priorities (short, high signal)
2. **[cycle_02_gradient_space_mechanism_study.md](cycle_02_gradient_space_mechanism_study.md)** — the current cycle: research question, threat model, phases
3. **`../active_reading/`** — the literature active-reading cards the current cycle is built on (BadNets, Krum, DBA, Neurotoxin, FoolsGold, FLAME, STRIP)

The Cycle-01 representation-space documents (`../cycle01_docs/`) and the retired Cycle-02-pivot documents (`../cycle_02_pivot/`) are historical context. Their quantitative findings were produced on the pre-audit codebase and are **not** current ground truth — Cycle 02 re-baselines from scratch.

## Document conventions

- **Cycle files** are named `cycle_NN_short_descriptive_label.md` where `NN` is the two-digit cycle number and the label is 3–5 words describing the theme. Filenames should tell you what the cycle did without opening the file.
- **Status** is one of `planned`, `active`, or `closed`.
- **Headline result** is filled in when a cycle closes and must be a single sentence that captures the most important scientific finding from that cycle.
- **Paper reading lists** are grouped by theme (not prioritized), with a one-sentence rationale per paper explaining why it matters to us.
- **Success criteria** are quantitative thresholds against the representation-space framework metrics wherever possible.
- **Closed-cycle documents are never edited** except to fix factual errors — they are the research trajectory and rewriting them would lose the historical context.
