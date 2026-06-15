# Cycle 04 — Orchestration Guide

**The durable plan** is `../roadmap/cycle_04_fusion_layer_backdoors.md`. Read it first. This folder
holds the *orchestration* scaffolding that lets serial Claude + Codex sessions execute it.

## Session model

- **Orchestrator session** (the one that created this folder): does NOT implement. It owns the plan,
  the decisions, the per-task SPEC contracts, and the kickoff prompts. It updates this folder as
  tasks complete.
- **Per-task build session (Claude):** one fresh session per task T0…T7. It reads the task SPEC,
  implements, writes tests + viz, and drives its GATE to green.
- **Per-task review session (Codex):** one fresh Codex session per task. It reviews the build
  session's diff against the SPEC + the paper/reference for **scientific correctness only**, and
  writes a `REVIEW.md`. It never commits code.

## The per-task loop

1. Orchestrator hands the build session `tasks/T<N>_SPEC.md` + the kickoff prompt.
2. Build session implements → writes `fl_v3/collab/T<N>/SPEC.md` (filled from the template),
   tests, viz; drives the GATE.
3. Codex session reviews → writes `fl_v3/collab/T<N>/REVIEW.md` (severity-tagged).
4. Build session (with the user) triages → fixes → Codex re-reviews the delta.
5. Decisions/rationale land in `fl_v3/collab/findings_log.md`. Orchestrator marks the task done and
   issues the next task's SPEC + kickoff.

## Where the working files live

- **Durable, here (orchestrator-owned):** the plan (roadmap), `decisions.md`, the `collab/`
  templates, the `tasks/T*_SPEC.md` contracts, the `kickoff/` prompts.
- **Working, in the new codebase (`fl_v3/`, created in T0):** `fl_v3/collab/T<N>/{SPEC,REVIEW}.md`
  and `fl_v3/collab/findings_log.md`. T0 establishes `fl_v3/collab/` and copies the templates there.

## Task sequence (gated; pace by gates, not days)

T0 scaffold+determinism → T1 nuScenes data+V1 → T2 fusion model+V2/V3 → T3 FedAvg baseline
(PLATFORM MILESTONE) → T4 eval+ASR+V4 → T5 attack suite+V5 → T6 defense suite+V6 → T7 matrix+analysis.
**Platform (T0–T4) before attack (T5)/defense (T6); mini = engineering smoke, trainval = science.**

## Crown-jewel review targets (every task)

Defense-algorithm parity (paper + `fl_v2` oracle, *implementation equivalence only*), bit-determinism,
null-config, ASR/utility metric definitions, coordinate/box conventions.
