# Research Cycle Index (active — fl_v3 / AD era)

The current roadmap for the project. As of **Cycle 04** the active platform is **`fl_v3/`** (federated
multimodal AD perception) and the research docs live with it, in `fl_v3/docs/`. Earlier cycles' full
docs are **frozen** in their codebase of origin (`fl_v2/docs/`) and linked below — docs freeze with
their era, and each era's roadmap references the prior frozen one.

**Codebase eras**
- **`fl_v2/`** — frozen GTSRB-era platform (cycles 01–03). `fl_v2/docs/` is the historical record; do
  not modify. Used in Cycle 04 only as an *implementation oracle*.
- **`fl_v3/`** — the active federated multimodal AD-perception platform (cycle 04+).

## Cycles

| Cycle | Dates | Theme | Status | Headline result |
|---|---|---|---|---|
| [01](../../../fl_v2/docs/roadmap/cycle_01_platform_and_representation_baseline.md) | 2026-03-25 → 04-15 | Platform + representation-space baseline | **closed** (frozen) | Pixel trigger is a "joint weak attack"; replacement amplifies but doesn't change the mechanism; probe direction ~orthogonal to top-PC, so unsupervised spectral defenses miss it. |
| ~~02-pivot~~ | 2026-04-15 → 05-12 | Designed attacks + representation framework (pre-audit) | **retired** | Findings not trusted; superseded. |
| [02](../../../fl_v2/docs/roadmap/cycle_02_gradient_space_mechanism_study.md) | 2026-05-13 → 05-29 | Gradient-space backdoor mechanism study (GTSRB) | **closed** (frozen) | FLAME drives ASR→0.000 on all 3 static attacks; MultiKrum bimodal; FoolsGold evaded by DBA; NormClip a no-op. FLAME's clean honest-rejection FPR = 0.469. FLAME ≡ HDBSCAN majority filter. |
| [03](../../../fl_v2/docs/roadmap/cycle_03_stronger_adaptive_attacks.md) | 2026-05-28 → 06-15 | Stronger adaptive attacks to break FLAME's 0.000 (GTSRB) | **paused** | Small-LR didn't break FLAME under Adam/GTSRB; GTSRB saturated (α=0.2 doesn't fragment the honest cluster). Thesis pivoted to real AD. **Paused, not closed** — LP/A3FL/3DFed unimplemented; a separate session writes the summary. |
| [04](cycle_04_fusion_layer_backdoors.md) | 2026-06-15 → TBD | **Federated multimodal AD perception platform + backdoor attack/defense benchmark** — bit-deterministic camera+LiDAR BEV/3D detection on nuScenes (`fl_v3/`); fusion-aware attack suite × general defense suite (FLAME, FoolsGold, MultiKrum, FedMedian, NormClip). Platform-first; not bonded to FLAME. | **active** | — |

## Cycle 04 working folder

Orchestration + execution scaffolding is in [`../cycle_04/`](../cycle_04/): the
[orchestration model](../cycle_04/README.md), [confirmed decisions D1–D8](../cycle_04/decisions.md),
and the per-task contracts + kickoffs:
- **T0** (done, Codex-PASS): [contract](../cycle_04/tasks/T0_SPEC.md) · [kickoff](../cycle_04/kickoff/T0_kickoff.md)
- **T1** (done, Codex-PASS): [contract](../cycle_04/tasks/T1_SPEC.md) · [kickoff](../cycle_04/kickoff/T1_kickoff.md)
- **T2** (done, Codex-PASS): [contract](../cycle_04/tasks/T2_SPEC.md) · [kickoff](../cycle_04/kickoff/T2_kickoff.md)
- **T3** (issued): [contract](../cycle_04/tasks/T3_SPEC.md) · [kickoff](../cycle_04/kickoff/T3_kickoff.md)

Per-task working `SPEC.md`/`REVIEW.md` + `findings_log.md` and the templates live in `fl_v3/collab/`.

## Reading order for a new collaborator (Cycle 04)

1. [`cycle_04_fusion_layer_backdoors.md`](cycle_04_fusion_layer_backdoors.md) — the approved plan.
2. [`../cycle_04/README.md`](../cycle_04/README.md) — the orchestrator + serial-worker session model.
3. [`../cycle_04/decisions.md`](../cycle_04/decisions.md) — the locked D1–D8.
4. The frozen Cycle-02/03 context in `fl_v2/docs/` (why GTSRB was exhausted) — linked above.

## Document conventions

- Cycle files: `cycle_NN_short_descriptive_label.md`. Status ∈ {planned, active, paused, closed}.
- Closed/paused cycle docs are edited only to fix factual errors — they are the research trajectory.
- Active research docs live with the active codebase (`fl_v3/docs/`); AI operating instructions
  (`CLAUDE.md`, `AGENTS.md`) live at the repo root; per-task working collab files live in
  `fl_v3/collab/`.
