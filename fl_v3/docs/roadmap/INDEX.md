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

> **2026-07-10 execution update:** `docs/cycle_04/` remains the Cycle-04
> experimental-design/decision record. Active cross-session coordination has moved
> to [`../../usenix27_orchestra/ORCHESTRA.md`](../../usenix27_orchestra/ORCHESTRA.md),
> with [session contracts](../../usenix27_orchestra/SESSIONS.md) and
> [copy-ready kickoffs](../../usenix27_orchestra/KICKOFFS.md). `fl_v3/collab/` is
> read-only historical evidence for the new stage.

Orchestration + execution scaffolding is in [`../cycle_04/`](../cycle_04/): the
[orchestration model](../cycle_04/README.md), [confirmed decisions D1–D8](../cycle_04/decisions.md),
and the per-task contracts + kickoffs:
- **T0** (done, Codex-PASS): [contract](../cycle_04/tasks/T0_SPEC.md) · [kickoff](../cycle_04/kickoff/T0_kickoff.md)
- **T1** (done, Codex-PASS): [contract](../cycle_04/tasks/T1_SPEC.md) · [kickoff](../cycle_04/kickoff/T1_kickoff.md)
- **T2** (done, Codex-PASS): [contract](../cycle_04/tasks/T2_SPEC.md) · [kickoff](../cycle_04/kickoff/T2_kickoff.md)
- **T3** (done, Codex-PASS): [contract](../cycle_04/tasks/T3_SPEC.md) · [kickoff](../cycle_04/kickoff/T3_kickoff.md)
- **T4** (done, Codex-PASS): [contract](../cycle_04/tasks/T4_SPEC.md) · [kickoff](../cycle_04/kickoff/T4_kickoff.md)
- **T5** (built + reviewed; **PAUSED** — camera-only backdoor non-viable, pilot negative; null uninterpretable on the undertrained/diluted checkpoint, see D14/D15): [contract](../cycle_04/tasks/T5_SPEC.md) · [kickoff](../cycle_04/kickoff/T5_kickoff.md)
- **Speedup + Clean-Baseline Diagnostics** (**DONE** — D15/D16, historical Alvis/A40 context): `determinism-level` knob ~3×/step; overcommit a measured dead end; weak model = FL-undertraining + FedAvg dilution, not architecture. Arrhenius supersedes the old bf16 default with `fp32` reference and `fp16` AMP + GradScaler for sparse training. [charter](../cycle_04/kickoff/speedup_kickoff.md) · [findings](../../collab/speedup/speedup_session_findings.md) · [decision](../../collab/speedup/D15_D16_decision_for_orchestrator.md)
- **Model Capability + Recipe (MCR)** (**historical input to the active Orchestra** — D17): its results and audits remain evidence; the corrected CL backbone and Protocol-B work are now governed by `fl_v3/usenix27_orchestra/`. [charter](../cycle_04/kickoff/model_capability_kickoff.md)

Historical per-task `SPEC.md`/`REVIEW.md` and `findings_log.md` remain in the read-only
`fl_v3/collab/`. New worker/reviewer packages live in
`fl_v3/usenix27_orchestra/handoffs/Sxx/`.

## Reading order for a new collaborator (Cycle 04)

1. [`../../usenix27_orchestra/ORCHESTRA.md`](../../usenix27_orchestra/ORCHESTRA.md) — active objective/protocol/gates.
2. [`../../usenix27_orchestra/SESSIONS.md`](../../usenix27_orchestra/SESSIONS.md) and [`../../usenix27_orchestra/KICKOFFS.md`](../../usenix27_orchestra/KICKOFFS.md) — active collaboration contracts.
3. [`cycle_04_fusion_layer_backdoors.md`](cycle_04_fusion_layer_backdoors.md) — approved Cycle-04 experimental design.
4. [`../cycle_04/decisions.md`](../cycle_04/decisions.md) — historical locked decisions.
5. The frozen Cycle-02/03 context in `fl_v2/docs/` (why GTSRB was exhausted) — linked above.

## Document conventions

- Cycle files: `cycle_NN_short_descriptive_label.md`. Status ∈ {planned, active, paused, closed}.
- Closed/paused cycle docs are edited only to fix factual errors — they are the research trajectory.
- Experimental-design/roadmap docs live in `fl_v3/docs/`; AI operating instructions
  live at the repo root; active session work lives in `fl_v3/usenix27_orchestra/`.
  `fl_v3/collab/` is read-only historical evidence.
