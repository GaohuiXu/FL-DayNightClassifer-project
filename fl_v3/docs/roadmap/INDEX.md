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
| 04 | 2026-06-15 → TBD | **Clean federated multimodal AD perception foundation** — camera/LiDAR/fusion detection on nuScenes with resolved runtime, official evaluation, and clean FedAvg. The active protocol and security sequencing are governed only by the Orchestra. | **active** | Centralized detector repair/validation is the current gate. |

## Cycle 04 working folder

> **2026-07-10 execution update:** `docs/cycle_04/` remains the Cycle-04
> experimental-design/decision record. Active cross-session coordination has moved
> to [`../../usenix27_orchestra/ORCHESTRA.md`](../../usenix27_orchestra/ORCHESTRA.md),
> with [session contracts](../../usenix27_orchestra/SESSIONS.md) and
> [copy-ready kickoffs](../../usenix27_orchestra/KICKOFFS.md). `fl_v3/collab/` is
> read-only historical evidence for the new stage.

The older `docs/cycle_04/` task contracts and kickoffs are frozen experimental-
design history, not active code or launcher routes. The active model, data,
runtime, evaluation, and clean-FL work is defined by the Orchestra documents
above. Historical speedup/model-capability findings remain read-only evidence;
Arrhenius supersedes the older Alvis/A40 environment assumptions.

`docs/roadmap/cycle_04_fusion_layer_backdoors.md` is likewise a frozen historical
roadmap file. Its retired task names and security routes are not active authority,
configuration, or launcher surfaces.

Historical per-task `SPEC.md`/`REVIEW.md` and `findings_log.md` remain in the read-only
`fl_v3/collab/`. New worker/reviewer packages live in
`fl_v3/usenix27_orchestra/handoffs/Sxx/`.

## Reading order for a new collaborator (Cycle 04)

1. [`../../usenix27_orchestra/ORCHESTRA.md`](../../usenix27_orchestra/ORCHESTRA.md) — active objective/protocol/gates.
2. [`../../usenix27_orchestra/SESSIONS.md`](../../usenix27_orchestra/SESSIONS.md) and [`../../usenix27_orchestra/KICKOFFS.md`](../../usenix27_orchestra/KICKOFFS.md) — active collaboration contracts.
3. [`../cycle_04/decisions.md`](../cycle_04/decisions.md) — historical locked decisions.
4. The frozen Cycle-02/03 context in `fl_v2/docs/` — linked above.

## Document conventions

- Cycle files: `cycle_NN_short_descriptive_label.md`. Status ∈ {planned, active, paused, closed}.
- Closed/paused cycle docs are edited only to fix factual errors — they are the research trajectory.
- Experimental-design/roadmap docs live in `fl_v3/docs/`; AI operating instructions
  live at the repo root; active session work lives in `fl_v3/usenix27_orchestra/`.
  `fl_v3/collab/` is read-only historical evidence.
