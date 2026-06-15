# Cycle 04 — Decisions Record (D1–D8)

Resolve these in the **orchestrator session** so every downstream task inherits them. Each has a
**recommendation**; the user **confirms or overrides**. Status: `RECOMMENDED (unconfirmed)` until the
user answers, then `CONFIRMED` with a date.

| # | Decision | Recommendation | Needed by | Status |
|---|---|---|---|---|
| **D1** | What gets FL-trained | **Frozen ImageNet camera backbone; FL-train LSS-depth + LiDAR-enc + fusion + BEV-neck + head.** Full-model FL = generality ablation. | T3 | **CONFIRMED** |
| **D2** | Attack vector | **Start with data-poison (BadFusion-style camera trigger).** Constrained fusion-only update later (for the Q2 dilution test). | T5 | **CONFIRMED** |
| **D3** | Fusion design | **BEV-concat `ConvFuser`** (clean named module). Point-decoration is the escape hatch if T5's cond-4 ablation shows degeneration. | T2 | **CONFIRMED** |
| **D4** | ASR headline | **Disappearance** primary; phantom secondary. | T4/T5 | **CONFIRMED** |
| **D5** | Defense breadth | **Minimum first:** {FedAvg, FLAME, FoolsGold, MultiKrum, NormClip} + the **random-drop control**. {FedMedian, FreqFed} as fast follow-ons. | T6 | **CONFIRMED** |
| **D6** | Normalization policy | **Frozen camera-backbone BN in eval mode; new fusion/neck/head modules use GroupNorm/LayerNorm.** FedBN as a diagnostic. | T2 | **CONFIRMED** |
| **D7** | Utility-collapse tolerance `δ` | **DEFERRED — set empirically.** Once T3/T6 reveal the baseline clean/poisoned NDS spread, pick `δ` so "utility preserved" = clean **and** poisoned NDS drop ≤ `δ` vs the FedAvg/benign-defense baseline. | T6/T7 | **DEFERRED** (set during exps) |
| **D8** | Target class | **Primary = car/vehicle** (densest, highest clean recall → stable eligibility). Pedestrian/cyclist secondary, only if eligible-count + clean-recall pass the floor. | T4/T5 | **CONFIRMED** |

## How blocking each is for T0

**None of D1–D8 block T0** (scaffold + carry-over). The earliest binding decisions are **D3 + D6 (T2)**
and **D1 (T3)**. **D7** needs a concrete `δ` value before T6/T7 interpretation. Confirming all now is
good hygiene so no downstream session re-litigates.

## Confirmed answers

**2026-06-15 — user confirmed ALL recommendations.**
- **D1** frozen ImageNet camera backbone; FL-train LSS-depth + LiDAR-enc + fusion + neck + head. Full-model FL = generality ablation.
- **D2** start with data-poison (BadFusion-style); constrained fusion-only update later (Q2 dilution test).
- **D3** BEV-concat `ConvFuser`; point-decoration is the escape hatch if T5's cond-4 ablation degenerates.
- **D4** disappearance ASR primary; phantom secondary.
- **D5** minimum defense set {FedAvg, FLAME, FoolsGold, MultiKrum, NormClip} + random-drop control; {FedMedian, FreqFed} fast follow-ons.
- **D6** frozen camera-backbone BN in eval mode; new fusion/neck/head modules GroupNorm/LayerNorm; FedBN diagnostic.
- **D7 (δ) DEFERRED** — set empirically once experiments reveal the baseline NDS spread (T3/T6), before interpreting any defended cell.
- **D8** primary target class = car/vehicle; pedestrian/cyclist secondary only if eligible-count + clean-recall pass the floor.
