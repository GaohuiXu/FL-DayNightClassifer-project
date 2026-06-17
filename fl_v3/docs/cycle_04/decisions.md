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

---

## D9 — Execution model for T5–T7: concurrent actors are determinism-safe (T3 follow-up)

**RECOMMENDED (needs user/supervisor confirm + a Swin-T-scale re-validation before full reliance).**

**Question (raised at T3):** the platform inherited fl_v2's `num-gpus=1.0` single-actor execution
("concurrent Ray actors diverge at round 2"). Run strictly serially, the T5–T7 attack×defense matrix
looked expensive.

**Finding (validated on the A40, T3 follow-up):** that divergence was an fl_v2 finding on a
**different, atomic-using model**. Our T2 detector is **atomic-free by construction** (deterministic
splat/scatter, no `scatter_add`), so concurrent-actor **kernel interleaving changes only timing, not
values**. Two parallelism methods were tested — both produced a final aggregated checksum
**byte-identical to the single-actor baseline** `d82ef5001b88…c08b236` (gate config: resnet18, N=8,
3 rounds, fraction-train=0.5; eval curves matched to all 16 digits):

- **Path A — multi-GPU** (`local-simulation-gpu-4x`, `num-gpus=1.0`, ONE client per GPU on a 4-GPU
  node): jobs 6764253 / 6764255 — **PASS-STRONG** (checksum == reference).
- **Path B — concurrent / shared-GPU** (`local-simulation-gpu-shared`, `num-gpus=0.25`, multiple
  actors sharing ONE GPU): job 6764256 — **PASS-STRONG** (checksum == reference).

Validation harness: `scripts/run_parallel_validation_a40.sh` + the two federations in
`configs/flwr_config.toml`. (Both are bit-identical to serial ⇒ future experiments are fully
reproducible with either, and they can be combined.)

**Path A vs Path B — what they actually buy (the speed distinction):**
- **Path A (multi-GPU) multiplies COMPUTE.** N GPUs → N clients truly parallel → **~N× wall-clock**.
  This is the real per-cell speedup. (Measured on the bring-up config: Path A 126 s vs Path B 176 s
  on identical work; the gap widens at headline scale.)
- **Path B (concurrent / shared-GPU) shares ONE GPU's compute** — it only fills idle gaps, so it
  helps when the GPU is *under-utilized* (small batch / I/O-bound) and gives **≈ no speedup when one
  client already saturates the GPU** (headline Swin-T at batch≥16 measured ~100% SM). Its residual
  value is hiding the per-round model-build / first-batch latency.

**Recommendation for T5–T7:**
1. **Matrix (many cells, T7):** the dominant lever is **across-cell fan-out** — each attack×defense
   cell is an independent FL run → one SLURM job per cell, run in parallel across the cluster
   (hundreds of idle A40 GPUs) → **matrix wall-clock ≈ one cell's time**, not the sum. Give each cell
   **1 GPU** (optionally **Path B** `num-gpus=0.25` to hide inter-round latency) to maximize
   cells-in-flight.
2. **A single heavy run** (a long trainval baseline): use **Path A (multi-GPU)** `num-gpus=1.0` on a
   4-GPU node → ~4× that run.
3. **Combine** = Path A (multi-GPU, the N× multiplier) + a *mild* Path B overcommit (`num-gpus=0.5`,
   2 actors/GPU) to hide latency — NOT aggressive overcommit, which just thrashes a saturated GPU.
4. **Keep `num-gpus=1.0` single-actor for the determinism GATES** and null-config byte-parity proofs
   (already so); the relaxation is for the heavy scientific cells.

**Caveats (do NOT skip before T5/T7 reliance):** validated at the **bring-up scale** (resnet18, N=8,
3 rounds, mini, 4 actors). **Re-confirm byte-identity AND measure the actual speedup at the headline
Swin-T + trainval scale and at the actor count you actually use** (e.g. Path A on a 4-GPU node;
`num-gpus=0.2` → 5 concurrent) before committing the matrix to it — the atomic-free argument is
architecture-wide, but the proof + the speedup number should be at the operating point. A
`single-actor vs Path A vs Path B vs combined` Swin-T benchmark (~4 short A40 jobs) would lock this in.

### Hardware tier for FULL-MODEL-FROM-SCRATCH (future) — consider A100 / A100-fat, and RE-VALIDATE first

The **current primary setting (D1) is memory-light** — the ImageNet backbone is FROZEN, so only the
**62 trainable tensors** are optimized (no backbone gradients / Adam state / backbone-backward
activations); the headline Swin-T trains at ~21 GB on the A40's 46 GB even at batch 16. For that
regime the **A40 is the right pin** and a bigger GPU buys nothing (memory is not the constraint).

**But a future task may train a FULL model from scratch** — the **full-model-FL generality ablation**
(unfreeze the backbone, ImageNet-init or random-init) trains **all ~28 M+ params**, which adds backbone
gradients + Adam moments + the full backbone-backward activation graph → **far higher VRAM** (and it
wants larger batches). The A40's 46 GB may be insufficient at useful batch sizes. **For that regime,
consider A100 (80 GB) / A100-fat** — the larger memory is what enables full-model training + bigger
batches.

**HARD REQUIREMENT before any full-model-from-scratch experiment on a new GPU tier:** re-establish the
determinism contract on that GPU exactly as we did for the A40 — an **A100 analog of the A40 gate**
(`det_gate_a40` / `run_fedavg_a40` patterns): assert the device, run two same-seed runs → **byte-identical**,
and **commit a new A100 checksum**. Determinism is **architecture-pinned** (T4 ≠ A40 ≠ A100 ≠ ARM H200,
per `docs/determinism.md`), so an A100 result is **NOT** byte-comparable to an A40 result — keep each
experiment's GPU tier **homogeneous and recorded**, and re-validate the **concurrent-actor (Path A/B)**
byte-identity on A100 too (the atomic-free argument is architecture-wide, but prove it at the operating
point). **Never run a full-scale exp on an unvalidated GPU tier** — that is the same FALSE-PASS trap the
A40 gate exists to prevent, one level up at the hardware tier.

---

## D10 — Full participation (fraction-train=1.0) is the primary scientific-training regime; sampling is secondary

**CONFIRMED 2026-06-17 (user-directed; verified by workflow `wf_25deb3c5-769`).**

**Decision.** The **clean scientific baseline and all T5–T7 attack×defense cells run at FULL participation**
(`fraction-train=1.0`, all `N` clients aggregated every round). Per-round random client **sampling
(`fraction<1`) is demoted to a SECONDARY realism setting only.** Wall-clock is bought with **D9 Path-A
multi-GPU** (N× compute, byte-identical to serial), **not** by starving participation.

**Why (the T3 weak-log-group diagnosis).** T3's clean log-group model (recall@2m **0.1455**) is **not** an
architecture-capacity ceiling — it conflates three confounds, all removed by full participation:
1. **Partial-participation variance** ~ `O(E²G²/(m·µ·T))` (Li et al., *On the Convergence of FedAvg on
   Non-IID Data*, ICLR 2020) — inversely ∝ the sampled count `m`, and **maximized under location-coherent
   shards** (large gradient-dissimilarity `G`). At `m=N` it **vanishes**.
2. **Objective-subset bias** (FedNova, Wang et al. NeurIPS 2020) — averaging over a shifting heterogeneous
   subset converges toward a *mismatched* objective.
3. **Severe under-training** — at 5-of-25 × 4 rounds each shard is selected only **≈0.8 times** (whole
   geographic regions trained zero times) and the eval curve was **still climbing** at the last round.
   The same architecture reaches recall **0.35–0.50** in IID settings → capacity exists.

**Consistency with the threat model.** Full participation **realizes the plan's PRIMARY "controlled
participation" regime as its corner case `n_r=N`** (it is NOT the "random sampling = secondary" path).
Derived `N=25, ρ=0.2` → `m_r=round(0.2·25)=5`, `h_r=20`: satisfies `1 ≤ m_r < n_r/2` (5<12.5) and
`m_r ≤ m=floor(ρN)=5`. It is **strictly better for the clean-vs-poisoned 2×2 fairness**: identical roster
clean vs poisoned ⇒ the `poison_rate=0` null-config is **exactly bit-identical** to clean (plan
§Verification), which sampled participation only approximates.

**Binding conditions (record + enforce):**
1. **Strict readiness sequencing (the one BLOCKER if mis-ordered):** (a) re-run the clean log-group
   trainval FedAvg at **full participation FIRST** (same batch/seed/partition; rounds bumped to a
   convergence target, ≤20 a floor not a ceiling); (b) **re-judge T4 `benchmark_readiness.json` on THAT
   checkpoint**; (c) escalate to **architecture strengthening (deeper LiDAR PFN / full-model-from-scratch
   on A100 per D9) ONLY if STILL NOT-READY.** Full participation must NOT be used to paper over a real PFN
   weakness — if it clears the floor, stamp the verdict **"cleared by participation, architecture not
   independently validated"**; if it does not, that IS the architecture signal. Report the 5-of-25 and the
   full-participation recall **side by side**.
2. **Re-label the T3 numbers:** `recall@2m log_group 0.1455` and the **non-IID gap `+0.2073`** are
   **sampled-regime (5-of-25, fraction-0.2, 4-round)** measurements — tag `scale=trainval-sampled`. They
   remain evidence a large non-IID gap *exists*, but are **NOT** the full-participation baseline. The T4
   readiness anchor and the T5-attacked checkpoint **must BOTH be the full-participation log-group
   checkpoint (same `FL_TRAINABLE_CHECKSUM`)** — a sampled anchor + a full-participation attack is the
   §0.2 partition-mismatch trap generalized to a **participation-regime mismatch**.
3. **Malicious roster at full participation:** a **fixed, seed-derived, recorded** subset of size
   `m=floor(ρN)` of the `0..N-1` partition-ids; `m_r=m` constant every poisoned round; still report `m_r`
   (ground truth) vs the defense-assumed `f_r` (a defender hyper-parameter). The clean baseline uses the
   **identical roster** with that subset behaving honestly (`poison_rate=0`).
4. **2×2 at identical participation:** clean and every attack×defense cell run at full participation;
   **forbid** comparing a full-participation clean baseline against a `fraction<1` sampled attack cell (it
   confounds the participation regime with the attack and voids the `δ` utility-collapse interpretation).
5. **Keep the DT3-B determinism gate at `fraction<1`** (`fl_gate_a40.py` FATALs on `fraction≥1`) — that
   gate exists to exercise **sampler drift** and is independent of the science participation regime; do
   NOT relax it to `1.0`.

**Codebase status (verified).** `select_partition_ids(...,1.0,...)` already returns `[0..N-1]` every round
(no special-casing; trivially bit-deterministic). **One build task:** there is no launcher today for the
T4 reference at `fraction=1.0` + log-group + trainval + Path-A — add a `t4_reference.json` + launcher, and
generalize `fl_stamp_supernodes` to stamp the `local-simulation-gpu-4x` (Path-A) federation (it currently
stamps only `local-simulation-gpu` and `-4x` hardcodes `num-supernodes=8`), or fall back to single-GPU.
Wall-clock: full 25/round × ≤20 rounds via Path-A on a 4-GPU A40 node ≈ **3–6 h** (one SLURM job).
