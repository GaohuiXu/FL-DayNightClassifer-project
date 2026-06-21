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

---

## D11 — Per-run speedup backlog (PROPOSED — build-session-surfaced at T4; needs orchestrator + a profiling session)

**Status: PROPOSED / unconfirmed.** Surfaced by the T4 build session from the full-participation
reference run (job 6764630); recorded here per user request for a future deep-dive/profiling session
to formalize. **Not a confirmed decision** — the orchestrator owns whether/when to act, and any change
that touches numerics MUST re-pass the determinism gate before a scientific run.

**Measured profile (the reference run, 4×A40 Path-A, full participation, Swin-T frozen).** ~22 min/round
× 15 ≈ 5.5 h. The run is **compute-bound, NOT Flower-bound**: all 4 A40s pinned at 100 % during the
training phase (verified by `srun --overlap nvidia-smi`); aggregation averages only the 62 trainable
tensors across 25 clients (ms); Flower/Ray overhead (actor scheduling, ~tens-of-MB update-vector
serialization) is negligible vs ~150 s/client of GPU compute. The dominant per-step cost is the
**frozen Swin-T (ViT) forward over 6 camera images** — the *headline* backbone (D1), recomputed every
step/epoch/round even though it never updates. (`resnet18` is only the bring-up/mini-smoke fallback.)
See `collab/findings_log.md` for the full breakdown.

**Candidate levers, ranked, with determinism implications:**
1. **Frozen-backbone feature caching (the biggest determinism-SAFE win; recommended for the T5–T7 matrix).**
   D1 freezes the camera backbone, so its multi-scale feature maps for a given (deterministically
   preprocessed) image are **invariant across steps/epochs/rounds** — recomputing them is pure waste.
   *Proposal:* precompute the frozen Swin-T feature maps once per camera image on the login node and
   cache to Mimer (the info-cache pattern: DATAROOT-relative keys, fixed dtypes, a host-portable content
   hash), then have training read cached features straight into the **trainable** camera-neck. Eliminates
   the dominant cost → plausibly **~3–5× per-step** (Swin-T forward dominates). *Open questions for the
   profiling session:* (a) cache the **backbone** multi-scale output (feeds the trainable neck), NOT the
   neck output (trainable, must stay live); size it — backbone strides [4,8,16,32] × channels [96,192,384,
   768] over 6 cams × ~28 k keyframes is the storage budget (estimate + decide f16-vs-f32, which affects
   determinism); (b) **REQUIRES no per-step image augmentation** on the camera path (verify the preprocess
   is deterministic resize+normalize only — if any random aug is added later, caching breaks); (c) the
   cached features must be **bit-identical** to the on-the-fly forward → a new determinism gate
   (precompute-twice byte-identity + cached-vs-live equivalence) before any scientific run; (d) ARM-rebuild
   portability of the cache (Arrhenius) — likely NOT host-portable (CUDA conv/attn kernels differ), so the
   cache is a **per-GPU-tier artifact** rebuilt on each tier (consistent with D9's architecture-pinning).
2. **A100 hardware (~2× per-GPU for Swin-T; per-cell lever for the matrix).** A100 (HGX) is ~2–2.5× faster
   than A40 for the transformer forward (memory bandwidth + TF32). All Alvis nodes are 4-GPU (no extra
   parallelism per run; A40 has no InfiniBand so no multi-node), so this is a **per-GPU** speedup, ~2×
   wall-clock at ~2× units/hr (≈ cost-neutral per result). **Blocked on D9's hard requirement:** establish
   the A100 determinism gate (assert device, two same-seed runs → byte-identical, commit a NEW A100
   checksum) BEFORE any A100 scientific run; A100 results are NOT byte-comparable to A40. A100fat (80 GB)
   is only needed for the full-model-from-scratch ablation (memory-bound), not the frozen-backbone setting.
3. **Across-cell fan-out (already D9 — the dominant MATRIX lever).** Each attack×defense cell is an
   independent FL run → one SLURM job/cell, 1 GPU/cell, fan out across the cluster → matrix wall-clock ≈
   one cell, not the sum. This dwarfs any single-cell speedup for T7.
4. **REJECTED for determinism:** AMP/fp16 (non-deterministic accumulation), `torch.compile` (can introduce
   nondeterminism + violates the pure-PyTorch/no-fragile-kernel posture). NOT applicable: GPU overcommit
   (>1 client/GPU — the GPU is already saturated, D9), larger batch (held fixed at 16 per §0.2; changes
   convergence).

---

## D12 — Dedicated speedup/infra track + the storage-allocation decision (T5 follow-up)

**CONFIRMED 2026-06-18 (user-directed; verified by workflow `wf_433c694a-c1c`).** Promotes the D11
backlog into an actual **dedicated speedup session, run OUTSIDE the T<N> task sessions** (an
infrastructure track parallel to T5–T7). Charter: `kickoff/speedup_kickoff.md`. Diagnosis confirmed:
each run is **compute-bound on the frozen Swin-T backbone** (re-run every step though D1 freezes it).

**The headline lever (feature caching) is blocked on PERSISTENT STORAGE — and the one apparent escape
hatch was a mirage.** Verified live: `mimer-weka` is **Mimer itself** — the group's 500 GB
`/mimer/NOBACKUP/groups` area and the 51 TB / 3.6 TB-free `/mimer/NOBACKUP/Datasets` area are the **same
WEKA filesystem, different exports**; the 3.6 TB free is the **read-only Datasets quota**, not obtainable
as writable. So D11 §H option-2 is dropped. Node-local `/tmp` (255 GB) is per-job ephemeral; `/cephyr`
home (30 GB) is backed-up and tiny. **The ONLY unblocker for the full ~3–5× cache is a SUPR storage
allocation.** Cache math reconfirmed exactly: 48.66 MB/keyframe (Swin-T 4-scale fp32, 6 cams @ 256×704)
→ **1.37 TB train + 0.29 TB val = ~1.66 TB** for the full clean-keyframe cache (96 GB free today).

**STORAGE DECISION (SUPERSEDED IN PART BY D13 — no longer urgent; caching is de-prioritized in favor of TF32. A modest allocation for run outputs is still fine, but the 5 TB cache rush is dropped.):** file a **NAISS Small storage request, 5 TB, on the EXISTING compute project
`naiss2024-22-991`** via SUPR (supr.naiss.se → the project → extend the Mimer storage component — the
2026 combined-proposal model, NOT a separate Storage project → attach a 1-paragraph DMP, mandatory above
default). **Small** = 5 TB cap, **WEEKLY** review (plausibly days) and covers the ~1.66 TB need with
headroom; Medium (40 TB cap, monthly) is the roomier-but-slower fallback. Also open a C3SE support ticket
(support@c3se.chalmers.se) asking for an interim quota bump in parallel.

**Corrections + additions to D11 (from the red-team) the session must honor:**
1. **The "80–90% backbone" share is INFERRED, not measured** (90%·step = 68 ms > the measured 55.4 ms
   forward — internally over-stated). The session's **first** task is a 5-min CUDA-event/nvtx microbench
   to size the share honestly → the cache ceiling may be **~2.5×, not ~3–5×**. (Caching is still #1.)
2. **Two FREE wins, no storage, apply now:** (a) **fan the eval wider** (D1/C4: ~2 h → ~40 min); (b) an
   **fp16 `/tmp` cache for the DEV/DEBUG loop ONLY** — dev iterations need NOT match `a80466c3` (only the
   recorded scientific run does), so A4's fp16 cache (was "science-banned") is **reclassified
   `dev-loop-only`** and cuts the 5.5 h edit-run-inspect cycle ~3×. This directly addresses the
   "each run unacceptable" pain today.
3. **Caching stays fp32** to preserve the `a80466c3` null (the cache stores the exact frozen-backbone
   bytes; cached-vs-live byte-identity is the det-gate). **TF32 is a real ~2× lever on the matmul-heavy
   frozen backbone but is mutually exclusive with the `a80466c3` family** (it changes the frozen output →
   a fresh reference cycle, like A100) — defer to a future-cycle, **never a global flag on the current
   null**.
4. **A node-local fp32 PARTIAL `/tmp` cache** (bit-identical, no allocation) is a free fractional stopgap
   (~1.2–1.4×) while the storage request is in flight; per-job ephemeral (helps a single heavy run / the
   eval more than the fanned-out matrix). So "storage is the *only* blocker" → "persistent storage blocks
   the *full* 3–5×; partial/dev wins exist today."

**Determinism guardrail (unchanged, D11 #4 + the §0 reality):** any cache/accel that feeds the
**scientific** runs needs its own bit-identity det-gate (precompute-twice byte-identity + cached-vs-live
equivalence; the null still reproduces `a80466c3`); per-GPU-tier artifact (A40 ≠ A100 ≠ ARM). `torch.compile`,
AMP/fp16 **training**, flash-attn stay banned for science.

---

## D13 — Precision policy: adopt TF32 at the GH200/Arrhenius re-baseline; it supersedes caching as the primary speed lever

> **TIMING SUPERSEDED BY D14 (2026-06-18):** TF32 is adopted **now on A40** (the T5 pause removed the `a80466c3`-protection rationale). D13's TF32-is-safe finding, the defense seed-robustness check, and the adoption mechanics REMAIN in force; only the "wait for GH200" timing is overridden.

**CONFIRMED 2026-06-18 (user-raised; verified by workflow `wf_bdd72d51-cd9`).** Resolves the speed
strategy. **TF32 is the precision policy for the GH200/Arrhenius re-baseline**, NOT a mid-cycle A40
switch. This **supersedes** D12's "caching is the #1 lever, file storage now" framing.

**Why TF32 over caching (the user's two points are correct):** caching only works because D1 *freezes*
the backbone and is voided by any backbone/config change; it is storage-blocked (~1.66 TB) + needs a
cache det-gate. **TF32 is strictly more general** — it accelerates the backbone whether frozen *or*
trained (so it survives "we train our own encoder" / the full-model ablation), survives config changes,
needs no storage. They are complementary (TF32 is multiplicative on the residual matmul after any cache),
but TF32 is the primary bet. **Caching is de-prioritized; the SUPR 1.66 TB cache request is NO LONGER
URGENT** (a modest allocation for run outputs is still fine — but drop the rush + the 5 TB ask).

**Is TF32 scientifically safe? YES (verified):**
- **Model quality:** no meaningful change — TF32 keeps FP32's exponent (range), uses a 10-bit mantissa
  for matmul/conv inputs, accumulates in FP32; converges to accuracy indistinguishable from FP32 for an
  Adam-trained detector. Published nuScenes/BEVFusion SOTA is trained at the *lower* FP16-AMP precision.
- **Venue acceptability:** TF32 is field-standard, **not** a reviewer objection (USENIX-Sec/NeurIPS/CVPR);
  reduced-precision training is the norm. The binding norm is seed-variance discipline, not IEEE-FP32.
- **Internal consistency:** all conditions (clean baseline + every attack + every defense) use identical
  precision (TF32 trainable path, **fp64 defense cores**) → all comparisons are differential and cancel
  the common-mode perturbation at first order.
- **The ONE required check (the defense-decision knife-edge):** FLAME-HDBSCAN / MultiKrum / cosine
  admit/drop/select are **discrete** operations; "TF32 doesn't change the conclusion" must be
  **demonstrated, not asserted**. Before committing TF32 to the science path, run a one-time
  **seed-bracketing check**: ≥3 seeds of one FLAME cell + one MultiKrum cell, log the per-round
  admitted/dropped/selected client sets + the headline ASR, and show they are **seed-stable above the
  ~2e-3 precision floor** (any flip confined to clients whose membership also flips across FP32 seeds =
  seed-noise, not precision-noise; report any genuinely precision-sensitive headline cell in FP32).
  The fp64 cores already shrink the exposure to "did training land a borderline client across the
  boundary," which this check directly measures. **Pre-register it in the thesis methods.**

**GH200 timing — why "at the migration" is zero-extra-cost:** determinism is architecture-pinned (D9:
A40 ≠ A100 ≠ ARM H200), so the Arrhenius cutover **forces a fresh re-baseline anyway** (new det-gate +
new reference checksum + rebuilt frozen-ASR subset). TF32's only marginal cost over FP32 is "it needs its
own reference" — which the migration already buys. **Decide now so we re-baseline ONCE** (stamp the GH200
reference TF32 from round zero, not FP32-then-TF32). HW ratio: A40 TF32:FP32 ≈ **2×** peak (GA102 has a
halved TF32 path — the worst card for TF32); GH200/Hopper ≈ **7–15×** peak → realistic **end-to-end
~1.3–1.8×** (gated by non-matmul LSS/BEV/norm/dataloader + Hopper memory bandwidth — NOT the peak; confirm
with the D12 nvtx microbench on-tier).

**REJECTED — mid-cycle A40 FP32→TF32 switch:** it invalidates the in-flight `a80466c3` null + the frozen
subset + everything T5 stamped, for only a weak ~2× peak (~1.3× end-to-end) on the worst-case A40 card.
**Keep FP32 + `a80466c3` for ALL remaining Alvis/A40 T5(–T7-on-Alvis) work.**

**Adoption mechanics (at GH200, under the D9 re-baseline gate):** set BOTH
`torch.backends.cuda.matmul.allow_tf32=True` + `cudnn.allow_tf32=True` inside `enforce_determinism`
(= `set_float32_matmul_precision('high')`; **never** `'medium'`/bf16x3; never per-call); keep
`use_deterministic_algorithms(True)` + `CUBLAS_WORKSPACE_CONFIG=:4096:8` + the static AST ban + flash-attn
ban; run the **two-same-seed-runs byte-identity gate** on the GH200 + commit a NEW GH200-TF32 reference
checksum. A gate scaffold exists at `scripts/tf32_det_gate_a40.py` (asserts cc≥8, no-raise, run-to-run
byte-identity, TF32≠FP32; tested only on the login T4 so far → must run on real TF32 hardware).

---

## D14 — Pause camera-only T5; diagnostics-first; adopt TF32 NOW on A40 (supersedes D13's "wait for GH200")

**CONFIRMED 2026-06-18 (user-directed, orchestrator handoff).** T5's first implementation + Codex review
are complete; the **camera-only relocation backdoor did NOT reach viability** (relocation/trigger_only ≈ 0
ASR; label_only weak; delete some-but-sub-threshold). **Treat T5 as a pilot negative-leaning result with
identified confounds — do NOT overclaim "BadFusion does not transfer to BEV-concat."** The null is
uninterpretable until the confounds are excluded: (a) architecture robustness (LiDAR-dominant fusion
outvotes a camera-only trigger), (b) FL undertraining (15 rounds, proxy still climbing → not a convergence
proof), (c) weak recipe (official mAP/NDS ~0.13 — marginal clean detection), (d) implementation issues
(relocation validity, aligned/non-aligned semantics, LiDAR-sparse control, trigger budget).

**Diagnostics-first ordering (the dedicated speedup session, re-scoped — `kickoff/speedup_kickoff.md`):**
**A** profile per-stage runtime (settles D12's *inferred* "80–90% backbone") → **B** config-gated server
eval (`none|final|every_n|all`; default `none` for trainval; ASR stays post-hoc; training checksum
UNCHANGED when only eval disabled = eval is RNG-neutral) → **C** adopt TF32 + its A40 det-gate → **D**
centralized baseline (clean first; attack ONLY if clean clears the readiness bar; **matched budget =
centralized epochs == FL rounds**) → **E** clean-FL convergence (15 vs 30 rounds, official metrics not
proxy). **Output = the 5 questions: where is runtime spent; how much does eval-disable save; does TF32
help under a reproducible regime; is the weak T5 due to FL-undertraining / weak-recipe / attack-design;
what is the correct baseline before the next attack.**

**TF32 NOW on A40 (supersedes D13's timing).** The pause removes the `a80466c3`-protection rationale (the
clean reference is being re-established by D/E anyway), and ~12 days of Alvis + plentiful GPU-hours make
velocity worth the re-baseline. So: adopt TF32 as the numeric regime for the remaining Alvis/A40 work;
the **D/E baselines are run IN TF32 and become the new TF32 reference** (determinism gate → clean FL ref →
readiness → frozen ASR subset → null-config identity → all later cells, same mode; **no mixing regimes**).
D13's TF32-is-scientifically-safe finding, the **defense-decision seed-robustness check** (≥3 seeds × one
FLAME + one MultiKrum cell), and the adoption mechanics (both `allow_tf32` flags inside `enforce_determinism`
= `set_float32_matmul_precision('high')`, never 'medium', never per-call; A40-style two-same-seed-runs
byte-identity gate + a NEW checksum) all REMAIN in force. **FP16/bf16 is a MEASURED contingency only** (if
profiling shows TF32's A40 win — ~1.3× end-to-end — is insufficient): prefer **bf16** over raw FP16-AMP, or
**FP16/bf16 on the FROZEN BACKBONE only** (targets the bottleneck, keeps trainable gradients at TF32);
decide the form AFTER profiling; the larger perturbation makes the seed-robustness check more important.

**Forward (parked, NOT now):** if the diagnostics confirm camera-only is genuinely non-viable (not a
confound), the likely pivot is a **camera+LiDAR** poison — a **D2 / threat-model** decision, held until the
diagnostics land. **Caching stays de-prioritized (D13); the storage request stays non-urgent (D12).**

**Boundaries:** (1) no major T5 camera-only attack dev until profiling + clean baselines are complete;
(2) no T6/T7 defense matrices on a non-viable attack; (3) mini/smoke ≠ scientific evidence; (4) no mixing
numeric regimes; (5) no feature caching without storage + a cache-vs-live bit-identity gate.

---

## D15 — Speedup/diagnostics session outcome: a `determinism-level` knob buys ~3×/step; overcommit is a MEASURED dead end; the weak model is FL-undertraining + FedAvg dilution (NOT architecture)

**RECORDED 2026-06-21 (outcome of the D14 speedup/diagnostics session; full trace + evidence:
`collab/speedup/speedup_session_findings.md` + `collab/speedup/D15_D16_decision_for_orchestrator.md`).**
The infra landed on `v3-ad-perception` (HEAD `1bf9015`); everything is behind a `determinism-level =
strict | relaxed` knob (default `strict` = byte-identical, 247 tests pass).

**Speed (measured, A40 unless noted):**
- **~3.1× per training step** under `relaxed` (1443 → 460 ms). The dominant, **precision-independent** win is
  the LSS view-transform rewrite — **`scatter_add` splat + mask-then-lift, 602 → 14 ms (42.8×)** — which
  drops the argsort+cumsum+1.3 GB materialization that was pure determinism tax (scatter_add atomics are
  non-deterministic → only available once strict is relaxed). Plus bf16-AMP forward (3.0× backbone) +
  `torch.compile` + free scheduling.
- **Round-level ladder:** strict 15.6 min/round → relaxed+compile+2-clients/GPU **5.6 min/round (2.76×)**.
  **A100 1/GPU also = 5.6 min/round** — only **1.14× over A40 1/GPU** despite **1.63×/step**, because each
  round's overhead (Ray actor spin-up ×25, dataloader, aggregation) is launch/CPU-bound and GPU-speed-invariant.
- **bf16 NaN caught + fixed:** focal `log(sigmoid)` on bf16 logits diverged → loss + BEV-accumulation kept
  in **fp32** (forward stays bf16). Re-test: bf16 within ~0.5% of deterministic at every epoch; trains
  cleanly at centralized AND FL scale. ⇒ keep one lightweight **trains-clean reasonableness gate** even
  after dropping bit-identity (speed that NaNs is worthless).

**Overcommit is a DEAD END (measured — E1/E2/E3 teardown, job 6769136).** More clients/GPU gives **~1.0×
throughput** (K=1→4): without CUDA MPS, separate processes **time-slice** the GPU, and each step is
**launch/latency-bound** (only ~10% is GPU compute at batch-16 + a **frozen** backbone) → packing clients
only serializes them; rising util (10→52%) is busy-fraction inflation. The A100's **12% util** is structural,
NOT fixable by packing clients (this answers the orchestrator's "smaller `num-gpus` failed" — it was never
going to work). 2/GPU is the A40 ceiling; 3/GPU thrashes; A100 4/GPU OOMs (compile inflates VRAM) and 2/GPU
hit a shared-`TORCHINDUCTOR_CACHE_DIR` cache-race. The dataloader is independently starved (nw=2 < one
client's need; 0.60× shared-FS contention) and widening it OOMs host RAM. **The real speed lever is CHEAPER
STEPS** (`torch.compile(mode="reduce-overhead")` / CUDA graphs + num-workers↑ + node-local staging) — **and
training the backbone makes the step GPU-heavy so util rises on its own and the A100's 1.63×/step finally
lands.** Overcommit + big-batch are OFF the path.

**Diagnostics answered Q1–Q5 (the key scientific result):** the weak T5 model was **FL-undertraining +
FedAvg dilution, NOT architecture/recipe.** Matched budget (epochs == rounds), official mAP/NDS:

| setting | budget | mAP | NDS | car recall |
|---|---|---:|---:|---:|
| **Centralized (D1)** | 15 ep | **0.360** | 0.357 | 0.93 |
| FL (E-15) | 15 r | 0.126 | 0.169 | 0.85 |
| FL (E-30) | 30 r | 0.196 | 0.226 | 0.89 |

Centralized reaches a strong detector on the same model/data/budget ⇒ architecture+recipe are fine at the
*centralized* level; FL-15 was undertrained (15→30 = +55% mAP, still climbing at r30); even FL-30 is ~1.8×
below centralized ⇒ **FedAvg dilution** over location-coherent non-IID shards. The T5 attack ran on a
doubly-compromised (undertrained + diluted) checkpoint ⇒ its null was uninterpretable. **Next clean
reference must be ≥30 rounds** (find the plateau past 30). *Note for D17: the centralized 0.36 ceiling is
itself well below SOTA (frozen ImageNet Swin-T, batch 16, single-sweep, modest resolution) — raising it is
a SEPARATE capability problem from the FL gap.*

**Other items banked:** server eval gated (`server-eval-mode=none|final|every_n|all`; byte-identity-safe,
default `none` for trainval; ASR + official mAP/NDS stay post-hoc). A100/A40 nodes have **identical 244 GB
host RAM** (the OOMs were VRAM/cache-race, not node size). **Hazard:** `t5_attack_eval.py` does NOT thread
the numeric/precision mode → would eval a relaxed checkpoint in the wrong regime (mirror
`t4_readiness_eval.py` before any T5 eval — assigned to D17 Phase 0).

---

## D16 — Precision + criteria: bf16-AMP is the single science regime; multi-seed claim-reproducibility replaces byte-identity (RATIFIES the speedup proposal; AMENDS Standing Rule #1)

**RATIFIED 2026-06-21 (orchestrator session; user explicitly signed off on relaxing the "bit-determinism is
sacred" standing rule).** Collapses the messy 4-axis space (fp32/tf32/bf16 × strict/loose) to **one regime +
one criterion**.

1. **Precision = bf16-AMP** (bf16 heavy ops + fp32 stability ops: focal `log(sigmoid)`, L1-over-log-dims, BEV
   scatter accumulation, optimizer). Field-standard (BEVFusion/BEVDet train fp16-AMP; bf16 is strictly safer
   — fp32 range, no GradScaler), fastest, and **verified to train comparably** (D15). **TF32 is dropped as a
   separate regime** — under bf16-AMP it is provably redundant (relaxed step 460 ms tf32-base vs 466 ms
   fp32-base = noise). **This supersedes D13 and D14's TF32-now framing.**
2. **Criterion = claim-reproducibility, NOT byte-identity.** Same-seed run-to-run variance is allowed
   (scatter_add atomics break byte-identity under `relaxed`); report results over **≥3 seeds (mean ± std)**
   at T5–T7; a claim is valid if it **clears the seed-variance floor**. **Retire strict byte-identity +
   checksum stamping from the science path.** Keep the **strict knob + static-AST ban as an offline
   dev-regression tool ONLY** (it caught two real bugs this session, incl. the lever-1 backward break).
3. **The new per-run science gate** (replaces the byte-identity gate): (a) **trains-clean reasonableness**
   (no NaN/divergence; the fp32-loss/accumulation guard), (b) **precision logged** into every run manifest,
   (c) for *reported* numbers, **multi-seed mean±std** + the characterized seed-variance floor, (d) the
   **offline strict-knob byte-identity regression** still green (dev-time, not a science bar).
4. **Config-collapse (authorized; executed in D17 Phase 0):** collapse `numeric-mode {fp32,tf32}` ×
   `determinism-level {strict,relaxed}` → ONE `precision = bf16 | fp32` knob (bf16 = science/relaxed; fp32 =
   dev/deterministic). ~8 call sites + provenance + gate scripts.
5. **Caching is permanently DROPPED** (D11/D12): it only ever worked because D1 *froze* the backbone, and
   D17 **trains** the backbone — the frozen-cache premise no longer exists. (D13 already foresaw bf16/TF32
   "survives 'we train our own encoder'"; caching does not.) The SUPR 5 TB cache request is cancelled.
6. **Consequence — re-baseline:** the D1/E reference checkpoints (tf32-strict) are **superseded**; the clean
   references (centralized + the ≥30-round FL) are **re-run in bf16-AMP** by D17 before T5–T7 bind to them.
   (Determinism is architecture-pinned anyway — D9; bf16 is one more re-baseline.)
7. **Standing rules amended** (CLAUDE.md): Rule #1 "bit-determinism is sacred / byte-identical" → the bf16-AMP
   + multi-seed regime above; Rule #4 null-config "bit-for-bit" → "within the seed-variance band."

### D16 addendum — the "banned ops" list, re-derived under claim-reproducibility (verified 2026-06-21, workflow `wf_be1c09cf-537`)

The old bans (Rule #1/#2) mixed **determinism** (relaxed by D16) with **portability/maintenance** (still
binding). Re-derived against current (2026) library state + ARM/H200 build feasibility (6 adversarial probes).
**The new binding bar = `maintained` + `builds on the target tier (x86 now, aarch64/H200 next)` + `doesn't
NaN` — NOT bit-determinism.** Use modern **in-tree** acceleration aggressively; avoid **out-of-tree fragile
CUDA extensions**.

| Verdict | Items | Why |
|---|---|---|
| **ADOPT (in-tree, portability-safe)** | **SDPA fused attention** (the **#1 missing lever** — D17 unfreezes the backbone, so the manual-fp32 windowed attention is the new GPU-dominant hotspot; route via `F.scaled_dot_product_attention`); **bf16-AMP**, **channels_last**, **fused Adam**, **activation checkpointing**, **EMA** (`swa_utils` — a missing capability lever that counters dilution + tightens the seed band) | All ship in the standard torch cu12x wheel incl. aarch64/H200; zero extra build. SDPA caveat: Swin's additive rel-pos bias rejects the FLASH backend → EFFICIENT backend, **~1.3–2×** not 2–4×; torchvision `swin_t` needs a rewrite. This **un-bans flash-attn-class fused attention** — via SDPA, not the external pkg. |
| **ADOPT-WITH-CAVEATS (in-tree, validate on ARM first)** | `torch.compile` (default, opt-in flag + eager fallback); `reduce-overhead`/CUDA-graphs on **static-shape camera/BEV subgraphs only** | Not bitwise-eager-equal; Inductor/Triton-on-aarch64 least-burned-in; pin a **release** cu128 aarch64 wheel (never nightly). `max-autotune` not worth it. |
| **GATED IN-TREE (measured ablation only)** | **dynamic voxelization** via native `scatter_reduce` as an **order-free `amax`** (not `scatter_add`); **LiDAR-capacity → in-tree dense upgrade** (PillarNet-style) | dyn-vox gain is modest (~≤1–2 mAP, small-object-biased, likely inside the seed band) → run only if pillar-cap point-dropping is shown to limit accuracy; watch the `torch.compile` dynamic-shape interaction. |
| **KEEP OUT (portability/maintenance, NOT determinism — none costs us speed)** | `flash-attn` pkg (SDPA covers it; x86-only build); **spconv/torchsparse** (no aarch64 wheel, stale single-maintainer, breaks the strict dev tool — the ~+5–8 mAP LiDAR gain isn't our binding constraint and is recoverable in-tree); `mmdet3d`/`mmcv` (framework not kernel; unmaintained + CUDA-13 build break); **FP8/Transformer-Engine** (external ext + 1–2% accuracy hit on small conv nets — can't trade accuracy at 0.36 mAP); **DALI** (out-of-tree; fix dataloading in-tree); **NestedTensor** (prototype) | These fail the maintained-or-portable bar; the strict offline dev tool also has no deterministic knob for spconv/mmcv kernels (a second reason out). |

**The offline strict dev tool keeps working** by pinning deterministic paths (SDPA→`sdpa_kernel(MATH)`,
compile/graphs/`channels_last`/fused off, `use_deterministic_algorithms(True)`). **Amends CLAUDE.md Rule #1
banned-list + Rule #2** (spconv downgraded from "banned for determinism" to "keep-out for portability";
mmdet3d/mmcv unchanged-but-reasoned; SDPA explicitly allowed). Full evidence: the workflow verdicts in the
session transcript; the operational list is in `kickoff/model_capability_kickoff.md` §Tooling envelope.

---

## D17 — The Model Capability + Recipe (MCR) session: raise the model, fix the FL recipe, produce the new ≥30-round bf16 FL clean reference (AMENDS D1)

**CONFIRMED 2026-06-21 (user-directed; scope chosen = "all-in-one to a new FL reference").** A dedicated
session (sibling to the speedup session, OUTSIDE the T<N> numbering) that fixes the two problems D15 exposed
**before** T5 restarts or T6/T7 begin. Full charter: **`kickoff/model_capability_kickoff.md`**; deliverables
land in `collab/model_capability/`.

**Why both problems are ONE program.** D15 showed the util gap (Issue 1) and the weak model (Issue 2) share a
root: a too-light, frozen-backbone model that (a) can't keep a GPU busy (12% util, overcommit dead) and (b)
caps centralized mAP at 0.36. **Training the backbone fixes BOTH** — the step becomes GPU-heavy (util rises,
A100 1.63×/step lands) and the detector gets stronger. So the session raises model capability and harvests
the throughput as a side effect, then closes the FL dilution gap.

**This AMENDS D1.** D1 froze the ImageNet camera backbone and labelled full-model FL "the generality
ablation." D17 **promotes full-model training to PRIMARY** — the camera backbone is now **trained**. The
frozen-backbone setting is demoted to a comparison point. The **federate-the-backbone vs.
central-pretrain→freeze→federate-fusion+head** fork has T5–T7 threat-model consequences (a federated backbone
lets a backdoor live in the backbone = broader/realistic surface; a frozen backbone keeps the attack surface
in fusion = preserves the D1/D3 fusion-aware framing) — the session **measures the frozen-vs-trained mAP cost
and brings this back as a data-driven D-decision**, it does NOT pre-commit.

**Sequencing (value-ordered for the Alvis sunset, 2026-06-30 = 9 days):**
- **Phase 0 — land the ratification (mechanical):** execute the D16 config-collapse (one `precision` knob,
  bf16 default for science) **on the CLEAN science path ONLY — T5 untouched** (`attacks/`, `t5_*`,
  `tests/test_attack_*` not modified; keep the lazy dormant imports resolving); confirm the strict-knob
  regression still green. The `t5_attack_eval.py` precision-threading hazard is **re-parked as a T5-restart
  prerequisite**, not MCR.
- **Phase 1 — raise the CENTRALIZED ceiling (the core capability search, in bf16-AMP):** unfreeze/train the
  camera backbone (LR-grouped) **+ the SDPA attention rewrite** (the now-trained windowed attention is the
  GPU-dominant hotspot); LiDAR multi-sweep accumulation; image-resolution + BEV-grid; fusion-layer redesign
  (D3 ConvFuser depth/attention) + BEV-neck; optimizer/LR/schedule + **EMA** (`swa_utils` — counters dilution
  + tightens the seed band); data aug — ablate one-at-a-time + a combined recipe, measure official mAP/NDS
  each step. Target: a detector **clearly above 0.36** with high car recall / ASR-eligible count
  (attack-credibility, not a round number).
- **Phase 2 — throughput to AFFORD it (interleaved):** in-tree eager wins FIRST (SDPA, `channels_last`, fused
  Adam, activation checkpointing); then **opt-in** `torch.compile(reduce-overhead)`/CUDA-graphs on static-shape
  camera/BEV subgraphs only (validate on ARM first, leave ragged loss + variable-count voxelizer out);
  num-workers↑ + node-local staging; move heavy runs to **A100/A100-fat** (a trained backbone needs the VRAM,
  and the A100 finally pays off once util rises). NO overcommit, NO reflexive big-batch, NO out-of-tree exts
  (D16 envelope).
- **Phase 3 — FL recipe + the new reference:** transfer the strong recipe to FL; close the dilution gap with
  **server-side momentum (FedAdam/FedOpt/FedYogi)**, round budget **≥30** (find the plateau), local-epoch
  tuning, on the location-coherent log-group non-IID (the threat model). Produce the **new clean ≥30-round
  bf16-AMP FL reference**, **multi-seed (≥3, mean±std)**.
- **Phase 4 — re-baseline the bindings:** re-judge T4 readiness on the new checkpoint; rebuild the frozen
  held-out ASR subset (content-hashed, bound to the new checkpoint); record the new provenance under
  claim-reproducibility (seed-band, not a single checksum).

**Alvis-sunset realism (built into the charter).** Determinism is architecture-pinned (D9) ⇒ the *final*
reference must be (re)produced on the GPU tier T5–T7 will run on. So the session's Alvis job is to **LOCK THE
RECIPE** (centralized ceiling + FL choices + throughput) — the portable, reusable artifact — and the **final
multi-seed ≥30-round FL reference is produced at the locked recipe on whichever tier is live** (A40/A100 now,
H200/Arrhenius after migration; a re-baseline is forced regardless). Every heavy run is **checkpoint-resumable**
(optimizer state saved) so a forced migration mid-run resumes; the venv stays reproducible from the pinned
manifest (`docs/env.md`). This honors "all-in-one" (the session owns the whole pipeline through the reference)
while not betting the deliverable on finishing a backbone-training ≥30-round FL run inside 9 A40-days.

**Absorbs the speedup doc's 5 open items:** D16 ratified (✓), config-collapse authorized (Phase 0, clean path
only), the "fresh FL-recipe session" (Phase 3), the multi-seed protocol (D16 + Phase 3), re-baseline references
in bf16 (Phase 4). (The `t5_attack_eval.py` hazard is **re-parked to T5-restart** — it belongs to T5, which
MCR does not touch.)

**Boundaries:** (1) no T5 restart / T6 / T7 until the new clean reference + its readiness verdict exist;
(2) **T5 is OUT OF SCOPE — do not modify `attacks/`, `t5_*`, `tests/test_attack_*`;** if a capability change
*requires* touching a T5-shared interface (ConvFuser signature, the `maybe_wrap_for_client` call site),
**escalate to the orchestrator, don't refactor T5**; (3) every reported number in bf16-AMP, multi-seed — no
mixing precision regimes (D16); (4) mini/smoke ≠ scientific evidence; (5) stay inside the **D16-addendum
tooling envelope** (maintained + builds-on-target-tier + no-NaN; in-tree over fragile extensions), NOT
bit-determinism; (6) the federate-vs-freeze threat-model choice returns as a data-backed decision, not a
unilateral session call.
