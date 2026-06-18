# T5–T7 speedup analysis — every lever, with feasibility under our constraints

> Build-session note (2026-06-18), written while the T5 trainval runs execute. Grounds the D11
> speedup backlog with concrete numbers + the storage reality. **The two constraints that gate
> everything:** (1) **determinism is sacred** — anything that perturbs numerics breaks either the
> byte-identical-null GATE (the null must reproduce `a80466c3` bit-for-bit) or the scientific
> equivalence of a cached/accelerated run vs a live one; (2) **storage** — the group's Mimer
> allocation is 500 GB total, **96 GB free** (405 GB used, of which 344 GB is other group members:
> Sheng 230 GB + chengyiming 114 GB — not reclaimable).

## The bottleneck (measured, D11)

A run is **compute-bound on the frozen Swin-T backbone**: a 6-camera ViT forward, recomputed every
training step though the backbone never updates (D1 freezes it). ≈ **22 min/round × 15 ≈ 5.5 h/run**;
all 4 A40s pinned at 100 %; Flower/aggregation negligible. The backbone is ~80–90 % of per-step cost.

**Legend:** ✅ already applied · 🟢 available now (free + safe) · 🟡 possible w/ work · 🔴 blocked ·
⛔ banned (breaks determinism/science).

---

## A. Eliminate the redundant frozen-backbone compute (the "caching" family) — the #1 lever

| # | Method | Mechanism | Gain | Status | Risk / blocker |
|---|---|---|---|---|---|
| A1 | **Disk feature cache** (D11 #1) | Precompute the frozen Swin-T 4-scale features per (keyframe, cam) once; training reads them into the trainable neck instead of re-running the backbone | **~3–5×/run** | 🔴 | **Storage**: 28,130 train kf × 48.7 MB = **1.37 TB fp32** (274 GB even for the 2 coarse scales) ≫ 96 GB free. Needs a new **bit-identical** determinism gate or it breaks the null. |
| A2 | In-run cache across rounds (RAM/local) | Same keyframes recur every round → compute once (round 1), reuse 2–15 | ~5–10×/run | 🔴 | Same storage wall (all 28 k kf recur → 274 GB–1.37 TB); RAM/`/tmp` (255 GB) can't hold it at fp32. |
| A3 | In-client cache across local epochs | Reuse within a client's local epochs (RAM, no disk) | (E−1)/E | 🔴 | `num-local-epochs=1` here → **no within-client reuse**. Raising epochs changes FL dynamics + the `a80466c3` binding. |
| A4 | fp16 / quantised cache on `/tmp` | 2-scale fp16 ≈ 137 GB fits node `/tmp` (255 GB) | ~2–3× | ⛔/🟡 | **Not bit-identical** → cached model ≠ live model (breaks the null + scientific equivalence); per-job ephemeral (no amortisation across the T7 fan-out, which lands on different nodes). |
| A5 | Partial cache (coarse scales only) | Cache stride-16/32 to save storage | **~0×** | 🔴 | Useless: to *skip* the backbone you must cache **all 4 scales it emits**; caching only coarse scales still forces the full forward (the expensive early high-res attention runs anyway). |
| A6 | Cache the preprocessed images (not features) | Skip decode+resize+normalize | <5 % | 🟢-ish | The decode is cheap vs the backbone; near-useless (the backbone forward still runs). Not worth it. |
| A7 | Content-hash cache (clean reuse, live for triggered) | Honest clients + non-poisoned samples (the majority) hit the clean cache; triggered samples compute live | ~3–5× on the clean majority | 🔴 | Still storage-blocked (A1). The triggered minority is small, so this is the *right design* once storage exists. |

**Verdict:** caching is the real lever but **blocked on persistent storage** — a *resource request*, not
code. It pays off most for **T6/T7** (dozens of cells reuse one cache). See §H.

---

## B. Faster hardware

| # | Method | Gain | Status | Risk / blocker |
|---|---|---|---|---|
| B1 | **A100** for the ViT forward | ~2–2.5×/run (training) | 🔴 for T5 | The whole task is **A40-pinned** (the READY ckpt `a80466c3` + frozen subset `2ad8f8da` were built on A40; A100 numerics aren't byte-comparable → breaks the **null GATE** + shifts subset scoring). The A100 det-gate already passed, so a **fresh A100-tier cycle** (re-establish the reference on A100) is viable for T6/T7 — but not *mixable* with the A40 artifacts. |
| B2 | A100 for the **eval** only | ~1.2× | 🔴 | The bs=1 eval is I/O/CPU-bound (T4 §5c: A100≈A40 there); + mixing tiers vs the A40 subset is the cross-tier trap. |
| B3 | H100 (Arrhenius, post-2026-06-30) | ~2–3× | 🟡 future | ARM re-build + a new det-gate + new reference; the cycle moves there anyway. |
| B4 | More GPUs **per run** | — | ✅ maxed | Alvis nodes are 4-GPU, no A40 InfiniBand → 4× (Path-A) is the per-run ceiling, already used. |

---

## C. Parallelism / fan-out (mostly already applied — these don't cut per-cell time, they overlap it)

| # | Method | Effect | Status |
|---|---|---|---|
| C1 | **Across-cell fan-out** (D9) — one SLURM job per independent run/cell | **Matrix wall-clock ≈ one cell**, not the sum | ✅ T5 (5 runs ∥) / the dominant **T7** lever |
| C2 | Path-A multi-GPU (4 clients ∥ on 4 A40s) | ~4×/run vs serial | ✅ applied |
| C3 | Path-B shared-GPU overcommit (`num-gpus<1`) | hides inter-round latency only | 🟡 ~0 gain here (Swin-T at batch≥16 ≈ 100 % SM, D9) |
| C4 | **Wider eval fan-out** (more ablation shards) | eval wall-clock = total/shards → **~2 h → ~40 min** | 🟢 **free now** (GPU-bound, not storage-bound) |
| C5 | More 4-GPU nodes for one run (multi-node) | — | 🔴 no A40 InfiniBand on Alvis |

---

## D. Eval-specific (the bs=1 decode is GPU-under-utilised → easy parallel wins)

| # | Method | Gain | Status | Note |
|---|---|---|---|---|
| D1 | **Fan the ablation wider** (40–80 shards) | ~2–4× eval wall-clock | 🟢 now | The eval is ~137 k forwards over the subset; more shards = less wall-clock, same total GPU-h (plentiful). |
| D2 | Share the backbone across a target's conditions | ~15–20 % eval | ✅/🟡 | cond-4+cond-5a already share one forward; **cond-1 & cond-3 use the same clean camera image** → their camera-backbone output is identical and could be computed once (a real micro-opt). |
| D3 | Batch-invariant **batched** decode | ~5–10× eval | 🟡 hard | T4 chose bs=1 for batch-invariance (cuDNN conv varies with batch composition). A deterministic batched decode would need solving that — a T4-domain change. |
| D4 | Cache **clean val features** for stealth + cond-1/3 | ~2× on the clean-image eval | 🔴 | 6019 val kf × 48.7 MB ≈ 293 GB ≫ 96 GB; triggered conds are per-target (low reuse) anyway. |

---

## E. I/O & data loading

| # | Method | Gain | Status | Note |
|---|---|---|---|---|
| E1 | More dataloader workers (`num-workers`) | marginal | 🟢-ish | Compute-bound (GPUs 100 %), so loading isn't the bottleneck; a small bump is harmless. |
| E2 | Faster image decode (opencv/turbojpeg) | small | ⛔ | PIL is **pinned for determinism** (opencv decodes the same JPEG to different pixels) — banned. |
| E3 | Stage data to node-local NVMe | small | 🟡 | Mimer is already fast (weka); decode/forward dominate, not read latency. |

---

## F. Reduce the work (config / algorithmic) — almost all break the science

| # | Method | Gain | Status | Why blocked |
|---|---|---|---|---|
| F1 | Fewer rounds | linear | ⛔ | 15 rounds is the D10 setting `a80466c3` binds to; the null must match + clean-vs-poisoned fairness. |
| F2 | Client sampling (`fraction<1`) | ~N/m × | ⛔ | D10 mandates **full participation**; sampling confounds the 2×2 + voids the δ interpretation. |
| F3 | Smaller image / batch / fewer cams | large | ⛔ | Changes the model → not the `a80466c3` family. |
| F4 | resnet18 backbone | ~2–3× | ⛔ | Mini-only; the science needs Swin-T (D1). |
| F5 | Subsample the ASR subset | linear on eval | ⛔ | The frozen subset is hard-pinned (N=27,432, **rebuild forbidden**, §0.C6). |
| F6 | Controls at fewer rounds | — | ⛔/➖ | Unfair vs the 15-round attack; and they run **in parallel** (no wall-clock cost) anyway. |
| F7 | Defer the controls/paired as fast-follow | frees capacity | 🟢 | No wall-clock gain (parallel), but frees cluster slots / fair-share if contended. |

---

## G. Banned outright (determinism posture, D11 #4)

`torch.compile` ⛔ (can inject nondeterminism + fragile kernels) · AMP/fp16 **training** ⛔
(non-deterministic accumulation) · flash / mem-efficient attention ⛔ (Swin uses fp32 manual
attention) · non-stable sort/topk, atomic scatter, `grid_sample` backward ⛔ (already banned in the
model). Gradient checkpointing = saves memory, **costs** time (we're not memory-bound). N/A.

---

## H. The real unblocker = persistent storage (turns A1/A7 from 🔴 → 🟢)

The only thing standing between us and the ~3–5× cache is **bytes to put it in**. Options, in order:

1. **Request a larger Mimer/scratch allocation** for the group (NAISS/C3SE) — even **+400 GB** unlocks
   the 2-scale fp32 cache; **+1.5 TB** unlocks the full 4-scale cache → ~3–5× across **all of T6/T7**.
2. A **dedicated fast-tier scratch** on the `mimer-weka*` 51 TB filesystem (3.6 TB free) **if a
   user-writable area** can be provisioned there (the Datasets mount is read-only).
3. Free space socially (Sheng 230 GB / chengyiming 114 GB) — out of our hands.

Once storage exists, the build is well-scoped: precompute the frozen 4-scale features keyed by
`(sample_token, cam)` (content-hash for triggered variants), a `det-feature-cache` det-gate
(precompute-twice byte-identity + cached-vs-live equivalence), and a `forward` switch that reads the
cache for clean images / computes live for triggered ones (A7). The null still reproduces `a80466c3`
(proving bit-identity), so the GATE survives.

---

## Bottom line

- **T5 (now):** ~7 h wall-clock is the floor — it already uses every *free* lever (Path-A 4-GPU + 5
  runs in parallel + sharded eval). The only un-applied free win is **D1/C4 — fan the eval wider**
  (~2 h → ~40 min). Per-cell 5.5 h is fixed by the frozen-Swin-T compute; only caching moves it, and
  caching needs storage.
- **T6/T7 (the real compute sink):** across-cell fan-out (C1) already makes the *matrix* wall-clock ≈
  one cell (~7 h). To beat the **per-cell** floor, the prerequisite is a **storage allocation** → then
  the feature cache (A1/A7) gives ~3–5× across the whole matrix. That is the single highest-ROI action,
  and it's a resource request to file **before** T7.
