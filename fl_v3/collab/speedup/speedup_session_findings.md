# Speedup + diagnostics session — findings, trace & open problems (D14/D15)

> Consolidated record of the dedicated speedup/diagnostics session (charter D14, extended D15). Merges the
> former `A_profiling_report.md`, `Q1_runtime_and_community.md`, `Q1_speedup_roadmap_D15.md`,
> `SYNTHESIS_5questions.md`, and the overcommit teardown (`overcommit_diag/SUMMARY.txt`). The
> orchestrator-facing **decision + asks** live in [`D15_D16_decision_for_orchestrator.md`](D15_D16_decision_for_orchestrator.md).
>
> **Labeling note (read first).** The fast regime is **bf16-AMP**. In the code it is reached by
> `determinism-level=relaxed` (which turns on `torch.autocast(bf16)` over the forward — `loop.py`); the
> separate `numeric-mode={fp32,tf32}` knob only sets the TF32 flags for the *residual* float32 ops. So a
> run logged `LEVEL=relaxed NUMERIC_MODE=fp32` **is a bf16-AMP run** (bf16 forward + fp32 stability ops:
> focal `log(sigmoid)`, L1-over-log-dims, BEV scatter accumulation, optimizer). This 2-axis naming is the
> mess D16 proposes to collapse into one `precision = bf16 | fp32` knob — see open problems §.
>
> Code committed on `claude/focused-diffie-1da582` (NOT v3-ad-perception). Everything is behind the
> `determinism-level` knob; **default `strict` is byte-identical and unchanged** (247 tests pass).

## The journey (trace)

1. **D14 diagnostics** — profile where runtime goes (A); gate server eval (B); test TF32 (C); centralized
   matched-budget baseline (D); 15-vs-30-round FL convergence (E). Answered the 5 questions below.
2. **D15 speedup** — relax bit-determinism behind a knob to unlock atomics + bf16 + compile; build the
   `determinism-level=relaxed` regime; measure the per-step and per-round speedup.
3. **Overcommit investigation** — the A100 is fast per-step but low-util; tried more clients/GPU to fill
   it. It failed (OOM, then ~30% oscillating util). Built the E1/E2/E3 teardown to find out *why*.
4. **The pivot** — the teardown shows overcommit is a dead end (GPU time-slices serialize without MPS);
   the real lever is **cheaper steps** (CUDA graphs), not more clients. See open problems §.

## Q1 — where runtime is spent (the bottleneck FLIPPED after the rewrite)

**Before (strict, fp32, A40, batch-16, frozen Swin-T; job 6767120), mean step ≈ 1931 ms:**

| stage | %step | why it costs what it does |
|---|---:|---|
| camera_backbone | 30.5% | frozen Swin-T over 6 cams × batch — matmul/attention. TF32 1.33×. |
| **LSS view_transform** | **31.2%** | materializes a ~1.3 GB lifted tensor (depth⊗context outer product) + **argsort + cumsum over ~4 M frustum points** — the *deterministic* substitute for the banned atomic `bev_pool`. Memory-bandwidth bound → TF32 ≈ 1.00×. |
| loss (CenterPoint) | 15.9% | not compute — `build_targets` looped over every GT box calling `.item()` → a CPU↔GPU sync per box. TF32 ≈ 1.00×. |
| backward | 12.2% | follows the view-transform outer-product + cumsum backward (gather). TF32 1.23×. |
| rest (dataloader 3.5%, neck/lidar/fusion/bev_neck/head) | <10% | small. GPU util ~76%. |

The D11/D12 *inferred* "80–90% backbone" is **false** — backbone is ~30%, co-equal with the LSS
view-transform. The run is GPU-compute-bound (dataloader 3.5%), confirming Flower/Ray overhead is
negligible.

**After the D15 rewrite (relaxed, bf16-AMP, A40), mean step ≈ 460 ms (compile) / 572 ms (FL, no compile):**

| stage | strict tf32 | relaxed bf16 | factor |
|---|---:|---:|---:|
| LSS view_transform | 602 ms | **14 ms** | **42.8×** (scatter_add splat + mask-then-lift; precision-independent) |
| camera_backbone | 442 ms | 147 ms (compile) / 264 (FL) | 3.0× / 1.7× (bf16 + compile) |
| backward | 191 ms | 112 ms | 1.7× |
| loss | 91 ms | 87 ms | 1.04× (already vectorized) |
| **mean step** | **1443 ms** | **460 / 572 ms** | **~3.1× / ~2.5×** |

The bottleneck flipped: the LSS view-transform (was a 31% co-bottleneck) collapsed to **2.4%**; the frozen
backbone is now the dominant stage (~46% in the FL no-compile path).

## What we built (the levers, all behind `determinism-level=relaxed`)

`strict` (default) = byte-identical, unchanged. `relaxed` unlocks what bit-identity banned:
- **scatter_add splat + mask-then-lift** (the big, precision-independent win — drops the argsort + cumsum
  + two ~1.3 GB copies that were pure determinism tax).
- **bf16-AMP** autocast over the forward (loss + BEV-splat accumulation kept fp32 for stability — the NaN
  fix below).
- **torch.compile(backbone)** for long/centralized runs (and FL with a pre-warmed cache — see cache-race).
- free scheduling: `cudnn.benchmark`, per-step `.item()` sync removal, dataloader pin/persistent/prefetch.

**The bf16 NaN — caught, diagnosed, fixed.** The profiler showed bf16 ~3× faster, but a 3-epoch smoke
diverged to NaN: the CenterPoint focal loss `log(sigmoid(logit))` ran on bf16 logits. Fix: loss + BEV
accumulation in **fp32** (forward stays bf16). Re-test: bf16-AMP 3.216/2.362/2.139 vs deterministic
3.199/2.365/2.145 — within ~0.5% at every epoch. *This is why we keep one lightweight reasonableness gate
even after dropping bit-identity: speed that NaNs is worthless.* (Separately, a `torch.compile`
checkpoint-save bug — `_orig_mod.` key prefix — was found + fixed in `5c2dfc2`.)

## Round-level speedup ladder (MEASURED, FL, full participation, 3-round runs)

| config | per-round | vs strict | note |
|---|---:|---:|---|
| A40 strict tf32 (reference) | **15.6 min** | 1.0× | E-15/E-30 |
| A40 relaxed bf16, 1/GPU, no compile | 7.0 min | 2.2× | job 6767474 |
| A40 relaxed bf16 + compile, 1/GPU | 6.4 min | 2.4× | job 6767561 |
| **A40 relaxed bf16 + compile + 2/GPU** | **5.6 min** | **2.76×** | job 6767562 — best on A40 |
| **A100 relaxed bf16 + compile, 1/GPU** | **5.6 min** | 2.76× | job 6767755 — **= A40 2/GPU**; only 1.14× over A40 1/GPU despite 1.63×/step |

**The A100 surprise:** per-*step* the A100 is 1.63× the A40 (283 vs 460 ms) but per-*round* only 1.14×
(6.4→5.6) — because each round's overhead (Ray actor spin-up, dataloader, aggregation) is launch/CPU-bound
and GPU-speed-invariant. A100 util at 1/GPU is only **12%**. This is what motivated the overcommit attempt.

## Q2–Q5 (the diagnostic answers, condensed)

- **Q2 server eval:** gating it (`server-eval-mode=none|final|every_n|all`) is byte-identity-safe (same-seed
  none-vs-all → identical checksum `0eed9236…`, job 6767126). Trainval default `none`. The gated metric is
  only the server proxy; ASR + official mAP/NDS stay post-hoc.
- **Q3 TF32:** real but modest on A40 (~1.12× end-to-end, confined to matmul stages; LSS+loss ~1.00×). Under
  bf16-AMP it is **redundant** (relaxed step 460 ms tf32-base vs 466 ms fp32-base = noise) → D16 drops it.
- **Q4/Q5 the weak T5 = FL-undertraining + FedAvg dilution, NOT architecture/recipe.** Official mAP/NDS
  (matched budget, epochs==rounds):

  | setting | budget | mAP | NDS | car recall |
  |---|---|---:|---:|---:|
  | **Centralized (D1)** | 15 ep | **0.360** | **0.357** | 0.93 |
  | FL (E-15) | 15 r | 0.126 | 0.169 | 0.85 |
  | FL (E-30) | 30 r | 0.196 | 0.226 | 0.89 |

  Centralized reaches a strong detector on the same model/data/budget → architecture+recipe are fine. FL-15
  was **undertrained** (FL-15→30 mAP +55%, still climbing at r30). Even FL-30 is ~1.8× below centralized →
  **FedAvg dilution** over location-coherent non-IID shards (caveat: also includes FL's per-round optimizer
  reset). **Next clean reference must be ≥30 rounds** (check the plateau past 30; r27→r30 slope still
  +0.005/round). The T5 camera-only attack ran on a doubly-compromised (undertrained + diluted) checkpoint —
  the null was uninterpretable.

## Overcommit investigation — why A100 4/GPU failed, and the teardown (E1/E2/E3)

Naive attempts first: **4/GPU OOMed** (compile inflates VRAM to ~10–12 GB/client → 4×≈40 GB), **2/GPU
failed on an inductor cache-race** (shared `TORCHINDUCTOR_CACHE_DIR` + concurrent round-1 compiles → corrupt
metadata JSON), and the compile-off 4/GPU run ran but oscillated at **~30% util**. (Correction banked: the
A100 and A40 nodes have **identical 244 GB host RAM** — the OOMs were VRAM/cache-race, not node size.)

The per-step teardown (one A100:4 node, job 6769136, `overcommit_diag/`):

**E1 — GPU-sharing ceiling (data + Ray removed, fixed batch):**

| K clients/GPU | per-proc step | agg steps/s | scale vs K=1 | util |
|---:|---:|---:|---:|---:|
| 1 | 354 ms | 2.83 | 1.00× | 10% |
| 2 | 788 ms | 2.62 | 0.93× | 30% |
| 3 | 1008 ms | 3.05 | 1.08× | 38% |
| 4 | 1331 ms | 3.01 | **1.06×** | 52% |

**Overcommit gives ~ZERO throughput gain (~1.0× across K), even with the dataloader bypassed.** Per-proc
step scales ~linearly with K → the clients **serialize on the GPU**. Without **CUDA MPS**, separate
processes' GPU contexts are time-sliced, not run concurrently; and since each step is **latency/launch-bound**
(only ~10% of the K=1 step is GPU compute), interleaving latency-bound work yields no throughput. Rising
util (10→52%) is busy-fraction inflation, not useful work.

**E2 — dataloader-only:** single loader nw=2 → 27.1, nw=4 → 60.9, nw=8 → 76.3 samples/s (worker-bound below
8). 16 concurrent loaders @ nw=2 → 259 agg vs ideal 434 = **0.60 contention factor → shared-FS I/O wall**.
A single client needs ~50 samples/s, so **nw=2 starves even one client**; nw=4 feeds it.

**E3 — realistic 16-actor:** nw=2 → per-proc step 5197 ms, **83% dataloader_wait**, util 22%, host 121 GB,
16/16 ok (reproduces the failed run). nw=4 → 3581 ms, 53% wait, util 24%, host 130 GB, **6/16 survived
(10 host-OOM-killed)** — confirms the num-workers↔host-RAM wall.

**Verdict:** A100 overcommit is a dead end here — it can't gain throughput (E1 serialization), the data
pipeline is independently starved (E2), and widening it OOMs host RAM (E3). The A100 sits at low util not
for lack of clients but because each step is **launch/latency-bound at a tiny batch** with a **frozen**
backbone. (Caveat: E1's procs ran simultaneously, so absolute times carry node-wide CPU load, but the
per-GPU K-trend → flat-aggregate conclusion is robust.)

## Community / SOTA context

- **BEVFusion (MIT):** "BEV pooling alone takes >80% of runtime" — the LSS view-transform was their
  bottleneck too. Their fix: a custom CUDA **bev_pool** (precompute + interval reduction) → >40×. The speed
  comes from **atomicAdd** → non-deterministic → **banned** for us. The determinism-safe *ideas* (avoid
  frustum materialization, precompute geometry) are our levers and we took them.
- **BEVPoolv2 (BEVDet):** +15× by not materializing the large frustum feature (= our mask-then-lift).
- **Training setup they use:** 8 GPUs × batch 4, 20–24 epochs, **FP16/AMP**, A100-class. vs us: 4 GPUs,
  bf16-AMP (deterministic-optional), ≥30 rounds, no custom CUDA op. The small batch (16) + frozen backbone
  is why our A100 util is low — SOTA uses much bigger effective batches.

## Open discussion & UNSOLVED problems

**The strategic pivot (current thinking).** Util is a means; the goal is a well-trained FL model in
acceptable wall-clock. Since (a) the model is undertrained → needs *more* gradient updates, and (b) each
step is latency/launch-bound (12% util, even inside the backbone = many small kernels with launch gaps),
the right lever is **cheaper steps, not fewer steps and not more clients**:

1. **CUDA graphs / `torch.compile(mode="reduce-overhead")`** on the backbone+forward — attacks the ~90%
   launch-bound directly, keeps update count high (good for an undertrained model). The most promising
   per-step lever. *Open:* needs static shapes — image tensors qualify, the ragged-box loss does not, so
   graph the forward and leave the loss out. **Untested — the natural next probe.**
2. **Bigger batch (16→48/64)** uses the idle VRAM (8/40 GB) and raises util, but **trades away gradient
   updates** → needs LR scaling and, to preserve quality, *more rounds* (not more local epochs — that
   worsens FedAvg dilution). A throughput lever, not a free quality lever. Use *with* graphs, not instead.
3. **Training our own backbone (planned for Cycle-04)** makes the step GPU-heavy → util rises on its own →
   the A100's 1.63×/step finally lands at 1/GPU. The current 12% util is partly a frozen-backbone artifact.
4. **Dataloader at 1/GPU** can afford `num-workers=8` (only 4 loaders/node, no host-RAM wall) + node-local
   data staging (/dev/shm or NVMe) to beat the 0.60× Mimer FS contention.
5. **CUDA MPS** would let overcommit actually overlap (undo E1's serialization) — fallback only; graphs +
   backbone-training make it unnecessary.

**Recommended combo for us:** 1/GPU + modest batch + cheap steps (CUDA graphs + pre-warmed compile +
num-workers↑ + node-local data), then spend the saved wall-clock on **more rounds** to fix undertraining.
Overcommit and big-batch are NOT on this path.

**Other open items / hazards (for the orchestrator):**
- **D16 config-collapse (top open item):** collapse `numeric-mode × determinism-level` → one
  `precision = bf16 | fp32` knob. One-way (touches ~8 call sites + provenance + re-baselines references);
  **held for the orchestrator.** Until then, read every "fp32" relaxed run as **bf16-AMP** (labeling note).
- **FL recipe (separate session):** FedAvg dilution is structural (centralized 0.36 ≫ FL 0.20 at matched
  budget) → server-side momentum (FedAdam/FedOpt), non-IID severity, local-epochs, round budget.
- **Re-baseline the clean references in bf16-AMP** at ≥30 rounds before T5–T7 bind to them.
- **Multi-seed protocol** for T5–T7 (≥3 seeds; the defense knife-edge needs it).
- **`t5_attack_eval.py` does not thread numeric-mode** → would evaluate a relaxed checkpoint in the wrong
  regime; mirror `t4_readiness_eval.py` before any T5 attack eval.

## Artifacts / provenance
- Profiler: `fl_v3/scripts/profile_stages_a40.py` (+ `--fixed-batch`, `--num-workers`).
- Overcommit teardown: `fl_v3/scripts/run_overcommit_diag_a100.sh` + `bench_dataloader_a100.py` +
  `agg_overcommit_diag.py`; raw results were in `overcommit_diag/` (SUMMARY folded in above).
- FL launcher: `run_clean_fl_tf32_a40.sh` (LEVEL/COMPILE/UTIL_SAMPLE/NUM_WORKERS/PREWARM knobs).
- Bit-identity regression tool (strict knob): `verify_levers.py`.
