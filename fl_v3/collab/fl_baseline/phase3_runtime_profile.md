# MCR Phase-3 — FL runtime teardown + speed-up (before the heavy run, R1)

> Profile bb02d's FL per-client step on **client-0's real shard** (1075 kf), A100-SXM4-40GB, batch-4 (the FL
> per-client batch), bf16. Tool: `p1_profile_a100.py --opt-compare` (job 6778238, 6:17). Raw: `p3_fl_profile.json`.

## Per-module teardown (baseline = the un-optimized FL config: sdpa on, compile off)

| stage | mean ms | share | note |
|---|---:|---:|---|
| dataloader | 0.25 | 0.1% | **NOT loader-bound** at batch-4 / 4 workers (the multi-sweep loader keeps pace) |
| forward_total | 120.6 | 26% | |
| — lidar_backbone | 36.4 | 7.8% | the dense 2D LiDAR trunk — the biggest single forward module |
| — camera_backbone (Swin-T) | 31.2 | 6.7% | |
| — lidar_encoder (PFN) | 16.7 | 3.6% | |
| — fusion | 15.8 | 3.4% | |
| — bev_neck / view_transform / head / camera_neck / preprocess | ~20 | 4% | |
| **backward** | **214.8** | **46%** | the dominant cost (1.8× the forward) |
| loss + optimizer | 11.0 | 2% | |
| **wall step** | **353** | — | util **95.5%** ⇒ **compute-bound**; peak VRAM 18.1 GB |

Kernel breakdown (top self-CUDA): `aten::copy_` (LSS splat + bf16 casts), `native_group_norm` fwd+bwd,
`convolution_backward`, `cudnn_convolution` — i.e. the **GroupNorm + conv BEV stack** dominates, exactly what
`torch.compile` of those subgraphs fuses.

## Speed-up A/B (batch-4, the lever)

| regime | step ms | speedup | util | peak VRAM |
|---|---:|---:|---:|---:|
| baseline (sdpa, no compile) | 353 | 1.00× | 95.5% | 18.1 GB |
| + compile (camera_backbone) + sdpa | 297 | 1.19× | 97.1% | 15.1 GB |
| **+ compile BEV-stack (camera_neck+fusion+bev_neck+head+lidar_backbone)** | **244** | **1.45×** | 96.0% | 15.5 GB |

The BEV-stack subgraphs have **static** shapes (512²/256² BEV) ⇒ no recompilation; the variable-pillar PFN +
ragged LSS splat stay eager. VRAM drops with compile (15.5 GB ≪ 40 GB) ⇒ trivially fits at batch-4.

## Optimization applied

- `client_app` now compiles the **BEV stack + lidar_backbone** (not just camera_backbone) under a new
  `det-compile-bev` knob (registered in pyproject; default off ⇒ byte-identical).
- `fl_bb02d_fedadam.json`: **`compile-backbone=true` + `det-compile-bev=true`** ⇒ the 1.45× path.
- Launcher sets a **persistent `TORCHINDUCTOR_CACHE_DIR`** so the kernel compile is a cache-hit after the
  first client (only the cheap dynamo re-trace recurs per client/round).
- BLAS thread-pin (`OMP/OPENBLAS/MKL=1`) already in the launcher (loader thread-oversubscription guard).

## Per-round estimate (to be confirmed by the R=2 speed-test, job 6778269 with compile ON)

At 244 ms/step, mean shard 1125 kf / batch-4 = 281 steps ⇒ ~69 s/client training; the largest shard (1449 kf)
~88 s. 25 clients / 4 GPUs = 7 waves ⇒ **~11–12 min/round** steady-state (+ a one-time first-client compile
~1–2 min, cached thereafter). **R=15 ≈ ~3 h** (vs ~4.1 h un-compiled — compile saves ~1 h). The R=2 speed-test
measures the *real* per-round wall-time + the model-rebuild-vs-train split (the FL overhead the single-GPU
profiler can't see) + confirms compile is net-positive in the Ray path (re-trace per client) + trains-clean.
**Decision gate:** if round-2 (warm) wall-time × 15 is acceptable → submit the R=15 training.

## FL-path speed-up (the Ray-run reality — single-GPU profiler can't see these)

Two R=2 Ray runs (compile ON / OFF) with `build_s`/`train_s`/`round-prof` telemetry exposed the *real* FL
bottlenecks (the single-GPU 244 ms/step was misleading):

1. **Launcher `PYTHONPATH` bug (caught + fixed).** The shared `.venv_v3` editable-installs `fl_v3` from ONE
   worktree; the FL launcher didn't prepend this worktree's `src`, so a bare `flwr run` imported **stale
   sibling code** — the first 52-min R=2 silently ran plain FedAvg (no FedAdam/EMA/snapshots/recipe). Fix:
   `export PYTHONPATH=$REPO/fl_v3/src` in the launcher (propagates to forked Ray actors). Tell-tale: the
   re-run now logs `round-prof`, saves the server-EMA + snapshots, and emits `build_s`/`train_s`.

2. **`torch.compile` is NET-NEGATIVE in FL — disabled.** The model is rebuilt per (client, round), so compile
   re-traces every client: `train_s` = **262 s** (compile) vs **97 s** (no-compile) — the per-client compile
   cost is not amortized over a ~281-step epoch. It also **breaks the update-vector key contract** (compile
   adds an `_orig_mod.` prefix ⇒ "reply key order differs from global"). → `compile-backbone=false`,
   `det-compile-bev=false`. (The 1.45× single-GPU win needs a persistent **compiled** model on the actor —
   a future optimization, not worth the risk for the first baseline.)

3. **Aggregation was 169 s/round = the gradient-space DEFENSE metrics.** `compute_gradient_space_metrics`
   (n×n pairwise cosine + top-k energy over the 33M-param flattened update, ×25 clients) is O(n²·d) and
   USELESS for the clean baseline. Gated behind `log-gradient-metrics` (default true; clean baseline = false).
   → ~150 s/round saved. The cheap per-client update **norms** still log (the divergent-client signal).

4. **`_load_info` re-unpickled the 580 MB info-cache PER CLIENT.** Memoized on the Task instance (persists per
   Ray-actor process) → each actor reads the pickle once (part of the ~9 s `build_s`).

**Measured (compile-OFF R=2, before the agg fix):** round ≈ **929 s** = ~760 s client-training (7 waves ×
~104 s: build 9 s + train ~95 s) + **169 s aggregation**. With the agg-metrics gated off + memoized loader:
**~11 min/round ⇒ R=15 ≈ ~3 h**. Trains-clean confirmed (loss dropping, no NaN/OOM). R=15 submitted (job
6778297) with the optimized config; rounds 1–2 monitored before letting it run.
