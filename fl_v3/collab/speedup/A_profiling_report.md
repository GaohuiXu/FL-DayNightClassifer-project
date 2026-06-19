# A — Per-stage runtime profiling report (Cycle-04 D14 Phase-1)

> Measured on a real A40 (alvis7-01, cc 8.6, job 6767120), the headline config
> (`t4_reference.json`: nuScenes trainval log-group, **frozen Swin-T**, batch-16, client 0),
> 20 timed steps after 5 warm-up, both numeric regimes. Raw: `profile_stages_a40.json`.
> Instrumentation is determinism-neutral (proven by `tests/test_profiling_neutral.py`:
> profiling-on == profiling-off, byte-identical params + RNG). **This settles D11/D12's
> *inferred* "80–90% backbone" with a measurement — and the measurement is very different.**

## Headline: the per-step breakdown is NOT backbone-dominated

| Stage | FP32 ms/step | share of step | TF32 ms/step | TF32 speedup |
|---|---:|---:|---:|---:|
| dataloader_wait | 66.9 | 3.5% | 66.9 | 1.00× |
| forward.preprocess | 30.3 | 1.6% | 30.3 | 1.00× |
| **forward.camera_backbone** | **589.8** | **30.5%** | 442.2 | **1.33×** |
| forward.camera_neck | 19.5 | 1.0% | 17.5 | 1.11× |
| **forward.view_transform (LSS)** | **603.4** | **31.2%** | 601.8 | **1.00×** |
| forward.lidar_encoder | 18.3 | 0.9% | 18.3 | 1.00× |
| forward.fusion | 34.6 | 1.8% | 28.4 | 1.22× |
| forward.bev_neck | 19.3 | 1.0% | 16.2 | 1.19× |
| forward.head | 4.6 | 0.2% | 4.2 | 1.07× |
| forward_total | 1320.0 | 68.4% | 1159.1 | 1.14× |
| loss (CenterPointLoss) | 307.5 | 15.9% | 309.3 | 0.99× |
| backward | 235.4 | 12.2% | 191.1 | 1.23× |
| optimizer_step | 1.3 | 0.1% | 1.3 | 1.00× |
| **mean step** | **1931.1** | 100% | **1727.8** | **1.12×** |

Peak GPU mem ≈ 6.8 GB (1 client, batch-16 — far under the A40's 46 GB; the run is compute-bound,
not memory-bound). Sampled GPU util ≈ 76–77% (the memory-bound LSS/loss stages leave the SMs partly
idle — consistent with their TF32-immunity).

## The four findings that change the speedup strategy

1. **The frozen camera backbone is ~30% of the step, NOT 80–90%.** D11/D12 inferred 80–90% from a
   single forward-time estimate; the measurement puts `camera_backbone` at **30.5%** of the step
   (and 44.7% of the *forward*). **Feature-caching the frozen backbone therefore has a hard ceiling
   of ~1.4× end-to-end** (remove 590 ms from a 1931 ms step → ~1341 ms), not the "~3–5×" the
   storage request was sized for. **This independently vindicates D13/D14's de-prioritization of
   caching** — the lever was never worth 1.66 TB of storage + a cache det-gate.

2. **The LSS view-transform is the co-equal bottleneck (31.2%) and is TF32-immune (1.00×).** This is
   the surprise. The depth-splat (`DepthLSSTransform`: outer-product context×depth, cumsum-trick
   scatter to BEV) is **memory-bandwidth-bound, not matmul-bound**, so neither TF32 nor backbone
   caching touches it. Any future speedup plan must target the view-transform, not just the backbone.

3. **TF32's A40 win is ~1.12× end-to-end — even below D13's ~1.3× estimate.** The gain is real but
   confined to the matmul/conv-heavy stages: backbone 1.33×, backward 1.23×, fusion 1.22×, bev_neck
   1.19×. The two largest stages (view_transform 31%, loss 16%) get ~0%. So on the A40 — the
   worst-case TF32 card (GA102 halved TF32 path) — TF32 buys ~12%. **This is a MEASURED contingency
   trigger for D14**: the A40 TF32 win is modest, so bf16-on-frozen-backbone is worth considering
   (see §Recommendation). The picture flips on Hopper/GH200 (full-rate TF32), where the backbone +
   matmul share would see the 7–15× peak ratio → a much larger end-to-end win at the migration.

4. **The loss is a surprising 16% (307 ms) and TF32-immune.** `CenterPointLoss` (BEV-grid target
   assignment + Gaussian splatting of GT centers + focal/reg on a dense [C,H,W] grid) is heavier than
   expected and memory-bound. A deterministic vectorization of the target assignment is a real,
   caching-free, regime-independent lever (~16% headroom) worth a follow-up.

## Answers this gives to the D14 questions

- **Q1 (where is runtime spent):** forward 68% of the step, split **~30% camera_backbone + ~31% LSS
  view-transform** (co-equal), then loss ~16%, backward ~12%, dataloader ~4%, everything else <2%.
  The "80–90% backbone" premise is **false**.
- **Q3 (does TF32 help under a reproducible regime):** yes, deterministically (A40 det-gate PASS),
  but only **~1.12× end-to-end on the A40** — concentrated in the backbone/backward/conv stages; the
  memory-bound LSS + loss see ~0%. Reproducible (no mixing regimes), determinism-neutral.

## Caveats (honest scope)

- Per-stage times are `synchronize()`-bracketed for clean attribution; this serializes stages that
  could otherwise overlap, so the **summed** step (1931 ms) is an upper bound on an un-instrumented
  step — but the **relative** breakdown (the deliverable) is sound and the cross-regime comparison is
  apples-to-apples (same instrumentation both regimes).
- Single client (client 0), single A40, batch-16. Shard size varies per client but per-step cost does
  not; the breakdown is representative of the headline regime.
- `dataloader_wait` is low (3.5%) with `num-workers=4` prefetch — the run is genuinely GPU-compute-
  bound, confirming D9/D11 that Flower/Ray overhead is not the bottleneck.

---

# Q1 — RELAXED regime (D15), measured on the current codebase (NEW; does not replace the strict Q1 above)

> After the D15 waves (scatter_add splat + mask-then-lift + bf16-AMP + free scheduling), profiled in the
> **FL-path regime**: `determinism-level=relaxed`, bf16-AMP, **no torch.compile** (FL doesn't compile —
> recompile/round). Job 6767475, A40, batch-16, 20 steps. `precision_state`: determinism_level=relaxed,
> cudnn_benchmark=True, tf32_engaged=False (clean bf16-AMP: bf16 heavy ops + fp32 stability/loss/splat).

| stage | strict tf32 (old Q1) | **relaxed bf16 (D15, FL regime)** | share of relaxed step | factor |
|---|---:|---:|---:|---:|
| camera_backbone | 442 ms | **263.8 ms** | 46.1% | 1.7× (bf16; +compile → 147 ms outside FL) |
| LSS view_transform | 602 ms | **13.9 ms** | 2.4% | **42.8×** (scatter_add + mask-then-lift) |
| backward | 191 ms | 112.1 ms | 19.6% | 1.7× |
| loss | 91 ms | 83.5 ms | 14.6% | 1.04× |
| fusion | 35 ms | 27.7 ms | 4.8% | 1.3× |
| camera_neck | 19 ms | 17.8 ms | 3.1% | — |
| lidar_encoder | 18 ms | 17.0 ms | 3.0% | — |
| bev_neck | 19 ms | 15.6 ms | 2.7% | — |
| preprocess | 30 ms | 13.9 ms | 2.4% | 2.2× |
| head | 5 ms | 4.4 ms | 0.8% | — |
| dataloader_wait | 67 ms | 0.3 ms | 0.1% | (prefetch hides it) |
| optimizer_step | 1 ms | 1.4 ms | 0.3% | — |
| **mean step** | **1443 ms** | **572 ms** | 100% | **~2.5×** (FL, no compile) |

**Reading the new Q1:** the bottleneck has FLIPPED. The LSS view-transform (was a 31–35% co-bottleneck)
is now **2.4%** — the scatter_add rewrite collapsed it. The **frozen camera backbone is now the dominant
stage at 46%** (bf16-only in FL; `torch.compile` would take it to ~147 ms but is excluded from the FL
per-round path). So the *next* speedup target, if needed, is the backbone (compile-in-FL despite the
recompile cost, or the frozen-feature cache). Step **572 ms** (FL regime, no compile) → ~2.5× over strict
tf32; **460 ms** with compile (centralized/long runs) → ~3.1×.

**FL per-round wall-clock (MEASURED, 3-round relaxed-bf16 FL, 4×A40):** strict tf32 15.6 min/round →
relaxed bf16 7.0 (1/GPU) → 6.4 (+compile) → **5.6 min/round (+compile +num-gpus=0.5, 2 clients/GPU) =
~2.76×**. GPU memory has room for ~5–6 clients/GPU but compute saturates at ~2/GPU (the 2nd client added
only +14%; 3/GPU expected to thrash). Full ladder + jobs in `D15_D16_decision_for_orchestrator.md`.

## Recommendation for the speedup strategy (feeds the §Synthesis go/no-go)

1. **Adopt TF32 on A40** (det-gate PASS): a free ~1.12×, regime-logged, determinism-safe. Worth it.
2. **Drop feature-caching entirely on Alvis** (confirms D13/D14): ceiling ~1.4×, needs 1.66 TB +
   a cache det-gate — not worth it given the backbone is only 30%.
3. **The real per-cell levers are now (a) the LSS view-transform (31%, memory-bound) and (b) the
   CenterPointLoss (16%)** — both caching-free and regime-independent. Defer optimization unless the
   diagnostics show the matrix needs it; the dominant matrix lever stays D9 across-cell fan-out.
4. **bf16-on-frozen-backbone** (the D14 measured contingency) would add at most the backbone's share
   (~30% → maybe ~1.5× combined with TF32) — marginal on A40; revisit only if a single heavy run
   becomes the critical path. The big precision win is deferred to the GH200/Hopper re-baseline.
