# Q1 runtime decomposition — the WHY per stage + speedup options + community context

> Grounded in our code (`view_transform.py`, `losses.py`) + the measured A40 profile (`A_profiling_report.md`)
> + a community check (BEVFusion / BEVDet / BEVPoolv2). All speedups tagged by determinism status:
> **[bit-identical]** = same math, reproduces the reference checksum exactly (safest); **[regime]** = a
> precision change (needs the D13 seed-robustness check); **[config]** = changes the model (new reference);
> **[BANNED]** = breaks bit-determinism.

## The measured per-step breakdown (FP32, batch-16, A40) + the cause of each

| Stage | %step | Why it costs what it does (from the code) |
|---|---|---|
| camera_backbone | 30.5% | Frozen Swin-T ViT forward over 6 cams × batch — pure matmul/attention. TF32 1.33×. |
| **LSS view_transform** | **31.2%** | (a) materializes a **~1.3 GB** lifted tensor `[BN,Cc,D,fH,fW]=[96,80,59,16,44]≈319 M floats` (the depth⊗context outer product); (b) **argsort + cumsum over ~4 M frustum points** — the *deterministic* substitute for the banned atomic `bev_pool`. Memory-bandwidth bound → TF32 ≈ 1.00×. |
| **loss (CenterPointLoss)** | **15.9%** | NOT compute — `build_targets` **loops over every GT box in Python calling `.item()`** (col, row, radius, class) → a **CPU↔GPU sync per box**, hundreds per batch. The focal math over `[B,10,128,128]` is cheap; the syncs are the cost. TF32 ≈ 1.00×. |
| backward | 12.2% | Follows the forward graph; dominated by the view-transform's outer-product + cumsum backward (gather). TF32 1.23×. |
| dataloader/preprocess/neck/lidar/fusion/bev_neck/head | <10% total | small. |

## Q: can we speed up the LSS view-transform?  YES — three determinism-safe levers

1. **[bit-identical] Don't materialize the full frustum feature — multiply only at valid points.**
   Today: `lifted = depth_prob ⊗ context` builds the full `[96,80,59,16,44]` (~1.3 GB), THEN reshapes,
   THEN keeps only valid points (`x_pt = feats_pt[m]`). Reorder: compute the valid mask FIRST, gather
   `depth_prob`+`context` at the ~1–2 M *surviving* points, multiply there → `[P_valid, Cc]`. This is
   **~150–300× less memory traffic** for the lift and is **byte-identical** (multiplying `a[i]*b[i]` for
   valid `i` gives the same bits whether or not you also computed the discarded ones). This is exactly
   **BEVPoolv2's headline idea** — *"omitting the calculation and preprocessing of the large frustum
   feature"* (15× over the first bev_pool) — and the determinism-safe part of it is free to us.
2. **[bit-identical] Cache the geometry (ranks / valid / sort order / img2lidar) per sample.**
   These depend ONLY on `(sample, calibration)` — fixed across epochs/rounds, independent of the learned
   features — so the per-forward **argsort + frustum→lidar projection** is recomputed wastefully every
   step. Precompute once per sample → cache (int64 indices, small vs feature caching) → skip the sort.
   This is literally **bev_pool's "precomputation"** half (the determinism-safe half; the *atomic
   interval-reduction* half is what we ban). Needs the standard cache-vs-live bit-identity gate.
3. **[config] Fewer depth bins.** `D=59` (1–60 m @ 1 m) drives `P`. Coarser depth (e.g. step 2 m → D≈30)
   ~halves `P` → ~2× the view-transform — but changes the model (depth resolution) → a new reference.
4. **[BANNED] the atomic `bev_pool` CUDA kernel itself** — its 40× comes from **atomicAdd**
   (non-associative float sum → non-deterministic; confirmed). This is the one we cannot use; it is *why*
   our view-transform is structurally heavier than the community's.

## Q: loss + backward — only a better GPU?  NO — both have non-hardware headroom

- **loss (16%) is an algorithmic bug, not a compute wall.** The per-GT `.item()` syncs serialize the
  GPU. Two **[bit-identical]** fixes: (a) **vectorize `build_targets`** — compute col/row/radius/reg for
  ALL GTs as tensors, no `.item()`; the Gaussian max-overlay is order-independent (max is commutative →
  same bytes), so a loop-free overlay reproduces the heatmap exactly; (b) **precompute targets per
  sample** (feature-independent, like the geometry cache) → skip `build_targets` at train time. Either
  removes most of the 16% with **no precision change**.
- **backward (12%) shrinks when the forward shrinks** — it is the backward of the same view-transform
  materialization + cumsum; levers 1–2 above cut it too. Plus TF32 already gives 1.23×, and an H200
  (~3–4× the A40's memory bandwidth) helps the memory-bound parts. So: **algorithmic AND hardware**, not
  hardware-only.

## Community context — they hit the SAME bottleneck; their fix is (mostly) banned for us

- **BEVFusion (MIT, ICRA'23):** *"BEV pooling alone takes >80% of runtime"* — the LSS view-transform was
  THE bottleneck for them too. Their headline engineering contribution: a custom CUDA **bev_pool** kernel
  (**precomputation + interval reduction**) → **>40× speedup**, ~1.9× lower total compute.
- **BEVPoolv2 (BEVDet):** further **15×** by **not materializing the large frustum feature** → 0.82 ms at
  640×1600. (= our lever 1.)
- **Determinism tension:** bev_pool's speed is **atomicAdd**-based → non-deterministic by construction
  (float non-associativity) → **banned** under our "bit-determinism is sacred" rule. So the community's
  *fast* path is off-limits; the determinism-safe **ideas** (precompute geometry, avoid frustum
  materialization) are available and are levers 1–2.
- **Training setup they use:** **8 GPUs × batch 4 (8xb4), 20–24 epochs, FP16/AMP** (`--amp`), A100-class
  (4–8× A100 / DGX-A100). vs us: **4× A40, TF32 (deterministic), 15 rounds**, no custom CUDA op.

## So is our 3.9 h "acceptable"? Reckoning + the path down

3.9 h (15 rounds, 4× A40, deterministic, no bev_pool, with the eval already gated) is **reasonable given
the constraints** — but there is clear, determinism-safe headroom we have NOT yet taken:
- lever 1 (no frustum materialization) + lever 2 (geometry cache) attack the 31% view-transform;
- the loss `.item()` fix attacks the 16%; backward shrinks with them.
- Plausible: cut view-transform+loss by ~half → step ~1.9 s → ~1.4 s, i.e. another **~1.3–1.5×** on top
  of the eval-gating win, all **bit-identical** (no accuracy risk, reproduces the reference checksum).
- The community's remaining edge (FP16, bev_pool atomics, 8× A100) is precisely the part we trade away
  for determinism + the A40 pin; the big precision/bandwidth jump comes "for free" at the GH200/Hopper
  migration (full-rate TF32 + ~3–4× bandwidth).

**Recommendation:** these are **post-diagnostics build tasks** (each needs a cache-vs-live / null
bit-identity gate before feeding science), NOT part of D14's current diagnostics. Lever 1 (mask-then-
multiply) + the loss `.item()` fix are the highest ROI and are bit-identical refactors — cheapest to land.

## Sources
- BEVFusion: <https://arxiv.org/pdf/2205.13542> · repo <https://github.com/mit-han-lab/bevfusion>
- BEVPoolv2: <https://arxiv.org/abs/2211.17111>
- mmdet3d BEVFusion (8xb4, 20e, `--amp`): <https://github.com/open-mmlab/mmdetection3d/blob/main/projects/BEVFusion/README.md>
- atomicAdd non-determinism (float non-associativity): <https://arxiv.org/pdf/2408.05148>
