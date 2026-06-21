# MCR Phase-1 — UNFROZEN model per-component profile (single A100-40GB, bf16)

Job 6769823 (A100-SXM4-40GB, cc 8.0). Config: `p1_unfrozen.json` (Swin trained, batch 16, bf16),
4 dataloader workers, 20 steps + 6 warmup. Pooled centralized set = **28,130 keyframes** (25 log-group
clients) → **1759 steps/epoch** at batch 16. Raw: `p1_profile_a100.json`.

## Headline: compute-bound, but only ~70% mean util — real headroom

GPU util (job 6769833, fixed streaming-nvidia-smi sampler) **during the timed training steps**:
**util_mean ≈ 70% but median 91–96%, p90/max 100%** (batch 16/24/32). Whole-run bash sampler = 45.5%
mean, but that is diluted by the 5 model-builds/setup between regimes — the steady-state TRAINING number
is **~70% mean / ~93% median**. So: the GPU is near-saturated MOST of the time but the mean sits at 70% →
periodic dips (launch gaps + the 82 ms loss + sync). **NOT fully used — ~30% mean-vs-median headroom that
`torch.compile` (kernel fusion, fewer launches) should fill.** Compute-bound confirmed by three signals:
- **Per-sample time FLAT across batch** (49.6 / 49.4 / 49.1 ms at batch 16/24/32) → not launch-bound.
- **profiler self-CUDA (8533 ms) > self-CPU (4990 ms)** → GPU-time dominates.
- **dataloader wait = 0.26 ms/step (0.0%)** → 4 workers fully hide data; NOT data-bound.

## Per-step teardown (batch 16, ckpt ON, 794 ms/step)

| stage | ms | % of step |
|---|---:|---:|
| **backward (all)** | **485** | **50.0%** |
| forward total | 199 | 20.5% |
|   — camera_backbone (Swin) | 146 | 15.1% |
|   — fusion / lidar_enc / cam_neck / bev_neck / preproc / view_xform / head | ~52 | ~5% |
| **loss** (CenterPoint target-build + focal + L1) | **82** | **8.4%** |
| optimizer (Adam) | 6 | 0.6% |
| dataloader | 0.36 | 0.0% |

Backward (50%) + Swin forward (15%) dominate — i.e. **the trained backbone is now the hotspot** (exactly
why util rose). Loss at 8.4% is surprisingly large.

## Memory + the ckpt decision (40 GB A100)

| regime | step ms | per-sample | peak GPU |
|---|---:|---:|---:|
| batch 16, **ckpt OFF** | 657 | **41.0 ms** | 33.3 GB |
| batch 16, ckpt ON | 794 | 49.6 ms | 12.0 GB |
| batch 24, ckpt ON | 1185 | 49.4 ms | 17.8 GB |
| batch 32, ckpt ON | 1570 | 49.1 ms | 23.5 GB |

**activation-checkpoint costs ~21% speed** (49.6 vs 41.0 ms/sample) and is **NOT needed at batch 16** —
ckpt-OFF fits in 33 GB of the 40 GB card. So the baseline run should be **batch 16, ckpt OFF (657 ms/step)**.
ckpt only buys headroom for bigger batch/resolution (which, being compute-bound, don't speed wall-clock).

## Optimization targets (from the CUDA-kernel breakdown)

Top kernel by far: **`aten::copy_` = 1157 ms, 18,921 calls** (~3150 copies/step) — wasted memory traffic
from autocast casts + NCHW layout conversions + Swin `.contiguous()`/`roll`. Then `native_layer_norm` +
`layer_norm_backward` (~890 ms, Swin LN), the attention `bmm`/`addmm`/`mm` (~760 ms, Swin attention +
linears), elementwise.

Levers (re-profile after each), ranked by expected win:
1. **`channels_last`** on the conv stack — kill the layout-conversion copies + tensor-core NHWC convs. Cheap.
2. **Drop activation-ckpt at batch 16** — immediate −21% (fits in 33 GB). Free.
3. **SDPA on Swin windowed attention** — fuse the attention `bmm`+softmax (~267 ms) + cut activation mem.
4. **`torch.compile`** (opt-in, eager fallback; validate on A100) — fuse the copy/cast/elementwise chains.
5. fused Adam — optimizer is 0.6%; negligible, skip.
6. investigate the **82 ms loss** (target-building) — 8.4% of the step.

## Wall-clock projection (single A100-40GB, 1759 steps/epoch)

| run | now (657 ms/step, ckpt-off) | after ~30% step optimization | + 4-GPU DDP (~3.5×) |
|---|---:|---:|---:|
| 5 ep (directional) | ~1.6 h | ~1.1 h | ~0.5 h |
| 15 ep | ~4.8 h | ~3.4 h | ~1.0 h |
| 30 ep | ~9.6 h | ~6.7 h | ~2.0 h |

## Optimization sequence — MEASURED (single A100-40GB, batch 16, bf16)

| step config | step ms | speedup vs ckpt-off | vs original | util (post-warmup) | peak GPU |
|---|---:|---:|---:|---:|---:|
| original (ckpt-ON, no opt) | 793 | — | 1.00× | — | 12 GB |
| ckpt OFF | 657 | 1.00× | 1.21× | ~85% | 33.5 GB |
| + torch.compile(backbone) | 446–528* | 1.25–1.46× | — | ~68–81% | 30.5 GB |
| + SDPA (Swin attention) | 598 | 1.10× | — | ~85% | 28.9 GB |
| **+ compile + SDPA** | **415** | **1.59×** | **1.91×** | ~77% | **27.1 GB** |

*compile alone is autotune-noisy run-to-run; **compile+SDPA (415 ms) is consistently the best**. Kernel
proof: the manual `aten::bmm` attention is **gone** under SDPA → fused `fmha_cutlass_bf16` (efficient-attn
fwd 55 ms + bwd 123 ms). channels_last was measured neutral (0.99×) and dropped. SDPA numerically validated
vs torchvision (fp32 max|Δ|=2.2e-07; bf16 rel=5.6e-03). util drops with optimization simply because there
is less work to fill the same launch/sync gaps — **absolute step time is what improved (1.91×)**.

**Locked optimized recipe** (`p1_unfrozen.json` + launcher `COMPILE=1`): unfreeze + LR-groups(0.1) +
grad-clip(35) + **ckpt OFF** + **SDPA** + **compile**. **~12 min/epoch** → 5 ep ≈ 1.0 h, 15 ep ≈ 3.0 h,
30 ep ≈ 6.1 h single-GPU. Residual headroom (not pursued): `aten::copy_` (238 ms, still #1 by count) +
Memcpy HtoD (102 ms, → pinned-mem/non_blocking) — diminishing returns vs compile+SDPA.

## Recommendation

1. **Optimize the single-GPU step FIRST** (channels_last → drop-ckpt → SDPA → compile), re-profiling after
   each via the same short A100 job. No new infra; biggest leverage; needed before DDP regardless.
2. **Then DDP** for the node: worth adding for the final converged + ≥3-seed runs (cuts them to ~1–2 h);
   for the lever *search*, optimized single-GPU (5 ep ≈ 1 h, under the 3 h gate) already gives fast feedback.
3. Fix the util sampler (use pynvml / `torch.cuda.utilization`) so the next profile reports util% directly.
