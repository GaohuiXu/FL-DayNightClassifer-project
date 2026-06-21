# MCR Phase-1 — UNFROZEN model per-component profile (single A100-40GB, bf16)

Job 6769823 (A100-SXM4-40GB, cc 8.0). Config: `p1_unfrozen.json` (Swin trained, batch 16, bf16),
4 dataloader workers, 20 steps + 6 warmup. Pooled centralized set = **28,130 keyframes** (25 log-group
clients) → **1759 steps/epoch** at batch 16. Raw: `p1_profile_a100.json`.

## Headline: the unfrozen model is COMPUTE-BOUND (the frozen 12%-util problem is gone)

Three independent signals, all consistent (the nvidia-smi util sampler returned no samples — a bug to
fix — but it isn't needed for the verdict):
- **Per-sample time is FLAT across batch** (49.6 / 49.4 / 49.1 ms at batch 16/24/32) → not launch-bound;
  bigger batches do not raise throughput.
- **profiler self-CUDA (8533 ms) > self-CPU (5000 ms)** → GPU-time dominates → compute-bound.
- **dataloader wait = 0.36 ms/step (0.0%)** → 4 workers fully hide data; NOT dataloader-bound.

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

## Recommendation

1. **Optimize the single-GPU step FIRST** (channels_last → drop-ckpt → SDPA → compile), re-profiling after
   each via the same short A100 job. No new infra; biggest leverage; needed before DDP regardless.
2. **Then DDP** for the node: worth adding for the final converged + ≥3-seed runs (cuts them to ~1–2 h);
   for the lever *search*, optimized single-GPU (5 ep ≈ 1 h, under the 3 h gate) already gives fast feedback.
3. Fix the util sampler (use pynvml / `torch.cuda.utilization`) so the next profile reports util% directly.
