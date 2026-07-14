# Arrhenius Mini-Only Speedup Log

Date: 2026-07-01

Scope: engineering-only mini-data optimization on Arrhenius GH200. These runs
validate correctness, stability, and profiling behavior for the
`voxel_fp16_main` path. They are not mAP/NDS/ASR/defense evidence.

> **Historical architecture boundary (2026-07-14 rebaseline).** Jobs `211502`
> and `211722` predate the current S03-S06/S07 six-task model. Their stable path
> ran outer FP16 AMP while keeping voxelization/VFE/spconv in FP32. They do not
> prove stability for the current `second_075` production resolver, which at the
> S07 anchor enables sparse-conv FP16 when `precision=fp16`. Current D1 Job
> `389356` found direct nonfinite L-S075/F-U gradients even at scale 1. Use this
> document as old-path performance evidence and S08 input, not as a current
> precision contract.
>
> Job `211502` needs one additional terminal-status qualifier: its comparable
> `voxel_fp16_main` profile cell completed and produced the baseline numbers below,
> but the overall multi-cell Slurm job later ended `OUT_OF_MEMORY` in the separate
> experimental sparse-conv-FP16 cell. The table is retained as completed main-cell
> evidence; the whole job must not be labelled an unqualified PASS.

## Runtime Policy

- Historical main engineering path: `det-lidar-encoder=voxel`,
  `precision=fp16`, AMP + `GradScaler(init_scale=512)`.
- Stable sparse path: VFE/voxelization/spconv run in fp32 under fp16 AMP.
- Experimental sparse-conv fp16 path: opt-in only via
  `det-sparse-conv-fp16=true`; not the default because bs1 real-data smoke can
  hit a spconv tuner failure.
- Direct sparse bf16 remains unsupported and should fail explicitly.

## Comparable Profile Baseline

The most comparable before/after numbers use the same canonical cell:

- Cell: `voxel_fp16_main`
- Camera backbone: `resnet18`, frozen, no pretrained weights
- Batch size: 16
- Data workers: 8
- Mini data only
- Baseline profile: Slurm job `211502`
- Final profile: Slurm job `211722`

The final profile also fixes mini iterator-reset accounting by selecting enough
tokens for the measured window:

- `num_tokens=256`
- `warmup_iters=4`
- `profile_iters=8`
- tokens required without iterator reset: 192
- measured iterator resets: 0

## End-To-End Result

| Metric | Before (`211502`) | After (`211722`) | Change |
| --- | ---: | ---: | ---: |
| Compute step latency | 417.68 ms | 329.98 ms | 21.0% lower, 1.27x faster |
| Compute samples/sec | 38.31 | 48.49 | 26.6% higher |
| End-to-end step latency | 558.06 ms | 330.21 ms | 40.8% lower, 1.69x faster |
| End-to-end samples/sec | 28.67 | 48.45 | 69.0% higher |
| Peak allocated GPU memory | 11630 MiB | 11040 MiB | 5.1% lower |

Important interpretation: the compute-step improvement is the cleaner model-side
speedup number. The end-to-end improvement also includes removing a mini-profile
measurement artifact where the DataLoader iterator reset inside the measured
window.

## Stage Breakdown

| Stage | Before (`211502`) | After (`211722`) | Change |
| --- | ---: | ---: | ---: |
| Forward total | 116.04 ms | 103.65 ms | 10.7% lower |
| Sparse LiDAR encoder total | 83.59 ms | 71.56 ms | 14.4% lower |
| Voxelize + VFE | 29.80 ms | 20.07 ms | 32.7% lower |
| spconv backbone | 49.67 ms | 47.31 ms | 4.8% lower |
| Backward | 273.42 ms | 202.42 ms | 26.0% lower |
| Loss | 21.14 ms | 18.00 ms | 14.9% lower |
| Data fetch | 140.38 ms | 0.23 ms | reset artifact removed |

## Optimization Steps

### 1. Opt-In Sparse Conv fp16 Path

Added `det-sparse-conv-fp16` as an explicit experimental switch. VFE and
voxelization stay fp32. Sparse conv fp16 can improve the bs16 profiling path,
but it is not stable enough to be the default because a bs1 real-data smoke hit
a spconv tuner failure. The stable main path disables outer autocast around
spconv so fp16 AMP does not silently change sparse conv execution.

Correctness checks:

- Synthetic sparse LiDAR smoke for fp32 and fp16 AMP paths.
- Mini one-batch eval/train.
- Tiny-overfit finite and decreasing.
- GradScaler scale stable at 512 with no skips on the stable path.

### 2. Sparse Batch Point Grouping

Replaced per-batch boolean filtering with a batch-contiguous fast path plus a
stable-sort fallback for interleaved batch indices.

Effect:

- Compute latency: 417.68 ms to 407.63 ms.
- Voxelize/VFE stage: 29.80 ms to 27.55 ms.
- Sparse metadata confirms real mini batches use contiguous grouping.

### 3. Valid-Only VFE

Changed VFE to run point MLP and pooling only over valid point slots, not padded
slots. This preserves voxel max-pool semantics while removing padded activation
work from forward and backward.

Effect:

- Compute latency: 407.63 ms to 353.76 ms.
- Voxelize/VFE stage: 27.55 ms to 25.76 ms.
- Backward: 271.41 ms to 207.87 ms.
- Peak allocated memory: 11630 MiB to 11075 MiB.

### 4. Low-Frequency Training Telemetry

Added `train-telemetry-interval`. The default `0` avoids per-step loss-term
`.item()` synchronization and grad-norm synchronization in normal training.

Centralized mini benchmark:

- `train-telemetry-interval=1`: 46.03 s
- `train-telemetry-interval=0`: 31.13 s
- Wall-clock improvement: 32.4%, 1.48x faster

This benchmark is a training-loop engineering result only. It is not a model
quality comparison.

### 5. Data/Profile Window Fix

Updated the Arrhenius mini profiler defaults to a GH200-oriented engineering
profile:

- `batch-size=16`
- `num-workers=8`
- `num-tokens=256`
- `warmup-iters=4`
- `profile-iters=8`
- persistent workers and pin memory enabled

Added explicit iterator-reset telemetry:

- per-step `data_iterator_reset`
- aggregate measured reset counts
- reset-free end-to-end means
- data-window metadata in the manifest

Final profile `211722` had zero measured iterator resets. Data fetch dropped to
0.23 ms in the measured window, showing that the earlier 140-151 ms data-fetch
mean was mostly a mini profiling artifact rather than the steady-state full
training input cost.

## Current Bottleneck Assessment

After these optimizations, the main measured costs in `211722` are:

- backward: 202.42 ms
- forward total: 103.65 ms
- sparse LiDAR encoder total: 71.56 ms
- spconv backbone: 47.31 ms
- voxelize/VFE: 20.07 ms
- loss: 18.00 ms

The next performance work should focus on:

- sparse LiDAR branch backward cost;
- spconv backbone algorithm/layout behavior;
- voxel/VFE allocation and scatter behavior;
- larger batch sizes to use more of GH200 memory and improve GPU saturation;
- lower-overhead profiling for long runs, because synchronized stage profiling
  depresses utilization.

Do not use mini data to choose a scientific best config or to claim final model
quality.
