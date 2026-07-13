# Clean determinism and reproducibility contract

This document describes the active clean training and aggregation contract. The
runtime and validated Arrhenius stack are defined in `docs/env.md`.

## Precision regimes

- `fp32` is the strict development and regression regime. It disables TF32,
  enables deterministic cuDNN behavior, and asks PyTorch to reject unsupported
  nondeterministic operations.
- `fp16` uses CUDA autocast and `GradScaler` for supported Arrhenius sparse
  training. It is a scientific runtime regime, not a byte-identity promise.
- Direct sparse `bf16` is unsupported by the validated cumm/spconv stack and is
  rejected by the runtime helpers.

Do not mix precision regimes within a comparison. Record the resolved precision,
hardware, software stack, seeds, data/split manifest, and checkpoint identity for
every result. Strict byte identity is a regression tool; scientific claims require
the declared multi-seed protocol.

## Randomness and client identity

- `seed_everything(seed)` seeds Python, NumPy, PyTorch, and all CUDA devices.
- `derive_seed(run_seed, client_id, server_round)` derives a stable per-client,
  per-round seed with SHA-256.
- DataLoader workers receive deterministic derived seeds.
- Federated clients are identified by partition ID. Driver-local node IDs never
  define sampling or aggregation order.
- `select_partition_ids` deterministically samples clients from the run seed,
  round, fraction, participant floor, and train/evaluation salt.

## Clean aggregation

The clean Flower strategy and in-process runner share
`strategy.aggregation_core.fp32_weighted_average`:

1. valid replies are sorted by partition ID;
2. every update is converted to FP32;
3. each update is weighted by its declared number of examples;
4. accumulation follows the fixed sorted order;
5. server optimization is applied only after the clean weighted average.

There is one production strategy, `CleanFedAvgStrategy`. The local runner exposes
only `run_clean_round` and `run_clean_rounds`; neither API has an aggregation-mode
selector. Server optimizer, EMA, checkpoint, resume, and trainable-only state are
orthogonal clean runtime state and are preserved explicitly.

## Model and data checks

The camera/LiDAR model uses one shared metric-to-BEV convention. Tests anchor that
mapping to real geometry and cover stable decode order, permutation invariance,
sparse LiDAR edge cases, trainable-state layout, and full-checkpoint loading.

The nuScenes data path binds cache identity to resolved dataset, split, sweep
depth, and source provenance. Deterministic partitioning owns complete log/scene
groups and records the derived client count and partition seed.

## Verification scope

Fast local checks cover:

- same-seed clean single- and multi-round checksums;
- deterministic client sampling at partial participation;
- FP32 number-of-examples weighted parity with Flower;
- server optimizer and trainable-state integration;
- resolved-config, precision, checkpoint, and resume contracts.

GPU, real-data, and multi-worker checks must run only in the validated target
environment under an explicitly approved execution request. A local smoke or mini
dataset result is engineering evidence, never scientific evidence.
