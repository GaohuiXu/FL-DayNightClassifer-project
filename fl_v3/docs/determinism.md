# fl_v3 determinism contract (bit-determinism is sacred)

Same-seed runs MUST be byte-identical at every gate. This is the load-bearing
property of the platform (T2/T3 depend on it; the null-config and oracle-parity
gates assume it).

## The harness (`fl_v3/src/fl_v3/utils/runtime.py`)

- **`derive_seed(run_seed, client_id, server_round)`** — SHA-256 of the
  colon-joined decimals, first 4 bytes → 32-bit seed. Portable across Python
  builds / `PYTHONHASHSEED`. Byte-identical to the fl_v2 oracle.
- **`seed_everything(seed)`** — seeds `random`, `numpy`, `torch` (+ all CUDA).
- **`seeded_worker_init`** — propagates each DataLoader worker's torch seed to
  numpy + stdlib `random`.
- **`enforce_determinism(strict=True)`** — sets `CUBLAS_WORKSPACE_CONFIG=:4096:8`,
  `cudnn.deterministic=True`, `cudnn.benchmark=False`,
  `torch.use_deterministic_algorithms(True, warn_only=not strict)`.

## How determinism is enforced per scope

- **Server startup / round:** `seed_everything(seed)` + `enforce_determinism()`.
- **Per client / per round:** `seed_everything(derive_seed(seed, client_id, server_round))`
  BEFORE the model build + local training (each Ray actor would otherwise seed
  torch from the OS clock).
- **Aggregation order:** sort replies by the deterministic 0..N-1
  `reply-meta/partition-id`, NEVER the per-driver-random `src_node_id` (the
  residual-ε source fl_v2 fixed). The in-process runner sorts identically.
- **Single Ray actor on the GPU:** `num-gpus=1.0` (concurrent actors diverge at
  round 2 — fl_v2 V4 finding). See `configs/flwr_config.toml`.

## Banned ops (strict mode makes them RAISE)

These have no deterministic implementation (or are atomic-order-dependent) and
are FORBIDDEN in the AD model (T2) and everywhere else:

- **atomic scatter** / `scatter_add` / `index_add` on CUDA (voxelization,
  pillar scatter) — use dense `torch.max` + dense scatter instead (PointPillars).
- **`grid_sample` backward** — avoid in the LSS camera→BEV path; use the
  `cumsum_trick` splat.
- **non-stable `sort` / `topk`** — always `stable=True` / `kind="stable"`.
- **flash-attention** — Swin-T runs fp32 window attention, no flash-attn.

`enforce_determinism(strict=True)` is the default precisely so any of these
RAISE at the call site rather than silently producing run-to-run drift. The
fl_v2 oracle used `warn_only=True`; fl_v3 tightens to strict for the AD build —
flip to `determinism-strict=false` in config only for the deliberate bring-up of
an op being made deterministic.

## What T0 proves

- TinyMLP trained twice at the same seed → identical weights (`torch.equal`).
- An in-process FL round run twice → identical aggregated-state SHA-256 checksum
  AND identical eval, across every defense in the suite.
- FLAME's seeded Gaussian noise is byte-reproducible (parity fixture matches the
  oracle's noisy aggregate exactly).

(See `fl_v3/tests/test_determinism_*.py` and `test_fl_round_smoke.py`.)
