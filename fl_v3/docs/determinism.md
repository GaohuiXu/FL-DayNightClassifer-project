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

## Banned ops (and why `strict` mode is NOT a sufficient detector on torch 2.7)

These are FORBIDDEN in the AD model (T2) and everywhere else:

- **atomic scatter** / `scatter_add` / `index_add` / `index_put(..., accumulate=True)` on CUDA
  (voxelization, pillar scatter) — use a **collision-free `index_copy_`/`index_put_(accumulate=False)`**
  dense scatter + `torch.max` (PointPillars), or the `cumsum_trick` (LSS).
- **`grid_sample` backward** — avoid in the LSS camera→BEV path; use the `cumsum_trick` splat.
- **non-stable `sort` / `argsort`** — always `stable=True`; **`torch.topk` has NO `stable` kwarg** —
  use a max-pool-mask + monotone-tiebreak composite or `sort(stable=True)` + slice (CenterPoint decode).
- **`AdaptiveAvgPool2d` / `AdaptiveMaxPool2d`** in any trainable module — their CUDA *backward* has no
  deterministic kernel and RAISES under strict mode. Use fixed-kernel pooling.
- **flash-attention / non-deterministic SDPA** — Swin-T uses manual fp32 math attention, no
  `scaled_dot_product_attention`; any future SDPA module wraps in `sdpa_kernel(SDPBackend.MATH)`.
- **`canvas[:, idx] = src` advanced-indexing assignment on CUDA** — silently **no-ops** under
  deterministic mode (PyTorch #76176); use an explicit `index_copy_`/`index_put_`.

> **IMPORTANT (verified empirically on torch 2.7.1 / CUDA 12.6, workflow `wf_f35d6cff-9be`):**
> `enforce_determinism(strict=True)` makes **only `grid_sample` backward (and adaptive-pool backward)
> RAISE.** `scatter_add` / `index_add` / `index_put(accumulate=True)` / non-stable `topk`/`sort` now
> have **registered deterministic CUDA kernels and do NOT raise.** So strict mode is **necessary but not
> sufficient** — a stray `scatter_add` passes silently, and **same-seed-twice on ONE GPU is bit-identical
> even with it present** (the drift only surfaces cross-architecture: T4 ≠ A40 ≠ ARM H200). We still ban
> these because (a) bit-identity of the deterministic-scatter path is **not guaranteed across the ARM
> rebuild**, and (b) the model must have **zero summation-order/atomic-ordering dependence by
> construction**. **Enforcement is therefore by (1) a static AST/grep ban test over the model package,
> (2) a permutation-invariance test (permuted input order → byte-identical output), and (3) the GPU
> guards (#76176 index-copy, float-CUDA-cumsum) — NOT by the runtime `strict` raise alone.**

`enforce_determinism(strict=True)` is still the default (`warn_only=False`; fl_v2 used `warn_only=True`)
— it catches the ops that *do* raise and forces `CUBLAS_WORKSPACE_CONFIG` + `cudnn.deterministic`. Flip
`determinism-strict=false` only for the deliberate bring-up of an op being made deterministic.

## What T0 proves

- TinyMLP trained twice at the same seed → identical weights (`torch.equal`).
- An in-process FL round run twice → identical aggregated-state SHA-256 checksum
  AND identical eval, across every defense in the suite.
- FLAME's seeded Gaussian noise is byte-reproducible (parity fixture matches the
  oracle's noisy aggregate exactly).

(See `fl_v3/tests/test_determinism_*.py` and `test_fl_round_smoke.py`.)

## What T2 adds (the AD model — §0 enforcement made concrete)

Because `strict` mode is necessary-but-not-sufficient (above), the BEVFusion-class model
is held to the banned-op contract by **four** dedicated artifacts (NOT the runtime raise):

- **Static AST ban** over `models/fusion/**` — `tests/test_model_determinism.py::
  test_static_ban_over_models_fusion` parses every model source file and fails on any
  `scatter_add`/`index_add`/`index_put(accumulate=True)`/`grid_sample`/`topk`/non-stable
  `sort`/`argsort`. This is the real banned-op gate.
- **Permutation-invariance** — `test_splat_permutation_invariant` +
  `test_pillar_scatter_permutation_invariant`: a permuted input order yields a
  **byte-identical** splat / pillar-scatter (a `scatter_add` would drift; the
  `cumsum_trick` with a canonical `(rank, geom_id)` sort and the `torch.max` pillar pool
  pass). `test_decode_tie_break_is_canonical` pins the decode tie order.
- **GPU op guards** (`tests/test_model_task.py`) — the #76176 `index_copy_` scatter
  yields the correct non-zero cell under strict mode, and float-CUDA `cumsum` does not
  raise and is `torch.equal` fwd+bwd. *(Re-run these on the ARM/H200 rebuild.)*
- **The A40-pinned bit-identity gate** — `scripts/det_gate_a40.py` (via
  `scripts/run_det_gate_a40.sh`): asserts the device is an A40, runs two same-seed
  K-step trainings to `torch.equal` on every parameter, and commits the per-param
  SHA-256 weight checksum into `collab/T2/SPEC.md`. **Errors LOUD (exit 2) on
  no-CUDA / a non-A40 device — a CPU / login-node (Tesla T4) pass is a FALSE PASS.**
