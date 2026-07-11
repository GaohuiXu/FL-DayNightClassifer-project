# S04 RESULTS — SECOND sparse LiDAR engineering validation

## Overall result

Job `335566` is a preserved **FAILED** engineering result: scheduler state
`FAILED`, exit `1:0`, with exactly `10` collected tests, `5 passed`, `5 failed`,
and zero skips. No retry, requeue, resubmission, follow-on, dataset access,
optimizer step, parameter update, profile, metric, or scientific run occurred.

All execution/source/request identity checks and artifact checksum checks passed.
All five failures have one implementation cause: `_SparseResidualBlock` was nested
inside `spconv.SparseSequential`. spconv therefore treated the arbitrary
`nn.Module` as a dense feature module and called it with `input.features`; the
block expected a `SparseConvTensor` and failed on `x.features` with
`AttributeError: 'Tensor' object has no attribute 'features'`.

This failure occurs before the full B=4 case reaches its output, loss, backward,
or memory evidence. No B=4 peak CUDA memory or finite-gradient claim is available.

## Approved request and execution identity

- One-time S00-approved request SHA-256:
  `00aea9398736471b3a68a1e1fade00fb7e639457795109cc8d9ad6971c956b7c`.
- Executed HEAD/tree:
  `49efb05dd341dbfbcc2d373508772e5b214aa726` /
  `d8e3d39ea5e6f2992002794bd41799b489aaf8e9`.
- Branch: `codex/s04-lidar-second`.
- Runtime source SHA-256:
  `4816f0de0a653b667e20a79d20b11862bb56423428c374f88e3a66fb6d6209df`.
- Launcher SHA-256:
  `53b763940175dbeb48fccbb85b6314bb67c48f1c6ca2de59ace384c272971c1e`.
- Output root:
  `/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s04_second_49efb054cb48`.
- Runtime: CPython `3.11.15`, Torch `2.11.0+cu128`, spconv `2.3.8`,
  cumm `0.7.13`, NumPy `1.26.4`, pytest `9.1.1`, aarch64 node `n507`.
- `execution_identity.json` records `synthetic_only=true` and exact job
  `335566`; its request/source/Git identities match the approval.

## Scheduler result

| Field | Value |
|---|---|
| job/name/node | `335566` / `flv3_s04_second` / `n507` |
| state/exit | `FAILED` / `1:0` |
| submit/start/end | `2026-07-11T18:16:38` / `18:16:38` / `18:18:19` |
| elapsed/timelimit | `00:01:41` / `00:20:00` |
| allocation | one GH200, eight CPUs, one node, `mem=11672M` |
| restarts | `0` |
| batch MaxRSS/MaxVMSize | `1216128K` / `18894016K` |
| batch disk read/write | `102.76M` / `0.60M` |
| batch TotalCPU | `00:34.330` |
| elapsed GPU allocation | approximately `0.0281` GPU-hours |

The job name had one submission. No active S04 job remained after failure.

## Test result

JUnit is exact: `tests=10`, `failures=5`, `errors=0`, `skipped=0`. Runtime was
`86.04s` (`5 passed, 5 failed, 1 warning`). Passed cases were the five
CPU/static checks (four contract fixtures plus sparse-precision policy). Failed
cases were:

1. `test_sparse_second_shape_stats_backward_and_reduced_occupancy`;
2. `test_per_sample_caps_extreme_occupancy_and_point_permutation`;
3. `test_empty_sample_batch_isolation_and_batch_permutation`;
4. `test_fp32_and_fp16_sparse_paths_have_finite_outputs_and_gradients`;
5. `test_s04_reference_b4_fp16_forward_backward_memory_bound`.

Each stack reaches `second_sparse_backbone.py:_SparseResidualBlock.forward`, where
`x` is a dense feature `Tensor`; accessing `x.features` raises. The first real
spconv test therefore caught the bug before B=4 completed. The stderr file contains
only the normal module-purge notice; pytest failures are in stdout/pytest log.

## Artifacts and checksums

The launcher's post-test `sha256sum -c` passed for every checksummed artifact even
though pytest failed, preserving the negative result intact.

| Artifact | Bytes | SHA-256 |
|---|---:|---|
| `execution_identity.json` | 819 | `4b57ff440d678d83615c50249ecb1b42982eb38897b30eee64590042936c5659` |
| `runtime_source_sha256s.txt` | 1,753 | `4816f0de0a653b667e20a79d20b11862bb56423428c374f88e3a66fb6d6209df` |
| `pytest.log` | 22,480 | `c6d27aa2e14f2535ccb5c0e6ea1fe39e305dceb8af2886889a70835a52bda5ee` |
| `pytest.junit.xml` | 22,137 | `a130a7ae347de462f49e802be4ca2d3aefad705d36e780adf16991e4ba591ada` |
| `sha256sums.txt` | 747 | `a9115f43e3c539867ec6c5d4440c5b07f00315bb7aeefc9a84ba264ac6bb040f` |
| Slurm stdout | 23,376 | `a8bd24753f806fb244e06c1090c43361c9773b7362c4b93c87aac8a2af0187c7` |
| Slurm stderr | 123 | `ae6330855ac405b2e19691ca1681d7f9eeedc6216718d1516023d9376d891b57` |

## Manual remediation after the failed job

S00 authorized a manual implementation correction but no retry. Commit
`2b5cf2f` removes custom residual blocks from `spconv.SparseSequential` and
forwards four `nn.ModuleList` residual stages explicitly as complete sparse
tensors; downsample modules remain sparse sequential conv/norm/activation. A
focused structural assertion was added to the first runtime fixture to reject any
future `_SparseResidualBlock` nested under `SparseSequential`. The test count and
all geometry/cap/channel/precision scopes remain unchanged.

Local-only checks after remediation:

- `python3 -m py_compile` on S04 source/tests: PASS;
- launcher `bash -n`: PASS;
- `git diff --check`: PASS;
- fusion-wide static banned-op/stable-sort audit: PASS, zero findings;
- exact test-function count: `10`.

The remediated code has **not** run on spconv/GH200. A new exact request is pending
S00 review; no rerun is authorized by these results.

## Interpretation limits

Allowed:

- the static shape/RF/metric/reference-channel fixtures passed on the exact
  failed-job source;
- Job 335566 proves the exact pre-remediation composition is incompatible with
  spconv `SparseSequential` custom-module dispatch;
- identity, request binding, scheduler record, and negative artifacts are intact;
- the manual remediation is statically valid and keeps densification after the
  reduced sparse grid.

Forbidden:

- runtime PASS for the remediated module;
- B=4 memory, fp16/fp32 gradient, sample isolation, cap, or empty-input PASS;
- production detector/S07-B readiness, full-data/mini behavior, throughput,
  profile, training convergence, mAP/NDS, fusion gain, FL, attack/defense,
  generalization, or publication claims.
