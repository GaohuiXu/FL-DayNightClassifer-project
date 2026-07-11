# S04 RESULTS — SECOND sparse LiDAR engineering validation

## Overall result

Both exact S04 jobs are preserved **FAILED** engineering results. Job `335566`
found the sparse composition bug (`5 passed / 5 failed`); manual remediation then
closed that bug, but Job `335579` found a final fp16-output dtype mismatch
(`8 passed / 2 failed`). No third job, retry, requeue, resubmission, or follow-on
occurred. Worker status remains **CHANGES-REQUESTED**, not runtime or integration
PASS.

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

The remediated code subsequently ran exactly once as Job `335579`; its result is
recorded below. Job `335566` remains a negative and is not superseded.

## Manual-remediation validation Job 335579 — FAILED

### Identity and scheduler

- Approved request SHA-256:
  `4acc45db2c6b1e5b0f4aaf5e3247e2e409217090edc62ec013b2c598eaa3354b`.
- Executed HEAD/tree:
  `0d6ea005fe138aaa4cb39cfab005431abb622acf` /
  `b9514e12eb5255602e9f7d0da6671a9be8e45c68`.
- Sparse fix: `2b5cf2f5da9a123c313780bbdd52b1202b62cd38`.
- Runtime source SHA-256:
  `2e5755522cff0aa2899a035f45440fb5ecdb71f2cb5156c96403dd818bba9886`.
- Job/name/node: `335579` / `flv3_s04_second` / `n507`.
- State/exit: `FAILED` / `1:0`.
- Submit/start/end: `2026-07-11T18:24:33` / `18:24:36` / `18:25:22`.
- Elapsed/timelimit: `00:00:46` / `00:20:00`.
- Allocation: one GH200, eight CPUs, one node, `mem=11672M`; restarts `0`.
- Batch MaxRSS/MaxVMSize: `36M` / `6672960K`; TotalCPU `00:33.016`.
- Elapsed GPU allocation: approximately `0.0128` GPU-hours; cumulative S04
  elapsed allocation approximately `0.0409` GPU-hours.

Execution identity again records the same CPython/Torch/spconv/cumm dependency
versions and `synthetic_only=true`. Git/ref/source/request checks passed before
pytest.

### Exact tests and new failure

JUnit: `tests=10`, `failures=2`, `errors=0`, `skipped=0`; pytest summary
`8 passed, 2 failed, 6 warnings in 31.14s`.

Passed evidence now includes:

- all static shape/RF/metric/dense-bound fixtures and precision-policy rejection;
- the structural regression proving custom residuals are not nested in
  `SparseSequential`;
- real spconv output shape, reduced occupancy and finite backward at small shape;
- exact per-sample train/eval caps, extreme occupancy and point-order behavior;
- empty input plus sample/batch isolation.

Failed cases:

1. `test_fp32_and_fp16_sparse_paths_have_finite_outputs_and_gradients`;
2. `test_s04_reference_b4_fp16_forward_backward_memory_bound`.

Both expected `torch.float16` for `sparse_conv_fp16=True`, but final output was
`torch.float32`. The B=4 case reached `[4,256,180,180]`, loss construction,
backward, and finite-gradient assertion before its dtype assertion. It did not
reach the subsequent peak allocated/reserved capture, so no B=4 peak-memory value
may be claimed. The likely seam is the dense/collapse/output boundary retaining or
promoting fp32 under the current normalization/identity projection; no manual fix
is authorized or made in this handoff.

### Job 335579 artifacts

In-job `sha256sum -c` passed all four primary artifacts.

| Artifact | Bytes | SHA-256 |
|---|---:|---|
| `execution_identity.json` | 819 | `e9e2a513a2ece734c98bc7ad4866368b780f732c47a865bc8b60505f90912dc2` |
| `runtime_source_sha256s.txt` | 1,753 | `2e5755522cff0aa2899a035f45440fb5ecdb71f2cb5156c96403dd818bba9886` |
| `pytest.log` | 5,129 | `4b30beadf77a822c9b8edd4b5a6010c403cb81b4c4226e881b544e8ddfc5bf01` |
| `pytest.junit.xml` | 5,047 | `0c97e228bdaac48a423c14532771191d2c3953e195c25eb2c7b209905538f1f8` |
| `sha256sums.txt` | 747 | `8ff9002e4045360ddb5d50187fce36e2769da8c84bbc7c876bdcde399edf509f` |
| Slurm stdout | 6,025 | `c309169861d7fa83cf05bd3f5c6b7a2849390b0d2b7cf2284951d79bc2278576` |
| Slurm stderr | 123 | `ae6330855ac405b2e19691ca1681d7f9eeedc6216718d1516023d9376d891b57` |

Output root:
`/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s04_second_0d6ea000d99d`.

## Interpretation limits

Allowed:

- the static shape/RF/metric/reference-channel fixtures passed on the exact
  failed-job source;
- Job 335566 proves the exact pre-remediation composition is incompatible with
  spconv `SparseSequential` custom-module dispatch;
- identity, request binding, scheduler record, and negative artifacts are intact;
- the manual remediation is statically valid and keeps densification after the
  reduced sparse grid;
- Job 335579 supports the explicitly listed eight passing contracts and shows
  that the remaining blocker is final fp16 output dtype.

Forbidden:

- overall runtime PASS for the remediated module;
- final fp16-output contract or B=4 peak-memory PASS;
- production detector/S07-B readiness, full-data/mini behavior, throughput,
  profile, training convergence, mAP/NDS, fusion gain, FL, attack/defense,
  generalization, or publication claims.

## Scoped fp16-output remediation — prepared, not executed

S00 returned S04 for the one remaining implementation defect without authorizing
compute. Executable `72184e9ed3d2a9ea4fcd9f1a8dc473312a09a52d` explicitly
casts the active sparse-AMP output to fp16 only after the low-resolution
projection, records pre-contract/output dtype, preserves the fp32 reference path,
and extends the existing tests without deleting or weakening any dtype assertion.
The exact test inventory remains ten.

Local preparation evidence: Python compilation PASS; launcher `bash -n` PASS;
exact test-function count 10; `git diff --check` PASS. The login-node system Python
does not have pytest, and the x86 login node is not the validated spconv runtime,
so no local pytest or GH200 claim is made.

A new `RUN_REQUEST.md` is `PENDING_S00_EXACT_O009_APPROVAL_NOT_SUBMITTED`.
No Job 335566/335579 status or interpretation has been overwritten, and no third
job, retry, requeue, resubmission, or follow-on has occurred at this delivery.
