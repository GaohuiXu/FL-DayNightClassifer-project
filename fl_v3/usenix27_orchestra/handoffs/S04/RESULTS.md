# S04 RESULTS — SECOND sparse LiDAR engineering validation

## Overall result

All three exact S04 jobs are preserved **FAILED** engineering results. Job `335566`
found the sparse composition bug (`5 passed / 5 failed`); Job `335579` closed that
bug but found a final fp16-output mismatch (`8 passed / 2 failed`); Job `336718`
closed the exercised dtype/B=4 cases but exposed a same-model train/backward-to-eval
spconv tuner failure (`9 passed / 1 failed`). No retry, requeue, resubmission, or
follow-on occurred. Worker status remains **CHANGES-REQUESTED**, not runtime or
integration PASS.

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
compute. Code/test commit `72184e9ed3d2a9ea4fcd9f1a8dc473312a09a52d` explicitly
casts the active sparse-AMP output to fp16 only after the low-resolution
projection, records pre-contract/output dtype, preserves the fp32 reference path,
and extends the existing tests without deleting or weakening any dtype assertion.
The exact test inventory remains ten.

S00's request audit then identified a fail-closed provenance flaw before any
submission: `sbatch --chdir` does not itself guarantee snapshot-valued
`SLURM_SUBMIT_DIR`. Executable `2729f45144053e1b554a0bf04640b8bbc1ff43e4`
corrects the launcher/request so submission occurs from inside the immutable
snapshot and the job validates actual `pwd`, `SLURM_SUBMIT_DIR`, and a read-only
identity binding executable SHA/tree plus source/request hashes. No job consumed
the flawed request.

Local preparation evidence: Python compilation PASS; launcher `bash -n` PASS;
exact test-function count 10; `git diff --check` PASS. The login-node system Python
does not have pytest, and the x86 login node is not the validated spconv runtime,
so no local pytest or GH200 claim is made.

A new `RUN_REQUEST.md` was pending at that delivery. No Job 335566/335579 status
or interpretation was overwritten, and no third job had occurred yet.

## Final-output remediation validation Job 336718 — FAILED

### Verdict and exact test result

S00 approved the attested snapshot request exactly once. Job `336718` ended
`FAILED 1:0`: JUnit reports exactly `10` tests, `1` failure, `0` errors, and `0`
skips in `88.124s`; pytest reports `9 passed, 1 failed, 6 warnings in 88.12s`.

The prior final-output defect is closed for the exercised paths:

- the focused test passed `out32.dtype == torch.float32` and
  `out16.dtype == torch.float16` before its later failure;
- the B=4 test passed final fp16 dtype, debug projection/output dtype assertions,
  output `[4,256,180,180]`, dense boundary `[4,128,2,180,180]`, scalar loss,
  backward, finite intended gradients, voxel-drop checks, and memory bounds;
- B=4 evidence: loss `0.01097890641540289`, peak allocated
  `1,017,576,960`, peak reserved `1,109,393,408`, device total
  `102,005,473,280` bytes, and `optimizer_or_parameter_update=false`.

The sole failure is later in
`test_fp32_and_fp16_sparse_paths_have_finite_outputs_and_gradients`. After the
same fp16 model had completed train-mode forward/backward, the new fixture called
`fp16.eval()` and attempted another six-voxel non-empty forward before its empty
comparison. The stem's inference SubMConv reached spconv `ConvTunerSimple` with
fp16 features, fp32 filters, `is_train=False`, `is_subm=True`, and
`output_dtype=torch.float32`, then raised:

```text
!all_profile_res.empty() assert faild. can't find suitable algorithm for 0
```

Therefore same-model train/backward-to-eval reuse and the new fp16 empty/non-empty
comparison are not established. This is not an overall runtime PASS. The raw
failure does not undo the separate B=4 and initial fp16-output passes.

### Identity and scheduler

- Approved delivery/executable/tree:
  `465788332b1a431d808d509b484525d0092e4d95` /
  `2729f45144053e1b554a0bf04640b8bbc1ff43e4` /
  `2fdb42c97995112b3defc7e78ea148daa6ee7786`.
- Request/source/identity SHA-256:
  `120a33111a42a7f3bb1e0fa5f5fceb5eb924ac58e23f51b15ed874fe04fe7104` /
  `a9b6fd7f6a5d72cc7691cb6118b001ac4221d6d5cffe4b6799d75ef32fa58c06` /
  `4106a780694a2d6e1b1cf036b3a72b1c98bafa93134486799abd534a346ae12d`.
- Job/name/node: `336718` / `flv3_s04_second` / `n593`.
- State/exit: `FAILED` / `1:0`; restarts `0`.
- Submit/start/end: `2026-07-11T18:58:03` / `18:58:04` / `19:00:57`.
- Elapsed/timelimit: `00:02:53` / `00:20:00`.
- Allocation: exactly one node, one `nvidia_gh200_120gb`, eight CPUs,
  `mem=11672M`; the launcher and Torch one-device fail-closed checks passed.
- Batch MaxRSS/MaxVMSize: `1,142,016K` / `18,891,648K`; TotalCPU `00:37.851`;
  disk read/write `117.50M` / `0.27M`.
- Elapsed allocation approximately `0.0481` GPU-hours; cumulative S04 elapsed
  allocation approximately `0.0890` GPU-hours.
- Runtime: CPython `3.11.15`, Torch `2.11.0+cu128`, spconv `2.3.8`, cumm
  `0.7.13`, NumPy `1.26.4`, pytest `9.1.1`, aarch64.

The immutable snapshot remained free of write bits after execution. The output
root is
`/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s04_second_72184e9ed3d2_fp16remediation_v1`.

### Raw artifacts and checksums

In-job `sha256sum -c` passed every primary artifact despite pytest failure.

| Artifact | Bytes | SHA-256 |
|---|---:|---|
| `execution_identity.json` | 2,907 | `a0d59d11bc16b801fba625d6ecadec9beba2b46c3438cbdd8d553376b8dd73e3` |
| `runtime_source_sha256s.txt` | 1,753 | `a9b6fd7f6a5d72cc7691cb6118b001ac4221d6d5cffe4b6799d75ef32fa58c06` |
| `pytest.log` | 11,365 | `3353d78a6f73ea38093b2a19a7453dba3f0fde46cee56fe27af097a858d96265` |
| `pytest.junit.xml` | 10,585 | `0c7756f1ed801b0268bd4b32fda0224bf55bedd973e3d6df749f57f5a95d7439` |
| `sha256sums.txt` | 823 | `373dfb5063c060dc1fd4f7b407f0de759f60b2edeeb81894633e2a2e16e35730` |
| Slurm stdout | 12,453 | `af6ae66979f98ad12e083e6959fcb54d660437f20f10d36afc6d500aa5e8d303` |
| Slurm stderr | 123 | `ae6330855ac405b2e19691ca1681d7f9eeedc6216718d1516023d9376d891b57` |

No dataset, optimizer/scaler/parameter step, metric, profile, retry, requeue,
resubmission, or follow-on occurred. Allowed evidence is limited to the explicit
nine passes and preserved failure above; production/S07-B/scientific readiness and
all model-quality, throughput, convergence, FL, attack/defense, or publication
claims remain forbidden.

## Source-only diagnosis and lifecycle-matrix preparation

No compute followed Job `336718`. Inspection of installed spconv 2.3.8 shows a
specific training/eval divergence rather than a final-BEV cast failure:

- S04 supplies fp16 sparse features whenever its sparse-fp16 path is active.
- Training routes through `SparseImplicitGemmFunction` with Torch
  `custom_fwd(cast_inputs=float16)`, so both features and fp32 master filters reach
  `ops.implicit_gemm` as fp16.
- Eval bypasses that autograd wrapper, calls `ops.implicit_gemm` directly, leaves
  filters fp32, and requests filter-dtype fp32 output. Job 336718 therefore asked
  the tuner for fp16 input / fp32 filter / fp32 output.
- The generated C++ tuner descriptor/cache key includes all three dtypes. A
  half/half/half training cache entry cannot satisfy the mixed eval key; the
  observed `all_profile_res.empty()` means no compatible descriptor was available.

This explains why backward itself is unlikely to be the direct cause, but does not
yet establish whether fresh eval, occupancy, or process-local order changes the
outcome. Commit `bd1fc9af139cce85240c5908d6704c38425f3c1f` added a seven-cell,
fresh-process synthetic diagnostic that traces exact implicit-GEMM call state. At
that delivery its request was pending; Job `336728` subsequently executed it as
recorded below.

Potential remedies are deliberately not implemented:

- converting an eval model to half may make the kernel tuple half/half/half, but
  mutates parameter precision and complicates resume/re-entry to training;
- falling back to fp32 sparse eval preserves fp32 weights but creates a
  train/eval sparse-precision mismatch;
- forcing the training autograd path under `no_grad` changes index/memory and
  framework semantics;
- patching spconv to cast inference filters requires a maintained dependency fork
  and changes numerical/runtime behavior.

Each is a material precision/lifecycle choice requiring reviewed diagnostic
evidence and S00/owner approval. None is an authorized fix or current claim.

## Lifecycle diagnostic Job 336728 — COMPLETED matrix, S04 still blocked

### Exact outcome and mechanism evidence

Job `336728` completed `0:0` because every required process returned a valid
structured envelope and all provenance/artifact gates passed. This is diagnostic
completion, not S04 PASS. Cell outcomes were six `error` and one `success`:

| Cell | Outcome | Decisive observation |
|---|---|---|
| fresh fp16 eval, 6 active voxels | error | first SubMConv fp16/fp32/fp32, no algorithm |
| same fp16 model train→eval, no backward | error | train half/half succeeds; eval mixed tuple fails |
| same fp16 model train/backward→eval | error | finite train gradients; eval mixed tuple fails |
| fresh fp16 eval, 256 active voxels | error | same mixed tuple fails, excluding tiny occupancy |
| fresh fp32 eval, 6 active voxels | success | all 21 sparse calls fp32/fp32/fp32 succeed |
| fp32 eval then fresh fp16 eval | error | fp32 cache/order does not satisfy mixed tuple |
| fp16 train then distinct fresh fp16 eval | error | half training cache does not satisfy mixed tuple |

Every failed fp16-eval cell stopped in the first stem SubMConv with
`is_train=False`, `is_subm=True`, fp16 feature shape `[6,4]` (or `[256,4]` for the
large case), fp32 filter `[16,3,3,3,4]`, requested fp32 output, and the same
`all_profile_res.empty()` assertion. Training traces used fp16 features/filters and
fp16 output successfully. This establishes a universal current fp16-eval dispatch
blocker in the tested environment, not a backward, reuse, occupancy, or cache-order
effect.

### Scheduler, identity and raw artifacts

- Authorization class: one validation-only diagnostic job under the owner's
  temporary S02-S05 authority; not a scientific matrix or O-009 expansion.
- Delivery/executable/tree:
  `7e3bf58d875bf72973c3990dc4fdf5697915ef40` /
  `bd1fc9af139cce85240c5908d6704c38425f3c1f` /
  `80b6f5cf5028faffa67b7510454a510e94b72f31`.
- Request/identity/repo-source/dependency-source SHA-256:
  `710eecb3cc10ae971ee8eca6f1bec421bc8f045c03df28b9d311cac5d65a63ab` /
  `a8069f9a2b3d7a1fa6f40d2cebd4e6f3d171e58ba5cc79b805f5f779d0890d7d` /
  `d2a5041c5177279f874bd788320053df679c5b8ad060f95d729e29ae0ebfbf63` /
  `e7e162a1f10b4e66c42c1bc07fae19248c42a5e198fbee2c546f3dc0a0d43141`.
- Job/name/node: `336728` / `flv3_s04_lifecycle` / `n593`.
- State/exit/restarts: `COMPLETED` / `0:0` / `0`.
- Submit/start/end: `2026-07-11T19:09:45` / `19:09:46` / `19:13:22`.
- Elapsed/timelimit: `00:03:36` / `00:20:00`; exactly one GH200, eight CPUs,
  one node, `mem=11672M`.
- Batch MaxRSS/MaxVMSize: `1,514,952K` / `24,302,208K`; TotalCPU `03:22.224`;
  disk read/write `572.85M` / `0.81M`.
- Elapsed allocation `0.0600` GPU-hours; cumulative S04 elapsed allocation about
  `0.1490` GPU-hours.
- CPython `3.11.15`, Torch `2.11.0+cu128`, spconv `2.3.8`, cumm `0.7.13`.
- Snapshot remained read-only; exactly one matching job was submitted and no S04
  lifecycle job remains active.

In-job `sha256sum -c` passed all primary artifacts:

| Artifact | Bytes | SHA-256 |
|---|---:|---|
| `execution_identity.json` | 2,752 | `0a40265c36c6854f777e7909c2a422f185e3222970dcc71b95be7ae3c6f66119` |
| `runtime_source_sha256s.txt` | 1,276 | `d2a5041c5177279f874bd788320053df679c5b8ad060f95d729e29ae0ebfbf63` |
| `dependency_source_sha256s.txt` | 1,262 | `e7e162a1f10b4e66c42c1bc07fae19248c42a5e198fbee2c546f3dc0a0d43141` |
| `diagnostic.log` | 276 | `ea800f13af1ca13c15f550e7fa68a497b85cdbbb5513acd248fa390e9420ca0f` |
| `lifecycle_matrix.json` | 174,979 | `3257e16b7bf8ed9b7afcfc252b284ece81595b5c45c83c296e9434e412e346e4` |
| `sha256sums.txt` | 983 | `038b0a93e4a8e084b2d4a9d06381361e8e4bea30ba89e25fb59e1073f5b102d0` |
| Slurm stdout | 1,220 | `ddfe1ec1a2b99509c33cef4188b3fc80dc4ae27d1073f35e22e8357d421e17f8` |
| Slurm stderr | 123 | `ae6330855ac405b2e19691ca1681d7f9eeedc6216718d1516023d9376d891b57` |

No data, optimizer/parameter step, metric, scientific cell, profile, retry,
requeue, resubmission, remedy, or follow-on occurred.

## Historical owner decision docket — remedy not yet authorized at Job 336728

### A — force only spconv convolutions through training dispatch in fp16 eval (recommended)

Keep the encoder in eval mode so eval voxel caps and surrounding module semantics
remain intact; GroupNorm has no running statistics. Under `torch.no_grad`, route
only spconv `SparseConvolution` modules through their training dispatch so the
existing custom-fwd casts fp32 master filters to fp16, matching successful training
compute. Do not call `.train()` on the whole encoder.

- Benefit: preserves fp32 master parameters/state dict and aligns sparse eval
  arithmetic with the already validated half/half/half training forward.
- Runtime risks: training dispatch may generate backward indice/mask state even
  under `no_grad`, increasing eval memory/time; it relies on spconv private dispatch
  behavior and must fail closed outside exact tested spconv/cumm versions.
- Scientific risks: a private dispatch workaround can silently drift after a
  dependency upgrade. Metadata must declare forced dispatch and dependency hashes;
  eval metrics are valid only after parity/lifecycle review.
- Required next validation: fresh 6/256-voxel and same-model before/after-backward
  eval, fp32 master/state-dict immutability, no eval gradients, forced-dispatch
  versus train-mode-`no_grad` numerical parity, fp32 control, B=4 eval memory, and
  explicit version guard rejection.

### B — run sparse eval in fp32, then cast final BEV to fp16

- Benefit: uses the proven fp32 eval path and avoids private spconv dispatch.
- Risks: training and evaluation use different sparse arithmetic; memory and speed
  rise, and fp32 eval may hide fp16 numerical failures. Any metrics must be labeled
  precision drift and cannot be compared as matched fp16 without an approved
  ablation.
- Required validation: fp32 eval lifecycle, final-BEV dtype, memory, parity delta
  versus fp16 training forward, and explicit resolved-config/manifest labeling.

### C — create an fp16 weight copy for evaluation

- Benefit: supplies the native half/half/half inference tuple without changing the
  installed dependency.
- Risks: duplicate model memory, conversion/caching complexity, stale-copy risk,
  GroupNorm/other parameter conversion scope, checkpoint/state semantics, and
  accidental loss of fp32 master weights if the training instance is mutated.
- Required validation: immutable fp32 source model, exact clone provenance,
  fresh/same-checkpoint eval parity, state-dict hashes, memory, and safe destruction
  before resume/training.

### D — patch the spconv inference path

- Benefit: can make inference autocast filters in the dependency at the intended
  boundary rather than altering module mode.
- Risks: highest maintenance and ABI/build risk; requires a maintained fork,
  source/build attestation, regression coverage across sparse ops, and reevaluation
  on every dependency upgrade. Numerical behavior becomes project-specific.
- Required validation: source/build hashes, full sparse-op forward/backward and
  inference regression, lifecycle/parity/memory tests, and independent dependency
  review.

Option A is the smallest current candidate because it preserves fp32 master weights
and matches the successful training dtype tuple. At this historical diagnostic
point it remained owner-locked; no option was implemented or approved by Job
`336728`. O-025 subsequently selected A as recorded next.

## O-025 option-A implementation validation — Job 341695

The owner subsequently selected option A in O-025, and canonical clarification
`04569c6` allowed the required one-shot synthetic forward/backward lifecycle
fixtures while forbidding optimizer/GradScaler/parameter updates and iterative
training. S00 independently approved the exact immutable tuple once. Job `341695`
consumed it and completed successfully; no retry, requeue, resubmission, or
follow-on occurred.

### Result and gate evidence

- Pytest/JUnit: `15 passed / 0 failed / 0 errors / 0 skipped`, `78.714s`.
- Exact version guard rejected an unsupported version, including before an empty
  fp16-eval return.
- Fresh 6-voxel, fresh 256-voxel, same-model eval-before-training, and same-model
  train/backward-to-eval fp16 paths all returned finite fp16 outputs.
- Eval required `torch.no_grad`; all parameter `.grad` fields were absent after
  eval, all master parameters remained fp32, and the complete state-dict SHA-256
  was unchanged across the lifecycle.
- Encoder, backbone, and GroupNorm remained eval. Exactly 21 spconv sparse
  convolutions used temporary training dispatch and every flag was restored,
  including the explicit exception path.
- Automatic option-A eval passed numerical parity against ordinary training
  dispatch under `no_grad`; the fresh fp32 eval control also passed.
- Existing coordinate/shape/RF, reduced densification, per-sample train/eval cap,
  over-cap/extreme occupancy, empty-input, permutation/isolation, fp16 train and
  B=4 forward/backward gates all passed without being removed or weakened.
- B=4 train/backward: output `[4,256,180,180]` fp16, dense
  `[4,128,2,180,180]`, finite loss `0.010624694637954235`, peak allocated/reserved
  `962,274,304` / `1,069,547,520` bytes.
- B=4 eval: output `[4,256,180,180]` fp16, dense
  `[4,128,2,180,180]`, dispatch count `21`, all masters fp32, all grads absent,
  peak allocated/reserved `371,167,232` / `411,041,792` bytes on a
  `102,005,473,280`-byte device.

### Exact identity, scheduler, and dependencies

- Approved delivery/tree: `2350335166e8f2407ff58f99e9aa5ca98c8acb23` /
  `671611cdd743d0b1d63bde5c861cc76ec55236db`.
- Executable/tree: `84985970f0f4b4acb8704ddbbd6ae9b2bf94ca9f` /
  `913fee67d405ed554b3f7df37c3c137f6f577c2d`.
- Request/source/identity SHA-256:
  `b242336e1696a68b6a01a90492ca0e58b7216ef8c4ab9a0eced1f983bf5d2110` /
  `a2608664abd6b69f09b96f19b915cdefe1431aa8b503985f2184b94817e92463` /
  `35d5547e68d2132d9bec1dc77202154b7faf6d2e1d767d68944f722bb0849d2c`.
- Job/name/node: `341695` / `flv3_s04_option_a` / `n412`.
- Submit/start/end: `2026-07-12T03:22:00` / `03:22:02` / `03:24:43`;
  scheduler elapsed `00:02:41`, state `COMPLETED`, exit `0:0`, restarts `0`.
- Allocation: one node, one `nvidia_gh200_120gb`, eight CPUs, `mem=11672M`,
  `OverSubscribe=OK`; batch MaxRSS/MaxVMSize `1,141,888K` / `18,892,480K`.
- CPython `3.11.15`, Torch `2.11.0+cu128`, spconv `2.3.8`, cumm `0.7.13`,
  NumPy `1.26.4`, pytest `9.1.1`.
- Snapshot:
  `/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/snapshots/s04_o025_84985970f0f4_v1`.
- Output:
  `/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s04_o025_84985970f0f4_v1`.

In-job `sha256sum -c` passed all four primary artifacts:

| Artifact | Bytes | SHA-256 |
|---|---:|---|
| `execution_identity.json` | 2,649 | `bdca744bf8d8380a6bb67f9ea48603e240278adc267b3ceb8c6d7d722c2e3342` |
| `runtime_source_sha256s.txt` | 1,869 | `a2608664abd6b69f09b96f19b915cdefe1431aa8b503985f2184b94817e92463` |
| `pytest.log` | 2,961 | `63ad1d176074b56020e072d56f93fcd496f7c59694a4ddde2ad954be39190a34` |
| `pytest.junit.xml` | 2,223 | `8969a3b40f39f65853d5fdae488a9f7c9e1acd22c6b28b2dffb7159afee466d6` |
| `sha256sums.txt` | 751 | `efc70b39763053662c82907ff7901488119f9d7b8c5b081a021f726c8958105d` |
| Slurm stdout | 3,535 | `f4e0bcee0900a9bfba727ec5d94b67f6a16c601da9cc7ac9fbd592802592661d` |
| Slurm stderr | 123 | `ae6330855ac405b2e19691ca1681d7f9eeedc6216718d1516023d9376d891b57` |

The stdout contains one Torch FX API warning plus the pytest warning summary:
one Python `locale.getdefaultlocale` deprecation from `ccimport` and seven repeats
of spconv's non-tuple multidimensional-indexing warning. Stderr contains only the
environment module-purge notice for `hpc_sysenv/.1` and `hpc/.2.1.0`. Neither
warning affected the exact test or checksum gates, but both dependency-compatibility
warnings remain visible for future Torch/Python upgrades.

### Interpretation limits and retained negatives

Job `341695` is bounded synthetic engineering evidence only for executable
`84985970` on the attested spconv-2.3.8 runtime. It does not establish production
detector wiring, full-data behavior, throughput/profile, metrics, convergence,
fusion gain, FL/security behavior, or scientific claims. Its pass does not erase
failed Jobs `335566` (sparse composition), `335579` (final fp16 dtype), or `336718`
(native fp16 eval dispatch), and does not reinterpret diagnostic Job `336728`:
all six native fp16-eval cells there remain recorded errors and are the evidence
that motivated O-025. Independent S04-R and Orchestra acceptance are still
required before integration.
