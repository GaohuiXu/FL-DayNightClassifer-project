# S04 RUN REQUEST — one synthetic GH200 SECOND validation

## Approval state

`APPROVED_EXACT_ONCE_CONSUMED_BY_JOB_335579_FAILED_NO_RETRY`

This request is governed by O-017 and the stricter S02-S05 rule in
`KICKOFFS.md`: even though the job fits O-009, S04 must stop and wait for explicit
S00 approval. Creating this file grants no execution permission.

## Immutable implementation and worktree state

- Session/branch: `S04` / `codex/s04-lidar-second`.
- Exact executable HEAD: `0d6ea005fe138aaa4cb39cfab005431abb622acf`.
- Exact executable tree: `b9514e12eb5255602e9f7d0da6671a9be8e45c68`.
- Initial implementation commit: `20d11e284f20fced3dbc33e7ac105c845da708a5`.
- Sparse-composition remediation:
  `2b5cf2f5da9a123c313780bbdd52b1202b62cd38`.
- Job-335566 evidence delivery:
  `0d6ea005fe138aaa4cb39cfab005431abb622acf`.
- Launcher commits: `a201245d7935ac9c385705a54d8eac8355b3df37`, request-binding
  correction `5676ff6ee0621ce8df26b50edcafa4a7a4f177f4`, and scoped
  forward/backward-only remediation `49efb05dd341dbfbcc2d373508772e5b214aa726`.
- Runtime source-state SHA-256 over the 17 locale-sorted files enumerated by the
  launcher: `2e5755522cff0aa2899a035f45440fb5ecdb71f2cb5156c96403dd818bba9886`.
- Expected ref: `codex/s04-lidar-second`.
- Expected worktree status during execution: exactly one untracked file,
  `fl_v3/usenix27_orchestra/handoffs/S04/RUN_REQUEST.md`; no tracked diff and no
  other untracked file.
- This request file's SHA-256 is intentionally recorded by S00 outside this file
  (a file cannot contain its own content hash). The launcher requires the exact
  S00-approved value through `EXPECTED_S04_REQUEST_HASH`, recomputes it in-job,
  and records it in `execution_identity.json`.

Any change to HEAD/tree, runtime source aggregate, request-file hash, branch/ref,
command, resources, input, output, or stop conditions invalidates approval.

The prior tuple at executable `5676ff6e`, runtime source `374f07b9...`, request
SHA-256 `e0d482e7...` was explicitly **NOT APPROVED** because it contained an
optimizer/GradScaler step. It is superseded and must never be submitted or
reinterpreted as this request.

## Preserved prior execution — Job 335566 FAILED, no retry

S00 approved exactly one remediated forward/backward-only request at executable
`49efb05dd341dbfbcc2d373508772e5b214aa726`, source
`4816f0de0a653b667e20a79d20b11862bb56423428c374f88e3a66fb6d6209df`, and request
SHA-256 `00aea9398736471b3a68a1e1fade00fb7e639457795109cc8d9ad6971c956b7c`.
It was consumed once as Job `335566` and ended `FAILED 1:0` after `00:01:41`:
exactly 10 tests, 5 passed, 5 failed, zero errors/skips. All failures were the
same spconv composition error: `_SparseResidualBlock` received a feature Tensor
from `SparseSequential` and then accessed `.features`. Identity and artifact
checksums passed. No optimizer/parameter update occurred; B=4 did not complete;
no memory evidence exists. No retry was submitted. Complete raw hashes and
scheduler fields are in `RESULTS.md`.

S00 then authorized manual implementation remediation only. Commit `2b5cf2f`
explicitly forwards residual `ModuleList` stages as `SparseConvTensor` and adds a
focused structural regression to the first runtime fixture. No voxel geometry,
stride, channels, caps, test intent/count, resources, or scientific contract
changed. This section preserves Job 335566 as a negative; the new request below is
a separately pending execution and is not authorized by the old approval.

## Remediation execution — Job 335579 FAILED, no further authorization

S00 independently approved this request exactly once at request SHA-256
`4acc45db2c6b1e5b0f4aaf5e3247e2e409217090edc62ec013b2c598eaa3354b`.
It was consumed as Job `335579`; no edit occurred between approval and submission.
The job ended `FAILED 1:0` after `00:00:46`: exact JUnit counts were 10 tests,
8 passed, 2 failed, zero errors/skips. The composition remediation worked: real
spconv shape/backward, per-sample caps/extreme occupancy, empty input, and sample/
batch isolation cases passed. Both remaining failures are the same precision
contract mismatch: the fp16 sparse path returns final BEV dtype `torch.float32`,
not required `torch.float16`.

The B=4 case reached output `[4,256,180,180]`, constructed loss, completed backward,
and observed finite intended gradients before failing the dtype assertion. Because
the test stops there, it did not record the peak CUDA memory evidence dictionary.
No optimizer, scaler, parameter update, dataset, profile, metric, retry, requeue,
resubmission, or follow-on occurred. Identity and checksums passed. Full scheduler,
test, artifact and log evidence is in `RESULTS.md`.

This approval is exhausted. The command below is retained only as exact execution
provenance and must not be run again. Any dtype remediation or validation requires
a new S00/owner decision and a new immutable request.

## Purpose and bounded input

Engineering-only validation of the S04 module contract:

1. CPU/static golden checks for `41x1440x1440 -> 2x180x180`, stride 8,
   receptive field, metric mapping, and the single reduced-resolution `dense()`;
2. real spconv fp32/fp16 finite forward/backward, sample/batch isolation,
   per-sample train/eval caps, empty input, over-cap occupancy, and point-order
   invariance on small synthetic shapes;
3. exactly one full-shape synthetic B=4 fp16-autocast forward, scalar loss
   construction, and backward with 4,096 generated points per sample, output
   `[4,256,180,180]`, expected dense boundary `[4,128,2,180,180]`, finite intended
   gradients, and printed peak allocated/reserved CUDA bytes.

The B=4 case constructs no optimizer or GradScaler and performs no `optimizer.step`,
`scaler.step/update`, scheduler/EMA action, or parameter update. The memory output
is one bounded engineering guard only, not a throughput/profile result.

Input is generated deterministically in the test. It reads no nuScenes mini,
trainval, ZIP, cache, checkpoint, or external model artifact. It computes no
mAP/NDS, throughput profile, scientific metric, 100/1000-step gate, epoch, matrix,
seed comparison, or detector/trainer result.

## Resources and budget

- Account/partition: `naiss2025-22-1113-gpu` / `gpu`.
- One node, one `nvidia_gh200_120gb`, eight CPUs.
- Walltime: `00:20:00`; maximum requested allocation `0.3334 GPU-hours`.
- One job, one concurrent S04 job, no array, DDP, retry, requeue, resubmission,
  spare-GPU expansion, or follow-on.
- S04 cumulative elapsed GPU allocation before this request: approximately
  `0.0281` GPU-hours from failed Job 335566.

## Exact command

After S00 records the exact SHA-256 of this request file as
`S04_APPROVED_REQUEST_SHA256`, the only authorized submission is:

```bash
test "$(git rev-parse HEAD)" = 0d6ea005fe138aaa4cb39cfab005431abb622acf
test "$(git branch --show-current)" = codex/s04-lidar-second
test "$(git status --short)" = "?? fl_v3/usenix27_orchestra/handoffs/S04/RUN_REQUEST.md"
test "$(sha256sum fl_v3/usenix27_orchestra/handoffs/S04/RUN_REQUEST.md | awk '{print $1}')" = "$S04_APPROVED_REQUEST_SHA256"
test ! -e /nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s04_second_0d6ea000d99d
sbatch --export=ALL,EXPECTED_S04_SHA=0d6ea005fe138aaa4cb39cfab005431abb622acf,EXPECTED_S04_REF=codex/s04-lidar-second,EXPECTED_S04_SOURCE_HASH=2e5755522cff0aa2899a035f45440fb5ecdb71f2cb5156c96403dd818bba9886,EXPECTED_S04_REQUEST_HASH="$S04_APPROVED_REQUEST_SHA256",S04_OUTPUT_ROOT=/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s04_second_0d6ea000d99d fl_v3/usenix27_orchestra/handoffs/S04/run_s04_second_smoke.sh
```

The launcher executes exactly:

```bash
python -m pytest -q -ra -s \
  fl_v3/tests/test_s04_second_contract.py \
  fl_v3/tests/test_sparse_voxel_encoder.py \
  fl_v3/tests/test_s04_second_smoke.py \
  --junitxml="$S04_OUTPUT_ROOT/pytest.junit.xml"
```

## Outputs and logs

- Unique output root (confirmed absent before request):
  `/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s04_second_0d6ea000d99d`.
- Slurm stdout/stderr:
  `/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/logs/s04_second_%j.{out,err}`.
- Expected compact artifacts: `execution_identity.json`,
  `runtime_source_sha256s.txt`, `pytest.log`, `pytest.junit.xml`, and
  `sha256sums.txt` with in-job `sha256sum -c`.

No checkpoint, dataset, cache, profile trace, or remote artifact is produced.

## Acceptance and stop conditions

Pass requires all of:

- exact in-job SHA/ref/status/request/source identity;
- validated Arrhenius Python/Torch/spconv/cumm versions recorded;
- JUnit has exactly `10` tests and zero failures, errors, or skips;
- full-shape B=4 output/dense shape and fp16 dtype match the contract;
- output, loss, and every intended gradient are finite; no parameter update occurs;
- per-sample caps/isolation, empty/over-cap, metric/shape/RF and no-fine-dense
  fixtures pass;
- artifact checksum verification passes.

Stop immediately on preflight drift, import/build error, OOM, non-finite value,
test-count drift, test failure/error/skip, checksum failure, scheduler timeout, or
output-root collision. Record the negative result. Do not retry, modify, resubmit,
or launch a follow-on without a new request and explicit approval.

## Interpretation boundary

Allowed if passed: bounded synthetic GH200 evidence for this exact standalone S04
module's shapes, finite fp16/fp32 behavior, gradients, per-sample voxelization, and
B=4 peak CUDA memory observation.

Forbidden regardless of result: production detector/S07-B readiness, full-data or
mini behavior, model quality, mAP/NDS, fusion gain, best voxel size, throughput,
training convergence, FL, attack/defense, generalization, or publication claims.
