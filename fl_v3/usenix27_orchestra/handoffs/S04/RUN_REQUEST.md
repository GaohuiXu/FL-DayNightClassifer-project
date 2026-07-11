# S04 RUN REQUEST — spconv fp16 lifecycle diagnostic

## Current approval state

`APPROVED_ONCE_CONSUMED_BY_JOB_336728_COMPLETED_DIAGNOSTIC_NO_FOLLOW_ON`

S00 approved the exact lifecycle diagnostic tuple once under the owner's temporary
S02-S05 authority for necessary validation-only Slurm work. Job `336728` consumed
it and completed the structured evidence matrix. This was one diagnostic job with
seven process-isolation observations, not a scientific experiment matrix and not
an expansion of O-009. The approval is exhausted; no remedy, retry, requeue,
resubmission, or follow-on is authorized.

## Prior final-output request state

`APPROVED_EXACT_ONCE_CONSUMED_BY_JOB_336718_FAILED_NO_RETRY`

S00 independently approved the exact tuple below once. It was consumed only by
Job `336718`, which ended `FAILED 1:0` with `9 passed / 1 failed`. The approval is
exhausted. There is no retry, requeue, resubmission, or follow-on authorization.

## Preserved negative executions

- Job `335566`: `FAILED 1:0`, `5 passed / 5 failed`, because a custom residual
  block inside `spconv.SparseSequential` received a feature Tensor rather than a
  SparseConvTensor. Commit `2b5cf2f5da9a123c313780bbdd52b1202b62cd38`
  corrected that composition. This remains a negative result.
- Job `335579`: `FAILED 1:0`, `8 passed / 2 failed`. The composition correction
  passed, but the active sparse-fp16 path returned final non-empty BEV as fp32
  after the low-resolution projection, while the approved contract and empty path
  require fp16. This also remains a negative result.

Complete scheduler fields, raw hashes, and interpretation limits remain in
`RESULTS.md`. Neither approval is reusable.

- Job `336718`: `FAILED 1:0`, `9 passed / 1 failed`. The original active-fp16
  output assertion and the B=4 dtype/forward/backward/memory case passed. The one
  failure occurred only when the focused test reused the already-trained/backward
  fp16 encoder after switching it to eval for a second non-empty forward: spconv
  `ConvTunerSimple` could not find an inference SubMConv algorithm for the
  six-voxel input. The new empty/non-empty consistency sub-check therefore did not
  complete. Identity, allocation, source/request/snapshot, and final artifact
  checksum gates passed. This approval is not reusable.

## Immutable implementation and source identity

- Session/ref: `S04` / `codex/s04-lidar-second`.
- Approved wave base: `372de9398ae435f82b83367a922fd302c0635738`.
- Exact remediation executable commit (dtype code/tests plus attested snapshot
  launcher): `2729f45144053e1b554a0bf04640b8bbc1ff43e4`.
- Exact executable tree:
  `2fdb42c97995112b3defc7e78ea148daa6ee7786`.
- Exact runtime source-state SHA-256 over the 17 locale-sorted files enumerated
  by the launcher:
  `a9b6fd7f6a5d72cc7691cb6118b001ac4221d6d5cffe4b6799d75ef32fa58c06`.
- Launcher SHA-256:
  `6486d8d42c56a4a91d02110b426e4df4a5b5b0357d01e0ac2d2dd5dede0eda9a`.
- The request-delivery commit and this request file's SHA-256 are intentionally
  supplied by S00 as `S04_APPROVED_DELIVERY_SHA` and
  `S04_APPROVED_REQUEST_SHA256`; the derived immutable identity-file hash is
  supplied as `S04_APPROVED_IDENTITY_SHA256`. A file cannot contain its own
  content hash or enclosing commit SHA without changing them.

The remediation is deliberately narrow. After the existing low-resolution
`to_bev` projection, it records the pre-contract dtype and casts only the active
sparse-AMP output to fp16. The fp32 reference path stays fp32; empty and non-empty
fp16 returns now share one dtype contract. Existing dtype assertions are retained
and extended to trace projection/output dtype. Sparse geometry, voxel caps,
channels, densification, backward/loss construction, and the ten-test inventory
are unchanged.

Any change to the delivery/executable SHA, request hash, source aggregate,
launcher, command, snapshot/output root, tests, resources, or stop conditions
invalidates approval.

## Immutable execution snapshot

The compute job does not execute from `/home` or query a Git worktree. The exact
command first archives executable `2729f45...` into a unique shared `/nobackup`
snapshot, replaces only the archived request with the S00-approved request bytes,
adds `.s04_snapshot_identity` containing the exact executable SHA/tree and
source/request hashes, then removes all write bits. The launcher fails unless its
actual working directory and `SLURM_SUBMIT_DIR` both equal that immutable snapshot,
and the identity/source/request hashes and contents match. Python bytecode and
pytest cache writes are disabled; all temporary/output writes go to the unique
output root.

- Snapshot root (absent before submission; now preserved read-only):
  `/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/snapshots/s04_72184e9ed3d2_fp16remediation_v1`.
- Output root (absent before submission; now preserves Job `336718` artifacts):
  `/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s04_second_72184e9ed3d2_fp16remediation_v1`.
- Slurm logs:
  `/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/logs/s04_second_%j.{out,err}`.

Expected compact outputs are `execution_identity.json`,
`runtime_source_sha256s.txt`, `pytest.log`, `pytest.junit.xml`, and
`sha256sums.txt` with in-job `sha256sum -c`. No checkpoint, dataset, cache,
profile trace, or remote artifact is produced.

## Purpose and bounded input

Execute the same exact ten deterministic synthetic tests used by Job `335579`:

```bash
python -m pytest -q -ra -s -p no:cacheprovider \
  fl_v3/tests/test_s04_second_contract.py \
  fl_v3/tests/test_sparse_voxel_encoder.py \
  fl_v3/tests/test_s04_second_smoke.py \
  --junitxml="$S04_OUTPUT_ROOT/pytest.junit.xml"
```

They cover static `41x1440x1440 -> 2x180x180`/stride/RF/metric mapping and one
reduced-grid `dense()`; small real-spconv fp32/fp16 forward/backward, finite
gradients, per-sample caps/extreme occupancy, empty input, and sample/batch
isolation; and exactly one B=4 full-shape fp16 forward, scalar loss construction,
backward, finite intended gradients, and peak CUDA allocation/reservation capture
using 4,096 generated points per sample.

The job reads no nuScenes mini/trainval/ZIP/cache/checkpoint or external model
artifact. It performs no optimizer/GradScaler/parameter step, scheduler/EMA action,
model step, metric, profile, 100/1000-step gate, epoch, matrix, seed comparison,
DDP, or scientific execution.

## Resources and budget

- Account/partition: `naiss2025-22-1113-gpu` / `gpu`.
- Shared scheduling request: one `nvidia_gh200_120gb`, eight CPUs, one task;
  there is deliberately no `--nodes` or exclusive-node request.
- Walltime: `00:20:00`; maximum requested allocation `0.3334 GPU-hours`.
- One job, one concurrent S04 job; no array, DDP, retry, requeue, resubmission,
  spare-GPU expansion, or follow-on.
- Prior cumulative S04 elapsed GPU allocation is approximately `0.0409` hours;
  the maximum cumulative total after this request is approximately `0.3743`
  GPU-hours, within O-009's two-hour session ceiling.
- The launcher fails closed unless Slurm reports exactly one node, eight CPUs,
  generic GPU count one, typed GH200 count one, and Torch sees exactly one CUDA
  device. Allocation drift produces a negative result and no test execution.

## Exact preparation and submission command

The exact command below was approved and consumed once by Job `336718`. It is
retained only as provenance and must not be run again:

```bash
set -euo pipefail
S04_APPROVED_DELIVERY_SHA=<exact S00-approved 40-character delivery SHA>
S04_APPROVED_REQUEST_SHA256=<exact S00-approved request SHA-256>
S04_APPROVED_IDENTITY_SHA256=<exact S00-approved snapshot-identity SHA-256>
EXEC_SHA=2729f45144053e1b554a0bf04640b8bbc1ff43e4
EXEC_TREE=2fdb42c97995112b3defc7e78ea148daa6ee7786
EXEC_SOURCE_SHA256=a9b6fd7f6a5d72cc7691cb6118b001ac4221d6d5cffe4b6799d75ef32fa58c06
REQUEST=fl_v3/usenix27_orchestra/handoffs/S04/RUN_REQUEST.md
SNAPSHOT=/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/snapshots/s04_72184e9ed3d2_fp16remediation_v1
OUTPUT=/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s04_second_72184e9ed3d2_fp16remediation_v1
TMP_SNAPSHOT="${SNAPSHOT}.tmp"

test "$(git rev-parse HEAD)" = "${S04_APPROVED_DELIVERY_SHA}"
test "$(git branch --show-current)" = codex/s04-lidar-second
test -z "$(git status --short)"
test "$(sha256sum "${REQUEST}" | awk '{print $1}')" = "${S04_APPROVED_REQUEST_SHA256}"
test "$(git rev-parse "${EXEC_SHA}^{tree}")" = "${EXEC_TREE}"
test ! -e "${SNAPSHOT}"
test ! -e "${TMP_SNAPSHOT}"
test ! -e "${OUTPUT}"
mkdir -p "$(dirname "${SNAPSHOT}")"
mkdir "${TMP_SNAPSHOT}"
trap 'chmod -R u+w "${TMP_SNAPSHOT}" 2>/dev/null || true; rm -rf "${TMP_SNAPSHOT}"' EXIT
git archive "${EXEC_SHA}" | tar -xf - -C "${TMP_SNAPSHOT}"
install -m 0444 "${REQUEST}" "${TMP_SNAPSHOT}/${REQUEST}"
printf '%s\n' \
  'schema=s04.snapshot.v1' \
  "exec_sha=${EXEC_SHA}" \
  "exec_tree=${EXEC_TREE}" \
  "source_sha256=${EXEC_SOURCE_SHA256}" \
  "request_sha256=${S04_APPROVED_REQUEST_SHA256}" \
  > "${TMP_SNAPSHOT}/.s04_snapshot_identity"
test "$(sha256sum "${TMP_SNAPSHOT}/.s04_snapshot_identity" | awk '{print $1}')" = "${S04_APPROVED_IDENTITY_SHA256}"
find "${TMP_SNAPSHOT}" -type f -exec chmod 0444 {} +
chmod 0555 "${TMP_SNAPSHOT}/fl_v3/usenix27_orchestra/handoffs/S04/run_s04_second_smoke.sh"
find "${TMP_SNAPSHOT}" -type d -exec chmod 0555 {} +
mv "${TMP_SNAPSHOT}" "${SNAPSHOT}"
trap - EXIT

(
  cd "${SNAPSHOT}"
  sbatch --chdir="${SNAPSHOT}" \
    --export=ALL,EXPECTED_S04_SHA="${EXEC_SHA}",EXPECTED_S04_TREE="${EXEC_TREE}",EXPECTED_S04_SOURCE_HASH="${EXEC_SOURCE_SHA256}",EXPECTED_S04_REQUEST_HASH="${S04_APPROVED_REQUEST_SHA256}",EXPECTED_S04_IDENTITY_HASH="${S04_APPROVED_IDENTITY_SHA256}",S04_SNAPSHOT_ROOT="${SNAPSHOT}",S04_OUTPUT_ROOT="${OUTPUT}" \
    "${SNAPSHOT}/fl_v3/usenix27_orchestra/handoffs/S04/run_s04_second_smoke.sh"
)
```

The three `<exact ...>` values are not executable placeholders and must be replaced
verbatim by the S00-approved tuple. S00 returns a fully expanded, placeholder-free
command when approving. S04 must report the returned job ID and stop;
it must not retry or modify the snapshot/request after submission.

## Acceptance and stop conditions

Pass requires all of:

- immutable snapshot, exact executable/source/request identity, dependency
  versions, and exact one-GPU allocation/visibility checks pass;
- JUnit reports exactly 10 tests with zero failures, errors, or skips;
- fp32 reference output is fp32; active sparse-fp16 empty and non-empty output is
  fp16; debug evidence records the fp32 projection-to-fp16 interface cast;
- B=4 output/dense shapes are `[4,256,180,180]` and `[4,128,2,180,180]`;
- output, scalar loss, and every intended gradient are finite; no parameter update;
- per-sample caps/isolation, empty/over-cap, metric/shape/RF, composition, and
  no-fine-dense fixtures pass;
- peak allocated/reserved bytes are recorded and bounded by visible device memory;
- final artifact checksum verification passes.

Stop on any identity/allocation/import/build drift, output collision, OOM,
non-finite value, test-count drift, failure/error/skip, checksum failure, or timeout.
Preserve the negative. No automatic retry, modification, resubmission, or
follow-on is authorized.

## Interpretation boundary

Allowed only if passed: bounded synthetic GH200 evidence for this exact standalone
S04 module's geometry, fp16/fp32 output contract, finite forward/backward,
per-sample voxelization/isolation, and one B=4 peak-memory observation.

Forbidden regardless of outcome: S07-B/production/full-data readiness, mini or
trainval behavior, throughput/profile, convergence, mAP/NDS, fusion gain, best
voxel size, FL, attack/defense, generalization, or publication claims.

## Executed source-isolation matrix after Job 336718

### Source diagnosis and purpose

This is diagnostic-only and does not alter the S04 encoder, dtype assertions, or
precision policy. Source inspection of the installed spconv 2.3.8 path explains
the observed mixed-dtype call state:

1. S04 explicitly supplies fp16 sparse features when `sparse_conv_fp16=True`.
2. In training, `SparseConvolution._conv_forward(training=True)` uses
   `SparseImplicitGemmFunction`, whose Torch `custom_fwd(cast_inputs=float16)`
   casts both features and fp32 master filters to fp16. Job 336718 confirms this
   training path succeeds.
3. In eval, `_conv_forward(training=False)` bypasses that autograd function and
   calls `ops.implicit_gemm` directly. Features remain fp16, filters remain fp32,
   and spconv sets output dtype to the filter dtype, producing the observed
   fp16/fp32/fp32 tuner request.
4. `ConvTunerSimple` descriptor and cache keys include input/filter/output dtypes,
   channels, architecture and mask properties. A cached fp16/fp16/fp16 training
   kernel therefore cannot satisfy the fp16/fp32/fp32 eval key. Job 336718 reached
   an empty compatible-descriptor set and asserted before output.

The bounded matrix determines whether the failure is fresh-eval universal,
backward-dependent, occupancy-sensitive, or affected by process-local tuning
order. It records every `ops.implicit_gemm` feature/filter/output dtype,
train/submanifold state, activation count, output or exception.

### Immutable diagnostic implementation

- Exact diagnostic executable: `bd1fc9af139cce85240c5908d6704c38425f3c1f`.
- Exact executable tree: `80b6f5cf5028faffa67b7510454a510e94b72f31`.
- Repo runtime source SHA-256 over the 12 locale-sorted launcher files:
  `d2a5041c5177279f874bd788320053df679c5b8ad060f95d729e29ae0ebfbf63`.
- Installed spconv source aggregate SHA-256 over `conv.py`, `functional.py`,
  `ops.py`, and the generated tuner availability/cache/tuning sources:
  `e7e162a1f10b4e66c42c1bc07fae19248c42a5e198fbee2c546f3dc0a0d43141`.
- Diagnostic script SHA-256:
  `381d16b22a224988230a1a03cadd00847fb5f725de80280df4628d28af8c01da`.
- Launcher SHA-256:
  `22ccbac6f63c8fccf4725f73b44f81445ee563135e358535ad7964caaac483dc`.
- Delivery/request/identity hashes are supplied externally by S00 because the
  request cannot contain its own hash or enclosing delivery SHA.

Any source, dependency source, request, cell, command, resource, or root change
invalidates approval.

### Exact seven subprocess-isolated cells

Each cell runs in a fresh Python process; one cell error cannot hide later cells.
The matrix is complete when all seven return a structured `success` or `error`
envelope. An observed cell error is diagnostic evidence, not by itself a launcher
failure or an S04 verdict.

1. fresh fp16 eval, exact six-voxel Job-336718 input;
2. same fp16 model train forward then eval, without backward;
3. same fp16 model train forward/backward then eval;
4. fresh fp16 eval with 128 generated points per sample;
5. fresh fp32 eval on the six-voxel control;
6. fp32 eval followed by fresh fp16 eval in one process;
7. fp16 train warm-up followed by a distinct fresh fp16 eval model in one process.

No workaround or fallback is exercised. In particular, the matrix does not cast
model parameters to half, force training mode for inference, disable fp16 sparse
inference, patch spconv, or weaken the eval case.

### Resources, input, and outputs

- Account/partition: `naiss2025-22-1113-gpu` / `gpu`.
- Shared request: one `nvidia_gh200_120gb`, eight CPUs, one task, no explicit
  `--nodes` or exclusive allocation; walltime `00:20:00`.
- Maximum `0.3334` GPU-hours. Prior cumulative S04 elapsed allocation is about
  `0.0890` GPU-hours; maximum cumulative total would be about `0.4224`, within
  O-009's two-hour session limit.
- One job, no array/DDP/retry/requeue/resubmission/follow-on.
- Deterministic synthetic points only; no mini/trainval/ZIP/cache/checkpoint,
  optimizer/parameter update, model step, metric, profile, seed comparison, or
  scientific run.
- Per-cell timeout 120 seconds; cells are sequential and process-isolated.
- Snapshot root, absent before submission and now preserved read-only:
  `/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/snapshots/s04_lifecycle_bd1fc9af139c_v1`.
- Output root, absent before submission and now preserving Job `336728` artifacts:
  `/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s04_lifecycle_bd1fc9af139c_v1`.
- Logs: `/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/logs/s04_lifecycle_%j.{out,err}`.
- Artifacts: execution identity, repo/dependency source lists, complete matrix JSON,
  diagnostic log, and checksum manifest with `sha256sum -c`.

### Exact preparation/submission template — consumed, not reusable

S00 supplied the three exact external hashes and the fully expanded form of this
command was consumed once by Job `336728`. The template is retained as provenance
and must not be reused:

```bash
set -euo pipefail
S04_DIAG_APPROVED_DELIVERY_SHA=<exact S00-approved delivery SHA>
S04_DIAG_APPROVED_REQUEST_SHA256=<exact S00-approved request SHA-256>
S04_DIAG_APPROVED_IDENTITY_SHA256=<exact S00-approved identity SHA-256>
EXEC_SHA=bd1fc9af139cce85240c5908d6704c38425f3c1f
EXEC_TREE=80b6f5cf5028faffa67b7510454a510e94b72f31
SOURCE_SHA=d2a5041c5177279f874bd788320053df679c5b8ad060f95d729e29ae0ebfbf63
DEP_SOURCE_SHA=e7e162a1f10b4e66c42c1bc07fae19248c42a5e198fbee2c546f3dc0a0d43141
REQUEST=fl_v3/usenix27_orchestra/handoffs/S04/RUN_REQUEST.md
SNAPSHOT=/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/snapshots/s04_lifecycle_bd1fc9af139c_v1
OUTPUT=/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s04_lifecycle_bd1fc9af139c_v1
TMP_SNAPSHOT="${SNAPSHOT}.tmp"

test "$(git rev-parse HEAD)" = "${S04_DIAG_APPROVED_DELIVERY_SHA}"
test "$(git branch --show-current)" = codex/s04-lidar-second
test -z "$(git status --short)"
test "$(sha256sum "${REQUEST}" | awk '{print $1}')" = "${S04_DIAG_APPROVED_REQUEST_SHA256}"
test "$(git rev-parse "${EXEC_SHA}^{tree}")" = "${EXEC_TREE}"
test ! -e "${SNAPSHOT}"
test ! -e "${TMP_SNAPSHOT}"
test ! -e "${OUTPUT}"
mkdir -p "$(dirname "${SNAPSHOT}")"
mkdir "${TMP_SNAPSHOT}"
trap 'chmod -R u+w "${TMP_SNAPSHOT}" 2>/dev/null || true; rm -rf "${TMP_SNAPSHOT}"' EXIT
git archive "${EXEC_SHA}" | tar -xf - -C "${TMP_SNAPSHOT}"
install -m 0444 "${REQUEST}" "${TMP_SNAPSHOT}/${REQUEST}"
printf '%s\n' \
  'schema=s04.lifecycle-snapshot.v1' \
  "exec_sha=${EXEC_SHA}" \
  "exec_tree=${EXEC_TREE}" \
  "source_sha256=${SOURCE_SHA}" \
  "dependency_source_sha256=${DEP_SOURCE_SHA}" \
  "request_sha256=${S04_DIAG_APPROVED_REQUEST_SHA256}" \
  > "${TMP_SNAPSHOT}/.s04_lifecycle_snapshot_identity"
test "$(sha256sum "${TMP_SNAPSHOT}/.s04_lifecycle_snapshot_identity" | awk '{print $1}')" = "${S04_DIAG_APPROVED_IDENTITY_SHA256}"
find "${TMP_SNAPSHOT}" -type f -exec chmod 0444 {} +
chmod 0555 "${TMP_SNAPSHOT}/fl_v3/usenix27_orchestra/handoffs/S04/run_s04_spconv_lifecycle_diagnostic.sh"
find "${TMP_SNAPSHOT}" -type d -exec chmod 0555 {} +
mv "${TMP_SNAPSHOT}" "${SNAPSHOT}"
trap - EXIT
(
  cd "${SNAPSHOT}"
  sbatch --chdir="${SNAPSHOT}" \
    --export=ALL,EXPECTED_S04_DIAG_SHA="${EXEC_SHA}",EXPECTED_S04_DIAG_TREE="${EXEC_TREE}",EXPECTED_S04_DIAG_SOURCE_HASH="${SOURCE_SHA}",EXPECTED_S04_DIAG_DEP_SOURCE_HASH="${DEP_SOURCE_SHA}",EXPECTED_S04_DIAG_REQUEST_HASH="${S04_DIAG_APPROVED_REQUEST_SHA256}",EXPECTED_S04_DIAG_IDENTITY_HASH="${S04_DIAG_APPROVED_IDENTITY_SHA256}",S04_DIAG_SNAPSHOT_ROOT="${SNAPSHOT}",S04_DIAG_OUTPUT_ROOT="${OUTPUT}" \
    "${SNAPSHOT}/fl_v3/usenix27_orchestra/handoffs/S04/run_s04_spconv_lifecycle_diagnostic.sh"
)
```

### Acceptance and interpretation

Diagnostic completion requires exact immutable identities, one-GPU allocation,
all seven subprocess envelopes in fixed order, no subprocess timeout/crash,
installed dependency versions/source hashes, and artifact checksum verification.
Individual structured `error` outcomes are accepted observations. The launcher
must fail on missing/malformed cells, identity drift, wrong allocation, timeout,
output collision, matrix validation failure, or checksum failure.

Job `336728` met diagnostic-completion acceptance: all seven envelopes, identities,
allocation checks, source/dependency hashes and artifact checksums passed. Six
fp16-eval cells returned structured errors and the fp32 control succeeded. Its
scheduler `COMPLETED` result means only that the diagnostic matrix is complete. It
does not establish S04 PASS, select a workaround, or authorize a precision change,
retry, production integration, full-data run, or scientific claim.
