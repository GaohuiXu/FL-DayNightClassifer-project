# S04 RUN REQUEST — final-output dtype remediation validation

## Approval state

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
