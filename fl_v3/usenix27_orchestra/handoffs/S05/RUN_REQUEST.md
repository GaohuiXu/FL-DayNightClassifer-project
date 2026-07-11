# S05-R2 RUN_REQUEST — exact synthetic CenterHead re-review runtime

## Approval state

- **Status:** `EXECUTED_ONCE_FAILED_NO_RETRY_AUTHORIZED`.
- This request was created because the x86_64 login interpreter has no `torch`,
  `pytest`, or `numpy`; at approval time all 44 authored S05 cases were `NOT RUN`.
- The requested job is a dependency/runtime engineering check only. It uses no
  dataset, model checkpoint, optimizer or parameter update, training step,
  scientific metric, profile, seed matrix, array, DDP, retry, or follow-on.
- S00 approved the exact immutable tuple once under O-009 and the owner's
  temporary S02-S05 validation authority. That approval was consumed by Job
  `336731`; it does not authorize retry, requeue, resubmission, remediation, or a
  follow-on job.

## Immutable source identity

- Reviewed worker/delivery SHA:
  `705216de097ae9eeb1813de6dcdc916e2844fcde`.
- Worker tree: `2d5cd99c004e3ebd83a748f84141c03739e8fd4b`.
- Worker branch which must still resolve to that exact commit at submission:
  `codex/s05-centerhead-decode`.
- Original reviewed delivery: `4561d3ef4d5dd1dcbfe71fdf0ca1eb38d61257d9`.
- Remediation implementation: `753944c199ceeace160732218f1b16dfdd15ac21`.
- Runtime source-list SHA-256 (31 C-locale-sorted paths):
  `bea19dd528010020a462b18cfaeedd2642fd0e0a147ac458e215bdb8718b1857`.
- Runtime source-state SHA-256 (SHA-256 of the 31 `sha256sum` lines):
  `2ff6389f0a556663e0cd2284c76c9fa11741bb0f44adb28eda4aebd33765c766`.
- Runtime source includes all selected tests; head/decode/NMS/BEV/eval sources;
  the complete eagerly imported `data/nuscenes/*.py` package; package
  initializers; environment bootstrap; and dependency/config manifests.
  `tests/conftest.py` is intentionally excluded because the exact invocation uses
  `--noconftest` and none of the selected cases needs its data fixtures.
- No working-tree path is executed. The launcher exports the exact Git object into
  a new immutable `/nobackup` snapshot, recomputes both source hashes there, and
  fails before environment activation on any mismatch.

## Exact bounded scope

The job runs exactly these four files and must collect exactly 44 cases:

1. `fl_v3/tests/test_head_capacity.py` — 6 cases;
2. `fl_v3/tests/test_s05_centerhead_decode.py` — 9 cases;
3. `fl_v3/tests/test_s05_nms.py` — 22 cases after parametrization;
4. `fl_v3/tests/test_s05_eval_roundtrip.py` — 7 cases.

The cases cover the six-task/GN topology; B=1/B>1 isolation; explicit name-to-
devkit ID mapping; per-class K=500/no second task K; fp16 adjacent-logit and strict
0.1 forced-FP32 boundaries; FP32 score/velocity output; canonical box/yaw/velocity
round trip; circle/rotate NMS geometry, units, budgets, input permutation and
cross-class suppression; submission conversion/order including the equal-score,
same-class, same-geometry velocity/attribute collision; invalid inputs; duplicate
samples; and the official 500-box cap.

## Resources and paths

- Account: `naiss2025-22-1113-gpu`.
- Partition: `gpu`.
- Nodes: scheduler-selected single node; the submission deliberately omits
  `--nodes=1` so it does not request an exclusive whole GH200 node.
- GPU: exactly one `nvidia_gh200_120gb`, allocation and CUDA visibility both
  verified fail-closed inside the batch step.
- CPUs: 8.
- Walltime: `00:15:00` (maximum 0.25 GPU-hours).
- Concurrency: one S05-R2 job maximum; no array, DDP, retry, resubmission, or
  follow-on.
- Common Git object store (read-only input):
  `/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/fl_weather_project/.git`.
- Fresh immutable snapshot root:
  `/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/execution_snapshots/s05r2_centerhead_705216de097a`.
- Fresh output root:
  `/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s05r2_centerhead_705216de097a`.
- Slurm logs:
  `/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/logs/s05r2_centerhead_%j.{out,err}`.
- Durable committed launcher (submitted directly; Slurm executes its spool copy):
  `fl_v3/usenix27_orchestra/handoffs/S05/run_s05r2_centerhead.sh`.
- S00-provisioned immutable copy of this exact request:
  `/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/launchers/s05r2_RUN_REQUEST_705216de097a.md`.

S00 must independently hash the committed launcher, make only the exact request copy
read-only under `/nobackup`, verify both fresh roots are absent and no
`flv3_s05r2_centerhead` job is queued/running, and bind the literal launcher/request
hashes in its approval. Slurm receives the committed launcher path directly and
executes its spool copy; S05-R2 does not synthesize launcher bytes from Markdown.

## Exact durable launcher

- Repository path:
  `fl_v3/usenix27_orchestra/handoffs/S05/run_s05r2_centerhead.sh`.
- SHA-256:
  `7ea5e8128fac4ddb471c27030b2d18b7e133297fca6a50fb336f27ee007a9e10`.
- `bash -n`: PASS.
- The launcher verifies the hash of its Slurm spool copy against the approved
  literal, exports worker `705216d` into the immutable snapshot, verifies the
  31-file worker-source closure, records allocation/runtime/JUnit evidence, and
  runs in-job `sha256sum -c`.

S00 must independently recompute this committed file hash before approval. Any
launcher-byte change invalidates the request.

## Exact staging/submission form after approval

The request SHA cannot be embedded literally in itself without a self-hash cycle.
The command below derives it from the clean committed request bytes, stages only
that exact request copy read-only under `/nobackup`, and passes the derived value to
the self-verifying committed launcher. S00 approval must still state the same
literal request SHA and review commit before this command may run.

```bash
REVIEW_ROOT=/home/gaohui/.codex/worktrees/s05r2/fl_weather_project
REQUEST_SOURCE="$REVIEW_ROOT/fl_v3/usenix27_orchestra/handoffs/S05/RUN_REQUEST.md"
LAUNCHER="$REVIEW_ROOT/fl_v3/usenix27_orchestra/handoffs/S05/run_s05r2_centerhead.sh"
REQUEST_COPY=/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/launchers/s05r2_RUN_REQUEST_705216de097a.md
LAUNCHER_SHA=7ea5e8128fac4ddb471c27030b2d18b7e133297fca6a50fb336f27ee007a9e10
REQUEST_SHA="$(sha256sum "$REQUEST_SOURCE" | awk '{print $1}')"
export LAUNCHER_SHA REQUEST_SHA
: "${S05R2_APPROVED_REVIEW_SHA:?S00-approved exact S05-R2 review delivery SHA is required}"

test "$(git -C "$REVIEW_ROOT" branch --show-current)" = codex/s05-r2-centerhead-review && \
test "$(git -C "$REVIEW_ROOT" rev-parse HEAD)" = "$S05R2_APPROVED_REVIEW_SHA" && \
test -z "$(git -C "$REVIEW_ROOT" status --short)" && \
test "$(sha256sum "$LAUNCHER" | awk '{print $1}')" = "$LAUNCHER_SHA" && \
test ! -e "$REQUEST_COPY" && \
test ! -e /nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/execution_snapshots/s05r2_centerhead_705216de097a && \
test ! -e /nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s05r2_centerhead_705216de097a && \
test -z "$(squeue -u "$USER" -h -o '%j' | awk '$1 == "flv3_s05r2_centerhead"')" && \
mkdir -p "$(dirname "$REQUEST_COPY")" && \
install -m 0444 "$REQUEST_SOURCE" "$REQUEST_COPY" && \
test "$(sha256sum "$REQUEST_COPY" | awk '{print $1}')" = "$REQUEST_SHA" && \
sbatch --export=ALL,EXPECTED_S05R2_SHA=705216de097ae9eeb1813de6dcdc916e2844fcde,EXPECTED_S05R2_TREE=2d5cd99c004e3ebd83a748f84141c03739e8fd4b,EXPECTED_S05R2_BRANCH=codex/s05-centerhead-decode,EXPECTED_S05R2_SOURCE_LIST_SHA=bea19dd528010020a462b18cfaeedd2642fd0e0a147ac458e215bdb8718b1857,EXPECTED_S05R2_SOURCE_SHA=2ff6389f0a556663e0cd2284c76c9fa11741bb0f44adb28eda4aebd33765c766,EXPECTED_S05R2_LAUNCHER_SHA="$LAUNCHER_SHA",EXPECTED_S05R2_RUN_REQUEST_SHA="$REQUEST_SHA",S05R2_REQUEST_COPY="$REQUEST_COPY",S05R2_SNAPSHOT_ROOT=/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/execution_snapshots/s05r2_centerhead_705216de097a,S05R2_OUTPUT_ROOT=/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s05r2_centerhead_705216de097a \
  "$LAUNCHER"
```

This exact block was run once only after S00's bound approval; the execution record
below consumes that approval. It must not be run again.

## Execution record — approval consumed

- The exact staging/submission block above was executed once with
  `S05R2_APPROVED_REVIEW_SHA=61e7fb14bc6f44fe681628a1fb0ed701ad4f7f28`.
- Read-only request-copy staging succeeded with executed request SHA-256
  `bcd8f426e5b95438f91973e9a3d9712193cf96a23f9254732114111fb68019c1`.
- Slurm accepted Job `336731`; no other S05-R2 job was submitted.
- Terminal state / exit / elapsed: `FAILED` / `1:0` / `00:01:15`.
- Exact allocation: `n570`, one node, eight CPUs, one
  `nvidia_gh200_120gb`, `OverSubscribe=OK`; batch MaxRSS `504M`.
- JUnit: 44 tests, one failure, zero errors, zero skips; pytest summary
  `1 failed, 43 passed in 22.88s`.
- The sole failure occurred after `forward == reverse` passed. The fixture then
  compared actual in-memory velocity tuples against expected lists. This is a
  test-fixture representation mismatch; it is not an observed submission-order or
  content defect. The zero-failure acceptance criterion nevertheless failed.
- All nine generated execution artifacts passed the launcher's in-job
  `sha256sum -c`; exact paths and hashes are recorded in `RESULTS.md`.
- No retry/requeue/follow-on is authorized. A test-only worker remediation, new
  durable SHA, independent review, and separately approved request are required
  before another runtime attempt.

## Acceptance and stop conditions

PASS requires all of the following:

- exact worker branch/SHA/tree, launcher/request hashes, source-list hash, and
  source-state hash;
- a fresh read-only `/nobackup` snapshot and fresh output root;
- scheduler allocation of exactly one node, one GH200 and eight CPUs, plus exactly
  one CUDA-visible GPU after environment activation;
- actual dependency versions and allocation identity recorded;
- exactly 44 tests, zero failures, errors, and skips;
- generated artifact checksums and a successful in-job `sha256sum -c`.

Stop/fail immediately on any identity/hash/allocation/output collision, missing
dependency, collection mismatch, failure/error/skip, exception, or walltime. There
is no automatic retry. Any negative result is preserved and returned to S00 before
any new request.

## Interpretation boundary

An accepted PASS can close the S05 authored synthetic runtime gate and support the
independent S05-R2 source verdict. It cannot establish production detector/loss/
config integration, official CUDA-kernel parity, CPU NMS performance, mini or
trainval model quality, mAP/NDS, full-run readiness, FL/security behavior, or any
scientific/publication claim.
