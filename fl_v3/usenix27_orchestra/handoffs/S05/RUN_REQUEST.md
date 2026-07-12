# S05 test-only remediation RUN_REQUEST — exact 44-case synthetic rerun

## Approval state and preserved negative

- **Status:** `EXECUTED_PASS_JOB_336738_NO_FOLLOW_ON`.
- Prior authorized S05-R2 Job `336731` is preserved as **FAILED 1:0**:
  exactly 44 cases collected, 43 passed, one failed, zero errors/skips. The sole
  failure was the expected container type in
  `test_submission_duplicate_geometry_orders_velocity_and_attribute_by_content`:
  the crucial `forward == reverse` assertion passed and the actual serialized
  velocity was the stable devkit tuple `(vx, vy)`, while the test expected a list.
  No production/source defect was observed.
- This request repeats exactly the same 44 synthetic cases after the two-token
  test-only expectation correction. It adds no test, resource, data, model step,
  metric, profile, seed, matrix, retry loop, or follow-on.
- **Approval class:** one-time focused rerun under the owner's explicit temporary
  delegation allowing S00 to approve necessary, reasonable validation-only Slurm
  jobs for S02-S05. This is **not O-009**: O-009 excludes reruns, and this request
  does not expand or reinterpret it.
- Before execution, preparing/committing this file and launcher did not grant
  permission. S05 waited for S00 to approve the exact immutable tuple below under
  the delegated validation authority. That approval was consumed by Job 336738
  and grants no follow-on.

## Exact approval and execution result

S00 approved this one-time request under the owner-delegated S02-S05 validation
authority. The approved/executed request bytes had SHA-256
`e4cb396bc550f08e92905903135f9ab0841ba1bd498f661ba731587a843a10b9`;
the launcher had SHA-256
`b86271e81ec41443232afab6a6ada5d1dbebfa72027946cea6547ee5c01598e5`;
the approved delivery was `98b71eca7684b50ece69afc36175564c7c283033`.

- Submission: exactly once; Job `336738`.
- Terminal: **COMPLETED 0:0**, node `n411`, elapsed `00:01:13`, batch MaxRSS
  `540M`.
- Allocation: one shared node, eight CPUs, exactly one GH200.
- Pytest/JUnit: `44 passed in 22.64s`; exactly 44 tests, zero
  failures/errors/skips, JUnit time `22.645s`.
- All nine in-job checksum targets passed `sha256sum -c`.
- No dataset, model/optimizer step, scientific metric, profile, array, DDP,
  automatic retry, or follow-on occurred.
- Prior Job `336731` remains preserved as the 43/44 negative; this PASS does not
  erase or relabel it.

Updating this record after execution grants no follow-on permission.

## Immutable execution source

- Test-only execution SHA:
  `96e509b71a3e22afb4de397132438fd3b9bbf5d8`.
- Execution tree: `aeaaad044199492b81c4383a013f3fb3c6596c02`.
- Parent reviewed/remediated worker delivery:
  `705216de097ae9eeb1813de6dcdc916e2844fcde`.
- Test-only diff SHA-256 (`705216d..96e509b`):
  `aed0033a6843212557b14bc0b950006e3b791cd2a75afb7fd5d40938e79fc700`.
- Corrected test file SHA-256:
  `e938dd34656e3ae5f5e9019748bea52a3ccc5cb99144492d6bf9f45e79c203c0`.
- Runtime source-list SHA-256 (31 C-locale-sorted paths):
  `bea19dd528010020a462b18cfaeedd2642fd0e0a147ac458e215bdb8718b1857`.
- Runtime source-state SHA-256 (SHA-256 of the 31 `sha256sum` lines):
  `7ac7ea66485b319672e9b975ffcd38caa2c607f8932d1ca2acc2a9c5159823b1`.
- The launcher exports the exact execution Git object into a fresh read-only
  `/nobackup` snapshot and recomputes both source hashes before environment
  activation. It never executes mutable working-tree source.
- `tests/conftest.py` remains intentionally excluded because the exact invocation
  uses `--noconftest` and none of the selected synthetic cases needs data fixtures.

## Exact unchanged test scope

The job runs exactly these four files and must collect exactly 44 cases:

1. `fl_v3/tests/test_head_capacity.py` — 6 cases;
2. `fl_v3/tests/test_s05_centerhead_decode.py` — 9 cases;
3. `fl_v3/tests/test_s05_nms.py` — 22 cases;
4. `fl_v3/tests/test_s05_eval_roundtrip.py` — 7 cases.

There is no dataset access, checkpoint, optimizer/parameter update, train/eval
model step, scientific metric, profiling, array, DDP, resubmission, or automatic
retry.

## Exact resources and fresh paths

- Account: `naiss2025-22-1113-gpu`.
- Partition: `gpu`.
- Nodes: scheduler-selected one shared node; `--nodes=1` is deliberately omitted.
- GPU: exactly one `nvidia_gh200_120gb`, allocation and CUDA visibility both
  verified fail-closed in the batch step.
- CPUs: 8.
- Walltime: `00:15:00` (maximum 0.25 GPU-hours).
- Concurrency: one S05-R3 job maximum; no array or follow-on.
- Common Git object store:
  `/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/fl_weather_project/.git`.
- Fresh snapshot root:
  `/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/execution_snapshots/s05r3_centerhead_96e509b71a3e`.
- Fresh output root:
  `/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s05r3_centerhead_96e509b71a3e`.
- Logs:
  `/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/logs/s05r3_centerhead_%j.{out,err}`.
- Immutable request copy:
  `/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/launchers/s05r3_RUN_REQUEST_96e509b71a3e.md`.

## Exact launcher

- Repository path:
  `fl_v3/usenix27_orchestra/handoffs/S05/run_s05r3_centerhead.sh`.
- SHA-256:
  `b86271e81ec41443232afab6a6ada5d1dbebfa72027946cea6547ee5c01598e5`.
- `bash -n`: PASS.
- The launcher checks its Slurm spool-copy hash, the immutable request-copy hash,
  execution SHA/tree, source closure, fresh roots, one-node/one-GH200/eight-CPU
  allocation, one visible GPU, exact JUnit counts, and final `sha256sum -c`.

Any launcher-byte change invalidates this request.

## Exact staging/submission form after S00 approval

The request cannot embed its own SHA without a self-hash cycle. S00 must approve
the final clean delivery SHA, this file's independently computed SHA-256, the
launcher SHA above, and approval class
`S00_OWNER_DELEGATED_S02_S05_VALIDATION_RERUN`. Only then may the following form
run:

```bash
WORKER_ROOT=/home/gaohui/.codex/worktrees/63f8/fl_weather_project
REQUEST_SOURCE="$WORKER_ROOT/fl_v3/usenix27_orchestra/handoffs/S05/RUN_REQUEST.md"
LAUNCHER="$WORKER_ROOT/fl_v3/usenix27_orchestra/handoffs/S05/run_s05r3_centerhead.sh"
REQUEST_COPY=/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/launchers/s05r3_RUN_REQUEST_96e509b71a3e.md
LAUNCHER_SHA=b86271e81ec41443232afab6a6ada5d1dbebfa72027946cea6547ee5c01598e5
REQUEST_SHA="$(sha256sum "$REQUEST_SOURCE" | awk '{print $1}')"
export LAUNCHER_SHA REQUEST_SHA
: "${S05R3_S00_DELEGATED_VALIDATION_APPROVED_DELIVERY_SHA:?exact S00 delegated-validation approval is required}"

test "$(git -C "$WORKER_ROOT" branch --show-current)" = codex/s05-centerhead-decode && \
test "$(git -C "$WORKER_ROOT" rev-parse HEAD)" = "$S05R3_S00_DELEGATED_VALIDATION_APPROVED_DELIVERY_SHA" && \
test -z "$(git -C "$WORKER_ROOT" status --short)" && \
test "$(sha256sum "$LAUNCHER" | awk '{print $1}')" = "$LAUNCHER_SHA" && \
test ! -e "$REQUEST_COPY" && \
test ! -e /nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/execution_snapshots/s05r3_centerhead_96e509b71a3e && \
test ! -e /nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s05r3_centerhead_96e509b71a3e && \
test -z "$(squeue -u "$USER" -h -o '%j' | awk '$1 == "flv3_s05r3_centerhead"')" && \
mkdir -p "$(dirname "$REQUEST_COPY")" && \
install -m 0444 "$REQUEST_SOURCE" "$REQUEST_COPY" && \
test "$(sha256sum "$REQUEST_COPY" | awk '{print $1}')" = "$REQUEST_SHA" && \
sbatch --export=ALL,EXPECTED_S05R3_SHA=96e509b71a3e22afb4de397132438fd3b9bbf5d8,EXPECTED_S05R3_TREE=aeaaad044199492b81c4383a013f3fb3c6596c02,EXPECTED_S05R3_SOURCE_LIST_SHA=bea19dd528010020a462b18cfaeedd2642fd0e0a147ac458e215bdb8718b1857,EXPECTED_S05R3_SOURCE_SHA=7ac7ea66485b319672e9b975ffcd38caa2c607f8932d1ca2acc2a9c5159823b1,EXPECTED_S05R3_LAUNCHER_SHA="$LAUNCHER_SHA",EXPECTED_S05R3_RUN_REQUEST_SHA="$REQUEST_SHA",EXPECTED_S05R3_APPROVAL_CLASS=S00_OWNER_DELEGATED_S02_S05_VALIDATION_RERUN,S05R3_REQUEST_COPY="$REQUEST_COPY",S05R3_SNAPSHOT_ROOT=/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/execution_snapshots/s05r3_centerhead_96e509b71a3e,S05R3_OUTPUT_ROOT=/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s05r3_centerhead_96e509b71a3e \
  "$LAUNCHER"
```

This exact staging/submission form ran once and produced Job `336738`. It must not
run again. The read-only staged request copy remains at the declared path.

## Acceptance and stop conditions

PASS requires exact immutable identities; one node/GH200/eight CPUs; exactly 44
tests with zero failures/errors/skips; recorded dependency/allocation identity;
checksummed logs/JUnit/source manifests; and successful in-job `sha256sum -c`.

Stop/fail on any identity/hash/allocation/output collision, missing dependency,
collection mismatch, test failure/error/skip, exception, or walltime. There is no
automatic retry.

## Interpretation boundary

Job 336738 PASS may close only the S05 authored synthetic runtime gate after the stable tuple
test correction. It cannot establish production detector/loss/config integration,
official CUDA-kernel parity, CPU NMS performance, mini/trainval model quality,
mAP/NDS, full-run readiness, FL/security behavior, or any scientific claim.
