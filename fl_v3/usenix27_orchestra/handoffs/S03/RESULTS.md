# S03 RESULTS — camera architecture focused validation

## Overall result

**SCHEDULER-REMEDIATION JOB FAILED IN LAUNCHER PROVENANCE PREFLIGHT; RUNTIME
GATES NOT EXECUTED.**

The exact-once O-009 request was approved by S00 and invoked once.  Slurm rejected
it during submission with an account/partition error.  No job, allocation, output,
log, JUnit, execution identity, runtime source record, or model/runtime evidence was
created.  A separately approved scheduler remediation then created job `335630`,
which failed in six seconds before output creation because the launcher attempted
Git discovery from Slurm's spool-script directory.  Environment activation, CUDA,
pytest, and camera code were never reached.  The implementation and synthetic
tests remain available for independent review, but S03 does not claim a runtime
PASS.

## First approved identity and rejected attempt

- Approved executable HEAD:
  `871db182c5fdcdda46e242d911ac9dcbf393683a`.
- Implementation:
  `6dfd2c775f54e488f3930996b303ce21f9b8e8b7`.
- Branch: `codex/s03-camera-architecture`.
- Approved RUN_REQUEST SHA-256:
  `bd33d91f65558ba97a4ab24f80783133349a8d39b6efa02c925ebd238525b547`.
- Approved launcher SHA-256:
  `9473b830776d478c14c55bcb4991bed329a8273cab6c04bbc8681649f33addfc`.
- Approved source-list SHA-256:
  `d4eb8d29da926c88bbcf5c9bbbf9b3e9197f9eda4478ea956ec4c7cfaf664742`.
- Approved source-state SHA-256:
  `71b0c708325548ad9d09e68e41a0f225bc81f741fd9b4924938317ed591b5b9f`.
- Attempt time: 2026-07-11 approximately 18:17 Europe/Stockholm.
- Preflight: exact HEAD/branch/clean status and request/launcher hashes matched;
  output was absent and no active S03 job existed.
- Submission count: exactly one.
- Exact approved command invoked:

```bash
sbatch --export=ALL,EXPECTED_S03_EXECUTABLE_SHA=871db182c5fdcdda46e242d911ac9dcbf393683a,EXPECTED_S03_IMPLEMENTATION_SHA=6dfd2c775f54e488f3930996b303ce21f9b8e8b7,EXPECTED_S03_BRANCH=codex/s03-camera-architecture,EXPECTED_S03_SOURCE_LIST_SHA=d4eb8d29da926c88bbcf5c9bbbf9b3e9197f9eda4478ea956ec4c7cfaf664742,EXPECTED_S03_SOURCE_SHA=71b0c708325548ad9d09e68e41a0f225bc81f741fd9b4924938317ed591b5b9f,EXPECTED_S03_LAUNCHER_SHA=9473b830776d478c14c55bcb4991bed329a8273cab6c04bbc8681649f33addfc,EXPECTED_S03_RUN_REQUEST_SHA=bd33d91f65558ba97a4ab24f80783133349a8d39b6efa02c925ebd238525b547,S03_OUTPUT_ROOT=/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s03_camera_contract_6dfd2c775f54 fl_v3/usenix27_orchestra/handoffs/S03/run_s03_camera_contract.sh
```

- Client-side Slurm result:

```text
sbatch: error: Batch job submission failed: Invalid account or account/partition combination specified
```

## First attempt scheduler and resource reconciliation

| Field | Result |
|---|---|
| Job ID | none returned |
| `squeue -n flv3_s03_camera_contract` | empty before and after attempt |
| `sacct --name flv3_s03_camera_contract` | no record |
| State / exit code | not applicable; job was never created |
| Node / GPU | none allocated |
| Actual GPU-hours | `0` |
| Retry/requeue/resubmit | none |
| Output root | absent |
| Slurm stdout/stderr files | absent |
| Runtime artifacts/checksums | absent |

Approved but unused resource ceiling was one node, one GH200, eight CPUs,
`00:15:00`, at most 0.25 GPU-hours.

## First attempt failure diagnosis

The committed S03 launcher declares node/GPU/CPU/time/log directives but no Slurm
account or partition.  A read-only repository audit found the active Arrhenius GPU
launchers consistently declare:

```text
#SBATCH -A naiss2025-22-1113-gpu
#SBATCH -p gpu
```

Examples include `run_s07a_provenance_tests.sh`,
`run_s01_nuscenes_zip_tests.sh`, `run_arrhenius_smoke.sh`, and
`run_arrhenius_mini_matrix.sh`.  This supports the missing account/partition as the
direct submission failure, not a test, CUDA, source-attestation, or model failure.

The consumed exact-once authorization did not cover a corrected request.  Under a
subsequent scoped S00 decision, S03 prepared a new candidate whose only execution
changes were the two directives above and a fresh output root.  Camera code, tests,
resources, provenance closure, and acceptance remained unchanged.  S00 separately
reviewed and approved that tuple; it was not an automatic retry or resubmission.

## Scheduler-remediation execution

- Account: `naiss2025-22-1113-gpu`.
- Partition: `gpu`.
- Implementation:
  `6dfd2c775f54e488f3930996b303ce21f9b8e8b7` (unchanged).
- Source-list SHA-256:
  `d4eb8d29da926c88bbcf5c9bbbf9b3e9197f9eda4478ea956ec4c7cfaf664742`.
- Source-state SHA-256:
  `6163d27c7f264902a1ac7688b4a13a704d2b98fc6597ca39c0da8b2a115157c1`.
- Launcher SHA-256:
  `d6f236d35f290b4552f3c3e93bb2d92438481100c8fa7726812ea0d658d12983`.
- New output root:
  `/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s03_camera_contract_schedfix_6dfd2c775f54`.
- Executable HEAD: `ddadd2ec8423e4d68fd434abf0554a7a2eb1377d`.
- Tree: `2be4655b8533a83251e95435a9d10fa43dfd6a11`.
- Approved RUN_REQUEST SHA-256:
  `d248d8f40f5ed917ec05cbd4681c36d2cfd56a43e25cf96775d47d78cff763f5`.
- Submission count under this remediation: exactly `1`.
- Job ID: `335630`.
- State / exit: `FAILED` / `1:0`.
- Node / elapsed: `n32` / six seconds.
- Retry/requeue/resubmit/follow-on: none.

Exact terminal error:

```text
fatal: not a git repository (or any of the parent directories): .git
```

The Slurm spool executes a copied script, so `BASH_SOURCE[0]` did not resolve to
the committed launcher path.  The launcher's first repository lookup used
`git -C "$SCRIPT_DIR"` and failed.  The unique output root remained absent, hence
no `execution_identity.json`, runtime source list/state, approved-source copies,
JUnit, pytest log, or test summary was created.  This is a provenance-launcher
failure, not a CUDA, pytest, projection, gradient, or model failure.

### Job `335630` resource reconciliation

| Field | Recorded value |
|---|---|
| Requested TRES | 1 GH200, 8 CPUs, 1 node, `00:15:00` ceiling |
| Allocated TRES | 4 GH200, 288 CPUs, 1 node |
| Elapsed | 6 seconds |
| Allocation-equivalent GPU-hours | `4 * 6 / 3600 = 0.006667` |
| GPU workload | none; failure preceded environment/CUDA access |
| Batch MaxRSS | 36 MiB |
| Output root | absent |
| Slurm stdout | empty, SHA-256 `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| Slurm stderr | 69 bytes, SHA-256 `d2b35cacadc66b7fad8dedd1be5244e17150a6625978c9330f831bd7a9cd6b1e` |

The requested tuple remained one GPU/eight CPUs.  Slurm nevertheless reported a
whole-node allocation of four GPUs/288 CPUs (`OverSubscribe=NO`).  This unexpected
scheduler allocation is preserved rather than silently reported as one GPU.

Raw durable evidence is under `artifacts/job_335630/`: `sacct.txt`, normalized
one-line `scontrol.txt`, exact empty `stdout.txt`, exact `stderr.txt`, and
`sha256sums.txt`.

## Scheduler-only remediation proposal (not executed)

S03 prepared a revised launcher/request for S00 review only.  No snapshot was
created and no job was submitted.  Commit
`2496fecaa1a5daa4a60d7354d06a69ab6ea7d918` was specifically not executed because
its launcher retained `--nodes=1`.  The current proposal keeps its replacement of
Git discovery from the
Slurm spool path with an exact `git archive` of the approved commit/tree from the
shared `/nobackup/.../fl_weather_project/.git` object store.  It verifies the spool
launcher, archived launcher/request, branch ref, tree, implementation ancestry,
and unchanged 15-file source list/state; then it makes the unique snapshot
recursively read-only before any output, environment, CUDA, or pytest action.

The scheduler-only change removes `--nodes=1`, replaces legacy `--gres` spelling
with the S04-job-335579-proven
`--gpus-per-node=nvidia_gh200_120gb:1`, and fail-closes before snapshot/output if
`scontrol` does not report one node/eight CPUs/one typed and generic GPU with
`OverSubscribe=OK` or if Slurm does not expose exactly one CUDA device.  Torch
rechecks `device_count() == 1` before tests.  If it reaches output creation, the
verified record is written to `slurm_allocation.txt` and checksummed.

- Proposed launcher SHA-256:
  `dc61bd2ebd2a88c8be717c8deb2bdfb848971bcf29fe3995e43e1f139f2bfaee`.
- Source-list SHA-256 (unchanged):
  `d4eb8d29da926c88bbcf5c9bbbf9b3e9197f9eda4478ea956ec4c7cfaf664742`.
- Proposed source-state SHA-256:
  `197e5692e6d3c4477a3595cff39d831240b4419954bf929c7ff61e55b65a687e`.
- Proposed snapshot root:
  `/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/execution_snapshots/s03_camera_contract_snapshotfix_6dfd2c775f54`.
- Proposed output root:
  `/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s03_camera_contract_snapshotfix_6dfd2c775f54`.

The four-GPU allocation from job `335630` remains preserved negative evidence.
S04 job `335579` independently demonstrated that the revised scheduler directives
produce `OverSubscribe=OK` and exact one-GPU/eight-CPU `AllocTRES`; the launcher's
runtime checks prevent silent drift.  The proposal therefore requests only the
O-009 ceiling of 0.25 GPU-hour, but it remains unexecuted and unauthorized pending
S00's exact tuple approval.

## Local/static evidence

Executed locally before the request:

- `python3 -m py_compile` over all four modified camera Python modules and
  `test_s03_camera_contract.py`: PASS.
- Python AST parse of those files: PASS.
- `git diff --check`: PASS.
- `bash -n handoffs/S03/run_s03_camera_contract.sh`: PASS.
- C-locale source-list/source-state recomputation: matched the approved values.
- Login-node pytest: unavailable because `/usr/bin/python3` lacks torch,
  torchvision, and pytest; this is not treated as a PASS or failure of the code.

## Requested gates and actual status

| Gate | Status | Evidence / limit |
|---|---|---|
| Exact resize/crop/pad/flip/rotation projection residuals | TEST AUTHORED, NOT EXECUTED | Four independent scalar fixtures exist; no dependency-complete result. |
| Native 1600x900 deterministic validation geometry | TEST AUTHORED, NOT EXECUTED | Expected 0.48 resize -> 432x768, crop `(32,176)`. |
| Seed-replayable training augmentation | TEST AUTHORED, NOT EXECUTED | Explicit CPU generator path covered by test source only. |
| All intended FPN levels affect output | TEST AUTHORED, NOT EXECUTED | Gradient checks cover every input level and neck parameter. |
| Every intended Swin/FPN/LSS parameter finite gradient | TEST AUTHORED, NOT EXECUTED | CUDA fp16-autocast chain did not run. |
| Camera feature/pixel sensitivity | TEST AUTHORED, NOT EXECUTED | No runtime output. |
| LiDAR invariance / pure-camera API | STATIC PASS; RUNTIME NOT EXECUTED | View-transform signature has no point/depth/LiDAR feature input; hostile keyword fixture did not run. |
| Stride-8 / 0.5 m / dtype/shape contract | STATIC PASS; RUNTIME NOT EXECUTED | Constructor/contracts compile; no torch execution. |
| 100-step camera-only loss decrease | NOT AUTHORIZED / NOT RUN | Explicitly outside S03 request. |
| Profile / throughput / peak CUDA memory | NOT AUTHORIZED / NOT RUN | No performance claim. |

## Memory arithmetic (static contract only)

For the primary feature contract with `B=1`, six cameras, 80 context channels,
118 depth bins and `32x88` stride-8 features, an unmasked lift contains
`159,498,240` elements: approximately `304.22 MiB` in fp16 or `608.44 MiB` in
fp32.  This is exactly 8x the old stride-16/1 m contract (`19,937,280` elements).
The relaxed path is designed to mask before lift and accumulate in fp32; the strict
path materializes the full lift.  These are arithmetic/interface implications, not
measured GH200 allocation or throughput.

## Interpretation boundary

Allowed:

- the source compiles and the launcher/source/request provenance design passed
  static checks;
- the first approved submission failed before job creation for the recorded Slurm
  account/partition reason;
- the corrected job failed during launcher provenance preflight before environment,
  CUDA, pytest, or camera execution;
- an immutable-snapshot plus scheduler-only one-GPU remediation is prepared but
  unexecuted and has no exact S00 approval;
- no data, optimizer/model step, metric, or scientific evidence was produced.

Forbidden:

- any projection, gradient, sensitivity, invariance, CUDA/fp16, memory, or
  deterministic-validation runtime PASS;
- camera/model/full-data readiness, tiny-overfit/100-step acceptance, throughput,
  mAP/NDS, fusion gain, FL, attack/defense, generalization, scientific, or
  publication claims;
- treating either scheduler/provenance failure as a negative model or architecture
  result;
- claiming only one GPU was allocated for job `335630`; `sacct` reports four for
  six seconds despite the one-GPU request.
