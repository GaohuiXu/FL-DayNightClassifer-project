# S03 RUN REQUEST — focused camera-contract validation

## Approval state

`PENDING_S00_EXACT_APPROVAL_SCHEDULER_ONLY_REMEDIATION_DO_NOT_SUBMIT`

S00 approved the exact executable/request tuple recorded below.  S03 performed
all preflight checks and invoked the approved command once at
`2026-07-11T18:17` Europe/Stockholm.  Slurm rejected the submission before job
creation:

```text
sbatch: error: Batch job submission failed: Invalid account or account/partition combination specified
```

No job ID was returned; `squeue` and `sacct` contain no S03 job; the output root
and Slurm logs were not created; actual GPU allocation is zero.  The exact-once
approval is consumed.  There is no retry/requeue/resubmission/follow-on
authorization.

Read-only diagnosis found that every active Arrhenius GPU launcher inspected uses
both `#SBATCH -A naiss2025-22-1113-gpu` and `#SBATCH -p gpu`, while the consumed S03
launcher omitted both.  S00 subsequently authorized preparation only of a fresh
scheduler-remediation request: add exactly those two directives, preserve all
camera/test/resource/provenance/acceptance scope, and use a new output identity.
That preparation was not an automatic resubmission.  S00 later issued one corrected
exact approval after explicitly superseding an erroneous message.  The corrected
tuple was submitted once as job `335630`; that approval is now consumed.

Consumed rejected-attempt identity (preserved, never reusable):

- executable HEAD: `871db182c5fdcdda46e242d911ac9dcbf393683a`;
- implementation: `6dfd2c775f54e488f3930996b303ce21f9b8e8b7`;
- RUN_REQUEST SHA-256:
  `bd33d91f65558ba97a4ab24f80783133349a8d39b6efa02c925ebd238525b547`;
- launcher SHA-256:
  `9473b830776d478c14c55bcb4991bed329a8273cab6c04bbc8681649f33addfc`;
- source-list SHA-256:
  `d4eb8d29da926c88bbcf5c9bbbf9b3e9197f9eda4478ea956ec4c7cfaf664742`;
- source-state SHA-256:
  `71b0c708325548ad9d09e68e41a0f225bc81f741fd9b4924938317ed591b5b9f`;
- output identity:
  `/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s03_camera_contract_6dfd2c775f54`.

The client-side rejection produced no job ID, allocation, output, or logs and used
zero GPU-hours.  It was not a runtime, CUDA, test, or model failure.

Scheduler-remediation execution identity and outcome:

- approved executable HEAD: `ddadd2ec8423e4d68fd434abf0554a7a2eb1377d`;
- approved tree: `2be4655b8533a83251e95435a9d10fa43dfd6a11`;
- approved RUN_REQUEST SHA-256:
  `d248d8f40f5ed917ec05cbd4681c36d2cfd56a43e25cf96775d47d78cff763f5`;
- job: `335630`, node `n32`, state `FAILED`, exit `1:0`, elapsed six seconds;
- stderr: `fatal: not a git repository (or any of the parent directories): .git`;
- output root: absent; environment activation, CUDA, pytest, and camera code were
  never reached;
- retry/requeue/resubmit/follow-on: none and not authorized.

The launcher resolved `BASH_SOURCE[0]` from Slurm's spool copy, so `SCRIPT_DIR` was
not the committed worktree and the initial `git -C "$SCRIPT_DIR" rev-parse` failed.
This is a launcher/provenance-preflight failure, not a camera test, CUDA, or model
failure.  The request asked for one GPU/eight CPUs, while `sacct` reports an
unexpected whole-node allocation of four GPUs/288 CPUs for six seconds.  No GPU
workload started; allocation accounting is nevertheless recorded exactly in
`RESULTS.md` and the raw artifacts.

## Scheduler-only immutable-snapshot remediation proposal

This is a proposal only.  It does not authorize snapshot creation or `sbatch`.
Job `335630` is not retried.  Commit `2496fecaa1a5daa4a60d7354d06a69ab6ea7d918`
was not executed because its launcher still declared `#SBATCH --nodes=1`, the
directive implicated by the job's `OverSubscribe=NO` whole-node allocation.  The
camera implementation, test file, exact 10-case scope, fp16 forward/backward,
dataset exclusion, acceptance, requested one GPU/eight CPU/15-minute limits, and
source closure remain unchanged.

The revised committed launcher no longer attempts to discover Git from its Slurm
spool path or access `/home/.../.codex/worktrees/.../.git`.  It also removes
`--nodes=1` and uses the exact S04-job-335579-proven scheduler form
`--gpus-per-node=nvidia_gh200_120gb:1`, `--cpus-per-task=8`, and 15 minutes.  If
separately approved, it would:

1. require externally approved executable HEAD, tree, branch, implementation,
   request, launcher, source-list/source-state hashes, snapshot root, output root,
   and Slurm job identity;
2. resolve the branch commit and tree only from the shared object store
   `/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/fl_weather_project/.git`;
3. verify the approved implementation is an ancestor and that the spool launcher
   bytes match the approved launcher hash;
4. before snapshot/output creation, query `scontrol` and require actual
   `NumNodes=1`, `NumCPUs=8`, `OverSubscribe=OK`, generic GPU count 1, typed GH200
   count 1, `SLURM_CPUS_PER_TASK=8`, and exactly one non-disabled
   `CUDA_VISIBLE_DEVICES` entry; echo that record to the Slurm log;
5. require both the unique snapshot and output roots to be absent;
6. `git archive` the exact approved commit into a job-unique temporary directory,
   verify the archived RUN_REQUEST, launcher, 15-file C-locale list/state, and
   branch/tree identities before any output creation;
7. write a snapshot identity manifest, remove all write bits recursively, and
   atomically rename the temporary tree to the approved shared snapshot root;
8. write `slurm_allocation.txt`, then after environment activation require
   `torch.cuda.device_count() == 1` before the unchanged pytest command runs from
   that read-only snapshot.

Proposed unique roots, both currently absent:

```text
/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/execution_snapshots/s03_camera_contract_snapshotfix_6dfd2c775f54
/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s03_camera_contract_snapshotfix_6dfd2c775f54
```

The prior job's whole-node allocation remains preserved negative evidence.  The
new scheduler form is supported by job `335579`, whose `scontrol` recorded
`OverSubscribe=OK` and whose `ReqTRES` and `AllocTRES` both contained exactly one
GH200/eight CPUs.  The new launcher additionally refuses to proceed if the actual
allocation or CUDA visibility differs, so the proposed ceiling is again one
GPU for 15 minutes (0.25 GPU-hour) under O-009.  This evidence makes the request
reviewable; it is not compute authorization.

S00 returned the first request for provenance remediation without approving
compute: its exact `sbatch` body existed only as a mutable Markdown here-doc and
neither the launcher nor approved request identity was bound in-job.  The consumed
scheduler-remediation snapshot used a committed launcher and external approval
hashes, but failed before those checks could identify the repository.  This
post-result revision grants no compute authority.

## Proposed immutable implementation and executable model

- Base: `372de9398ae435f82b83367a922fd302c0635738`.
- Implementation commit:
  `6dfd2c775f54e488f3930996b303ce21f9b8e8b7`.
- Worker branch: `codex/s03-camera-architecture`.
- Durable launcher:
  `fl_v3/usenix27_orchestra/handoffs/S03/run_s03_camera_contract.sh`.
- Launcher SHA-256:
  `dc61bd2ebd2a88c8be717c8deb2bdfb848971bcf29fe3995e43e1f139f2bfaee`.
- Proposed executable HEAD/tree and final RUN_REQUEST SHA-256: computed and
  reported after this scheduler-only proposal commit, then subject to external
  review.

Before creating the immutable snapshot, output, or importing the runtime, the
proposed launcher fails closed unless:

1. actual HEAD equals the externally approved executable SHA;
2. the shared branch ref equals that HEAD and its tree equals the externally
   approved tree SHA;
3. implementation `6dfd2c7...` is an ancestor of executable HEAD;
4. spool and archived launcher plus archived request bytes match their externally
   approved SHA-256;
5. archived C-locale source-list and content hashes match the approved values;
6. `scontrol` reports one node/eight CPUs/one generic and typed GH200 allocation
   with `OverSubscribe=OK`, while Slurm exposes exactly one CUDA device;
7. both exact snapshot and output roots do not exist;
8. the exported snapshot contains no writable path before runtime begins;
9. after environment activation, torch sees exactly one CUDA device before tests.

Any edit or new commit after approval invalidates it.

## Source-state and actual import closure

The committed pytest invocation uses `--noconftest`, empty `PYTEST_ADDOPTS`, and
`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`.  Therefore `fl_v3/tests/conftest.py` is
intentionally not imported.  The 15-file C-locale set covers the selected test's
actual local eager import closure, package initializers, read-only `bev_grid.py`,
pytest/dependency inputs, Arrhenius environment bootstrap, and durable launcher:

```text
fl_v3/pyproject.toml
fl_v3/requirements.lock.txt
fl_v3/requirements.txt
fl_v3/scripts/arrhenius_env.sh
fl_v3/src/fl_v3/__init__.py
fl_v3/src/fl_v3/models/__init__.py
fl_v3/src/fl_v3/models/fusion/__init__.py
fl_v3/src/fl_v3/models/fusion/bev_grid.py
fl_v3/src/fl_v3/models/fusion/camera_backbone.py
fl_v3/src/fl_v3/models/fusion/camera_neck.py
fl_v3/src/fl_v3/models/fusion/preprocess.py
fl_v3/src/fl_v3/models/fusion/swin_sdpa.py
fl_v3/src/fl_v3/models/fusion/view_transform.py
fl_v3/tests/test_s03_camera_contract.py
fl_v3/usenix27_orchestra/handoffs/S03/run_s03_camera_contract.sh
```

- C-locale sorted source-list SHA-256:
  `d4eb8d29da926c88bbcf5c9bbbf9b3e9197f9eda4478ea956ec4c7cfaf664742`.
- SHA-256 of the corresponding `sha256sum` source-state file:
  `197e5692e6d3c4477a3595cff39d831240b4419954bf929c7ff61e55b65a687e`.

`RUN_REQUEST.md` is not part of that aggregate because its final hash is a
separate mandatory externally approved input.  Including both its hash and the
expected source aggregate inside itself would create a self-reference cycle.  The
launcher independently recomputes and verifies request, launcher, list, and source
identities before output creation, then records all four in
`execution_identity.json`.

## Exact validation scope

- One file: `fl_v3/tests/test_s03_camera_contract.py`.
- Exactly 10 synthetic pytest cases after parametrization.
- Projection residual fixtures for resize/crop/pad/flip/rotation.
- Deterministic validation geometry and seeded train replay.
- Native 1600x900 -> 256x704 reference validation geometry.
- Every declared FPN level and parameter has finite gradient coverage.
- Pure-camera API, LiDAR-input rejection/invariance, camera feature and camera
  pixel sensitivity.
- Stride-8, 0.5 m depth-bin shape/dtype contract and theoretical memory arithmetic.
- One Swin-T -> FPN -> pure-camera LSS forward/backward.  Because CUDA is required
  by launcher preflight, this case runs on the allocated GH200 and uses fp16
  autocast.

Inputs are synthetic tensors only.  There is no nuScenes mini/trainval metadata,
payload, ZIP/cache/GT database, DataLoader, optimizer, scheduler, EMA, model
training step, tiny-overfit/100/1000-step gate, profile, evaluation, metric, matrix,
seed campaign, or scientific result.

## Resources, output, and command contract

- One job; actual allocation must be one node, one `nvidia_gh200_120gb`, and eight
  CPUs or the launcher exits before snapshot/output/model activity.
- Account `naiss2025-22-1113-gpu`; partition `gpu`.
- Walltime `00:15:00`; requested allocation ceiling 0.25 GPU-hours.
- Historical job `335630`: whole-node allocation of four GPUs/288 CPUs because
  the launcher still declared `--nodes=1`; negative evidence is retained.
- Scheduler control evidence: S04 job `335579` used
  `--gpus-per-node=nvidia_gh200_120gb:1` without `--nodes=1` and recorded
  `OverSubscribe=OK`, `AllocTRES=1 GPU/8 CPUs`.
- S03 allocation-equivalent use to date: approximately 0.006667 GPU-hour.
- No array, DDP, concurrent S03 job, retry, requeue, resubmission, follow-on, or
  spare-GPU expansion.

Unique immutable snapshot and output roots, both required absent:

```text
/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/execution_snapshots/s03_camera_contract_snapshotfix_6dfd2c775f54
/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s03_camera_contract_snapshotfix_6dfd2c775f54
```

Logs are fixed by committed `#SBATCH` directives:

```text
/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/logs/s03_camera_contract_%j.out
/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/logs/s03_camera_contract_%j.err
```

Any future approval would have to bind all values below after the proposal commit:

```text
EXPECTED_S03_EXECUTABLE_SHA=<post-commit 40-hex>
EXPECTED_S03_TREE_SHA=<post-commit 40-hex>
EXPECTED_S03_IMPLEMENTATION_SHA=6dfd2c775f54e488f3930996b303ce21f9b8e8b7
EXPECTED_S03_BRANCH=codex/s03-camera-architecture
EXPECTED_S03_SOURCE_LIST_SHA=d4eb8d29da926c88bbcf5c9bbbf9b3e9197f9eda4478ea956ec4c7cfaf664742
EXPECTED_S03_SOURCE_SHA=197e5692e6d3c4477a3595cff39d831240b4419954bf929c7ff61e55b65a687e
EXPECTED_S03_LAUNCHER_SHA=dc61bd2ebd2a88c8be717c8deb2bdfb848971bcf29fe3995e43e1f139f2bfaee
EXPECTED_S03_RUN_REQUEST_SHA=<post-commit 64-hex>
S03_SNAPSHOT_ROOT=/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/execution_snapshots/s03_camera_contract_snapshotfix_6dfd2c775f54
S03_OUTPUT_ROOT=/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s03_camera_contract_snapshotfix_6dfd2c775f54
```

The exact resolved command is returned to S00 after the proposal commit; it is not
authorized until S00 separately approves that immutable tuple.  The consumed
approval that produced job `335630` remains non-reusable.  This Markdown revision
contains no executable here-doc and does not authorize running the launcher.

## Recorded identity and artifacts

Before pytest the proposed launcher writes `execution_identity.json` containing:

- exact executable HEAD/tree, implementation SHA, branch, and snapshot root;
- runtime source-list and source-state hashes;
- launcher and externally approved RUN_REQUEST hashes;
- Slurm job ID/GPU/CPU environment, `CUDA_VISIBLE_DEVICES`, host, architecture,
  Python, torch, torchvision, pytest, CUDA runtime, GPU name, and GPU memory.

It also writes the exact source file list, per-file SHA-256 state, and the verified
`scontrol`/CUDA-visible allocation as `slurm_allocation.txt`.  Pytest emits log and
JUnit; a post-check requires exactly `10/0/0/0` tests/failures/errors/skips.  All
identity/allocation/source/test summary artifacts are checksummed and verified
in-job.

## Stop conditions and interpretation

The job fails on any missing approval variable, malformed hash, HEAD/branch/status/
ancestor mismatch, launcher/request/source drift, changed output, actual allocation
other than one node/eight CPUs/one GH200 with `OverSubscribe=OK`, CUDA visibility
other than one device, unavailable CUDA, pytest failure/error/skip/count drift, or
artifact checksum failure.  Any failure is returned to S00; it does not authorize
retry or scope change.

Allowed if PASS: the exact synthetic S03 camera geometry/interface/gradient suite
passes on the recorded GH200 runtime, including one CUDA fp16-autocast camera-chain
forward/backward.

Forbidden regardless of PASS: mini/trainval model readiness, tiny-overfit or
100/1000-step acceptance, throughput/profile, mAP/NDS/fusion gain, FL,
attack/defense, generalization, scientific, or publication claims.
