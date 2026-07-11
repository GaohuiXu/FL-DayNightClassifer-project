# S03 RUN REQUEST — focused camera-contract validation

## Approval state

`PENDING_S00_REAUDIT_DO_NOT_SUBMIT`

S00 returned the first request for provenance remediation without approving
compute: its exact `sbatch` body existed only as a mutable Markdown here-doc and
neither the launcher nor approved request identity was bound in-job.  This revision
uses a committed launcher and external post-commit approval hashes.  Preparing or
committing it is not approval; S03 must receive a new explicit S00 decision before
one `sbatch`.

## Immutable implementation and executable model

- Base: `372de9398ae435f82b83367a922fd302c0635738`.
- Implementation commit:
  `6dfd2c775f54e488f3930996b303ce21f9b8e8b7`.
- Worker branch: `codex/s03-camera-architecture`.
- Durable launcher:
  `fl_v3/usenix27_orchestra/handoffs/S03/run_s03_camera_contract.sh`.
- Launcher SHA-256:
  `9473b830776d478c14c55bcb4991bed329a8273cab6c04bbc8681649f33addfc`.
- Executable HEAD: the commit containing this final request plus the launcher.
  Its SHA cannot be embedded in its own tree; S03 reports it after commit and S00
  binds it externally through `EXPECTED_S03_EXECUTABLE_SHA`.
- Final RUN_REQUEST SHA-256: also computed and reported after commit, then bound
  externally through `EXPECTED_S03_RUN_REQUEST_SHA`.  It is deliberately not
  embedded here, avoiding a request self-hash cycle.

The launcher runs directly from the approved clean branch/HEAD.  Before creating
output or importing the runtime it fails closed unless:

1. actual HEAD equals the externally approved executable SHA;
2. actual branch equals `codex/s03-camera-architecture` and status is clean;
3. implementation `6dfd2c7...` is an ancestor of executable HEAD;
4. launcher and final request bytes match their externally approved SHA-256;
5. C-locale source-list and content hashes match the approved values;
6. the exact output root does not exist.

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
  `71b0c708325548ad9d09e68e41a0f225bc81f741fd9b4924938317ed591b5b9f`.

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

- One job; one node; one `nvidia_gh200_120gb`; eight CPUs.
- Walltime `00:15:00`; maximum requested allocation 0.25 GPU-hours.
- S03 cumulative GPU use before this request: 0 GPU-hours.
- No array, DDP, concurrent S03 job, retry, requeue, resubmission, follow-on, or
  spare-GPU expansion.

Unique output root, required absent:

```text
/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s03_camera_contract_6dfd2c775f54
```

Logs are fixed by committed `#SBATCH` directives:

```text
/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/logs/s03_camera_contract_%j.out
/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/logs/s03_camera_contract_%j.err
```

After committing this revision, S03 reports the executable HEAD and final request
hash alongside the fully resolved one-line command.  S00 approval must bind that
exact command and all values below:

```text
EXPECTED_S03_EXECUTABLE_SHA=<post-commit 40-hex reported by S03>
EXPECTED_S03_IMPLEMENTATION_SHA=6dfd2c775f54e488f3930996b303ce21f9b8e8b7
EXPECTED_S03_BRANCH=codex/s03-camera-architecture
EXPECTED_S03_SOURCE_LIST_SHA=d4eb8d29da926c88bbcf5c9bbbf9b3e9197f9eda4478ea956ec4c7cfaf664742
EXPECTED_S03_SOURCE_SHA=71b0c708325548ad9d09e68e41a0f225bc81f741fd9b4924938317ed591b5b9f
EXPECTED_S03_LAUNCHER_SHA=9473b830776d478c14c55bcb4991bed329a8273cab6c04bbc8681649f33addfc
EXPECTED_S03_RUN_REQUEST_SHA=<post-commit 64-hex reported by S03>
S03_OUTPUT_ROOT=/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s03_camera_contract_6dfd2c775f54
```

The only authorized submission form, if S00 later approves, is one `sbatch
--export=ALL,...` of the committed launcher.  The Markdown request contains no
executable here-doc and does not authorize running the launcher.

## Recorded identity and artifacts

Before pytest the launcher writes `execution_identity.json` containing:

- exact executable HEAD, implementation SHA, and branch;
- runtime source-list and source-state hashes;
- launcher and externally approved RUN_REQUEST hashes;
- Slurm job ID, host, architecture, Python, torch, torchvision, pytest, CUDA
  runtime, GPU name, and GPU memory.

It also writes the exact source file list and per-file SHA-256 state.  Pytest emits
log and JUnit; a post-check requires exactly `10/0/0/0` tests/failures/errors/skips.
All identity/source/test summary artifacts are checksummed and verified in-job.

## Stop conditions and interpretation

The job fails on any missing approval variable, malformed hash, HEAD/branch/status/
ancestor mismatch, launcher/request/source drift, changed output, unavailable CUDA,
pytest failure/error/skip/count drift, or artifact checksum failure.  Any failure is
recorded and returned to S00; it does not authorize retry or scope change.

Allowed if PASS: the exact synthetic S03 camera geometry/interface/gradient suite
passes on the recorded GH200 runtime, including one CUDA fp16-autocast camera-chain
forward/backward.

Forbidden regardless of PASS: mini/trainval model readiness, tiny-overfit or
100/1000-step acceptance, throughput/profile, mAP/NDS/fusion gain, FL,
attack/defense, generalization, scientific, or publication claims.
