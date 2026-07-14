# S08 RUN_REQUEST — replacement implementation smoke; Q1 deferred

## Current exact request

```text
REQUEST_ID: S08-SMOKE-3
REQUEST_STATE: CONSUMED / TERMINAL PASS
PREDECESSOR: S08-SMOKE-2 / Job 427800 / terminal focused-test FAIL
OWNER_REMEDIATION_AUTHORITY: O-102 permits narrow code/tests/snapshot/request preparation only
EXACT_TUPLE_APPROVAL: explicit owner approval on 2026-07-14
APPROVED_SUBMISSIONS: 1 / consumed by Job 428112
RETRY_OR_AUTOMATIC_RESUBMISSION: forbidden
Q1_STATE: NOT READY / NOT PART OF THIS REQUEST
```

The owner explicitly approved `S08-SMOKE-3` as written. S00 verified the frozen
binding and submitted it exactly once as Job `428112`; it completed `0:0` with
zero restarts and satisfied every acceptance condition. That approval is now
consumed. Any further submission or changed source/state/snapshot/selector/script/
resource/output/stop condition requires a new request and owner decision.

## Immutable source contract

```text
BASE_AUDIT_COMMIT: 733c84f8e3019fe4d683663821bd86918d3875a7
BRANCH_AT_FREEZE: codex/s08-s09-cl-readiness
SOURCE_KIND: read-only pre-commit remediated implementation snapshot
SNAPSHOT: /nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/execution_snapshots/s08_smoke3_3014cab90ed8
SNAPSHOT_TREE_SHA256: 3014cab90ed88b5705367fc1dd1a21740593acc3a186c72f9073bffe15247a43
SNAPSHOT_FILES: 583
SNAPSHOT_BYTES: 4444941
SNAPSHOT_WRITABLE_ENTRIES: 0
SMOKE_RUNNER_SHA256: 266b83f558b8d9c60f4086d633ac79326cd0dbf3e9c063837d653acf9d44cdf0
SUBMIT_SCRIPT_SHA256: 5fa6e31df27425fde7f04373519d39d30c01386fa3e9b487e406048f11bd6ac0
JOB_BODY_SHA256: 7ac8d0277576a665690408c6002e7438e311e1e66b3ab8721f2be61856d8003a
ARRHENIUS_ENV_SHA256: f57befbb5082aaf4d4bb186958a88420ea873e0fdee5c65da1091b73f566c2bf
PYPROJECT_SHA256: 29c5e81e56fdcb40a2caefdc8a91563ffcd1596df64fed6f4997eef3d58bab72
OUTPUT_ROOT: /nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s08_smoke3_3014cab90ed8
OUTPUT_ROOT_STATE_AT_REQUEST: absent
```

The tree identity covers every Git tracked or non-ignored untracked file present
at freeze. Files are ordered by relative-path bytes. Each record contains relative
path, executable bit, decimal byte count, and file SHA-256 separated by NUL bytes;
the record length is encoded as an eight-byte big-endian integer before the
record. The digest was reproduced independently from the worktree and frozen
snapshot before all snapshot entries were made read-only. The snapshot predates
the outer submit-script path rewrite; only its exact runner is executed, while the
outer submit script is bound separately above.

Relative to the consumed S08-SMOKE-2 snapshot, the only behavior/test remediation
is:

- disabled GradScaler diagnostics condition access to growth/backoff policy
  getters on `scaler.is_enabled()`, while preserving `enabled=false`, scale
  `1.0`, and recording the three unavailable policy fields as `None`;
- the real disabled-scaler test requires that exact record while continuing to
  exercise output/RNG neutrality and hostile-path cleanup;
- one test regex now matches the unchanged production error
  `must return 6 task dictionaries`.

There is no model, head, loss equation, target, optimizer, precision partition,
source-state verifier, runner selector/order, resource, data, or acceptance-gate
change.

## Exact environment and source-state contract

The runner fails before pytest unless the node is aarch64 with exactly one visible
`NVIDIA GH200 120GB` and all identities below verify:

| Component | Required identity |
|---|---|
| Torch | version `2.11.0+cu128`; executable build `a58ba749ac7947ce123a6af8d4cdc595d2aff5dccccec5d6e10bcfe522040f10`; source `70d99e998b4955e0049d13a98d77ae1b14db1f45` |
| spconv | version `2.3.8`; executable build `74934de877e07a8eef8edacd4e31ec0f06eff030b3bc7e06d01f41b1444687d8`; source HEAD `263d6b47425ef843c82f997b12d8b714013d216c` |
| spconv tracked state | format `git-tracked-regular-files.v1`; state `499efdbb5ab31c43109d48f11ee0ff79af847a3d378fd48bf9c79f8672da28db`; sole record `" M" pyproject.toml`; file `e2c84544b5b5d6fd8e149d88539c3a6e989a1824637fd6b0006891955cb7a7e9` |
| cumm | version `0.7.13`; executable build `0a7e3c1a8c3e8d41b3b40c4fb77d05bdec8ca2dfce5dbb8863626c4b45d8296d`; source HEAD `4dedaf43ff801e417c60c6bd7536a29d83d29ee0` |
| cumm tracked state | format `git-tracked-regular-files.v1`; state `f835ee22d539bbf0ab486fecf1188c3883c3cde5860913434cbcf945ee325662`; no changes |

The exact state is checked before and after first sparse-package import. Any
additional tracked file, path/status/content change, unsupported Git state, source
HEAD drift, import-origin drift, or installed executable-build drift fails closed.
The runner neither edits nor resets either external checkout.

## Exact pytest scope

One pytest process runs these selectors in fixed order:

1. all `test_s08_precision_partition.py` cases;
2. all `test_s08_source_identity.py` cases;
3. all `test_s08_precision_diagnostics.py` cases;
4. all S06 training-runtime, resolved-config, checkpoint/resume, eval-provenance,
   and model-mode regressions;
5. the five template fail-closed cases, sparse runtime-identity test, and two
   six-task loss reachability cases;
6. the existing tiny sparse FP32/FP16 forward/backward/eval-workaround test;
7. the tiny sparse FP32-island/autocast/boundary-gradient test.

There is no production training cell and therefore no single resolved experiment
config hash. The exact runner SHA and selectors are the execution configuration.
Unit fixtures bind their own `s08.v1` hashes; the canonical synthetic camera
fixture now resolves to
`e8303eb11ee9793ff8ccc552e9204135905f9cc94e21113b1a3e029fdc23219e`.

All dataset environment variables are unset. No nuScenes mini/trainval, ZIP,
cache, manifest, worker process, sample scan, or model metric is read or produced.
Sparse tests use only the fixed in-source eight-point/two-sample tensor and tiny
`41x16x16` contract. Diagnostic loop tests use at most two toy samples/windows.
Source-state tests create only temporary toy Git repositories beneath pytest's
bounded output directory and do not touch the Arrhenius dependency checkouts.

## Resources, command, and output

```text
account: naiss2025-22-1113-gpu
partition: gpu
nodes/tasks: 1/1
GPU: 1 x nvidia_gh200_120gb
CPU: 8
memory: 96 GiB
Slurm limit: 00:30:00
internal pytest timeout: 20 minutes + 30-second TERM kill grace
requeue: disabled
array/DDP/retry: none
```

The exact approved command was executed once from the repository root:

```bash
bash fl_v3/scripts/submit_s08_precision_smoke.sh
```

The submission script refuses an existing output root, creates it mode 0700,
submits one non-array job, and executes only the read-only snapshot runner. It
does not retry or select another GPU.

Expected artifacts are `environment.json`, `smoke.log`, `smoke.junit.xml`,
`smoke.exit`, `artifact_sha256s.txt`, and Slurm stdout/stderr. Success requires:

- Slurm `COMPLETED 0:0`, zero restarts;
- exact aarch64/GH200, dependency, source-state, and build identities;
- pytest exit zero with zero failed/error/skipped selected cases;
- `smoke.exit` exactly `0`;
- terminal `S08_PRECISION_SMOKE_PASS`;
- all expected artifacts present and checksum-verifiable.

Any identity mismatch, test failure/skip, timeout, OOM, missing artifact, or
nonzero exit is a bounded implementation-smoke FAIL. Preserve artifacts and stop.
Do not edit source/environment, resubmit, expand selectors, or consume another GPU
without a new exact owner decision.

Job `428112` completed in `00:03:28` on `n576` with `COMPLETED 0:0`, zero
restarts, 106 passed/0 failed/0 errors/0 skipped, `smoke.exit=0`, verified
artifact checksums, and terminal `S08_PRECISION_SMOKE_PASS`. Exact artifact
identities and interpretation limits are recorded in `RESULTS.md`.

## Allowed and forbidden interpretation

A pass establishes only that exact tracked-source attestation, focused
config/routing/checkpoint/loop/window-diagnostic contracts, and tiny sparse
FP32/FP16/island paths execute in the reviewed GH200 environment. It does not
establish a stable current six-task optimizer window, precision-policy acceptance,
LiDAR gradient health, convergence, performance, capability, production-data
readiness, mAP/NDS, Protocol A/B, attack, or defense.

## Consumed predecessor

`S08-SMOKE-2` was explicitly approved and submitted once as Job `427800`. It used
snapshot `s08_smoke2_935d0464b3bf` with tree SHA
`935d0464b3bf11fecb10b5e2c3d6a1f7896c44affc442f225fc4c8685d5a4ce1`.
Exact source-state and runtime attestation passed; pytest then completed all 106
selected cases with 103 passed and three focused failures. Two failures came from
calling enabled-only policy getters on a real disabled CPU GradScaler; the third
was a test regex mismatch against the unchanged six-task production error. Both
tiny sparse runtime tests passed. The request is consumed, grants no retry, and
its exact artifacts, hashes, and interpretation limits are preserved in
`RESULTS.md`.

`S08-SMOKE-1` was explicitly approved and submitted once as Job `426619`. It used
snapshot `s08_smoke_f963da5a620e` with tree SHA
`f963da5a620e38a479bf9cee3a80af489bb9d212db79848678ff0560a9555ec2`
and terminated before pytest because the old verifier required a wholly clean
checkout. It grants no retry. Exact artifacts, hashes, and interpretation limits
are preserved in `RESULTS.md`; an exact copy of its request is also inside both
the consumed and replacement snapshots.

## Q1 placeholder — no execution authority

The owner has approved a future Q1 resource ceiling of one GH200 and no more than
one hour. Q1 still requires a successful replacement smoke, immutable
implementation SHA, independent review readiness, exact eight-cell resolved
configs/hashes, accepted dependency identities, exact mini fixture/output/command
hashes, and a new exact owner binding. No S08-SMOKE request authorizes Q1.
