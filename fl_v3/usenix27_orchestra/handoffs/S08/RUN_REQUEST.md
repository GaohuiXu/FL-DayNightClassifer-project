# S08 RUN_REQUEST — review-remediation smoke and fixture attestation; Q1 deferred

## Current exact request

```text
REQUEST_ID: S08-SMOKE-5
REQUEST_STATE: CONSUMED / TERMINAL PASS
PREDECESSOR: S08-SMOKE-4 / Job 428889 / terminal Phase-1 test-construction FAIL
REVIEW_BASELINE: 791aba97f7bbe92e7708b63f94f2e7d8599f91be
REVIEW_VERDICT: REMEDIATE
OWNER_DIRECTION_BEFORE_FREEZE: initial Smoke-5 request was not treated as an exact approval
EXACT_TUPLE_APPROVAL_AFTER_FREEZE: explicit owner confirmation on 2026-07-14 / O-106
APPROVED_SUBMISSIONS: 1 / consumed by Job 429080
RETRY_OR_AUTOMATIC_RESUBMISSION: forbidden
Q1_STATE: BLOCKED / NOT PART OF THIS REQUEST
```

The owner first requested `S08-SMOKE-5` before its changed source, snapshot tree,
script hash, and output path existed; that message was not used as execution
authority. After S00 froze and presented the tuple below, the owner explicitly
confirmed that exact tuple under O-106. S00 reverified every identity and submitted
it once as Job `429080`. The job completed `0:0`, zero restarts, with Phase 1
116/116 PASS and Phase 2 1/1 PASS. The approval is consumed; there is no automatic
retry, alternate node, or spare-GPU authority.

## Terminal disposition

```text
JOB_ID: 429080
STATE/EXIT/RESTARTS: COMPLETED / 0:0 / 0
NODE/ELAPSED: n23 / 00:03:36
PHASE_1: 116 passed / 0 failed / 0 errors / 0 skipped
PHASE_2: 1 passed / 0 failed / 0 errors / 0 skipped
SMOKE_EXIT/FIXTURE_EXIT: 0 / 0
TERMINAL_MARKER: S08_PRECISION_SMOKE_PASS
AUTOMATIC_RETRY: none
```

This is terminal overall Smoke-5 **PASS** for its declared focused remediation and
fixture-attestation scope. All expected artifacts and their checksum manifest
verified. It is not Q1, a precision-policy decision, model-numerical evidence, or
S09 readiness. Complete artifact and fixture identities are in `RESULTS.md`.

## Immutable source contract

```text
BASE_IMPLEMENTATION_COMMIT: 791aba97f7bbe92e7708b63f94f2e7d8599f91be
BRANCH_AT_FREEZE: codex/s08-s09-cl-readiness
SOURCE_KIND: read-only uncommitted review-remediation snapshot
SNAPSHOT: /nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/execution_snapshots/s08_smoke5_51daec3e860e
SNAPSHOT_TREE_SHA256: 51daec3e860e6d412ad57d807efd78a08b03630afb37798880999fa039900a25
SNAPSHOT_FILES: 585
SNAPSHOT_BYTES: 4515200
SNAPSHOT_WRITABLE_ENTRIES: 0
SMOKE_RUNNER_SHA256: 08b74822862e6e91f14802426b76bfff29dfdd7ace85482a9882a94914941ff1
SUBMIT_SCRIPT_SHA256: 254064b207f004ae778f1c73c5e474f0cdf74642a1ba50724adec6e4911ffd40
JOB_BODY_SHA256: 7ac8d0277576a665690408c6002e7438e311e1e66b3ab8721f2be61856d8003a
ARRHENIUS_ENV_SHA256: f57befbb5082aaf4d4bb186958a88420ea873e0fdee5c65da1091b73f566c2bf
PYPROJECT_SHA256: 29c5e81e56fdcb40a2caefdc8a91563ffcd1596df64fed6f4997eef3d58bab72
REVIEW_MD_SHA256: 4385f1696d984d50cbdc5037b0384f70453237d78597d24374c4fa6ad4e32569
OUTPUT_ROOT: /nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s08_smoke5_51daec3e860e
OUTPUT_ROOT_STATE_AT_REQUEST: absent
```

The tree identity covers every Git-tracked and non-ignored untracked file present
at freeze, including the unchanged independent `REVIEW.md` and the new raw-input
manifest. Files are ordered by relative-path bytes. Each length-prefixed canonical
record contains relative path, executable bit, decimal byte count, and file
SHA-256 separated by NUL bytes. The digest, file count, byte count, and zero
writable entries were reproduced from both the worktree and frozen copy.

As in the consumed S08 smoke requests, the snapshot predates the final outer
submit-script path rewrite and this exact request text. The exact snapshot runner
is the only in-snapshot script executed; the outer submit script and here-document
job body are separately hash-bound above.

Relative to reviewed implementation SHA
`791aba97f7bbe92e7708b63f94f2e7d8599f91be`, the remediation is limited
to:

- predeclaring a bounded raw-input manifest and requiring exact full-batch,
  augmentation-field/order, augmentation-value, and fixture-manifest identities
  before future Q1 model/optimizer construction;
- a candidate-only fixture-attestation test that derives those values without
  constructing a model or optimizer and cannot execute Q1;
- pure-Python scheduler/EMA consistency records plus scheduler continuity/final-
  epoch and EMA-disabled qualification gates;
- focused positive and hostile-negative tests; and
- current canonical/handoff wording corrections required by `REVIEW.md`.

Relative to the consumed Smoke-4 snapshot, the **only** source/test/runner
behavior change is:

```text
changed[tensor_name].reshape(-1)[0] = 1.0
    -> changed[tensor_name].reshape(-1)[0].add_(1.0)
```

This guarantees a real finite value change for each parametrized synthetic input,
including the `torch.eye(4)` calibration diagonal. The runner, selectors, expected
116+1 counts, production diagnostics, precision paths, data manifest, resources,
and gates are otherwise byte/semantically unchanged.

No model architecture, sparse normalization, head/loss/target, optimizer recipe,
metric/decode/NMS, precision regime, camera/LSS boundary, dependency checkout,
environment, production data/cache, or scientific policy is changed.

## Exact environment and dependency-source contract

The runner fails before pytest unless the node is aarch64 with exactly one visible
`NVIDIA GH200 120GB` and all identities below verify:

| Component | Required identity |
|---|---|
| Torch | version `2.11.0+cu128`; executable build `a58ba749ac7947ce123a6af8d4cdc595d2aff5dccccec5d6e10bcfe522040f10`; source `70d99e998b4955e0049d13a98d77ae1b14db1f45` |
| spconv | version `2.3.8`; executable build `74934de877e07a8eef8edacd4e31ec0f06eff030b3bc7e06d01f41b1444687d8`; source HEAD `263d6b47425ef843c82f997b12d8b714013d216c` |
| spconv tracked state | format `git-tracked-regular-files.v1`; state `499efdbb5ab31c43109d48f11ee0ff79af847a3d378fd48bf9c79f8672da28db`; sole record `" M" pyproject.toml`; file `e2c84544b5b5d6fd8e149d88539c3a6e989a1824637fd6b0006891955cb7a7e9` |
| cumm | version `0.7.13`; executable build `0a7e3c1a8c3e8d41b3b40c4fb77d05bdec8ca2dfce5dbb8863626c4b45d8296d`; source HEAD `4dedaf43ff801e417c60c6bd7536a29d83d29ee0` |
| cumm tracked state | format `git-tracked-regular-files.v1`; state `f835ee22d539bbf0ab486fecf1188c3883c3cde5860913434cbcf945ee325662`; no changes |

The exact source state is checked before and after first sparse-package import.
The runner neither edits nor resets either external checkout.

## Exact bounded data and fixture-attestation contract

```text
BACKEND: nuScenes v1.0-mini directory backend only
DATAROOT: /nobackup/proj/disk/naiss2024-22-991/personal/gaohui/fl_weather_project/data/nuscenes_mini
SPLIT/SAMPLE: mini_train / 00889f8a9549450aa2f32cf310a3e305
N_SWEEPS: 10 (one key LiDAR plus exactly nine previous sweeps)
POINT_BOUND: first exact 4096 keyframe-only collated points
SEED: 20260713
RAW_INPUT_MANIFEST_SCHEMA: s08.q1-raw-input-manifest.v1
RAW_INPUT_MANIFEST_FILE_SHA256: 62a63cf6c3dd4295f8c246fdef6ba170e7685cab6930294b17633a1d448798b4
RAW_INPUT_MANIFEST_LOGICAL_SHA256: f95c0cd141c891f69f44a0ecc792e4878946a3cdc4a1a2ce7911df074b848316
RAW_INPUT_FILES/BYTES: 29 / 41085435
CACHE/ZIP/MANIFEST: none
MODEL/OPTIMIZER/BACKWARD/WINDOWS: none
```

The manifest names and hashes exactly 13 mini metadata JSON files, six camera
payloads, one key LiDAR payload, and nine previous LiDAR payloads. The attestation
reads only these 29 declared files and the single frozen sample; it does not walk
the mini tree, inspect trainval, create a cache, touch the stored-ZIP backend, or
scan full data. It derives and writes candidate identities for the complete batch
tensor manifest, augmentation field order, augmentation values, and canonical
fixture manifest. These derived values are outputs of this request and therefore
are **not** preaccepted Q1 inputs; a future Q1 request must bind them exactly after
immutable remediation review.

## Exact pytest phases and execution semantics

Phase 1 runs one pytest process with all dataset variables unset. It repeats the
106 selectors accepted by S08-SMOKE-3 and adds exactly ten focused cases:

1. expected fixture-identity environment schema is fail-closed;
2. changed `images`, `cam_intrinsics`, and `gt_boxes` each change complete batch/
   fixture identity and fail the pre-model gate;
3. augmentation field-order drift is rejected;
4. an overflow-then-three-accept scheduler timeline with EMA disabled qualifies;
5. hostile scheduler delta, continuity, final epoch, and EMA-enabled records each
   fail qualification.

Expected Phase-1 JUnit totals are exactly 116 tests, zero failures, zero errors,
and zero skips. It includes the same tiny sparse FP32/FP16/island paths and toy
training-loop diagnostics as Smoke-3, but no current full six-task optimizer cell.

Phase 2 starts a separate pytest process with only the exact mini dataroot and an
empty attestation output directory. It selects only
`test_s08_q1_fixture_attestation`. Expected JUnit totals are exactly one test,
zero failures, zero errors, and zero skips. Pytest success means the implementation
and fixture-attestation runner completed; it does not mean any precision regime
qualified. No Q1 cell is selected.

There is no single resolved experiment-config hash because no experiment or model
cell executes. The exact runner SHA, ordered selectors, raw-input identities, and
seed are the execution configuration.

## Resources, command, output, and stop conditions

```text
account: naiss2025-22-1113-gpu
partition: gpu
nodes/tasks: 1/1
GPU: 1 x nvidia_gh200_120gb
CPU: 8
memory: 96 GiB
Slurm limit: 00:30:00
per-pytest timeout: 20 minutes + 30-second TERM kill grace
requeue: disabled
array/DDP/retry: none
```

The only proposed command is:

```bash
bash fl_v3/scripts/submit_s08_precision_smoke.sh
```

The submit script refuses an existing output root, creates it mode 0700, submits
one non-array job, and executes only the read-only snapshot runner. It performs no
retry or node selection.

Expected artifacts are:

- `environment.json`;
- `smoke.{log,junit.xml,exit}`;
- `fixture-attestation.{log,junit.xml,exit}`;
- `fixture-attestation/{fixture_manifest,fixture_identity,fixture_attestation}.json`;
- `artifact_sha256s.txt`; and
- Slurm stdout/stderr.

Acceptance requires Slurm `COMPLETED 0:0`, zero restarts; exact runtime/source/data-
manifest identities; exact 116+1 JUnit totals with no failure/error/skip; both exit
files exactly `0`; all three fixture JSON artifacts; checksum-verifiable artifacts;
and terminal `S08_PRECISION_SMOKE_PASS`. The attestation JSON must state
`candidate_only=true`, `model_constructed=false`, `optimizer_constructed=false`,
and `q1_executed=false`.

Any identity mismatch, skip, failure, timeout, OOM, missing artifact, nonzero exit,
or unexpected test count is terminal bounded-smoke FAIL. Preserve all artifacts
and stop. Do not edit source/environment, retry, broaden selectors, submit Q1, or
consume another GPU without a new exact owner decision.

## Allowed and forbidden interpretation

A pass would establish only that the linear review remediation, focused prior S08
contracts, and exact bounded fixture-attestation path execute in the reviewed
GH200 environment, producing candidate fixture identities for a later request. It
would not establish a stable current six-task optimizer window, choose a precision
policy, explain or repair the large LiDAR gradients, establish convergence,
performance, capability, production-data readiness, mAP/NDS, Protocol A/B, attack,
or defense.

## Consumed predecessors

- `S08-SMOKE-4`, Job `428889`: exact preflight PASS, then terminal Phase-1
  focused-test FAIL at 115 passed/1 failed because the calibration drift assignment
  did not change its already-`1.0` value. Phase 2 did not start. No retry authority.
- `S08-SMOKE-3`, Job `428112`: exact runtime/source-state attestation and 106/106
  focused tests PASS, `COMPLETED 0:0`, zero restarts. This remains valid for its
  declared implementation-smoke scope.
- `S08-SMOKE-2`, Job `427800`: source-state attestation PASS, then terminal
  focused-test FAIL at 103 passed/3 failed. No retry authority.
- `S08-SMOKE-1`, Job `426619`: terminal pre-pytest provenance-policy FAIL. No
  model/diagnostic code ran and no retry authority.

Exact artifacts, hashes, and interpretation limits for all four remain in
`RESULTS.md`.

## Q1/Q2 completion authority — exact tuple freeze in progress

O-109 authorizes the exact Q1 primary and minimal Q2 L-P020/F-CBGS submissions
needed to close S08, after each immutable tuple is recorded here. Across all new
Q1/Q2 jobs, one-GPU elapsed allocation must remain at or below two GPU-hours;
unused elapsed time from an earlier job is the only budget available to a later
job. Smoke-5 has passed, remediation is sealed at `103c7389`, and independent R2
returned `PASS_WITH_RESIDUAL_RISK`. Before Q1 submission, this file must bind all
five fixture-identity SHA-256 values, exact source/snapshot/config and launcher
hashes, eight cells/order, output, resources, and stop conditions. O-109 permits
no extra cells, seeds, data, harnesses, scientific retries, merge, or push.
