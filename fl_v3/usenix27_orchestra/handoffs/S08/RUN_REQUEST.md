# S08 RUN_REQUEST — precision qualification execution ledger

> **Ledger state:** no active request. `S08-Q1` Job `431013` and `S08-Q2` Job
> `435151` are consumed and terminal under O-109. The immediately following
> `S08-SMOKE-5` block is preserved terminal history.

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

## Q1/Q2 completion authority — consumed terminal execution

O-109 authorized the exact Q1 primary and minimal Q2 L-P020/F-CBGS submissions
after each immutable tuple was recorded here. Q1 Job `431013` and Q2 Job `435151`
are now consumed and terminal; together they used `00:07:58` of the two-GPU-hour
ceiling. All five fixture identities, sources/snapshots/configs/launchers,
cells/order, outputs, resources, and stop conditions remain bound below. No extra
cell, seed, data, harness, scientific retry, merge, or push occurred.

## S08-Q1 exact primary precision qualification — approved under O-109

```text
REQUEST_ID: S08-Q1
REQUEST_STATE: CONSUMED / TERMINAL BOUNDED RESULT / Job 431013 COMPLETED 0:0
EXECUTION_SOURCE_SHA: e6e28bea43f7757347da2e460cdf24e9a32b791f
REVIEWED_IMPLEMENTATION_SHA: 103c7389a47938b1f9dd0cba60251df6dce9e5bb
SOURCE_RELATION: e6e28be differs from reviewed implementation only in canonical/review documentation
BRANCH_AT_FREEZE: codex/s08-s09-cl-readiness
SNAPSHOT: /nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/execution_snapshots/s08_q1_dbeee35dcd6d
SNAPSHOT_TREE_SHA256: dbeee35dcd6d7bcb919f549f03c42763d5d82b2b20740815743b7aa2b3f9bc9c
SNAPSHOT_FILES/BYTES/WRITABLE: 585 / 4544533 / 0
Q1_TEST_SHA256: 1de18962b3ac5d3b1a4b992f8c8de4fe75570af90b5c1be2f8f73e6117773b26
RAW_INPUT_MANIFEST_FILE_SHA256: 62a63cf6c3dd4295f8c246fdef6ba170e7685cab6930294b17633a1d448798b4
JOB_SCRIPT: /nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/execution_requests/s08_q1_dbeee35dcd6d/job.sh
JOB_SCRIPT_SHA256: 42cb555d518a6d7bb517c325c22c1f0ab8362c03da36b9cfd1f0b981d8b349e1
OUTPUT: /nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s08_q1_dbeee35dcd6d
OUTPUT_STATE_AT_FREEZE: absent; terminal artifacts preserved
DATA_BACKEND: nuScenes v1.0-mini directory backend; no ZIP/full-data scan
SAMPLE_TOKEN: 00889f8a9549450aa2f32cf310a3e305
LIDAR_BOUND: keyframe plus 9 prior sweeps, fixed 4096-point prefix
SEED: 20260713
CELL_ORDER: C1,C2,L1,L2,L3,F1,F2,F3
MAX_ATTEMPTED_WINDOWS: 99 across all cells
MAX_ACCEPTED_UPDATES: 24 across all cells; exactly 3 required per qualifying cell
Q1_RESOURCE_BUDGET: one GH200, 8 CPUs, 96 GiB, 01:00:00, no requeue
CUMULATIVE_NEW_Q1_Q2_BUDGET: <=2 one-GPU elapsed hours under O-109
RETRY: none; F2 bounded negative result is preserved
```

The read-only snapshot contains only the 585 Git-tracked files from exact source
SHA `e6e28be`. Its tree digest orders files by relative-path bytes and hashes a
sequence of eight-byte big-endian record lengths followed by records containing
`path NUL executable-bit NUL decimal-size NUL file-SHA256`. The source worktree was
clean at freeze. The one-shot job script lives outside the snapshot, is mode 0500,
and adds no repository harness.

### Prebound fixture and resolved-config identities

| Identity | SHA-256 |
|---|---|
| raw-input logical manifest | `f95c0cd141c891f69f44a0ecc792e4878946a3cdc4a1a2ce7911df074b848316` |
| complete batch tensor manifest | `de8b8f06c8c5b14871262fe56167ac52095f8e7cac42387de157b8e247a4e9da` |
| augmentation field order | `0495e2db0984cf3063ef5d0d84a2fd83b99b1b0cf3383f7a78534bbce8bb5de7` |
| augmentation values (`torch.float64`) | `57728184c564966e83d19214e192e8fc79fd84a2701b46b8299c237eb61dd9ea` |
| canonical fixture manifest | `f46a79c1cefa52a65d9e402b791cfce73fa194f20e6aa7cbfb3096957b6b9c89` |

| Cell | Mode/regime | Resolved-config SHA-256 |
|---|---|---|
| C1 | C-STR8 FP32 | `6cfc8f60d1116d1cb161c01d939ee54fac17f9c537ce58eb59fecc419ac25a64` |
| C2 | C-STR8 full FP16 | `f56d0e4bf5d88a96523976ff8bd1ad2cd1b6ecdad3ca835f0643808f21984757` |
| L1 | L-S075 FP32/sparse FP32 | `d2d3fee5a8a38bbfa5200a49cda7a1a31302ddd22e1bbf50af037a9a964da257` |
| L2 | L-S075 full FP16/sparse FP16 | `c77819da84bbfb5293b9044e5f41488d0dcec2f025d1da906632bf2307a3a80d` |
| L3 | L-S075 FP16/sparse FP32 island | `b38cf86fa061b54ef7b85e753a2c33ef5941f57f81a1394843c14f712834ca4b` |
| F1 | F-U FP32/sparse FP32 | `9f49479c96d643ebd2072df22b9a5808f6bcfca6d17ec90c00bddb5e6e5a8201` |
| F2 | F-U full FP16/sparse FP16 | `ee5eac7b7db660ca6e75d904f61579520daec64042c122a9ac82c21b10936d61` |
| F3 | F-U FP16/sparse FP32 island | `1b23d9907ffc6190062be285b203b18951d648ec293707e88f7904835fda9ee9` |

All cells use AdamW `lr=1e-4`, weight decay `0.01`, microbatch/effective global
batch/accumulation `1/1/1`, `num_workers=0`, EMA disabled, and the same per-mode
canonical initialized state and replayed forward RNG. FP32 allows exactly three
attempts. FP16 uses one persistent dynamic GradScaler per cell from scale 512,
backoff 0.5, up to 18 attempts, including scales below one, and stops after three
accepted updates or on a post-accept skip.

### Exact submission

```bash
OUTPUT=/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s08_q1_dbeee35dcd6d
install -d -m 0700 "$OUTPUT"
sbatch --parsable \
  --account=naiss2025-22-1113-gpu \
  --partition=gpu \
  --nodes=1 --ntasks=1 \
  --gpus-per-node=nvidia_gh200_120gb:1 \
  --cpus-per-task=8 --mem=96G --time=01:00:00 --no-requeue \
  --job-name=flv3_s08_q1 \
  --output="$OUTPUT/slurm-%j.out" --error="$OUTPUT/slurm-%j.err" \
  /nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/execution_requests/s08_q1_dbeee35dcd6d/job.sh
```

The runner executes only
`test_s08_q1_primary_precision_qualification`. Pytest success means all cells
completed and evidence was written; each regime is judged only by its explicit
`qualification_pass`. Identity/dependency/config/order/lifecycle/OOM/timeout or
artifact failure is terminal. Expected artifacts are environment/JUnit/log/exit,
the exact fixture and resolved configs, JSONL window records, partial/final Q1
summaries, Slurm stdout/stderr, and a checksum manifest.

Allowed interpretation is limited to stable or unstable bounded optimizer windows
for these eight regimes on one replay-frozen mini fixture, including localization
of nonfinite and large gradients through recorded SECOND/head boundaries. It is
not convergence, performance, capability, mAP/NDS, a final scientific precision
policy, Protocol A/B, attack, or defense evidence.

## S08-Q2 exact compatibility gate — approved under O-109

```text
REQUEST_ID: S08-Q2
REQUEST_STATE: CONSUMED / TERMINAL PASS / Job 435151 COMPLETED 0:0
EXECUTION_SOURCE_SHA: 3bb10d39c60e6fd2d0bfe480bb03a7c8cfc76fe9
Q1_JOB/ELAPSED: 431013 / 00:04:02
O-109 REMAINING BEFORE Q2: 01:55:58 one-GPU elapsed
SNAPSHOT: /nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/execution_snapshots/s08_q2_1d9191c2f623
SNAPSHOT_TREE_SHA256: 1d9191c2f6234199d31405f9690ffd2d83343889333efbe1e1ae47e6235a5c60
SNAPSHOT_FILES/BYTES/WRITABLE: 585 / 4566358 / 0
Q2_TEST_FILE_SHA256: d3bbeb457c7d8b77aa90096684cc2f7c7b5fe1504e97fca6868fb8bb8f3234b2
JOB_SCRIPT: /nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/execution_requests/s08_q2_1d9191c2f623/job.sh
JOB_SCRIPT_SHA256: ff14fd735788a4fa4691a473eb788276d901371160c28f447fe8819f33494d0d
OUTPUT: /nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s08_q2_1d9191c2f623
OUTPUT_STATE_AT_FREEZE: absent
DATA/FIXTURE/SEED: exact Q1 mini fixture and five identities / seed 20260713
CELL_ORDER: P1,B1
P1: L-P020, global FP16, sparse partition not_applicable, uniform
B1: F-CBGS, global FP16, SECOND sparse FP32 island, cbgs identity
P1_RESOLVED_CONFIG_SHA256: 7219c1a3978bf9c0d16efbaa10fa01448fd7e99793ae8c9eb58e492e8dc2d5dd
B1_RESOLVED_CONFIG_SHA256: 49d2ceb0d6a0ae4283c3459805267689d36f811be727cebf38b54d999e50b4b6
MAX_ATTEMPTED_WINDOWS: 18 per cell / 36 total
MAX_ACCEPTED_UPDATES: 1 per cell / 2 total
RESOURCE: one GH200, 8 CPUs, 96 GiB, 00:30:00, no requeue
RETRY: no scientific retry; preserve any bounded negative result
JOB/STATE/EXIT/RESTARTS: 435151 / COMPLETED / 0:0 / 0
SUBMIT/START/END: 2026-07-14T19:02:10 / 19:02:11 / 19:06:07 +02:00
ELAPSED/NODE: 00:03:56 / n207
Q2_RESULT: P1 accepted at scale 8; B1 accepted at scale 16; exact accounting PASS
CUMULATIVE_Q1_Q2_ELAPSED/REMAINING: 00:07:58 / 01:52:02
ARTIFACT_MANIFEST_SHA256: 36b9cbf1eab30f54799cf7abbe83056ac009b301a7817d604a0c8b9abea5fb2f
```

Q2 adds no generic harness and executes only the two compatibility cells in
`test_s08_q2_precision_compatibility`. Each cell must produce one accepted
production-loop window with finite loss/parameter/boundary gradients, exact
optimizer/scheduler/EMA/exposure accounting, and its declared precision route.
The F-CBGS cell binds `det-cbgs=true` in the resolved production config, but the
replay-frozen batch does not claim sampling-distribution or loader-performance
evidence.

```bash
OUTPUT=/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s08_q2_1d9191c2f623
install -d -m 0700 "$OUTPUT"
sbatch --parsable \
  --account=naiss2025-22-1113-gpu --partition=gpu \
  --nodes=1 --ntasks=1 --gpus-per-node=nvidia_gh200_120gb:1 \
  --cpus-per-task=8 --mem=96G --time=00:30:00 --no-requeue \
  --job-name=flv3_s08_q2 \
  --output="$OUTPUT/slurm-%j.out" --error="$OUTPUT/slurm-%j.err" \
  /nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/execution_requests/s08_q2_1d9191c2f623/job.sh
```

Allowed interpretation is only bounded precision/config/model compatibility for
L-P020 and F-CBGS on the exact Q1 fixture. Q2 is not convergence, capability,
sampling quality, performance, full data, metric, Protocol A/B, attack, or defense
evidence.
