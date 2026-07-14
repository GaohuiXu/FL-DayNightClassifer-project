# S08 precision qualification — implementation handoff

## State

```text
SESSION_ID: S08
MILESTONE_STATE: REVIEW REMEDIATION + FIXTURE-ATTESTATION PASS / REMEDIATION SEAL + RE-REVIEW AUTHORIZED
BASE_AUDIT_COMMIT: 733c84f8e3019fe4d683663821bd86918d3875a7
BRANCH: codex/s08-s09-cl-readiness
IMPLEMENTATION_COMMIT: 791aba97f7bbe92e7708b63f94f2e7d8599f91be
COMPUTE_EXECUTED: Jobs 426619/427800/428889 terminal negative evidence; Jobs 428112/429080 focused smoke PASS
INDEPENDENT_REVIEW: REMEDIATE; REVIEW.md SHA-256 4385f1696d984d50cbdc5037b0384f70453237d78597d24374c4fa6ad4e32569
```

The owner approved S08 envelope v1, local implementation/validation, one immutable
implementation commit after validation, and a resource ceiling of one bounded
GH200 smoke plus the later Q1 request at no more than one hour each. The owner
then explicitly bound `S08-SMOKE-1`. Its sole submission, Job `426619`, failed in
the dependency-provenance preflight before pytest. After O-100 remediation, the
owner explicitly bound `S08-SMOKE-2`; Job `427800` passed runtime attestation but
ended `103 passed, 3 failed`. O-102's narrow remediation was then bound by the
owner as exact `S08-SMOKE-3`; Job `428112` completed `0:0` with 106/106 tests
passing. No retry was attempted. The owner-authorized immutable implementation/
evidence seal was then created as
`791aba97f7bbe92e7708b63f94f2e7d8599f91be`. Independent review preserved the
focused-smoke PASS but returned `REMEDIATE`; Q1 is blocked and unapproved, and S09
remains dependency-only. O-104 authorizes only the linear review remediation,
local validation, and exact S08-SMOKE-4 fixture-attestation request preparation.
The owner separately approved that exact request under O-105. Its sole Job
`428889` passed preflight and 115/116 Phase-1 tests, then stopped on a synthetic
calibration-test no-op before fixture attestation. O-105 is consumed; it grants no
retry or remediation-commit authority. The owner then renewed approval against the
post-freeze Smoke-5 tuple under O-106. Its sole Job `429080` completed `0:0`, zero
restarts, with 116/116 focused tests and 1/1 candidate fixture attestation passing.
O-106 is consumed; it grants no Q1 or remediation-commit authority.
O-108 now separately authorizes one immutable remediation/evidence commit and an
independent re-review pinned to that commit. It grants no Q1 execution authority.

## Independent review and active O-104 remediation

The independent reviewer pinned exact implementation/evidence SHA
`791aba97f7bbe92e7708b63f94f2e7d8599f91be`, inspected the full diff and all three
smoke artifacts, and wrote `REVIEW.md` without changing implementation source.
The review found no P0 and retained Job `428112` as a valid focused-smoke PASS,
but issued:

- P1: the future Q1 runner dynamically recorded the complete replay fixture
  instead of comparing it with predeclared full identities before model
  construction;
- P2: scheduler transitions and the declared EMA-disabled state were recorded but
  not required by each cell's qualification predicate; and
- P2: active canonical/status wording lagged the explicit partition and terminal
  smoke/review state.

Under O-104, the unsealed remediation candidate now prebinds a frozen bounded-mini
raw-input manifest plus full batch, augmentation-field/order, augmentation-value,
and canonical fixture identities; all comparisons occur before task/model/
optimizer construction. It also records pure-Python scheduler/EMA consistency,
requires scheduler continuity/final epoch and EMA-disabled state in the Q1 cell
gate, adds hostile-negative tests, and reconciles active status wording. A
separate fixture-attestation test only derives candidate identities; it constructs
no model or optimizer and cannot be interpreted as Q1. Runtime validation passed
in Job `429080`; immutable resealing and independent re-review are authorized by
O-108.

O-107 also records a prospective collaboration simplification: a future exact
O-009 engineering-smoke request may opt into one capped mechanical remediation
loop, so obvious test/fixture/wrapper/provenance/artifact or output-neutral
diagnostics defects can be fixed, frozen, recorded, and resubmitted without a
second owner review. It does not apply retroactively to these smoke jobs and never
covers Q1 or a model/data/precision/recipe/scientific change.

The exact candidate-only validation request was separately approved once and is
now consumed:

```text
REQUEST_ID: S08-SMOKE-4
REQUEST_STATE: CONSUMED / TERMINAL PHASE-1 FOCUSED-TEST FAIL
JOB_ID/STATE/EXIT: 428889 / FAILED / 1:0
SNAPSHOT: /nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/execution_snapshots/s08_smoke4_425568c1c83d
SNAPSHOT_TREE_SHA256: 425568c1c83df06889c17d305b4ee8a9264b0535d7c204d0d34c8427aa18e90f
SNAPSHOT_FILES/BYTES/WRITABLE: 585 / 4503677 / 0
RUNNER_SHA256: 08b74822862e6e91f14802426b76bfff29dfdd7ace85482a9882a94914941ff1
SUBMIT_SHA256: cb6e3a3da2969d7c522db4a83e07202deef2a2434346aa544aa8094a6e1d2c29
OUTPUT_ROOT: /nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s08_smoke4_425568c1c83d
OUTPUT_ROOT_STATE: preserved terminal-failure artifacts
```

Its first phase repeats Smoke-3 plus ten review-remediation unit cases. Its second
phase hashes only 29 predeclared mini inputs and derives candidate fixture
identities without model/optimizer/Q1 execution. `RUN_REQUEST.md` binds the exact
data, selectors, runtime, command, resource, artifacts, stop conditions, and
interpretation limits. Job `428889` passed the exact runtime/raw-manifest preflight
and 115 tests; the `cam_intrinsics` negative case was a no-op because it assigned
`1.0` to the already-`1.0` first identity-matrix element. Phase 2 therefore did
not start. This is terminal overall FAIL and no retry is authorized.

## Consumed S08-SMOKE-5 replacement — terminal PASS

After Job `428889`, the owner requested S08-SMOKE-5 execution before a changed
immutable tuple existed. Under the fail-closed execution contract, that message
cannot bind a future snapshot. It was used only to authorize the narrow correction
and request preparation. The only source/test behavior change relative to Smoke-4
is:

```text
changed[tensor_name].reshape(-1)[0] = 1.0
    -> changed[tensor_name].reshape(-1)[0].add_(1.0)
```

This changes the actual value for all three parametrized tensors, including the
identity-matrix diagonal. Production source, runner selectors, diagnostics, data,
resources, and acceptance semantics are unchanged. The new request is:

```text
REQUEST_ID: S08-SMOKE-5
REQUEST_STATE: FROZEN / NOT APPROVED AFTER EXACT-TUPLE FREEZE / NOT SUBMITTED
SNAPSHOT: /nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/execution_snapshots/s08_smoke5_51daec3e860e
SNAPSHOT_TREE_SHA256: 51daec3e860e6d412ad57d807efd78a08b03630afb37798880999fa039900a25
SNAPSHOT_FILES/BYTES/WRITABLE: 585 / 4515200 / 0
RUNNER_SHA256: 08b74822862e6e91f14802426b76bfff29dfdd7ace85482a9882a94914941ff1
SUBMIT_SHA256: 254064b207f004ae778f1c73c5e474f0cdf74642a1ba50724adec6e4911ffd40
OUTPUT_ROOT: /nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s08_smoke5_51daec3e860e
OUTPUT_ROOT_STATE: absent
```

The owner renewed approval against this exact post-freeze tuple under O-106. It
was submitted once as Job `429080`:

```text
REQUEST_STATE: CONSUMED / TERMINAL PASS
JOB_ID/STATE/EXIT/RESTARTS: 429080 / COMPLETED / 0:0 / 0
NODE/ELAPSED: n23 / 00:03:36
PHASE_1: 116 passed / 0 failed / 0 errors / 0 skipped
PHASE_2: 1 passed / 0 failed / 0 errors / 0 skipped
TERMINAL_MARKER: S08_PRECISION_SMOKE_PASS
```

The job emitted all five complete fixture identities and its artifact checksum
manifest verified. Exact values and artifact hashes are in `RESULTS.md`. This is
candidate-only fixture evidence: no model, optimizer, Q1 cell, or precision regime
ran. No retry or additional submission was attempted.

## Consumed pre-remediation smoke source

The implementation candidate was copied once into a read-only content snapshot:

```text
SNAPSHOT: /nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/execution_snapshots/s08_smoke_f963da5a620e
SNAPSHOT_TREE_SHA256: f963da5a620e38a479bf9cee3a80af489bb9d212db79848678ff0560a9555ec2
SNAPSHOT_FILES: 577
SNAPSHOT_BYTES: 4386097
WRITABLE_FILES_OR_DIRECTORIES: 0
SMOKE_RUNNER_SHA256: aab4656a339598366e2e0d34927cdf2812119459e15444b6e6a7b7e82487c8c9
```

The tree identity hashes each relative path, executable bit, byte count, and file
SHA-256 in stable NUL-delimited order.  The snapshot contains all tracked files
plus the five then-untracked S08 implementation files.  It intentionally predates
this handoff and the outer submission script; neither is executed inside the job.

An initial local freeze attempt stopped before sealing because non-NUL Git output
quoted one historical Unicode `fl_v2` filename.  That S00-created incomplete
directory was verified to contain exactly the copied candidate, removed, and
replaced by the NUL-safe snapshot above.  No code, data, Git ref, or compute state
was changed by that local packaging error.

## Provenance remediation after Job 426619

O-100 authorizes a narrow source-state remediation without changing or resetting
the external spconv checkout and without executing a replacement job. The current
candidate now:

- replaces the blanket clean-checkout assertion with canonical
  `git-tracked-regular-files.v1` state objects;
- records every permitted change as exact Git two-character status, normalized
  relative path, and working-tree file SHA-256, plus a canonical state SHA-256;
- accepts only unstaged modifications (`" M"`) to existing regular tracked files;
  staged, added, deleted, renamed, copied, conflicted, symlink/directory, malformed,
  duplicate, unsorted, or digest-inconsistent states fail closed;
- continues to ignore untracked files, matching the prior
  `--untracked-files=no` contract;
- binds `spconv_source_state` and `cumm_source_state` into the strict resolved
  config, canonical config hash, run-config bridge, runtime manifest, and Q1 cells;
- verifies source HEAD and exact source state before executable/import hashing,
  then verifies HEAD, import origin, and state again after first import;
- preserves the independent installed executable-build hashes, so allowing the
  known build-metadata patch does not allow executable Python/CUDA/C++ drift.

The exact accepted current states are:

```text
spconv HEAD: 263d6b47425ef843c82f997b12d8b714013d216c
spconv state SHA256: 499efdbb5ab31c43109d48f11ee0ff79af847a3d378fd48bf9c79f8672da28db
spconv only change: " M" pyproject.toml
spconv file SHA256: e2c84544b5b5d6fd8e149d88539c3a6e989a1824637fd6b0006891955cb7a7e9
cumm HEAD: 4dedaf43ff801e417c60c6bd7536a29d83d29ee0
cumm state SHA256: f835ee22d539bbf0ab486fecf1188c3883c3cde5860913434cbcf945ee325662
cumm changes: []
```

No external dependency checkout, installed environment, model, loss, optimizer,
precision partition, diagnostic semantics, data, or Git ref was changed by this
remediation.

## Consumed replacement smoke source

```text
REQUEST_ID: S08-SMOKE-2
REQUEST_STATE: CONSUMED / TERMINAL FOCUSED-TEST FAIL
JOB_ID: 427800
SNAPSHOT: /nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/execution_snapshots/s08_smoke2_935d0464b3bf
SNAPSHOT_TREE_SHA256: 935d0464b3bf11fecb10b5e2c3d6a1f7896c44affc442f225fc4c8685d5a4ce1
SNAPSHOT_FILES: 583
SNAPSHOT_BYTES: 4431728
WRITABLE_FILES_OR_DIRECTORIES: 0
SMOKE_RUNNER_SHA256: 266b83f558b8d9c60f4086d633ac79326cd0dbf3e9c063837d653acf9d44cdf0
SUBMIT_SCRIPT_SHA256: da4bf059256715efa5270305977021edf1a7eefed88de10c690f8e703a209323
JOB_BODY_SHA256: 7ac8d0277576a665690408c6002e7438e311e1e66b3ab8721f2be61856d8003a
OUTPUT_ROOT: /nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s08_smoke2_935d0464b3bf
OUTPUT_ROOT_STATE_AT_REQUEST: absent
OUTPUT_ROOT_TERMINAL_STATE: preserved focused-test failure artifacts
```

The tree digest was reproduced from both the worktree file set and frozen copy
before sealing. The snapshot predates the outer submit-script path rewrite; its
exact runner is immutable and the outer submit script is separately hashed. The
complete selectors, source-state identities, resources, output, stop conditions,
and interpretation limits are in `RUN_REQUEST.md`. The owner separately approved
the exact tuple; its sole submission is Job `427800`, with complete terminal
evidence in `RESULTS.md` and no retry authority.

## Consumed S08-SMOKE-3 request — terminal PASS

O-102 authorized only the narrow remediation, local validation, immutable request
preparation, and no GPU execution. The owner subsequently gave a separate exact
approval for the frozen replacement request, which was submitted once:

```text
REQUEST_ID: S08-SMOKE-3
REQUEST_STATE: CONSUMED / TERMINAL PASS
JOB_ID: 428112
STATE/EXIT/RESTARTS: COMPLETED / 0:0 / 0
ELAPSED/NODE: 00:03:28 / n576
SNAPSHOT: /nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/execution_snapshots/s08_smoke3_3014cab90ed8
SNAPSHOT_TREE_SHA256: 3014cab90ed88b5705367fc1dd1a21740593acc3a186c72f9073bffe15247a43
SNAPSHOT_FILES: 583
SNAPSHOT_BYTES: 4444941
WRITABLE_FILES_OR_DIRECTORIES: 0
SMOKE_RUNNER_SHA256: 266b83f558b8d9c60f4086d633ac79326cd0dbf3e9c063837d653acf9d44cdf0
SUBMIT_SCRIPT_SHA256: 5fa6e31df27425fde7f04373519d39d30c01386fa3e9b487e406048f11bd6ac0
JOB_BODY_SHA256: 7ac8d0277576a665690408c6002e7438e311e1e66b3ab8721f2be61856d8003a
OUTPUT_ROOT: /nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s08_smoke3_3014cab90ed8
OUTPUT_ROOT_STATE_AT_REQUEST: absent
OUTPUT_ROOT_TERMINAL_STATE: complete PASS artifacts preserved
```

The worktree and snapshot tree digests were reproduced independently before all
snapshot entries were made read-only. Relative to S08-SMOKE-2, the only
behavior/test changes are disabled-scaler getter gating and exact `None` record
assertions, plus the corrected test regex; the runner selectors/order, dependency
attestation, resources, data boundary, and acceptance gate are unchanged. The
complete immutable tuple and interpretation limits are in `RUN_REQUEST.md`.
The job passed exact runtime/source-state attestation, all 106 selected tests,
`smoke.exit=0`, checksum verification, and the terminal PASS marker. Complete
artifact hashes and the three dependency-warning records are in `RESULTS.md`.

## Implemented semantics

### 1. Explicit, fail-closed precision partition

- The only accepted production schema is now `s08.v1`.
- Global `precision` remains exactly `fp32 | fp16`.
- New `sparse_conv_precision` is exactly `fp32 | fp16 | not_applicable`.
- Non-SECOND models require `not_applicable`.
- SECOND with global FP32 requires sparse FP32.
- SECOND with global FP16 may explicitly select sparse FP16 or the FP32 island.
- Production construction requires both global and sparse fields, rejects the
  legacy `det-sparse-conv-fp16` boolean, and never infers the partition.
- The partition is part of canonical config bytes/hash, checkpoint identity, and
  diagnostic identity.  Sparse BF16 remains rejected.

Files: `src/fl_v3/config/resolved.py`, `src/fl_v3/config/__init__.py`,
`src/fl_v3/training/tasks.py`, `src/fl_v3/utils/runtime.py`, the centralized
entry point, and the six current config/template JSON files.

### 2. SECOND FP32 island

The existing sparse encoder already disabled outer autocast for
voxelization/mean-VFE and spconv.  The implementation also disables inherited
autocast for the final low-resolution `to_bev` projection when sparse FP16 is not
requested.  Consequently the explicit island keeps voxelization, VFE, SECOND,
dense collapse, and `to_bev` FP32 while the eligible camera/fusion/head path stays
under global FP16 autocast.  Full sparse FP16 retains the reviewed spconv 2.3.8
evaluation workaround and its established FP16 output contract.

### 3. Opt-in window-end diagnostics

`training/precision_diagnostics.py` is a bounded diagnostic component, not a
generic observer framework:

- default-off; no hook registration, file writer, checkpoint state, or output
  mutation;
- supports only complete one-microbatch windows;
- validates source/config/mode/partition and exact optimizer parameter coverage
  before forward;
- captures only explicit `head.input`, `second.output`, `second.stage1`, and
  `second.stem` tensors, using `retain_grad()` and a `finally` cleanup;
- after GradScaler `unscale_` and before clip/step, records strict-JSON loss/task
  terms, finite/nonfinite counts, first bad parameter in named order, FP64 stable
  finite L2/RMS, max finite magnitude, prefix summaries, boundary gradients,
  sparse metadata/voxel counts, RNG identities, scale, and state counters;
- retained activation gradients are never mutated by `unscale_`; an FP64 copy is
  divided by the exact scale, including scales below one;
- every tensor reduction and JSON validation occurs before optimizer step;
  post-step finalization is preallocated pure-Python bookkeeping only;
- a hostile pre-step diagnostic failure clears gradients, records a discarded
  window, leaves parameters unchanged, and releases all retained references.

The training loop's return schema and disabled path remain unchanged.  Optimizer,
scheduler, EMA, exposure, and scaler accounting continue to advance only on
accepted windows.

### 4. Replay-frozen Q1 evidence runner (not executed)

`tests/test_s08_precision_qualification.py` declares eight ordered primary cells:

```text
C1 C-STR8 FP32
C2 C-STR8 global FP16
L1 L-S075 FP32/sparse-FP32
L2 L-S075 global FP16/sparse-FP16
L3 L-S075 global FP16/sparse-FP32 island
F1 F-U FP32/sparse-FP32
F2 F-U global FP16/sparse-FP16
F3 F-U global FP16/sparse-FP32 island
```

Each mode clones one exact initialized state into its regimes.  All attempts use
the same frozen mini sample, 4096 keyframe-only LiDAR point prefix, point order,
camera augmentation parameters, and restored forward RNG. Before any model or
optimizer construction, Q1 requires launcher-bound exact SHA-256 identities for
the bounded raw metadata/sensor inputs, full batch tensor manifest, augmentation
field order, augmentation values, and canonical fixture manifest. FP32 requires three
accepted windows.  FP16 uses one persistent dynamic GradScaler from 512, permits
up to 18 attempts (therefore reaching the predeclared 0.03125-and-below region if
continued backoff is needed), and requires three accepted windows with no later
skip.  Numerical failure in one cell does not suppress the remaining cells;
infrastructure/lifecycle/identity failures remain hard errors.  Raw records are
persisted after each completed cell to avoid losing earlier bounded evidence.
Every qualifying cell additionally requires scheduler continuity, exactly one
scheduler advance per accepted optimizer update, final `last_epoch == 3`, and the
recorded EMA-disabled state. Pytest success means runner completeness; numerical
acceptance remains per-cell `qualification_pass` evidence.

This runner deliberately uses the D1 numerical-isolation fixture: random camera
initialization, AdamW `1e-4/0.01`, constant scheduler, batch one, no EMA, no clip,
no 3D augmentation, no GT paste.  Its cache/manifest fields are explicit hashed
fixture identities and are not a claim that the production ZIP/cache route ran.

L-P020 and F-CBGS compatibility in v1 is limited to fail-closed schema/template
and constructor/sampling regression coverage.  They are not additional Q1
precision cells or scientific comparisons.

## Verification completed locally

PASS on the x86 login environment:

- `python3 -m py_compile` for every changed Python source and test;
- `bash -n` for both S08 shell scripts;
- `git diff --check`;
- pure-Python legal/illegal precision-matrix reconstruction;
- complete resolution of the current `configs/s06_synthetic_camera.json` as
  `s08.v1`, hash
  `e8303eb11ee9793ff8ccc552e9204135905f9cc94e21113b1a3e029fdc23219e`;
- all five S07 candidate templates remain rejected specifically because of
  `template_only`, rather than being silently promoted to runnable configs;
- exact live spconv/cumm source-state reconstruction and expected-state matching;
- pure temporary-Git positive/negative checks for clean state, the exact metadata
  patch, one-byte drift, an additional tracked file, and invalid state digest;
- strict config source-state roundtrip/hash-binding and malformed-state rejection;
- the consumed snapshots' digest/file-count/read-only checks, plus the frozen
  S08-SMOKE-3 digest, 583-file/4,444,941-byte inventory, zero writable entries,
  exact runner/submit/job-body hashes, and output root absent at request freeze.

For the O-104 remediation, local validation additionally passed:

- `python3 -m py_compile` for the changed diagnostics and both changed S08 tests;
- `bash -n` for the Smoke-4 runner and exact outer submit script;
- canonical JSON parse/logical-hash reconstruction for the new raw-input manifest;
- exact size/content verification of all 29 declared mini inputs (41,085,435
  bytes) without directory traversal or trainval access;
- worktree-to-snapshot tree identity reproduction for 585 files/4,503,677 bytes,
  followed by a zero-writable-entry check; and
- `git diff --check`, exact reviewer-artifact SHA preservation, and absent proposed
  output root.

The login node has no usable project Torch/pytest runtime and is x86_64, while
the validated environment is aarch64/GH200.  Therefore no Torch unit test,
spconv forward/backward, or optimizer window is claimed locally.

## Runtime result and gates still open

Job `426619` consumed S08-SMOKE-1 and stopped after 58 seconds because the runtime
verifier rejected the already-known `spconv/pyproject.toml` build-metadata patch.
The same status/diff/content SHA was captured by accepted S07-B-COMPLETE evidence;
no executable spconv source differs, but the pre-remediation verifier required a
wholly clean editable checkout. See `RESULTS.md` for the complete artifact
inventory, hashes, historical comparison, and strict interpretation boundary.

Job `427800` then proved the exact source-state remediation on GH200 and ran all
selected tests. It ended after 106 tests with 103 passed and three focused
failures: two diagnostics tests queried enabled-only GradScaler policy fields on
the real disabled CPU scaler, while one test expected `six task` instead of the
production error's `6 task dictionaries`. The first defect blocks FP32 Q1
diagnostics; the latter is a test-only regex mismatch. Both selected tiny sparse
tests passed. This is not an overall smoke PASS; see `RESULTS.md` for exact JUnit,
logs, hashes, and interpretation limits.

O-102 subsequently authorized only the narrow code/test remediation and
S08-SMOKE-3 request preparation, not GPU execution. After separate exact owner
approval, Job `428112` validated the resulting candidate. The implementation:

- evaluates `scaler.is_enabled()` once at window start;
- keeps disabled-scaler `enabled=false`, `scale_before/after=1.0`, and records
  backoff/growth factor/interval as `None` without calling enabled-only getters;
- retains enabled FP16-scaler policy recording unchanged;
- extends the real disabled-scaler output/RNG-neutrality test to require that
  exact record, allowing both previously blocked diagnostics tests to reach their
  intended paths;
- changes only the test regex from `six task` to the exact existing production
  message `must return 6 task dictionaries`; production loss code is unchanged.

1. Preserve Job `428889` as terminal negative evidence; do not retry it or treat
   115 passing cases as an overall Smoke-4 PASS.
2. Do not edit/reset the external spconv checkout; its remediated attestation
   passed Jobs 427800 and 428112 exactly.
3. Smoke-5 is terminal PASS and O-106 is consumed. Obtain separate owner authority
   before creating the immutable remediation/evidence commit.
4. The independent reviewer then re-reads that exact remediation SHA/diff,
   request, results, and raw artifacts; findings remain P0-P3 first and the
   reviewer does not fix.
5. Only after a re-review readiness verdict and a new exact owner binding may Q1
   be submitted. Precision-policy acceptance remains an owner decision after Q1.

## Explicit non-goals and forbidden interpretations

No architecture, normalization, head/loss/target, optimizer recipe, official
metric/decode/NMS, camera/LSS precision boundary, attack, or defense was changed.
No full trainval/cache scan, 100/1000-step run, profile, DDP, seed expansion,
Protocol A/B, mAP/NDS, capability, convergence, performance, or scientific claim
is authorized or established.  Large but finite LiDAR gradients remain a health
signal to diagnose from Q1 boundary evidence; they are not pre-labeled as a loss
semantic defect or fixed by this implementation.
