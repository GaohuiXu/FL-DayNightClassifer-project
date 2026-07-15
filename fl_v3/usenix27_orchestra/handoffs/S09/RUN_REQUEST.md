# S09 RUN_REQUEST — four-stop execution ledger

> **Ledger state:** O-112 STOP-1 was submitted exactly once as Job `441191` and is
> terminal technical PASS. The submission authority is consumed. First review
> passed raw evidence but returned `REMEDIATE` for durable documentation
> provenance; bounded re-review at `5252a59` closed every finding and returned
> `PASS_WITH_RESIDUAL_RISK`. O-113 owner-accepts STOP-1. The exact STOP-2 smoke
> request is frozen below. O-115 approved that exact tuple and explicitly enabled
> its recorded O-107 mechanical boundary. Initial Job `441293` consumed the
> submission, completed `0:0` in `00:01:04`, and required no replacement; evidence
> remediation `79f87dc` received independent `PASS_WITH_RESIDUAL_RISK` with no
> open P0-P3. O-116 owner-accepts/closes STOP-2. O-117 accepts the updated exact
> STOP-3 envelope, its linear commits, and one derived immutable G100 submission.
> Exact Job `441511` consumed that sole submission and failed before data/loader/
> model execution. O-118 now approves the exact serial recovery below: one
> dependency-attestation submission and, only after its PASS plus independent
> review, one strictly derived replacement G100. Neither phase permits retry.
> Phase A was consumed exactly once by Job `442152` and is terminal technical
> PASS. Independent review accepted evidence `82a0e53` with no open P0-P2 or
> material semantic concern. The strictly derived Phase-B source/snapshot/request
> tuple is frozen below and independent derivation confirmation returned
> `PASS_WITH_RESIDUAL_RISK` with no open P0-P3. The one exact submission was
> consumed by Job `446225` at `2026-07-15T11:09:11+02:00`; it started on `n450`
> one second later and completed `0:0` in `00:05:05`. The production lifecycle
> reports technical PASS for every frozen gate. Immutable evidence `c28d09c`
> received independent `PASS_WITH_RESIDUAL_RISK` with no P0-P2; closure re-review
> of remediation `84adfd0` found no open P0-P3. O-119 owner-accepts/closes
> STOP-3 and approves the serial STOP-4A-D envelope recorded at the end of this
> ledger; no submission occurs before its exact immutable tuple is frozen and
> reviewed. The STOP-4A tuple at request seal `6724762` received independent
> pre-submit `PASS_WITH_RESIDUAL_RISK / no open P0-P3 / SUBMIT GO`; its sole
> command produced Job `452520`, which completed `0:0` in `00:09:42` with all
> four cells and 59 focused tests passing, no replacement or retry. STOP-4B/4C
> implementation `6da4bb5` and evidence closure `1a0b7e3` now have no open P0-P3.
> STOP-4C request seal `131619f` received `REMEDIATE / SUBMIT NO-GO` and was
> never submitted: its old runner did not fail-close the recorded performance
> gates. That tuple and wrapper are forbidden. A source-level runner remediation
> passed immutable review. A completely new exact source/snapshot/config/wrapper/
> output tuple is frozen below and remains unsubmitted pending request review.

## Authorization state

```text
SESSION_ID: S09
S09_BASE_SHA: 28f79802c0868afa6290d74ae6aeb9d23c7d088f
STOP1_EXECUTION_SOURCE_SHA: 1f276b9d2cc54f705b0b6800a573258707711045
STOP1_REQUEST_COMMIT: d4b64964f56738ec388a39c277f01b3d45a4eeee
STOP1_FIRST_EVIDENCE_SHA: b35591b1a9ac64ea50ee3ad3257304baef07f8de
BRANCH: codex/s08-s09-cl-readiness
OWNER_DIRECTION: O-111 envelope + O-112/O-113 STOP-1 + O-114/O-115/O-116 STOP-2 + O-117/O-118 STOP-3 + O-119 STOP-4
APPROVED_COMPUTE: O-119 STOP-4A <=00:30:00 + STOP-4C <=00:30:00 + conditional STOP-4D <=01:00:00 / serial <=2 GPU-hours / no retry
APPROVED_SUBMISSIONS: prior STOP-1/2/3/4A/4C consumed; prospective conditional STOP-4D after exact freeze/review
ACTIVE_REQUEST: STOP-4D exact 5642884 tuple frozen / independent request review pending / not submitted / no retry
IMPLEMENTATION_COMMIT_AUTHORITY: STOP-4 implementation/request/evidence/review remediation within O-119
REQUEST_REMEDIATION/REVIEW: cad72621e0e3ba409ae19bb0b62829118134b2d0 / PASS_WITH_RESIDUAL_RISK / no open P0-P3
MERGE_OR_PUSH_AUTHORITY: none
```

O-111 approved preparation and review of the envelope. O-112 additionally starts
STOP-1 and authorizes its one bounded materialization job after this ledger records
the complete immutable tuple. Each later stop will receive a frozen request block
with exact immutable source/snapshot, resolved config hash, dataset/cache/manifest
identities, cells/order, sample/window/step
bounds, seed, command/script hashes, resources, output root, stop conditions, and
allowed/forbidden interpretation before the owner is asked for approval.

After that one exact approval, S00 creates the stop goal and executes continuously
within the frozen boundary. An altered material tuple requires a new owner
decision. No identical retry, spare-node/GPU expansion, unused-quota transfer, or
conditional next stop is implicit.

## STOP-1 — production `t1.v2` cache identity

```text
REQUEST_STATE: CONSUMED / TERMINAL / REVIEWED PASS_WITH_RESIDUAL_RISK
OBJECTIVE: materialize and attest exact train/val t1.v2 caches for n_sweeps=10
MODEL_OR_TRAINING: none
RESOURCE_CEILING: 1 GH200 / 8 CPU / 96 GiB host / 00:30:00 / 0.5 GPU-hours
SUBMISSIONS: exactly 1 / consumed by Job 441191
RETRY_OR_REPLACEMENT: forbidden
```

### Immutable source and command

```text
SOURCE_SHA: 1f276b9d2cc54f705b0b6800a573258707711045
SOURCE_GIT_TREE: c0d2ecac553e3f2ec81b52b85a633c20c64e5111
SNAPSHOT: /nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/execution_snapshots/s09_stop1_cache_1f276b9d2cc5
SNAPSHOT_REF_MODE: detached / clean / self-contained Git object database
SNAPSHOT_TRACKED_FILES/BYTES: 587 / 4618253
SNAPSHOT_WRITABLE_WORKTREE_ENTRIES: 0
RUNTIME_SOURCE_FILES: 23
RUNTIME_SOURCE_LIST_SHA256: eebaaf9528a56004b63cc2cb37fe6d312b75a52df450f374307e8e559cb1cbb5
RUNTIME_SOURCE_STATE_SHA256: c44db468cb65aaedab7152202ca49056147119b9ef970ffd191fdeeb4258bca8
CACHE_LAUNCHER_SHA256: 212e176df55e20b727c620bbabcd2950f8b64d0b160daa149b76bf5be2390c2e
CACHE_BUILDER_SHA256: 6b9ebf186e97c50c54289ed0b466544a743b469dee71de8e242866e6f8ef97c3
ARRHENIUS_ENV_SHA256: f57befbb5082aaf4d4bb186958a88420ea873e0fdee5c65da1091b73f566c2bf
SUBMIT_SCRIPT: /nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/execution_requests/submit_s09_stop1_cache_1f276b9d2cc5.sh
SUBMIT_SCRIPT_SHA256: aeabbab55b625594a6da9eb820f8b5dae1cdb7e70a6d1d447055a162e093856d
RESOLVED_EXPERIMENT_CONFIG: not applicable; no model or experiment cell
```

The snapshot is a clean detached local clone of the immutable source commit. It
was repacked into its own object database, its alternates file was removed, and
every worktree file/directory is non-writable; only internal `.git` bookkeeping
remains writable so the launcher's read-only `git status` preflight can operate.
The 23-file source set and aggregation algorithm are exactly the launcher's
`LC_ALL=C`-sorted contract. `bash -n` passed for the launcher/environment and
`py_compile` passed for the builder/cache module before freeze.

The only submission command is:

```bash
bash /nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/execution_requests/submit_s09_stop1_cache_1f276b9d2cc5.sh
```

The hash-bound wrapper rechecks detached/clean/source identity, fresh output,
manifest file SHA-256 and an empty exact-name queue, then invokes the unchanged
reviewed S07-A launcher from the snapshot. The historical `S07A_*` environment
variable names and launcher filename are retained to avoid a needless wrapper/
cache-semantic rewrite; the job name, logs, output, approval and interpretation
are S09 STOP-1.

### Exact dataset and accepted manifest input

```text
DATASET_MODULE: nuScenes-data/1.0-map-1.3-zip
DATASET_VERSION/SPLITS: v1.0-trainval / train val
DATAROOT: exact NUSCENES_DATA_DIR exported by the named module and captured in-job
ZIP_MANIFEST_PATH: /nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s01_zip_full_gate_v2_1fe651700bd0/nuscenes_trainval_zip_manifest.sqlite
ZIP_MANIFEST_FORMAT: s01.nuscenes-zip.v2
ZIP_MANIFEST_LOGICAL_SHA256: 023f72b4220bb0db587be00920308bf9074384740fe186d243be92f9a53119f6
ZIP_MANIFEST_FILE_SHA256: 228e2f5bab30007acb06eb61393d1fbacc88979490668ff800f8f7f9752a47fb
ZIP_MANIFEST_BYTES/MODE: 633106432 / 0444
ARCHIVES: exact trainval01_blobs.zip through trainval10_blobs.zip
MANIFEST_OCCURRENCES/UNIQUE/DUPLICATE_OCCURRENCES: 2631093 / 2631084 / 9
N_SWEEPS: 10 total, including the keyframe
EXPECTED_TRAIN_SAMPLES/BOXES: 28130 / 944881
EXPECTED_VAL_SAMPLES/BOXES: 6019 / 187528
```

The accepted manifest path was locally rechecked as present/read-only; its
physical SHA-256, metadata logical hash, format, archive rows/names, and occurrence
counts matched the accepted S01 evidence before request freeze. The job repeats
the physical/logical/archive-name checks before creating its output.

### Exact resources and outputs

```text
ACCOUNT/PARTITION: naiss2025-22-1113-gpu / gpu
NODES/TASKS: 1 / 1
GPU: 1 x nvidia_gh200_120gb
CPU/HOST_MEMORY: 8 / 96 GiB
WALLTIME/GPU-HOUR_CEILING: 00:30:00 / 0.5
JOB_NAME: flv3_s09_stop1_cache
REQUEUE/ARRAY/DDP/RETRY: disabled / none / none / none
OUTPUT_ROOT: /nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s09_stop1_cache_t1v2_1f276b9d2cc5
OUTPUT_STATE_AT_FREEZE: absent
OUTPUT_STATE_AFTER_JOB: complete / checksum-verified / read-only
STDOUT: /nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/logs/s09_stop1_cache_t1v2_%j.out
STDERR: /nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/logs/s09_stop1_cache_t1v2_%j.err
```

Expected output artifacts are:

- `info_cache_msweep10/nuscenes_info_v1.0-trainval_train_t1.v2_nsweeps10.pkl`
  and its `.meta.json` sidecar;
- `info_cache_msweep10/nuscenes_info_v1.0-trainval_val_t1.v2_nsweeps10.pkl`
  and its `.meta.json` sidecar;
- `execution_identity.json`, `runtime_source_sha256s.txt`,
  `cache_identity.json`, and `sha256sums.txt`; and
- exact Slurm stdout/stderr at the paths above.

### Acceptance, stop conditions, and interpretation

PASS requires Slurm `COMPLETED 0:0` with zero restarts; exact source/runtime/
module/manifest identity; fresh output; format `t1.v2`; declared and per-record
depth 10; exact train and val sample/box counts in both loaded records and metadata;
sidecar equality; recomputed canonical content hashes; all physical file SHA-256s;
and successful in-job `sha256sum -c`. STOP on any mismatch, exception, output
collision, or walltime. There is no retry or replacement submission.

The job may traverse the official metadata and previous-sweep metadata needed to
build these records. It does not open/extract sensor payloads, rebuild the ZIP
manifest, construct a model, create a loader sweep, profile, train, evaluate, or
compute mAP/NDS. A reviewed PASS permits proposing only these exact cache files as
S09 production inputs; it does not establish data decode parity, performance,
model readiness, convergence, or scientific capability.

The earlier compact proposal is superseded by the exact tuple above. Independent
data/provenance review is mandatory before STOP-2 can bind the resulting cache.

### Terminal execution record

```text
JOB_ID/STATE/EXIT/RESTARTS: 441191 / COMPLETED / 0:0 / 0
NODE/START/END: n125 / 2026-07-15T06:06:33 / 2026-07-15T06:09:39
ELAPSED/LIMIT/GPU_HOURS: 00:03:06 / 00:30:00 / 0.051667
BATCH_MAX_RSS/MAX_VM/TOTAL_CPU: 9287360K / 13743424K / 02:23.146
OUTPUT_FILES/BYTES/WRITABLE: 8 / 698280214 / 0
IN_JOB_SHA256SUM_CHECK: PASS
POST_JOB_SHA256SUM_CHECK: PASS
RUNTIME_SOURCE_23_FILE_CHECK: PASS
SIDECAR_VS_EMBEDDED_META: PASS
RETRY_OR_FOLLOW_ON: none
```

Both requested splits completed. Exact cache/log/artifact identities and the
allowed/forbidden interpretation are recorded in `RESULTS.md`. O-112's compute
authority is exhausted even though the job used only 0.051667 of its 0.5-GPU-hour
ceiling; unused time is not retry or later-stop authority.

Documentation-only remediation SHA
`5252a591983abb0013f19547e1d6ad20d3d6661f` corrected the first review's P2/P3
provenance/status findings. Bounded independent re-review found no open P0-P3 and
returned `PASS_WITH_RESIDUAL_RISK`; its residual risks and downstream hash-binding
requirements are preserved in `REVIEW.md`.

O-113 owner-accepts this reviewed STOP-1 gate and permits later S09 requests to
bind only these exact cache/manifest identities. It does not reactivate O-112,
authorize a retry, or approve STOP-2 implementation or compute.

## STOP-2 — minimal readiness instrumentation

```text
REQUEST_ID: S09-STOP2-SMOKE
REQUEST_STATE: CONSUMED / JOB 441293 / INDEPENDENT PASS_WITH_RESIDUAL_RISK / OWNER ACCEPTED AND CLOSED UNDER O-116
OWNER_CONFIRMATION: approved exact S09-STOP2-SMOKE tuple and enabled the recorded O-107 boundary
OBJECTIVE: execute the focused Torch/CUDA regression gate for the reviewed output-neutral readiness implementation
MODEL_OR_TRAINING: deterministic toy Linear/MSE loop only / no production detector or production training
RESOURCE_CEILING_PER_SUBMISSION: 1 GH200 / 4 CPU / 32 GiB host / 00:10:00
O107_CUMULATIVE_CEILING: enabled under O-115 / at most 3 submissions / at most 0.5 GPU-hours
CURRENT_SUBMISSIONS: 1 initial / 0 replacements
```

O-114 approved implementation, local/static validation, linear immutable commits,
and independent review, but not GH200 execution. O-115 subsequently approves the
following exact immutable tuple and explicitly opts into its bounded O-107 rule.
No altered source, selector, seed, resource, command family, or output is approved.

### Immutable source and snapshot

```text
PLANNING_BASELINE: 25a59a699fe88b8cec207d5281d6c3342d2d2db0
SOURCE_SHA: 37aef4d6b3f4679d6702d0acef2bb5bd1b57a952
SOURCE_GIT_TREE: d0626e313aab411bc5c71733afb41eca5b102693
FULL_IMPLEMENTATION_DIFF: 25a59a699fe88b8cec207d5281d6c3342d2d2db0..37aef4d6b3f4679d6702d0acef2bb5bd1b57a952
FULL_IMPLEMENTATION_DIFF_SHA256: cb55d4a46c21f3d508e5d73240367d06080de7b456751d802367b19ed055e7eb
SNAPSHOT: /nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/execution_snapshots/s09_stop2_smoke_37aef4d6b3f4
SNAPSHOT_REF_MODE: detached / clean / self-contained Git object database
SNAPSHOT_TRACKED_FILES/BYTES: 590 / 4722741
SNAPSHOT_WRITABLE_WORKTREE_FILES: 0
SNAPSHOT_ALTERNATES: absent
SNAPSHOT_COMMIT_GRAPH: absent
SNAPSHOT_FSCK: git fsck --full --no-reflogs exit 0 / dangling-tree notices only
```

The snapshot reproduces the exact source SHA and tree. Tracked executable files
retain their Git executable bit while all worktree files are non-writable; only
internal `.git` bookkeeping remains writable so the fail-closed status preflight
can run. No uncommitted S00 documentation is part of the execution source.

### Exact selectors and bounded data scope

The job runs exactly these selectors, in this order:

```text
fl_v3/tests/test_s09_readiness.py
fl_v3/tests/test_s06_resolved_config.py
fl_v3/tests/test_s06_training_runtime.py
fl_v3/tests/test_s07_b_integration.py::test_candidate_templates_name_exact_choices_and_fail_closed
```

The expected JUnit total is exactly `44` tests, with zero failures, errors, and
skips. The tests use deterministic toy tensors/configs only. `PYTHONHASHSEED=0` is
fixed; the output-neutral loop fixture uses Torch seed `711`, the bounded loader
fixture uses sampler seed `17`, the readiness lifecycle fixture uses sampler seed
`23`, and existing structural config/runtime fixtures retain their committed
seeds. There is no nuScenes metadata, ZIP/cache, sensor payload, model
qualification, production training window, decode, metric, or performance profile.

The CUDA case exercises direct event creation/resolution. With readiness timing
disabled versus enabled, the test compares final model, optimizer, scheduler,
EMA, scaler and `TrainingState`; all non-timing aggregate metrics; and host/device
RNG state. It does not retain or directly compare per-window model outputs or
gradient tensors. CPU cases cover the strict `s09.v1` contract, attempted-window
bounds, lifecycle refusal, terminal artifact behavior, and output-neutral bounded
loader accounting.

### Frozen scripts and only submission command

```text
REQUEST_DIR: /nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/execution_requests/s09_stop2_smoke_37aef4d6b3f4
JOB_SCRIPT: /nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/execution_requests/s09_stop2_smoke_37aef4d6b3f4/job.sh
JOB_SCRIPT_SHA256: 54bc788c97ed0cd9d0a24e9198043d8e2be18d533011930fb06417f5b8f7bc7f
SUBMIT_SCRIPT: /nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/execution_requests/s09_stop2_smoke_37aef4d6b3f4/submit.sh
SUBMIT_SCRIPT_SHA256: d652e5bea9dca8ada684cc0286d6e4a8a108572e95488798fdc7fdcca7677a8e
SCRIPT_STATE: bash -n PASS / mode 0500 / containing directory non-writable
```

The only authorized initial submission command is:

```bash
bash /nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/execution_requests/s09_stop2_smoke_37aef4d6b3f4/submit.sh
```

The wrapper refuses a changed SHA/tree, attached or dirty snapshot, changed job
script hash, existing output, or active exact-name job before calling `sbatch`.
The job activates the accepted Arrhenius environment from the snapshot, disables
third-party pytest autoload, uses no warnings-as-errors policy or profiler, and
caps pytest at eight minutes inside the ten-minute allocation.

### Exact resources and fresh output

```text
ACCOUNT/PARTITION: naiss2025-22-1113-gpu / gpu
NODES/TASKS: 1 / 1
GPU: 1 x nvidia_gh200_120gb
CPU/HOST_MEMORY: 4 / 32 GiB
WALLTIME: 00:10:00
JOB_NAME: flv3_s09_stop2_smoke
REQUEUE/ARRAY/DDP: disabled / none / none
OUTPUT_ROOT: /nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s09_stop2_smoke_37aef4d6b3f4_a1
OUTPUT_STATE_AT_FREEZE: absent
STDOUT: /nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/logs/s09_stop2_smoke_37aef4d6b3f4_%j.out
STDERR: /nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/logs/s09_stop2_smoke_37aef4d6b3f4_%j.err
```

Expected artifacts are `environment.json`, `selectors.txt`, `pytest.log`,
`pytest.junit.xml`, `pytest.exit`, `acceptance.json`, and
`artifact_sha256s.txt`. The completed output is made non-writable before the
terminal marker.

### Acceptance, stop conditions, and O-107 boundary

PASS requires Slurm `COMPLETED 0:0` with zero restarts; exact source/tree/script
identity; aarch64, CUDA-available GH200 environment; fresh output; pytest exit
zero; exactly `44/0/0/0` JUnit tests/failures/errors/skips; all expected artifacts;
and reproducible artifact hashes. Stop on any identity, environment, output,
selector, count, test, timeout, or artifact mismatch.

O-115 explicitly opts into O-107 only for this engineering smoke: the initial job
plus at most two derived replacements, each
within the same one-GH200/four-CPU/32-GiB/ten-minute ceiling and at most `0.5`
cumulative GPU-hours. A derived replacement may fix only an obvious test, fixture,
wrapper, provenance/artifact, or output-neutral timing-plumbing defect; before
submission it must freeze a new immutable source/snapshot/script/output identity
in this ledger. It may not be an identical retry.

The derived command/output family is prebound as follows:

```text
SUBMISSION_INDEX: integer in {1,2,3}; monotonically increasing; never reused
INITIAL_COMMAND: the exact submit.sh command frozen above
DERIVED_REQUEST_DIR: /nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/execution_requests/s09_stop2_smoke_<derived_source_sha12>_a<submission_index>
DERIVED_COMMAND: bash <DERIVED_REQUEST_DIR>/submit.sh
OUTPUT_NAME_RULE: /nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s09_stop2_smoke_<derived_source_sha12>_a<submission_index>
```

Index `1` is the current initial tuple and resolves exactly to the frozen `a1`
output above. A mechanical replacement uses index `2` or `3`; if source is
unchanged for a wrapper-only fix, the same source prefix is retained and the
index still advances. Every derived wrapper must execute the same four selectors
in the same order with the same toy scope, committed seeds, environment policy,
eight-minute pytest timeout and resource ceiling. Its source SHA/tree, snapshot,
job/submit hashes, request directory and fresh output must be recorded before
submission; an existing path is a hard stop, never an overwrite.

Any possible change to model outputs, losses, gradients, accepted updates, data,
precision policy, optimizer/scheduler/EMA, execution schema semantics, selector or
data scope, seeds, resources, metric, or scientific interpretation ends the loop
and returns to the owner. The same recurring blocker or the submission/GPU-hour
cap also ends it. No spare-GPU expansion or unused-quota transfer is permitted.

This smoke can establish only that the reviewed readiness instrumentation and
strict config/lifecycle regression suite execute correctly in the accepted
Torch/CUDA environment. It cannot establish production data-loader behavior,
throughput, memory headroom, model stability, convergence, mAP/NDS, recipe quality,
Protocol A/B, FL, attack, or defense. Those remain outside STOP-2 or require the
separately approved STOP-3 gate.

### Terminal execution record

```text
REQUEST_APPROVAL_COMMIT: 254872197c0a4b2b3d02ebd8b8e320a49b98a218
JOB_ID/STATE/EXIT/RESTARTS: 441293 / COMPLETED / 0:0 / 0
NODE/START/END: n120 / 2026-07-15T08:30:23 / 2026-07-15T08:31:27
ELAPSED/LIMIT/GPU_HOURS: 00:01:04 / 00:10:00 / 0.017778
PYTEST: 44 passed / 0 failed / 0 errors / 0 skipped / 15.19s
CUDA_OUTPUT_NEUTRAL_TEST: present / passed / 1.996s
OUTPUT_TOP_LEVEL_EVIDENCE_FILES/BYTES: 7 / 8453
OUTPUT_ALL_REGULAR_FILES/BYTES: 18 / 24797
OUTPUT_REGULAR_OR_DIRECTORY_WRITABLE: 0
O107_REPLACEMENTS: 0 / not warranted
```

The job reproduced the exact source/tree and reported aarch64, Python `3.11.15`,
Torch `2.11.0+cu128`, CUDA `12.8`, spconv `2.3.8`, cumm `0.7.13`, one available
`NVIDIA GH200 120GB`, and device memory `102005473280` bytes. Independent S00
parsing confirmed exactly 44 JUnit cases, including the CUDA event test, with no
failure/error/skip. `sha256sum -c artifact_sha256s.txt` passed after completion.

The output also retains pytest's read-only `pytest-tmp` scratch tree: 11 regular
scratch files, 29 scratch directories including its root, and 14 POSIX `current`
symlinks. Those scratch entries are not acceptance evidence and are not included
in the top-level checksum manifest. Every regular file and directory is
non-writable; symlink permission bits are not meaningful and their non-writable
parent directories prevent replacement. Raw output was preserved rather than
post-processed or used to justify a retry.

No O-107 replacement is warranted. The initial PASS terminates this request;
unused submission/time ceiling is not STOP-3 or other compute authority.

## STOP-3 — loader selection and G100

```text
REQUEST_STATE: CONSUMED / JOB 441511 TERMINAL FAILED 1:0 / NO RETRY
PRIMARY_CELL: F-U only
PRECISION: global FP16 autocast + explicit SECOND FP32 island
INITIALIZATION: random / engineering seed 0
OPTIMIZER: AdamW lr=1e-4 weight_decay=0.01
SCHEDULER: constant
EMA/GRAD_CLIP/BEV_AUG/GT_PASTE: disabled
MICROBATCH/ACCUMULATION/GPU: 1 / 1 / 1 GH200
RESOURCE_CEILING: 1 GH200 / 16 CPU / 96 GiB host / 01:00:00 / 1 GPU-hour
SUBMISSIONS: exactly 1 / consumed by Job 441511 / no retry or replacement
RESOLVED_CONFIG_SHA256: cb1723322c756579ab6740eb126de8455b65f808849ec977258c76b919f2c58c
EXECUTION_SOURCE_SHA: 4d6bd829450021aa0813bcece066fb1fac85f478
EXECUTION_SOURCE_TREE: affb4854689a0bf65d829a273d769c87c000174c
SNAPSHOT: /nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/execution_snapshots/s09_stop3_g100_4d6bd8294500
SNAPSHOT_REF_MODE: detached / clean / self-contained Git object database
SNAPSHOT_TRACKED_FILES/BYTES: 592 / 4776222
SNAPSHOT_WRITABLE_WORKTREE_ENTRIES: 0
CONFIG: fl_v3/configs/s09_stop3_f_u_g100.json
CONFIG_FILE_SHA256: e8a17b392c071e3d28c489264d7d051ddfed3d125038a41766250a56dde0083f
RUNNER: fl_v3/scripts/run_s09_stop3_g100.sh
RUNNER_SHA256: 18cca984a65aaa3d462037b931afa100ff046742e06d48fbbeee0c79d0067195
CENTRALIZED_TRAIN_SHA256: 9284d3950541d80417aa1a2a0a1c8e6f41dd4ae46febef53aae3359cbfa959c2
ARRHENIUS_ENV_SHA256: f57befbb5082aaf4d4bb186958a88420ea873e0fdee5c65da1091b73f566c2bf
SUBMIT_SCRIPT: /nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/execution_requests/s09_stop3_g100_4d6bd8294500/submit.sh
SUBMIT_SCRIPT_SHA256: 82790e4cc22c246dc5b458d652d5ca6cb4b9147a9aebb712349e0c0d5c482b1d
JOB_NAME: flv3_s09_stop3_g100
OUTPUT: /nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s09_stop3_g100_4d6bd8294500_a1
STDOUT: /nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/logs/s09_stop3_g100_4d6bd8294500_%j.out
STDERR: /nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/logs/s09_stop3_g100_4d6bd8294500_%j.err
DATAROOT: /dataset/easybuild/data/nuScenes-data/1.0-map-1.3-zip
TRAIN_CACHE_LOGICAL_SHA256: 310e1bba8f65912450e864b634a47b4ca2ea4feb20ed26018e087c93299eed0a
TRAIN_CACHE_PICKLE_SHA256: 57fce20f035a99c0c0ab96fdef418c1b0e04e28bd3e32d191a8298f99919be30
TRAIN_CACHE_SIDECAR_SHA256: f4c45dd12ea0db8ec35d9235de52e51981870b91f175c376d5c34747da661b6c
VAL_CACHE_LOGICAL_SHA256: bb692de4c1eb8b66e8c74f4e807eb208ad891b45ce8f233e8017dc4f3a3b6e2f
VAL_CACHE_PICKLE_SHA256: d4ed7aee9978c2294e2087c917006cbb3d69276453266d0f9c92591340084837
VAL_CACHE_SIDECAR_SHA256: 4f5390815720e14625be31b20fb1596cafe9869ad95b08dc098aea65413be432
ZIP_MANIFEST_LOGICAL_SHA256: 023f72b4220bb0db587be00920308bf9074384740fe186d243be92f9a53119f6
ZIP_MANIFEST_FILE_SHA256: 228e2f5bab30007acb06eb61393d1fbacc88979490668ff800f8f7f9752a47fb
```

The only authorized submission command is:

```bash
bash /nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/execution_requests/s09_stop3_g100_4d6bd8294500/submit.sh
```

That read-only script rechecks the exact detached/clean source/tree, config and
runner file hashes, fresh output path, and empty exact-name queue before calling
`sbatch --parsable` with account `naiss2025-22-1113-gpu`, partition `gpu`, one
node/task, `nvidia_gh200_120gb:1`, 16 CPUs, 96 GiB host memory, `01:00:00`,
`--no-requeue`, and the frozen snapshot as the Slurm working directory. It exports
only the exact snapshot/output/source/tree/resolved-config identities consumed by
the source-controlled runner. No array, DDP, second seed, retry, replacement, or
derived output is authorized.

Job `441511` consumed that sole command at `2026-07-15T09:43:23+02:00` and
terminated `FAILED 1:0` after `00:02:29`, with zero restarts. No replacement was
submitted. Exact source/config/execution identity preflights passed, but
`verify_runtime_dependency_identity` imported editable spconv under the runner's
incorrect `arrhenius_load_modules run` environment. ccimport/ninja could not find
`cublasLt.h` and failed before physical data hashing, loader profiling, model
construction, or an optimizer attempt. The binding environment contract requires
`arrhenius_load_modules build` for runtime jobs because editable sparse imports may
need nvcc/toolkit headers.

The source-controlled runner is corrected after the failed immutable source, but
its unexecuted SHA-256
`855bbd15877a4ceaa6919ccdf9d2ca369e1f3c84ee306415a41376c07d5d8b5d`
is neither an approved replacement tuple nor sufficient by itself: the failed
JIT attempt also changed cumm native artifacts, invalidating any assumption that
the frozen aggregate executable-build hash still holds. O-117 is exhausted. Any
runtime rebuild/attestation, new dependency/config hash, snapshot/request/output,
or replacement G100 requires a new exact owner decision.

The approved immutable request binds:

1. loader-only production ZIP/cache cells at `num_workers=0/2/4/8`, two persistent
   repeats per worker count, with 32 digest, 16 warm-up and 256 measured batches
   per repeat (2,432 bounded decoded samples total);
2. `num_workers=8` hash-bound in the future exact G100 config before execution,
   based on accepted S01 evidence; the fresh loader cells are observational and
   abort/report rather than silently changing that resolved config if they falsify
   the choice;
3. one fresh F-U run requiring 100 successful optimizer updates, stopping after at
   most 120 attempted windows, with the first ten successful updates excluded
   from timing summaries; and
4. stage/end-to-end p50/p95, throughput, data wait, peak allocated/reserved
   memory, optimizer/scheduler/exposure accounting, scaler/nonfinite status, and
   two bounded epoch-time estimates; and
5. one-Hz read-only aggregate GPU utilization, memory utilization/use, power,
   clocks and temperature telemetry. It is not a module/kernel/Tensor-Core
   profiler and does not attribute backward time to C/L/F subgraphs.

O-117 acceptance criteria are: exact data/runtime/config identity; all loader
digests equal; worker-8 warm throughput at least 90% of the best warm cell; exactly
100 successful updates within 120 attempts; zero nonfinite or discarded windows;
no optimizer/scheduler/EMA/exposure drift; at least 95% accepted-window ratio after
the declared warm-up; accepted `(data_wait + CUDA H2D-through-update)` p95/p50 no
greater than 1.5; measured data-wait share no greater than 10%; peak reserved
memory no greater than 86 GiB (`92,341,796,864` bytes); both the dataset-traversal
and accepted-exposure-equivalent estimates no greater than 24 hours; usable GPU
telemetry; and complete checksum-bound artifacts. Aggregate loss must be finite;
monotonic convergence is not a STOP-3 gate or claim.

The epoch formulas are frozen as:

```text
attempted_rate = measured_attempted_samples / measured_wall_seconds
accepted_rate = measured_exposure_samples / measured_wall_seconds
dataset_traversal_hours = 28130 / attempted_rate / 3600
accepted_exposure_equivalent_hours = 28130 / accepted_rate / 3600
```

Val cache pickle/sidecar and the manifest are streamed for physical identity
verification. Val is not deserialized into a model loader, decoded, evaluated, or
used by the 100-update gate.

This gate may report the engineering effect of the unresolved large LiDAR
gradients. It may not change normalization, model/head/loss/target, clipping, or
the base-uniform recipe in response. Such a blocker returns to the owner rather
than being silently optimized away.

If measured performance instead exposes a specific output-neutral bottleneck in
ZIP/cache access, loader lifecycle, H2D transfer, redundant conversion/sync/
allocation, or bounded logging/checkpoint overhead, STOP-3 reports an exact
candidate and equivalence plan. It does not change source or submit another G100
under unused quota. A replacement requires an owner amendment binding the files,
semantics, immutable source/config/command, and resource tuple; STOP-4 remains
blocked until that gate is independently accepted.

## STOP-3 — approved O-118 dependency recovery plus conditional replacement

```text
REQUEST_ID: S09-STOP3-O118-RECOVERY
REQUEST_STATE: O-118 CONSUMED / PHASE A REVIEWED PASS / PHASE B JOB 446225 INDEPENDENTLY REVIEWED TECHNICAL PASS / P3 CLOSED / OWNER-READY
OWNER_CONFIRMATION: "批准 O-118 条件式续行 envelope" / continuous execution within the frozen boundary
PURPOSE: re-attest the drifted editable sparse runtime, then conditionally execute the unchanged O-117 loader/G100 gate
ADDITIONAL_SUBMISSIONS: at most 2 / one dependency attestation + one conditional G100 replacement
ADDITIONAL_GPU_HOUR_CEILING: 1.333334
S09_CUMULATIVE_GPU_HOUR_CEILING_IF_FULLY_USED: 1.444167 / below the standing 2-hour ceiling
RETRY_OR_SPARE_GPU: forbidden
STOP4: forbidden
```

This is one approved serial envelope, not permission inferred from unused O-117
time. Phase B exists only if Phase A and its independent review pass every gate
below. O-118 approves both phases through the exact derivation rule. Phase-A
review passed; Phase-B submission now remains blocked only until independent
confirmation of the frozen derivation below.

### Phase A — exact dependency rebuild/warm and attestation

```text
PHASE_STATE: CONSUMED BY JOB 442152 / REVIEWED PASS_WITH_RESIDUAL_RISK / NO OPEN P0-P2 OR MATERIAL SEMANTIC CONCERN
MODEL/DATA/TRAINING: none / no nuScenes module or read / no model construction / zero training attempts
SOURCE_SHA: 788b493889bcf7be98f36b9cbb6686d51e8e5edf
SOURCE_TREE: 0bc61b3c2693f818ad0feb4e749af64a3947913e
SOURCE_PARENT: 42a9bff34b0517b11de144e7bda42b62524b7d3e
SNAPSHOT: /nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/execution_snapshots/s09_stop3_dep_attest_788b493889bc
SNAPSHOT_REF_MODE: detached / clean / self-contained Git object database
SNAPSHOT_TRACKED_FILES/BYTES: 593 / 4822669
SNAPSHOT_WRITABLE_WORKTREE_ENTRIES: 0
SNAPSHOT_ALTERNATES: absent
SNAPSHOT_FSCK: git fsck --full --no-reflogs exit 0
RUNNER: fl_v3/scripts/run_s09_stop3_dependency_attestation.sh
RUNNER_SHA256: a00d4631600ce92d4954fc8dbbcdc47cf16d636c4e8fa9ad116679a89888282b
CONFIG: fl_v3/configs/s09_stop3_f_u_g100.json
CONFIG_FILE_SHA256: e8a17b392c071e3d28c489264d7d051ddfed3d125038a41766250a56dde0083f
RESOLVED_CONFIG_SHA256: cb1723322c756579ab6740eb126de8455b65f808849ec977258c76b919f2c58c
ARRHENIUS_ENV_SHA256: a56758d72096a65708352e155d1c72adf261ae6cdaf5a56a38f7d2dd5472648f
RUNTIME_IDENTITY_SOURCE_SHA256: da84c73932584e9fe0a2ddd37fce6945e84686a625d3cef91be2bebcf08b78f7
SOURCE_IDENTITY_SOURCE_SHA256: fdda00cdbe910582385fc8bc2d6b2475037e8740c395f1fb55a8c080981b4986
SUBMIT_SCRIPT: /nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/execution_requests/s09_stop3_dep_attest_788b493889bc/submit.sh
SUBMIT_SCRIPT_SHA256: 93848490f485ab38a74ce9818a1ce9d8c35a5eaa17e389fc6b437e9238aa9706
COMMAND: bash /nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/execution_requests/s09_stop3_dep_attest_788b493889bc/submit.sh
JOB_NAME: flv3_s09_stop3_dep_attest
RESOURCE_CEILING: 1 GH200 / 8 CPU / 32 GiB host / 00:20:00 / 0.333334 GPU-hours
SUBMISSIONS: exactly 1 / consumed by Job 442152 / no retry or replacement
OUTPUT: /nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s09_stop3_dep_attest_788b493889bc_a1
STDOUT: /nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/logs/s09_stop3_dep_attest_788b493889bc_%j.out
STDERR: /nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/logs/s09_stop3_dep_attest_788b493889bc_%j.err
SPCONV_VERSION/HEAD/STATE: 2.3.8 / 263d6b47425ef843c82f997b12d8b714013d216c / 499efdbb5ab31c43109d48f11ee0ff79af847a3d378fd48bf9c79f8672da28db
CUMM_VERSION/HEAD/STATE: 0.7.13 / 4dedaf43ff801e417c60c6bd7536a29d83d29ee0 / f835ee22d539bbf0ab486fecf1188c3883c3cde5860913434cbcf945ee325662
PRECOMPUTE_REVIEW: 788b493 PASS_WITH_RESIDUAL_RISK / no open P0-P3
```

The read-only submit script rechecks the exact detached source/tree, no
alternates, zero writable worktree entries, runner/config file hashes, canonical
resolved-config hash, both external source HEADs/tracked states, absent output,
and empty exact-name queue before `sbatch --parsable`. It requests account
`naiss2025-22-1113-gpu`, partition `gpu`, one node/task, one
`nvidia_gh200_120gb`, eight CPUs, 32 GiB, `00:20:00`, `--no-requeue`, no array,
and the frozen snapshot as working directory.

The runner loads `arrhenius_load_modules build`, requires aarch64/GH200 and
`nvcc`, then fail-closes before sparse import on the active installed versions,
editable direct URLs, exact expected checkouts, HEADs, tracked states, and import
origins. The first cumm/spconv import is the only mutation-capable warm/build
operation. It may replace or create native build artifacts under the two exact
editable checkouts and their existing build/cache roots. It may restore only
newly modified tracked paths matching
`{spconv,cumm}/core_cc/**/*.pyi` that were clean in the pre-job tracked state;
every path is recorded separately for `post_warm` and EXIT. It never restores or
changes the accepted pre-existing `spconv/pyproject.toml` modification. Any other
new tracked path is a hard failure and is not silently restored.

PASS requires Slurm `COMPLETED 0:0` with zero restarts; exact identities above;
successful build-module warm import; final spconv/cumm tracked states equal the
accepted hashes; two fresh Python processes producing byte-identical config,
Torch, spconv and cumm identity JSON; stable executable-build hashes; an
`acceptance.json` declaring no data/model/training; original/cleanup/seal/final
statuses all zero; checksum-valid artifacts; and zero writable output entries.
Failure, timeout, unexpected mutation, probe mismatch, cleanup/seal failure, node
loss, missing artifact, or identity drift terminates Phase A and the entire O-118
chain. Native state after any failure remains explicit evidence; there is no
automatic repair or second dependency job.

#### Phase-A terminal execution record

```text
JOB_ID/STATE/EXIT/RESTARTS: 442152 / COMPLETED / 0:0 / 0
SUBMIT/START/END: 2026-07-15T10:37:28 / 2026-07-15T10:37:29 / 2026-07-15T10:49:21
NODE/ELAPSED/LIMIT/GPU_HOURS: n507 / 00:11:52 / 00:20:00 / 0.197778
ACCOUNT/PARTITION/RESOURCES: naiss2025-22-1113-gpu / gpu / 1 GH200 / 8 CPU / 32 GiB
BATCH_MAX_RSS/MAX_VM/TOTAL_CPU: 5041170K / 17851008K / 01:03:52
ACCEPTANCE_SHA256: 4b60f319660124d3bfac23a21bfbfa1b7c66ca920a0e4a4df03b1a512833e9b4
ARTIFACT_MANIFEST_SHA256: b176faa88df06ab955a295cac2ef63e09d51d59427b36c2c8bc11f3b27e73133
PROBE_A/B_SHA256: 52b956995d19b2598836d31246790d4c724cde97e089e84f63132e627ec3d97c / byte-identical
TORCH_BUILD_SHA256: a58ba749ac7947ce123a6af8d4cdc595d2aff5dccccec5d6e10bcfe522040f10
SPCONV_BUILD_SHA256: af42200511a53ce86d77cea0306924a2dc516a74f0483ef7cfe0a6e1dc84b100
CUMM_BUILD_SHA256: 0a7e3c1a8c3e8d41b3b40c4fb77d05bdec8ca2dfce5dbb8863626c4b45d8296d
OUTPUT_FILES/BYTES/DIRECTORIES/WRITABLE: 30 / 1637056 / 1 / 0
ORIGINAL/CLEANUP/SEAL/FINAL_STATUS: 0 / 0 / 0 / 0
DATA_LOADED/MODEL_CONSTRUCTED/TRAINING_ATTEMPTS: false / false / 0
```

The scheduler command, working directory, logs, resource tuple and no-requeue
state match the frozen request. All 29 files named by
`artifact_sha256s.txt` pass `sha256sum -c`; probe A and probe B are byte-identical
fresh-process manifests. Warm-import and both probe stderr files are empty. The
output and Slurm logs are read-only. Before and after the job, tracked external
state remained exact: spconv contains only the accepted modified
`pyproject.toml`, and cumm has no tracked change (the pre-existing untracked
`cumm/core_cc/common.pyi` is outside tracked-source identity). All four generated-
tracked-change inventories are empty, so no tracked stub was restored.

The two cumm `core_cc` copies are now byte-identical at 2,877,128 bytes and
SHA-256 `9970ccc54041c7aee3272604ca7bb4e0d339e7ff37698b74159862c1ba2eface`;
the two spconv copies are byte-identical at 45,180,616 bytes and SHA-256
`37f2ef8dfff8a199d7ffe18b765cd23c26fa1096a1b8ac77082024e086c816aa`.
The aggregate cumm build identity is restored to the S08-accepted value while
spconv has the new stable aggregate identity above. Phase A did not capture a
complete executable-artifact manifest before the warm import, so it does not
support an exact per-file mutation-delta claim. It establishes only the stable,
point-in-time post-warm identity reproduced by two fresh processes. The shared
editable runtime can still drift after the job; Phase B must recheck every
source/build identity before data or model work.

O-118 Phase-A authority is consumed even though the job used less than its
ceiling. No retry occurred or is authorized. Independent review of immutable
evidence `82a0e5315c9098056b6670afb490850cc71dc653` returned
`PASS_WITH_RESIDUAL_RISK`, no open P0-P2, and no material semantic concern. It
explicitly opened only the strictly derived Phase-B path recorded below.

### Phase B — strictly derived replacement loader/G100

Phase B is authorized by O-118 only if Phase A is
terminal PASS and an independent reviewer confirms its exact scheduler record,
raw artifacts, final external source state and emitted build identities with no
open P0-P2 or material semantic concern. Documentation-only P3 closure may be
sealed linearly; it may not alter the derived compute tuple.

The only permitted derivation is:

1. replace only `dependencies.spconv_build_sha256` and
   `dependencies.cumm_build_sha256` in
   `fl_v3/configs/s09_stop3_f_u_g100.json` with the two stable values emitted by
   Phase A; source HEAD/state, versions, Torch identity, data identities, model,
   precision, seed and recipe remain unchanged;
2. update S09 provenance documents and no production behavior; the corrected
   `fl_v3/scripts/run_s09_stop3_g100.sh` remains byte-identical at SHA-256
   `855bbd15877a4ceaa6919ccdf9d2ca369e1f3c84ee306415a41376c07d5d8b5d`;
3. run local config/static tests, create a new linear immutable commit and
   detached self-contained snapshot, then record exact source/tree, raw/resolved
   config hashes, snapshot inventory, runner/submit hashes and fresh paths in
   this ledger before submission; and
4. obtain independent confirmation that the derived diff follows items 1–3.

#### Frozen Phase-B derived tuple

```text
PHASE_STATE: JOB 446225 COMPLETED 0:0 / INDEPENDENTLY REVIEWED TECHNICAL PASS / NO OPEN P0-P3 / OWNER-READY / NO RETRY
PHASE_A_EVIDENCE_SHA/TREE: 82a0e5315c9098056b6670afb490850cc71dc653 / 7428f5978c8d423a7c1855d9e3f858eac718aeae
PHASE_A_REVIEW_SEAL: 386fdbd34c9fe5d420e3ac6c8e439bfe65f6f74d / PASS_WITH_RESIDUAL_RISK / Phase-B GO
DERIVATION_SOURCE_SHA: c200bac861a42fc4338973787d3700e28ddd6c7e
DERIVATION_SOURCE_TREE: c0cc4cb8c2e207e42dcc45a129ada28a3d40feb8
DERIVATION_PARENT: 386fdbd34c9fe5d420e3ac6c8e439bfe65f6f74d
DERIVATION_DIFF: exactly one JSON value / dependencies.spconv_build_sha256 74934de8... -> af422005...
SNAPSHOT: /nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/execution_snapshots/s09_stop3_g100_c200bac861a4
SNAPSHOT_REF_MODE: detached / clean / self-contained Git object database
SNAPSHOT_TRACKED_FILES/BYTES: 593 / 4851938
SNAPSHOT_WRITABLE_WORKTREE_ENTRIES: 0
SNAPSHOT_ALTERNATES/COMMIT_GRAPH: absent / absent
SNAPSHOT_FSCK: git fsck --full --no-reflogs exit 0
CONFIG: fl_v3/configs/s09_stop3_f_u_g100.json
CONFIG_FILE_SHA256: 6733a47203bdf7a4da6e39867e6319a7beb9322257e9149f31b7dff6edacf3ce
RESOLVED_CONFIG_SHA256: ba06b72e4c5f1e54f20472e3286a516e7d4328cfb0fccd8bfc7b13095f597ab6
RUNNER: fl_v3/scripts/run_s09_stop3_g100.sh
RUNNER_SHA256: 855bbd15877a4ceaa6919ccdf9d2ca369e1f3c84ee306415a41376c07d5d8b5d
CENTRALIZED_TRAIN_SHA256: 9284d3950541d80417aa1a2a0a1c8e6f41dd4ae46febef53aae3359cbfa959c2
ARRHENIUS_ENV_SHA256: a56758d72096a65708352e155d1c72adf261ae6cdaf5a56a38f7d2dd5472648f
SUBMIT_SCRIPT: /nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/execution_requests/s09_stop3_g100_c200bac861a4_a2/submit.sh
SUBMIT_SCRIPT_SHA256: 4801ddfee4cd3c04fbc7215c26ffc25efdafc1e6267599bee39bb87491de309e
COMMAND: bash /nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/execution_requests/s09_stop3_g100_c200bac861a4_a2/submit.sh
JOB_NAME: flv3_s09_stop3_g100_r1
RESOURCE_CEILING: 1 GH200 / 16 CPU / 96 GiB host / 01:00:00 / 1 GPU-hour
SUBMISSIONS: exactly 1 conditional replacement / consumed by Job 446225 / no retry
OUTPUT: /nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s09_stop3_g100_c200bac861a4_a2
OUTPUT_STATE_AT_FREEZE: absent
STDOUT: /nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/logs/s09_stop3_g100_c200bac861a4_a2_%j.out
STDERR: /nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/logs/s09_stop3_g100_c200bac861a4_a2_%j.err
EXACT_JOB_NAME_QUEUE_AT_FREEZE: empty
SPCONV_VERSION/HEAD/STATE/BUILD: 2.3.8 / 263d6b47425ef843c82f997b12d8b714013d216c / 499efdbb5ab31c43109d48f11ee0ff79af847a3d378fd48bf9c79f8672da28db / af42200511a53ce86d77cea0306924a2dc516a74f0483ef7cfe0a6e1dc84b100
CUMM_VERSION/HEAD/STATE/BUILD: 0.7.13 / 4dedaf43ff801e417c60c6bd7536a29d83d29ee0 / f835ee22d539bbf0ab486fecf1188c3883c3cde5860913434cbcf945ee325662 / 0a7e3c1a8c3e8d41b3b40c4fb77d05bdec8ca2dfce5dbb8863626c4b45d8296d
LOCAL_VALIDATION: exact semantic-diff assertion + config resolution + git diff --check + runner/submit bash -n and shellcheck PASS
DERIVATION_CONFIRMATION: PASS_WITH_RESIDUAL_RISK / no open P0-P3 / exact frozen wrapper SUBMIT GO
```

The derived config's cumm build value was already equal to Phase A and therefore
produces no textual diff; the only changed byte sequence is the spconv build
hash. Direct config comparison found no other semantic change. The resolver
reproduced fusion mode, seed `0`, 100 successful updates, eight training workers,
global `fp16`, and sparse `fp32`. The runner remains byte-identical. The new
read-only submit wrapper checks the exact detached source/tree, clean/no-alternate/
non-writable snapshot, runner/config hashes, fresh output and empty exact-name
queue before issuing the single frozen `sbatch` command.
Independent review reproduced the complete derivation, snapshot, hashes, current
external sparse state and wrapper preflights, and issued `SUBMIT GO` only for the
exact command above. Any pre-submit drift still cancels that GO.

Job `446225` consumed the command exactly once. Scheduler submission/start were
`2026-07-15T11:09:11/11:09:12+02:00`; the initial `scontrol` record matched
account, partition, job name, source-controlled runner, snapshot working
directory, output/error paths, one GH200, 16 CPUs, 96 GiB, `01:00:00`,
`Requeue=0`, and zero restarts.

#### Phase-B terminal execution record

```text
JOB_ID/STATE/EXIT/RESTARTS: 446225 / COMPLETED / 0:0 / 0
SUBMIT/START/END: 2026-07-15T11:09:11 / 2026-07-15T11:09:12 / 2026-07-15T11:14:17
NODE/ELAPSED/LIMIT/GPU_HOURS: n450 / 00:05:05 / 01:00:00 / 0.084722
ACCOUNT/PARTITION/RESOURCES: naiss2025-22-1113-gpu / gpu / 1 GH200 / 16 CPU / 96 GiB
BATCH_MAX_RSS/MAX_VM/TOTAL_CPU: 5497920K / 26820864K / 04:53.849
READINESS_STATUS: PASS / 100 successful updates within 103 attempts
SCALER_START/END/SKIPS: 512 / 64 / 3 initial overflow windows
NONFINITE/DISCARDED/POST_WARM_INVALID: 0 / 0 / 0
POST_WARM_ACCEPTED_RATIO: 90/90 = 1.0
LOADER_CONTENT_SHA256: 6c6d8f0674c66f756ae3003bc765596994ec1be0816a6db94f3b214ec7925feb / all cells equal
WORKER8_WARM_SAMPLES_PER_SECOND: 141.969756 / 100% of best warm cell
STEADY_WALL_SECONDS_PER_UPDATE: 0.210818
CUDA_ONLY_WINDOW_P50/P95_MS: 208.575935 / 224.153076
COMBINED_DATA_WAIT_PLUS_CUDA_WINDOW_MEAN/P50/P95_MS: 210.760627 / 208.745739 / 224.326678
COMBINED_WINDOW_P95_OVER_P50: 1.074641
DATA_WAIT_SHARE_OF_COMBINED_MEAN: 0.00076355 = 0.076355%
PEAK_ALLOCATED/RESERVED/HEADROOM_GIB: 3.256302 / 6.433594 / 88.566406
ATTEMPTED/ACCEPTED_EPOCH_ESTIMATE_HOURS: 1.647307 / 1.647307
READINESS_SHA256: 08e376e767f654bb38982127ad5ffd84d94ebaa48b3026ceba2ab7ef93a6c9b6
ARTIFACT_MANIFEST_SHA256: b229633889052c46bec5c05d6713e0102aea806a98f9170a65119f9864dbea4b
OUTPUT_FILES/BYTES/DIRECTORIES/WRITABLE: 12 / 3178950 / 2 / 0
```

All eleven files named by the artifact manifest pass `sha256sum -c`; the output
and both Slurm logs are read-only. Runtime/config/stdout JSON identities agree
structurally, and current external spconv/cumm artifact manifests independently
recompute the accepted `af422005...` and `0a7e3c1a...` builds after the job. The
three scaler overflows occurred before the ten-successful-window timing warm-up;
the following 100 optimizer updates were consecutive accepts at scale 64. They
are visible invalid/overflow windows, not hidden updates; the separately frozen
zero-nonfinite and zero-discarded gates both pass.

Every O-117 threshold passes as written, including the exact pairwise combined
`data_wait + CUDA window` ratio above. Detailed loader, timing, memory,
telemetry, counter, artifact and interpretation evidence is in `RESULTS.md`.
O-118 used `0.282500` GPU-hours across Phase A and B; all S09 jobs through this
point used `0.393333` GPU-hours. Unused quota is not retry or STOP-4 authority.

The resulting one-shot G100 keeps every O-117 material field and gate unchanged:
F-U, engineering seed `0`, global FP16 autocast plus SECOND FP32 island, random
initialization, AdamW `1e-4/0.01`, constant scheduler, EMA/clip/BEV augmentation/
GT-paste disabled, microbatch/accumulation `1/1`, production train cache and ZIP
manifest identities, observational loader cells `0/2/4/8` in the same order and
bounds, training workers `8`, 100 successful optimizer updates within 120
attempted windows, first ten successful updates excluded from timing, one-Hz GPU
telemetry, all existing timing/memory/epoch/numerical gates, one GH200, 16 CPUs,
96 GiB and `01:00:00`. The derived naming rule is:

```text
SNAPSHOT: /nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/execution_snapshots/s09_stop3_g100_<derived_source_sha12>
REQUEST_DIR: /nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/execution_requests/s09_stop3_g100_<derived_source_sha12>_a2
OUTPUT: /nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s09_stop3_g100_<derived_source_sha12>_a2
JOB_NAME: flv3_s09_stop3_g100_r1
SUBMISSIONS: exactly 1 conditional replacement / no retry
```

Any deviation from the four derivation items, Phase A evidence/review failure,
new dependency/source drift, changed data/cell/order/bounds/seed/precision/recipe/
resources, or an existing derived path cancels Phase B and returns to the owner.
Unused Phase A or Phase B time is not retry, extra-cell, STOP-4 or other authority.

## STOP-4 — O-119 optimize/G1000/close envelope

```text
REQUEST_STATE: OWNER-APPROVED O-119 / STOP-4A JOB 452520 PASS / STOP-4B REVIEW CLOSED / OLD STOP-4C 131619f SUBMIT NO-GO + NEVER SUBMITTED / REPLACEMENT JOB 455539 INDEPENDENTLY CLOSED PASS + NO RETRY / STOP-4D IMPLEMENTATION REVIEWED + REQUEST FREEZE PENDING
OWNER_DECISION: O-119
SERIAL_GPU_CEILING: 2 cumulative GH200-hours
RETRIES: none
MERGE/PUSH: forbidden
```

O-119 accepts/closes STOP-3 and supplies one prospective derivation rule for the
three serial STOP-4 jobs below. S00 may freeze each source/tree/config/script/
submit/output identity after the preceding implementation/evidence review without
another conversational approval, but may not change any material field. A model,
loss math, gradient/update, data/order, precision, recipe, seed, scope, gate or
resource change cancels remaining authority.

### STOP-4A — bounded profiler and capacity characterization

```text
IMPLEMENTATION_SOURCE_SHA/TREE: b509f5e527c2dd28d2db506c3f87b5a06b3b1b6a / 9c556d37d1e45ece7aad31b10881bb9eb8686424
IMPLEMENTATION_REVIEW: PASS_WITH_RESIDUAL_RISK / no open P0-P3
SNAPSHOT: /nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/execution_snapshots/s09_stop4a_profile_capacity_b509f5e527c2
SNAPSHOT_REF_MODE: detached / clean / self-contained / no alternates / zero writable worktree entries
SNAPSHOT_TRACKED_FILES/BYTES: 598 / 4968985
RUNNER: fl_v3/scripts/run_s09_stop4a_profile_capacity.sh
RUNNER_SHA256: 5b321a7001b7636331d40b5a6d34f55738551f834a77c22b694c7508d26f499e
CENTRALIZED_TRAIN_SHA256: 9a57bc72d620351724888632f93911ef7c19081149619abe108554e1e4c8a478
ARRHENIUS_ENV_SHA256: a56758d72096a65708352e155d1c72adf261ae6cdaf5a56a38f7d2dd5472648f
CONFIG_B1_PROFILE: fl_v3/configs/s09_stop4a_f_u_b1_profile.json
CONFIG_B1_PROFILE_FILE_SHA256: 1bd9bce1b1a34f603990f07d72ac250d38465d9dd5d0a7eb1188012ab7f2eaa6
RESOLVED_B1_PROFILE_SHA256: a0cb86122d607849f479fd04c70acac3b2b7c66d6e65875ad06c638e0db6ad2e
CONFIG_B1_NO_CKPT: fl_v3/configs/s09_stop4a_f_u_b1_no_ckpt.json
CONFIG_B1_NO_CKPT_FILE_SHA256: 555e2b7f278d39e2965cf19e1d15ecd4a8fa0ffe6e358eab2b43ea75219e98d3
RESOLVED_B1_NO_CKPT_SHA256: 5291290d0dbc372eb012bfcc2eeff4877e34db66aa654055c2ebfdf398820a87
CONFIG_B2_NO_CKPT: fl_v3/configs/s09_stop4a_f_u_b2_no_ckpt.json
CONFIG_B2_NO_CKPT_FILE_SHA256: b30e837cc26ef8ce3ec001f0e17a171eec67d0b2e0ba65090f2229acc346d6ff
RESOLVED_B2_NO_CKPT_SHA256: ac841713cf5c996705afb2ddf628965c4fffa169130925285f03c6113d669f6f
CONFIG_B4_NO_CKPT: fl_v3/configs/s09_stop4a_f_u_b4_no_ckpt.json
CONFIG_B4_NO_CKPT_FILE_SHA256: 8d3a3f7847f32c25c319b2ca77fd7a7702457e9c9fcbd02797048a06e8e88f4f
RESOLVED_B4_NO_CKPT_SHA256: cf6f4effe0c9532a45f3a2503a3f98423af2e340b16ae0419d6b287655709a48
DATA: exact accepted STOP-1 train t1.v2 n_sweeps=10 plus accepted ZIP manifest; val identity-bound/not iterated
COMMON_MODEL/PRECISION/SEED: F-U / random seed 0 / global fp16 + SECOND fp32 island
COMMON_RECIPE: AdamW 1e-4/0.01 / constant scheduler / EMA+clip+3D aug+GT paste off / uniform / world1 / accumulation1 / workers8
CELLS_IN_ORDER: B1 checkpoint-on + bounded profiler; B1 checkpoint-off; B2 checkpoint-off; B4 checkpoint-off
BOUNDS_PER_CELL: 20 successful updates / <=30 attempted / no loader profile / fresh process and initialization
PROFILER_SCHEDULE: B1 baseline only / wait5 + warmup2 + active3 attempted windows / record shapes+memory / 250 summary rows
FOCUSED_TESTS: exact selectors frozen in runner before model cells
RESOURCE: one GH200 / 16 CPU / 96 GiB / 00:30:00 / <=0.5 GPU-hours
SUBMISSIONS: exactly one / serial / no retry
OUTPUT: /nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s09_stop4a_profile_capacity_b509f5e527c2_a1
JOB_NAME: flv3_s09_stop4a
STDOUT/STDERR: /nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/logs/s09_stop4a_profile_capacity_b509f5e527c2_%j.{out,err}
SUBMIT_SCRIPT: /nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/execution_requests/s09_stop4a_profile_capacity_b509f5e527c2/submit.sh
SUBMIT_SCRIPT_SHA256: fb59ad993bca04ff3156813c9c584ad83ceefcc35352ca5b1b62ad372fdb315e
REQUEST_REVIEW: PASS_WITH_RESIDUAL_RISK / no open P0-P3 / exact-tuple SUBMIT GO received before Job 452520
```

The sole reviewed and executed submission command was:

```bash
bash /nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/execution_requests/s09_stop4a_profile_capacity_b509f5e527c2/submit.sh
```

The read-only wrapper rechecks the detached source/tree, clean snapshot, absent
alternates/writable worktree entries, runner/trainer and all four raw config
hashes, fresh output, and empty exact-name queue. It invokes one non-requeue
`sbatch` on account `naiss2025-22-1113-gpu`, partition `gpu`, one node/task, one
`nvidia_gh200_120gb`, 16 CPUs, 96 GiB host memory, and `00:30:00`. The snapshot,
output, source/tree and four resolved hashes are passed explicitly to the runner.
At freeze time the output is absent, the exact-name queue is empty, and `sacct`
contains zero prior `flv3_s09_stop4a` jobs since 2026-07-01. No array, DDP,
replacement, retry, extra cell, or derived output path exists.

The independent request reviewer task `/root/s09_stop4a_impl_reviewer` checked
request seal `6724762d1ae7...`, execution source/tree `b509f5e... /
9c556d3...`, the detached 598-file snapshot, four raw/resolved configs, runner,
trainer, caches/manifest, fresh output, empty exact-name queue, and the sole
wrapper. Its pre-submit terminal verdict was `PASS_WITH_RESIDUAL_RISK / no open
P0-P3 / SUBMIT GO`, restricted to the command above and wrapper SHA-256
`fb59ad99...`. This review preceded submission; it is recorded here durably after
the evidence reviewer found the earlier `pending` status had not been sealed.

Mandatory B1 outcomes are terminal PASS with one checksum-bound profiler trace/
summary, 20 accepted updates within 30 attempts, no nonfinite/discarded window and
valid counters. Checkpoint-off B1 must also PASS. B2/B4 PASS establishes bounded
capacity; a directly identified CUDA OOM is retained as `CAPACITY_LIMIT`, not
retried or disguised as a job failure. Any other nonzero cell is terminal FAIL.
Profiler-active windows are diagnostic; post-warm-up direct timing starts only
after the profiler cycle. No worker matrix is run.

Execution record:

```text
JOB_ID: 452520
STATE/EXIT/ELAPSED: COMPLETED / 0:0 / 00:09:42
NODE/RESTARTS: n495 / 0
SUBMISSIONS: 1 / no replacement / no retry
CELL_STATUS: B1-profile PASS; B1-no-checkpoint PASS; B2 PASS; B4 PASS
FOCUSED_TESTS: 59 passed / 0 failed / 0 skipped
ARTIFACT_MANIFEST_SHA256: fbd07beebcd9078c5a980995e05febc1efc873469d0ff4fe61f30c6748f5272f
SLURM_STDOUT_SHA256: e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
SLURM_STDERR_SHA256: 8db5d05b4abfa9c9cc1bd7028c410675c3e2d697af110ce6c6d9aa51f2e1e830
SLURM_LOG_MODE: 0444 / read-only
```

All 35 manifest entries pass checksum verification; all cell exit files and the
terminal exit are zero. `RESULTS.md` records the exact metrics, trace counts,
capacity interpretation, and prohibition on deriving an S09 batch-size change.

### STOP-4B/4C — measured output-neutral remediation and optimized G100

STOP-4B may remove only source/trace-confirmed synchronization or allocation and
make checkpoint-off explicit. The known source candidate is unconditional
per-task loss-term `.item()` recording when telemetry is disabled; remediation
must propagate `record_terms` across the six task losses while preserving S08
precision-diagnostic terms. A criterion-buffer device move or any other change is
allowed only if STOP-4A confirms it and equivalence tests show unchanged output,
loss, gradients, updates, RNG, data order, precision and counters.

```text
OLD_REQUEST_SEAL: 131619f0940bd3c453969f4d211bdaa775bacbb8 / REMEDIATE / SUBMIT NO-GO / NEVER SUBMITTED
OLD_TUPLE_STATUS: forbidden; retained below as negative request evidence only
IMPLEMENTATION_SHA: 6da4bb5016410708b1e731d26d898f24e6b315ac
IMPLEMENTATION_TREE: 721165340f2b5ab4cda222b4f3a86e951f9d7c14
EXECUTION_SOURCE_SHA: 1a0b7e38805d86fb42ff4fe84d67e1680de55015
EXECUTION_SOURCE_TREE: b76d9a480bcd9654ae63e72bdbb5d99191902829
IMPLEMENTATION_EVIDENCE_REVIEW: PASS_WITH_RESIDUAL_RISK / no open P0-P3 / closure GO
SNAPSHOT: /nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/execution_snapshots/s09_stop4c_g100_1a0b7e38805d
SNAPSHOT_REF_MODE: detached / clean / self-contained / no alternates / zero writable worktree entries
SNAPSHOT_TRACKED_FILES/BYTES: 600 / 5003986
CONFIG: fl_v3/configs/s09_stop4c_f_u_g100.json
CONFIG_FILE_SHA256: 8ca905ade59214822d9c5b894c02786af77f6f531299ed1ca25caf51d00a35ce
RESOLVED_CONFIG_SHA256: afcd002184e35158e129353dfb9b621c390555b5927a37fa5f5acd9547538980
RUNNER: fl_v3/scripts/run_s09_stop4c_g100.sh
RUNNER_SHA256: 614703d8a5f88a85838ba6fcceb3a3b4839fb9548efe877a79257e6bec1fd307
CENTRALIZED_TRAIN_SHA256: 9a57bc72d620351724888632f93911ef7c19081149619abe108554e1e4c8a478
ARRHENIUS_ENV_SHA256: a56758d72096a65708352e155d1c72adf261ae6cdaf5a56a38f7d2dd5472648f
DATA: exact accepted STOP-1 train t1.v2 n_sweeps=10 plus accepted ZIP manifest; val identity-bound/not iterated
CELL: F-U / B1 / random seed 0 / global fp16 + SECOND fp32 island / checkpoint off / no loader or operator profiler
RECIPE: AdamW 1e-4/0.01 / constant scheduler / EMA+clip+3D aug+GT paste off / uniform / world1 / accumulation1 / workers8
BOUND: 100 successful updates / <=120 attempted / ten-successful-window warm-up
FOCUSED_TESTS: s06 resolved config + full s09 readiness + full s08 precision diagnostics/partition + checkpoint switch + true six-task record_terms equality
OUTPUT: /nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s09_stop4c_g100_1a0b7e38805d_a1
JOB_NAME: flv3_s09_stop4c_g100
STDOUT/STDERR: /nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/logs/s09_stop4c_g100_1a0b7e38805d_%j.{out,err}
SUBMIT_SCRIPT: /nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/execution_requests/s09_stop4c_g100_1a0b7e38805d/submit.sh
SUBMIT_SCRIPT_SHA256: eec841bf452f2f5c8adc0908c67c538aaee1c2842322313e5beb4096e7ae00be
RESOURCE: one GH200 / 16 CPU / 96 GiB / 00:30:00 / <=0.5 GPU-hours
CUMULATIVE_O119: STOP-4A used 0.161667 GPU-hours; STOP-4C+4D ceilings would total <=1.661667 GPU-hours
SUBMISSIONS: 0 / old tuple forbidden / no retry
REQUEST_REVIEW: REMEDIATE / open P2 performance gates absent from frozen runner / SUBMIT NO-GO
```

The following historical command was never executed and is forbidden:

```bash
bash /nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/execution_requests/s09_stop4c_g100_1a0b7e38805d/submit.sh
```

After review-time mechanical mode repair, the old wrapper is mode `0555` and its
safe preflight (all lines before `sbatch`) passes. It rechecks the detached
source/tree, clean snapshot, absent alternates/
writable worktree entries, runner/trainer/environment/config hashes, fresh output,
and empty exact-name queue. It contains exactly one non-requeue `sbatch` on
account `naiss2025-22-1113-gpu`, partition `gpu`, one node/task/GH200, 16 CPUs,
96 GiB and `00:30:00`. Output, queue and history were fresh and no job was
submitted. However, the first snapshot freeze had incorrectly stripped 37
executable bits, so its original `clean` assertion was false; restoring Git-index
modes closed that packaging defect without changing contents/tree. More
importantly, old runner SHA `614703d8...` does not implement the performance gates
below, so the entire old tuple remains invalid regardless of the repaired modes.

Technical PASS requires all focused tests and runner evidence validation to pass;
100 accepted updates within 120 attempts; zero direct-nonfinite/discarded windows;
exact optimizer/scheduler/exposure/EMA/scaler accounting; post-warm accepted ratio
`>=95%`; combined `(data wait + CUDA window)` p95/p50 `<=1.5`; mean data-wait
share `<=10%`; peak reserved memory `<=86 GiB`; both steady epoch estimates
`<=24 h`; and finite aggregate loss. The exact combined p50/p95 and stage timings
are compared with accepted STOP-3 values `208.745739 / 224.326678 ms`. Either
combined p50 or p95 exceeding `1.10x` its reference (`229.620313 /
246.759346 ms`) is a material engineering regression and blocks STOP-4D. No job
retry follows any failure.

The replacement runner adds all of these checks to the existing terminal
validation artifact. A positive replay over accepted STOP-3 evidence reproduces
combined p50/p95 `208.745738839 / 224.326677561 ms`, ratio `1.074640751`,
data-wait share `0.000763551`, and both epoch estimates `1.647306968 h`, returning
PASS. A negative replay with measured CUDA windows multiplied by two returns
exit `4` and the frozen 1.10x-regression error. The new immutable source,
snapshot, hashes, wrapper and fresh output are now bound by the replacement tuple
below; no identity from the forbidden old tuple is reused.

Allowed interpretation is one exact B1 engineering before/after regression under
the accepted O-110 precision partition. It may establish retained 100-update
health and quantify checkpoint-off plus source-proven synchronization removal. It
cannot establish convergence, recipe quality, mAP/NDS, fusion gain, backward
branch attribution, multi-seed behavior, Protocol A/B, FL, attack, or defense.

#### STOP-4C replacement exact tuple — consumed by Job 455539 / evidence review remediation pending

```text
RUNNER_REMEDIATION_SHA/TREE: 72a09d5a503a258f3f257b208180585d16ee49d0 / 887d275b71a7f6ccd34cf67188e9fac0843393c1
RUNNER_REMEDIATION_REVIEW: PASS_WITH_RESIDUAL_RISK / no open P0-P3 / closure GO
EXECUTION_SOURCE_SHA: c7769901201b8c507997dfa9ff5154fbe6dbb297
EXECUTION_SOURCE_TREE: 1e2c4464d2582d81e7ef7fef4740c764d0a48e8c
SNAPSHOT: /nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/execution_snapshots/s09_stop4c_g100_c7769901201b
SNAPSHOT_REF_MODE: detached / clean / self-contained / no alternates / zero writable worktree entries / executable modes preserved
SNAPSHOT_TRACKED_FILES/BYTES: 600 / 5024857
CONFIG: fl_v3/configs/s09_stop4c_f_u_g100.json
CONFIG_FILE_SHA256: 8ca905ade59214822d9c5b894c02786af77f6f531299ed1ca25caf51d00a35ce
RESOLVED_CONFIG_SHA256: afcd002184e35158e129353dfb9b621c390555b5927a37fa5f5acd9547538980
RUNNER: fl_v3/scripts/run_s09_stop4c_g100.sh
RUNNER_SHA256: a899deb5a8a68541d2e7b361c816ae49bd873b16b3c18030e0fc54c65717daa5
CENTRALIZED_TRAIN_SHA256: 3dffb4fe70ab2c82ac0192a07b3bcebfbca5232e85c1e32f4e6e4a44b783530f
ARRHENIUS_ENV_SHA256: a56758d72096a65708352e155d1c72adf261ae6cdaf5a56a38f7d2dd5472648f
DATA/MODEL/PRECISION/RECIPE/CELL/BOUNDS/GATES: exactly unchanged from the reviewed replacement contract above
OUTPUT: /nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s09_stop4c_g100_c7769901201b_a1
JOB_NAME: flv3_s09_stop4c_g100
STDOUT/STDERR: /nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/logs/s09_stop4c_g100_c7769901201b_%j.{out,err}
SUBMIT_SCRIPT: /nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/execution_requests/s09_stop4c_g100_c7769901201b/submit.sh
SUBMIT_SCRIPT_SHA256: e44db31b1dd14960d346e4df5bcfdf108cedfdaa7561a6559dc2658b9e0d87a9
RESOURCE: one GH200 / 16 CPU / 96 GiB / 00:30:00 / <=0.5 GPU-hours
CUMULATIVE_O119: STOP-4A used 0.161667 GPU-hours; STOP-4C+4D ceilings total <=1.661667 GPU-hours
SUBMISSIONS: exactly one after independent exact-request GO / no retry
REQUEST_REVIEW: PASS_WITH_RESIDUAL_RISK / no open P0-P3 / SUBMIT GO
JOB_ID: 455539
SUBMIT_TIME: 2026-07-15T14:14:48+02:00
SUBMISSION_COUNT: 1 / exact replacement tuple / no retry
JOB_STATE/EXIT/RESTARTS: COMPLETED / 0:0 / 0
JOB_NODE/START/END/ELAPSED: n414 / 2026-07-15T14:14:49+02:00 / 2026-07-15T14:18:55+02:00 / 00:04:06
ACTUAL_GPU_HOURS: 0.068333
TECHNICAL_RESULT: PASS / independently reviewed / STOP-4D conditional release GO
```

The sole submitted replacement command was:

```bash
bash /nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/execution_requests/s09_stop4c_g100_c7769901201b/submit.sh
```

This is not a retry: the superseded tuple never submitted a job. The new snapshot
was cloned from the reviewed closure source and frozen by removing write bits
without changing Git executable modes; it is detached, clean and read-only. The
mode-0555 wrapper's safe prefix passes and binds all content hashes above, fresh
output, and an empty exact-name queue. It contains one non-requeue `sbatch` with
the unchanged O-119 resources and passes the exact config hash into the runner.
At freeze time the replacement output is absent and `squeue`/`sacct` contain no
exact-name job. Any drift or request-review finding cancels this tuple.

### STOP-4D — fresh optimized G1000

```text
IMPLEMENTATION_REVIEW: PASS_WITH_RESIDUAL_RISK / no open P0-P3 / exact request freeze GO
EXECUTION_SOURCE_SHA: 5642884cdbb16e1c9b3107f529dc70b3a1243c6a
EXECUTION_SOURCE_TREE: b13a08819b2e203dfe355309f1310c79f94f3023
SNAPSHOT: /nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/execution_snapshots/s09_stop4d_g1000_5642884cdbb1
SNAPSHOT_REF_MODE: detached / clean / self-contained / no alternates / zero writable worktree entries / executable modes preserved
SNAPSHOT_TRACKED_FILES/BYTES: 602 / 5055331
CONFIG: fl_v3/configs/s09_stop4d_f_u_g1000.json
CONFIG_FILE_SHA256: dfd46e1a179b3b10d98055762fe8cfc9f9f312f4faa5aec05c9f5b14a7b37928
RESOLVED_CONFIG_SHA256: c3b39a3f9dbfccd673a494f8ec976aa0cad1424a63cda3e56f836b4b733f7a1b
RUNNER: fl_v3/scripts/run_s09_stop4d_g1000.sh
RUNNER_SHA256: 43511df4f54265bfc9595aed424aa363a91ff3cc855ab0fb5fe43162885961dc
CENTRALIZED_TRAIN_SHA256: 3dffb4fe70ab2c82ac0192a07b3bcebfbca5232e85c1e32f4e6e4a44b783530f
ARRHENIUS_ENV_SHA256: a56758d72096a65708352e155d1c72adf261ae6cdaf5a56a38f7d2dd5472648f
DATA: exact accepted STOP-1 train t1.v2 n_sweeps=10 plus accepted ZIP manifest; val identity-bound/not iterated
CELL: exact accepted STOP-4C F-U B1 / fresh random seed-0 initialization / global fp16 + SECOND fp32 island / checkpoint off / no resume, loader profile or operator profiler
RECIPE: AdamW 1e-4/0.01 / constant scheduler / EMA+clip+3D aug+GT paste off / uniform / world1 / accumulation1 / workers8
BOUND: 1000 successful updates / <=1020 attempted / ten-successful-window warm-up
FOCUSED_TESTS: s06 resolved config including exact 4C->4D delta + full s09 readiness + full s08 precision diagnostics/partition + checkpoint switch + true six-task record_terms equality
GATES: all focused tests and exits PASS; exact 1000/1020 lifecycle and optimizer/scheduler/exposure/EMA/scaler accounting; zero direct-nonfinite/discarded windows; post-warm accepted ratio >=95%; pairwise combined p95/p50 <=1.5; mean data-wait share <=10%; reserved memory <=86 GiB; both steady epoch estimates <=24 h; finite aggregate loss; combined p50/p95 <=229.620313/246.759346 ms
OUTPUT: /nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s09_stop4d_g1000_5642884cdbb1_a1
JOB_NAME: flv3_s09_stop4d_g1000
STDOUT/STDERR: /nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/logs/s09_stop4d_g1000_5642884cdbb1_%j.{out,err}
SUBMIT_SCRIPT: /nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/execution_requests/s09_stop4d_g1000_5642884cdbb1/submit.sh
SUBMIT_SCRIPT_SHA256: e2c8c5b32b63bb50ccc8eedeb1fdcf1aa807ae30fc06661bf7b7864ba3a69e28
RESOURCE: one GH200 / 16 CPU / 96 GiB / 01:00:00 / <=1 GPU-hour
SUBMISSIONS: exactly one conditional on reviewed STOP-4C PASS / no retry
O119_ACTUAL_PLUS_STOP4D_CEILING: 0.230000 used + 1.000000 <= 1.230000 / 2.000000 GPU-hours
REQUEST_REVIEW: pending / do not submit before independent exact-tuple GO
```

The sole prospective command is:

```bash
bash /nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/execution_requests/s09_stop4d_g1000_5642884cdbb1/submit.sh
```

The mode-0555 wrapper passes syntax, ShellCheck and safe-prefix execution. It
rechecks the exact detached source/tree, clean self-contained snapshot, absence
of writable worktree entries and alternates, runner/trainer/environment/raw
config hashes, fresh output and empty exact-name queue before one non-requeue
`sbatch`. At freeze time the output, exact-name queue and same-name history since
2026-07-01 are empty. Any drift or request-review finding cancels this tuple.

Technical PASS records all 1000-success lifecycle, performance, memory and
telemetry evidence. Any nonzero test/training/validation/final exit, missed gate,
cap exhaustion or Slurm failure is retained as terminal negative evidence with no
retry. A PASS remains engineering health only, not convergence or model quality.

Independent review of the complete STOP-4 source, exact diffs, configs, jobs and
raw artifacts is required before S09 can close. B=2/4 results remain capacity
evidence for S10 and do not authorize a batch/learning-rate/recipe selection.

## Interpretation limits shared by all stops

No S09 stop establishes mAP/NDS, convergence, fusion gain, scientific training-
recipe optimality, a normalization fix, Protocol A/B validity, FL capability,
attack viability, or defense efficacy. Mini or bounded readiness evidence cannot
support those claims. Failed, skipped, or omitted cells and all attempted windows
must remain visible in `RESULTS.md`; a gate may not be weakened retroactively.
