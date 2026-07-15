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
> model execution; no retry is authorized, and a new owner amendment is required.

## Authorization state

```text
SESSION_ID: S09
S09_BASE_SHA: 28f79802c0868afa6290d74ae6aeb9d23c7d088f
STOP1_EXECUTION_SOURCE_SHA: 1f276b9d2cc54f705b0b6800a573258707711045
STOP1_REQUEST_COMMIT: d4b64964f56738ec388a39c277f01b3d45a4eeee
STOP1_FIRST_EVIDENCE_SHA: b35591b1a9ac64ea50ee3ad3257304baef07f8de
BRANCH: codex/s08-s09-cl-readiness
OWNER_DIRECTION: O-111 envelope + O-112/O-113 STOP-1 + O-114/O-115/O-116 STOP-2 + O-117 STOP-3
APPROVED_COMPUTE: STOP-1/2 consumed / STOP-3 Job 441511 consumed and terminal failed
APPROVED_SUBMISSIONS: STOP-1 1 consumed / STOP-2 1 consumed / STOP-3 1 consumed / no replacement
ACTIVE_REQUEST: none / STOP-3 pre-model bootstrap FAIL / new owner amendment required
IMPLEMENTATION_COMMIT_AUTHORITY: STOP-3 exact config/runner/request/evidence and review remediation authorized
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

## STOP-4 — conditional G1000

```text
REQUEST_STATE: CONDITIONAL / DEPENDS ON INDEPENDENT STOP-3 PASS AND NEW OWNER APPROVAL
OBJECTIVE: one fresh 1000-successful-step readiness run
WALLTIME_AND_GPU_QUOTA: unset
SUBMISSIONS: unset / no automatic continuation
```

STOP-4 starts from a fresh initialization and is not a mid-epoch resume of
STOP-3. It must freeze the accepted cache, precision policy, base-uniform recipe,
worker count, seed, counters, thresholds, source/config/script hashes, resources,
and output path from the reviewed STOP-3 decision. It remains a single-GH200
engineering readiness run unless measured STOP-3 evidence motivates a separate
owner-approved DDP amendment. It requires independent evidence review before S09
can close.

## Interpretation limits shared by all stops

No S09 stop establishes mAP/NDS, convergence, fusion gain, scientific training-
recipe optimality, a normalization fix, Protocol A/B validity, FL capability,
attack viability, or defense efficacy. Mini or bounded readiness evidence cannot
support those claims. Failed, skipped, or omitted cells and all attempted windows
must remain visible in `RESULTS.md`; a gate may not be weakened retroactively.
