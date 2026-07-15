# S09 RUN_REQUEST — four-stop execution ledger

> **Ledger state:** O-112 STOP-1 was submitted exactly once as Job `441191` and is
> terminal technical PASS. The submission authority is consumed. First review
> passed raw evidence but returned `REMEDIATE` for durable documentation
> provenance; bounded re-review at `5252a59` closed every finding and returned
> `PASS_WITH_RESIDUAL_RISK`. Owner STOP-1 acceptance is pending and no later stop
> is authorized.

## Authorization state

```text
SESSION_ID: S09
S09_BASE_SHA: 28f79802c0868afa6290d74ae6aeb9d23c7d088f
STOP1_EXECUTION_SOURCE_SHA: 1f276b9d2cc54f705b0b6800a573258707711045
STOP1_REQUEST_COMMIT: d4b64964f56738ec388a39c277f01b3d45a4eeee
STOP1_FIRST_EVIDENCE_SHA: b35591b1a9ac64ea50ee3ad3257304baef07f8de
BRANCH: codex/s08-s09-cl-readiness
OWNER_DIRECTION: O-111 envelope + O-112 STOP-1 execution
APPROVED_COMPUTE: STOP-1 only / <=0.5 GPU-hours
APPROVED_SUBMISSIONS: 1 / consumed by Job 441191
ACTIVE_REQUEST: none / S09-STOP1-DATA reviewed PASS_WITH_RESIDUAL_RISK / owner decision pending
IMPLEMENTATION_COMMIT_AUTHORITY: no production implementation; linear STOP-1 docs/evidence commits allowed
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

## STOP-2 — minimal readiness instrumentation

```text
REQUEST_STATE: PROPOSED / DEPENDS ON STOP-1 / NOT APPROVED
OBJECTIVE: implement and validate output-neutral performance/readiness accounting
PROPOSED_RUNTIME_SMOKE_CEILING: 1 GH200 / 8 CPU / 96 GiB host / 00:30:00 / 0.5 GPU-hours
PROPOSED_SUBMISSIONS: 1, or at most 3 only if the future exact request explicitly opts into O-107
```

The future implementation request will bind the file envelope in `HANDOFF.md`,
tests, local validation, immutable implementation-commit authority, and reviewer
scope. Any GH200 smoke must be a focused implementation/runtime check within
O-009, not a model-qualification or production training run. If the owner
explicitly opts into O-107, derived replacements are limited to obvious test,
fixture, wrapper, provenance/artifact, or output-neutral diagnostic-plumbing
defects and remain inside O-009's cumulative two-GPU-hour ceiling. Model output,
loss/gradient, precision, data, recipe, cell, seed, or resource changes end the
loop.

The accepted instrumentation must use direct bounded timestamps/CUDA events and
memory counters only. The S08 precision observer, forward/backward hooks,
activation retention, general profiler, sampler/checkpoint redesign, and retired
harnesses are forbidden.

## STOP-3 — loader selection and G100

```text
REQUEST_STATE: PROPOSED / DEPENDS ON REVIEWED STOP-2 / NOT APPROVED
PRIMARY_CELL: F-U only
PRECISION: global FP16 autocast + explicit SECOND FP32 island
INITIALIZATION: random / exact engineering seed to be frozen
OPTIMIZER: AdamW lr=1e-4 weight_decay=0.01
SCHEDULER: constant
EMA/GRAD_CLIP/BEV_AUG/GT_PASTE: disabled
MICROBATCH/ACCUMULATION/GPU: 1 / 1 / 1 GH200
PROPOSED_RESOURCE_CEILING: 1 GH200 / <= 01:00:00 / 1 GPU-hour
PROPOSED_SUBMISSIONS: 1
```

The exact future request will bind:

1. loader-only production ZIP/cache cells at `num_workers=0/2/4/8`, two persistent
   repeats per worker count, with 16 warm-up batches and 256 measured batches per
   repeat;
2. a single worker count selected by a predeclared rule and frozen before model
   training begins (8 is only the current provisional default);
3. one fresh F-U run requiring 100 successful optimizer updates, stopping after at
   most 120 attempted windows, with the first ten successful updates excluded
   from timing summaries; and
4. stage/end-to-end p50/p95, throughput, data wait, peak allocated/reserved
   memory, optimizer/scheduler/exposure accounting, scaler/nonfinite status, and
   a bounded epoch-time estimate.

Candidate acceptance criteria, to be finalized before approval, are: zero
nonfinite accepted windows; no optimizer/scheduler/exposure drift; at least 95%
accepted-window ratio after the declared warm-up; p95/p50 end-to-end ratio no
greater than 1.5; peak reserved memory no greater than 86 GiB; complete artifacts;
and a provisional estimated epoch ceiling of 24 hours. The denominator and exact
handling of data exhaustion/restarts will be explicit in the frozen request.

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
