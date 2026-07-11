# S01 RUN_REQUEST — bounded shared-ZIP reader smoke

## Approval state

- **Status:** `APPROVED_FOR_THIS_EXACT_BOUNDED_SMOKE`.
- **Owner-specific approval:** owner message in the active S01 session on
  2026-07-10 explicitly approved a short Slurm smoke to verify that the ZIP reader
  can discover/open the Arrhenius shared dataset and read a small number of camera,
  LiDAR, and sweep members.
- **Active-session amendment:** S00/owner subsequently confirmed that S01 may submit
  this limited engineering smoke directly, without waiting for S00, and instructed
  this record to cite the standing smoke policy `O-009`.
- This approval does **not** include the ten-archive exhaustive manifest/full
  train+val coverage scan, full-data throughput profile, full epoch/eval, job array,
  DDP/multi-node, seed expansion, matrix, rerun, or automatic resubmission.

## Immutable execution identity

- Repository root:
  `/home/gaohui/.codex/worktrees/1ab2/fl_weather_project`
- Git HEAD / BASE_SHA:
  `f262f6bea037580065a8505008773c04fdd259f5`
- Ref mode: detached HEAD.
- The active-session amendment explicitly permits this smoke to bind to HEAD plus
  the uncommitted working-tree state rather than requiring a worker commit.
- S01 runtime source-state SHA-256:
  `ee7c030911d5c2a99f7e60c73df4454d2f93d15de6881e0f081ceb04c1de0869`
  - Definition: SHA-256 of the sorted per-file SHA-256 list for every non-pyc file
    under `fl_v3/src/fl_v3/data/nuscenes/` plus
    `s01_nuscenes_zip_manifest.py`, `s01_nuscenes_zip_smoke.py`, and
    `run_s01_nuscenes_zip_smoke.sh`.
  - The launcher recomputes this value and fails before data access on mismatch.
- Tracked S01 diff SHA-256 before this request file:
  `a954185abcbcdf3d37bf33a915e8a8f779a61b8fbb22e0677b65d20f4f831515`.
- Pre-request `git status` + changed/untracked file-content manifest SHA-256:
  `8c6e11d00fe1abdfcc1e93ebc19b2a7b04926258488575ce46927719c5e7998c`.
- No model/config/checkpoint is used. Resolved config hash: `N/A`.
- Seed: loader seed `42`; no scientific/random-seed cell.

## Exact data scope

- Module: `nuScenes-data/1.0-map-1.3-zip`.
- Dataroot: module-provided `NUSCENES_DATA_DIR`.
- Version metadata: `v1.0-trainval`, read-only.
- Archive central directory scanned: **only** `trainval01_blobs.zip`.
- Payloads read: four metadata-complete keyframes whose six cameras, key
  `LIDAR_TOP`, and up to nine previous `LIDAR_TOP` records all resolve in that one
  archive. At least one selected keyframe must have all nine previous sweeps.
- Candidate search cap: first 4,000 metadata samples; membership queries only.
- No payload from trainval02..10 is read, apart from lightweight archive existence/
  size preflight performed by `verify_dataset`.
- No extraction and no write under the shared dataset.

## Exact resources and command

- Nodes: 1.
- GPUs: 1 x GH200 (environment/architecture access; no model computation).
- CPUs: 8.
- Walltime: 00:20:00.
- Concurrent S01 jobs: 1 maximum.
- Maximum charged scope: 0.333 GPU-hours. This remains below O-009's cumulative
  two-GPU-hour ceiling.
- Job array: forbidden.
- Exact output root (must not already exist):
  `/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s01_zip_smoke_ee7c030911d5`
- Slurm logs:
  `/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/logs/s01_zip_smoke_%j.{out,err}`

```bash
sbatch --export=ALL,\
EXPECTED_S01_SHA=f262f6bea037580065a8505008773c04fdd259f5,\
EXPECTED_S01_STATE_HASH=ee7c030911d5c2a99f7e60c73df4454d2f93d15de6881e0f081ceb04c1de0869,\
S01_OUTPUT_ROOT=/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s01_zip_smoke_ee7c030911d5 \
fl_v3/scripts/run_s01_nuscenes_zip_smoke.sh
```

The launcher runs exactly:

1. one-archive `s01_nuscenes_zip_manifest.py --archives trainval01_blobs.zip`;
2. `s01_nuscenes_zip_smoke.py` with `n_sweeps=10`, four samples, two persistent
   workers, and two repeated epochs;
3. SHA-256 generation for the SQLite manifest and JSON report.

## Acceptance and stop conditions

Pass requires all of the following:

- exact HEAD and runtime source-state hash match;
- output root resolves outside the dataset and did not previously exist;
- canonical module/root/table alias discovery succeeds;
- the one archive is stored, unencrypted, canonical, and duplicate-free;
- four real samples decode with six uint8 `900x1600` RGB cameras and finite float32
  key+history LiDAR;
- every payload passes length and CRC validation;
- 0-worker and 2-worker decoded hashes match across two persistent epochs;
- worker PIDs remain stable and worker-local read state advances without an
  inter-epoch archive reopen;
- command exits 0 and writes manifest/report/checksums.

The job stops immediately on any mismatch, missing member, access error, unsupported
ZIP feature, decode/schema/CRC failure, non-finite LiDAR, worker lifecycle mismatch,
or walltime. There is no automatic resubmission. Any follow-up job requires review
of this job's exact negative/positive result and must remain within O-009 or receive
new owner approval.

## Interpretation limit

This is engineering smoke evidence only. It cannot establish 100% train/val member
coverage, full-data throughput/data-wait, scientific readiness, model quality, or
any attack/defense claim.

## Execution record

- Submitted once, with the exact command and immutable source-state identity above.
- Slurm job: `330409` (`flv3_s01_zip_smoke`).
- Submit/start/end: `2026-07-10T17:45:10` / `17:45:12` / `17:46:58`
  Europe/Stockholm.
- Node/resources: `n408`, one GH200, eight CPUs, one node.
- State/elapsed/exit: `COMPLETED`, `00:01:46`, `0:0`.
- Charged upper bound from the requested walltime: 0.333 GPU-hours; actual elapsed
  usage was approximately 0.0294 GPU-hours. No second or concurrent S01 job was
  submitted.
- Outputs and checksums are recorded in `RESULTS.md`. The job stayed within O-009;
  no array, DDP, multi-node, full scan/profile, epoch/eval, matrix, seed expansion,
  rerun, or automatic resubmission occurred.

---

# S01 RUN_REQUEST — complete trainval ZIP coverage and loader gate

## Approval state

- **Status:** `APPROVED_FOR_THIS_EXACT_FULL_GATE`.
- **Exact owner approval:** the owner message in the active S01 task on 2026-07-11
  approved commit `011e4640d26330e2c8145fcdb56833fe19e7b67d`, one GH200,
  32 CPUs, walltime `01:35:00`, maximum 1.583 GPU-hours, the specified output
  directory, and no automatic resubmission.
- This job exceeds O-009's autonomous per-job walltime ceiling of 60 minutes; the
  approval above is the explicit owner-authorized expansion for this exact job.
- It remains one node, one GPU, one job, no array/DDP/seeds/retry. Its requested
  1.583 GPU-hours plus the earlier smoke's requested 0.333 GPU-hours remains below
  the two-GPU-hour cumulative ceiling (1.916 GPU-hours maximum requested).

## Immutable execution identity

- Git branch: `codex/s01-nuscenes-zip-backend`.
- Exact commit: `011e4640d26330e2c8145fcdb56833fe19e7b67d`.
- Commit tree: `3bf7344a6afa1b381162e43581c0f550a2924f79`.
- Full-gate runtime source-state SHA-256:
  `7c601b2818acec028c2d350e5feee9320021244417f9a75c3e516b232ad379df`.
  It hashes the sorted per-file SHA-256 list for the complete nuScenes data module,
  cache builder, manifest/audit/benchmark tools, and full-gate launcher.
- No model, checkpoint, resolved experiment config, scientific seed, training, or
  evaluation is involved.

## Exact data and validation scope

1. Load `nuScenes-data/1.0-map-1.3-zip` and resolve the module-provided
   `NUSCENES_DATA_DIR` read-only root.
2. Scan the central directories of exactly `trainval01_blobs.zip` through
   `trainval10_blobs.zip` once and atomically build an external read-only SQLite
   member manifest. Reject compression, encryption, duplicates within one archive,
   conflicting cross-archive copies, noncanonical names, malformed archives, or
   archive mutation. Identical cross-archive copies are retained as occurrences and
   routed deterministically.
3. Build extraction-free 10-sweep info caches for official train (28,130 samples)
   and val (6,019 samples) from metadata.
4. Reconcile every cache path against a fresh metadata traversal and the manifest:
   204,894 six-camera references, 34,149 key `LIDAR_TOP` references, and every
   actually requested previous sweep (up to nine per sample). Require zero missing
   members, exact train/val counts and disjointness, correct sensor prefixes, and
   matching cache hashes.
5. Read and CRC-check one deterministic payload sentinel from every one of the ten
   archives through the production `pread` path. This demonstrates every archive
   can serve payload bytes; it intentionally does not read/CRC every one of the
   millions of blobs, which would create unnecessary shared-filesystem load.
6. Profile decoded 10-sweep train samples at 0, 2, 4, and 8 DataLoader workers,
   two persistent-worker repeats each: 32 deterministic batches, 16 warm-up
   batches, and 256 measured batches per repeat. This decodes 2,432 sample reads in
   total and emits samples/s plus batch-wait p50/p95/min/max and cold/warm behavior.
7. Hash the manifest, train/val caches and metadata, coverage JSON, and loader
   profile JSON. No extraction or write beneath the shared dataset is permitted.

## Exact resources, command, and outputs

- Nodes: 1.
- GPU: 1 x GH200, used to access the validated aarch64 environment; there is no
  model/GPU computation.
- CPUs: 32.
- Walltime hard limit: `01:35:00` (the CLI overrides the launcher's conservative
  two-hour template default).
- Requested GPU budget: 1.583 GPU-hours maximum.
- Expected elapsed usage: approximately 0.75-1.25 GPU-hours; this is an estimate,
  not evidence. The 95-minute hard cap is authoritative.
- Concurrent S01 jobs: one maximum; no array, DDP, multi-node, seed expansion,
  automatic resubmission, or follow-on cell.
- Exact output root (must not exist before submission):
  `/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s01_zip_full_gate_011e4640d263`.
- Logs:
  `/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/logs/s01_zip_gate_%j.{out,err}`.

Exact proposed submission:

```bash
test "$(git rev-parse HEAD)" = "011e4640d26330e2c8145fcdb56833fe19e7b67d" && \
test -z "$(git status --short --untracked-files=no -- \
  fl_v3/src/fl_v3/data/nuscenes \
  fl_v3/scripts/build_nuscenes_cache.py \
  fl_v3/scripts/s01_nuscenes_zip_manifest.py \
  fl_v3/scripts/s01_nuscenes_zip_audit.py \
  fl_v3/scripts/s01_nuscenes_zip_benchmark.py \
  fl_v3/scripts/run_s01_nuscenes_zip_full_gate.sh)" && \
sbatch --time=01:35:00 \
  --export=ALL,S01_OUTPUT_ROOT=/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s01_zip_full_gate_011e4640d263 \
  fl_v3/scripts/run_s01_nuscenes_zip_full_gate.sh
```

## Estimated stage timing and stop conditions

- Ten central-directory scans + manifest: estimated 10-25 minutes.
- Full train/val 10-sweep cache + reference audit: estimated 10-25 minutes.
- Four worker-count profiles and artifact hashing: estimated 15-40 minutes.
- Expected total: about 45-75 minutes, with 95 minutes as the hard stop. These are
  planning estimates derived partly from the bounded one-archive smoke and must not
  be reported as measured throughput.

The job fails immediately on identity/output-path mismatch, module/access error,
archive/schema/integrity violation, cache disagreement, any missing reference,
train/val overlap/count mismatch, unreadable archive sentinel, nondeterministic
decoded hashes, output collision, or walltime. There is no automatic retry. A
timeout or failure is a negative result and requires a new owner decision before
any follow-up job.

## Acceptance and interpretation boundary

Pass requires exact ten-archive manifest coverage, zero missing official train/val
camera/key/sweep references, ten readable archive sentinels, deterministic decoded
hashes across worker counts/repeats, and an emitted loader wait/throughput report
with durable hashes.

Allowed interpretation after a pass: all official train/val paths requested by the
10-sweep data pipeline are represented in the shared stored archives, all ten
archives can serve real payload bytes, and bounded production-loader performance is
measured on the declared GH200 node/runtime.

Forbidden interpretation: every payload received a full CRC scan; scientific/model
readiness; acceptable end-to-end GPU utilization; model quality; or any FL,
attack/defense, metric, or publication claim. Independent S01-R remains mandatory.

## Execution record — failed attempt 332648

- Submitted once after the exact owner approval; Slurm job `332648`, node `n569`.
- Submit/start/end: `2026-07-11T04:57:36` / `04:57:37` / `04:58:47`.
- State/elapsed/exit: `FAILED`, `00:01:10`, `1:0`.
- Failure stage: manifest construction while scanning
  `trainval02_blobs.zip`; SQLite rejected a path already present from an earlier
  archive under the v1 `path PRIMARY KEY` schema.
- No manifest, cache, coverage report, profile, or checksum artifact completed.
  The output root contains only the empty `info_cache_msweep10/` directory created
  by the launcher before the manifest command.
- Batch-step resources: `MaxRSS=792M`, `MaxVMSize=6214208K`,
  `MaxDiskRead=140.03M`, `MaxDiskWrite=0.28M`, `TotalCPU=00:11.949`.
- Actual allocation was approximately 0.0194 GPU-hours. No retry or follow-up job
  was submitted.
- Negative conclusion: the shared layout contains at least one path occurrence in
  multiple archives. This attempt did not determine whether those copies have
  identical size/CRC, and it provides no full-coverage or throughput result.
- stdout SHA-256:
  `5f73f4b10fb5c4940fadcf375d9dbfbce4947da396fa1efd28f46a9772f6bc4b`.
- stderr SHA-256:
  `7b232bac2b86f95c23709cc70fbb87fbeb6ebfcabb83b7b3324cb155de97ff47`.
- Any v2 follow-up requires a new immutable commit, a revised exact request, and
  fresh owner approval. This record does not authorize resubmission.

---

# S01 RUN_REQUEST — v2 complete trainval follow-up

## Approval state

- **Status:** `APPROVED_FOR_THIS_EXACT_V2_FOLLOW_UP`.
- Failed job `332648` was not retried. Its approval is exhausted and did not cover
  this v2 commit, output directory, or command.
- **Exact owner approval:** the owner message in the active S01 task on 2026-07-11
  approved commit `1fe651700bd06a07707307c60ad4e31cc9d1e0ba`, one GH200,
  32 CPUs, walltime `01:35:00`, maximum 1.583 GPU-hours, the specified new output
  directory, and no automatic resubmission.

## Immutable execution identity

- Branch: `codex/s01-nuscenes-zip-backend`.
- Exact v2 commit: `1fe651700bd06a07707307c60ad4e31cc9d1e0ba`.
- Commit tree: `e8dcda9f8ee88daf10c2db7ad6629c925550e460`.
- Full-gate runtime source-state SHA-256:
  `64ba617eb2df8be49df89b83f691d6c91829c0cb91f85acbe665b499f5dab65c`.
- Runtime manifest format: `s01.nuscenes-zip.v2`.
- No model, checkpoint, scientific seed, resolved model config, training, or
  evaluation is involved.

## Exact scope and v2 acceptance

The data/cache/coverage/profile scope is unchanged from the approved failed attempt:

1. scan central directories from exactly `trainval01_blobs.zip` through
   `trainval10_blobs.zip` into an external read-only SQLite manifest;
2. retain every `(path, archive)` occurrence; permit a cross-archive copy only if
   its central-directory file size and CRC match every other occurrence; reject
   conflicting copies and duplicates within one archive;
3. route runtime reads deterministically to the lowest archive ID while retaining
   per-archive counts and one real CRC-checked sentinel from every archive;
4. build official train 28,130 + val 6,019 10-sweep caches from metadata without
   extracting payloads;
5. reconcile all 204,894 camera references, 34,149 key-LiDAR references, and every
   requested previous sweep against metadata, cache, and manifest; require zero
   missing paths, exact counts, train/val disjointness, and matching cache hashes;
6. profile decoded train samples at 0/2/4/8 workers, two repeats each, with 32
   determinism + 16 warm-up + 256 measured batches per repeat (2,432 total sample
   reads), reporting samples/s and batch-wait p50/p95/min/max;
7. hash manifest, caches/sidecars, coverage report, and loader profile. Never write
   below the shared dataset and never extract an archive.

The job fails on any identity/output collision, module/access error, unsupported ZIP
feature, intra-archive duplicate, cross-archive size/CRC conflict, malformed local
header, missing reference, cache/metadata disagreement, split overlap/count drift,
unreadable archive sentinel, nondeterministic decoded hash, or walltime. There is no
automatic retry.

## Resources, budget, output, and exact command

- Nodes: 1; GPU: one GH200; CPUs: 32.
- Walltime hard limit: `01:35:00`.
- Maximum new allocation: 1.583 GPU-hours.
- Expected elapsed allocation: approximately 0.75-1.25 GPU-hours; planning estimate
  only. The prior smoke and failed attempt used about 0.0489 actual GPU-hours, so
  cumulative actual allocation would remain at or below approximately 1.632
  GPU-hours even if this follow-up reaches its hard limit.
- Concurrent jobs: one; no array, DDP, seeds, training/evaluation, matrix, follow-on
  cell, or automatic resubmission.
- New immutable output root, currently absent:
  `/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s01_zip_full_gate_v2_1fe651700bd0`.
- Logs:
  `/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/logs/s01_zip_gate_%j.{out,err}`.
- Failed output root `s01_zip_full_gate_011e4640d263` is retained read-only as
  negative evidence and will not be reused or removed.

Exact proposed submission:

```bash
test "$(git rev-parse HEAD)" = "1fe651700bd06a07707307c60ad4e31cc9d1e0ba" && \
test -z "$(git status --short --untracked-files=no -- \
  fl_v3/src/fl_v3/data/nuscenes \
  fl_v3/scripts/build_nuscenes_cache.py \
  fl_v3/scripts/s01_nuscenes_zip_manifest.py \
  fl_v3/scripts/s01_nuscenes_zip_audit.py \
  fl_v3/scripts/s01_nuscenes_zip_benchmark.py \
  fl_v3/scripts/run_s01_nuscenes_zip_full_gate.sh)" && \
test ! -e "/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s01_zip_full_gate_v2_1fe651700bd0" && \
test -z "$(squeue -u "$USER" -h -o '%i %j' | \
  awk '$2 ~ /flv3_s01_zip/ {print}')" && \
sbatch --time=01:35:00 \
  --export=ALL,S01_OUTPUT_ROOT=/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s01_zip_full_gate_v2_1fe651700bd0 \
  fl_v3/scripts/run_s01_nuscenes_zip_full_gate.sh
```

## Interpretation boundary

A pass would establish that every official train/val path requested by the 10-sweep
pipeline resolves, all cross-archive repeated paths have matching size/CRC metadata,
all ten archives serve a real payload sentinel, repeated decoded hashes are stable,
and bounded loader wait/throughput is measured for this exact runtime.

It would not mean every payload was read/CRC-scanned, nor establish model/GPU-step
readiness, scientific quality, metrics, FL/attack/defense behavior, or publication
claims. Independent S01-R remains required after results are committed.

## Execution record — successful v2 follow-up 332651

- Submitted exactly once after the owner approval above; no concurrent S01 job or
  automatic resubmission.
- Job/node: `332651` / `n574` (`aarch64`).
- Submit/start/end: `2026-07-11T05:10:26` / `05:10:28` / `05:15:57`.
- State/elapsed/exit: `COMPLETED`, `00:05:29`, `0:0`.
- Actual allocation: approximately 0.0914 GPU-hours; cumulative actual allocation
  for jobs 330409, 332648, and 332651 was approximately 0.1402 GPU-hours.
- Manifest: ten archives, 417,774,430,886 archive bytes, 2,631,093 occurrences,
  2,631,084 unique members, nine duplicate occurrences. The only repeated path was
  `LICENSE`, present once in each archive with size 25,319 and CRC `48f670e8`.
- Coverage: 538,695/538,695 references resolved, 534,532 unique referenced members,
  zero missing paths; 204,894 camera, 34,149 key LiDAR, and 299,652 previous-sweep
  references. All ten real archive sentinels passed.
- Loader determinism digest
  `4e46534f92c7979c04667a72f8a6dd0b9c61bfe0a14808b5debb85c34e0b54f7`
  matched every 0/2/4/8-worker repeat.
- First-repeat throughput at 0/2/4/8 workers was
  18.94/46.81/89.08/154.36 samples/s; complete wait distributions and warm-repeat
  values are in `RESULTS.md` and the profile JSON.
- Batch-step resources: `MaxRSS=11048512K`, `MaxVMSize=71370624K`,
  `MaxDiskRead=21058.98M`, `MaxDiskWrite=6392.35M`, `TotalCPU=05:40.946`.
- Output root:
  `/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s01_zip_full_gate_v2_1fe651700bd0`
  (approximately 1.3 GiB). No shared-dataset write or extraction occurred.
- Artifact/log hashes and interpretation limits are recorded in `RESULTS.md`.

---

# S01 RUN_REQUEST — S01-R remediation focused tests

## Approval state

- **Status:** `APPROVED_FOR_THIS_EXACT_FOCUSED_TEST`.
- **Exact owner approval:** the owner message in the active S01 task on 2026-07-11
  approved commit `54a48f9102fd0de9a9abe97701550740b547e769`, one GH200,
  eight CPUs, walltime `00:20:00`, maximum 0.333 GPU-hours, the specified output
  directory, and no automatic resubmission.
- Requested execution basis: owner-specific approval for this exact entry plus
  standing bounded engineering-smoke policy `O-009`.
- Prior approvals for jobs `330409`, `332648`, and `332651` are exhausted and do
  not authorize this job or output root.

## Immutable execution identity

- Branch: `codex/s01-nuscenes-zip-backend`.
- Exact remediation commit:
  `54a48f9102fd0de9a9abe97701550740b547e769`.
- Focused-test runtime source-state SHA-256:
  `260560ef3c5904825ad384825ec6755877748bbb403f65b5d5d907f1b7db1cda`.
- Source-state set: every regular file under
  `fl_v3/src/fl_v3/data/nuscenes/` (excluding `__pycache__`), the four exact test
  files named below, and `fl_v3/scripts/run_s01_nuscenes_zip_tests.sh`.
- No model, checkpoint, resolved model config, scientific seed, training,
  evaluation, shared trainval archive, or old full-gate cache is involved.

## Exact scope and acceptance

Run exactly these four focused pytest files in the validated Arrhenius environment:

1. `fl_v3/tests/test_nuscenes_zip_backend.py`;
2. `fl_v3/tests/test_nuscenes_zip_dataset.py`;
3. `fl_v3/tests/test_nuscenes_zip_info_cache.py`;
4. `fl_v3/tests/test_nuscenes_info_cache.py`.

The only real dataset input is the existing extracted mini root
`/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/fl_weather_project/data/nuscenes_mini`.
The parity test deterministically selects two `mini_train` keyframes (one scene
start and one with nine previous LiDAR sweeps), copies only their six-camera,
key-LiDAR, and requested sweep payloads into small stored ZIP fixtures under the
job's node-local pytest temporary directory, and compares bytes plus decoded
arrays against directory mode. Cache tests traverse mini metadata and decode only
their declared bounded fixtures/selected samples. All other archives and payloads
are synthetic temporary fixtures. The shared trainval ZIP set is neither opened
nor scanned.

Acceptance requires:

- every selected test passes with no failure/error;
- real-mini directory/ZIP bytes, images, key-LiDAR, and 10-sweep decoded arrays
  match;
- repeated persistent two-worker reads are deterministic under both available
  `fork` and `spawn`, with PID-owned handles and stable lifecycle counts;
- `t1.v2` filenames, sidecars, per-record depth, hash, ambiguity checks, and
  2-sweep-versus-10-sweep rejection pass;
- same-length local-header filename mutation is rejected;
- duplicate-path sentinels open and CRC-check each exact archive occurrence;
- in-job Git SHA and runtime source-state hash equal the values above;
- JUnit, pytest log, execution identity, runtime file hashes, and artifact hashes
  are written to the unique output root.

The job fails closed on identity/hash drift, output collision, mini-root failure,
pytest failure/error, missing required artifact, or walltime. A platform-inapplicable
start method may be reported as a pytest skip; any other skip must be treated as a
negative result requiring review. There is no automatic retry.

## Resources, budget, output, and exact command

- Nodes: 1; GPU: one GH200; CPUs: 8.
- Walltime hard limit: `00:20:00`.
- Maximum new allocation: 0.333 GPU-hours.
- Expected elapsed allocation: approximately 0.05-0.17 GPU-hours (3-10 minutes),
  planning estimate only.
- Previous three S01 jobs consumed approximately 0.1402 GPU-hours, so worst-case
  cumulative S01 allocation after this job is approximately 0.4735 GPU-hours.
- Concurrent S01 jobs: at most one; no array, DDP, seed sweep, full trainval scan,
  throughput profile, model step, epoch/eval, matrix, or automatic resubmission.
- Unique output root, verified absent before submission and created by job 333206:
  `/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s01_zip_review_fixes_54a48f9102fd`.
- Logs:
  `/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/logs/s01_zip_tests_%j.{out,err}`.
- Pytest-created ZIP/cache fixtures live only in the compute node's temporary
  directory; the shared dataset is read-only and no extraction occurs there.

Exact proposed submission, once the owner approves this immutable entry:

```bash
test "$(git rev-parse HEAD)" = "54a48f9102fd0de9a9abe97701550740b547e769" && \
test -z "$(git status --short --untracked-files=no -- \
  fl_v3/src/fl_v3/data/nuscenes \
  fl_v3/tests/test_nuscenes_zip_backend.py \
  fl_v3/tests/test_nuscenes_zip_dataset.py \
  fl_v3/tests/test_nuscenes_zip_info_cache.py \
  fl_v3/tests/test_nuscenes_info_cache.py \
  fl_v3/scripts/run_s01_nuscenes_zip_tests.sh)" && \
test ! -e "/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s01_zip_review_fixes_54a48f9102fd" && \
test -z "$(squeue -u "$USER" -h -o '%i %j' | \
  awk '$2 ~ /flv3_s01_zip/ {print}')" && \
sbatch --time=00:20:00 --cpus-per-task=8 \
  --export=ALL,EXPECTED_S01_SHA=54a48f9102fd0de9a9abe97701550740b547e769,EXPECTED_S01_STATE_HASH=260560ef3c5904825ad384825ec6755877748bbb403f65b5d5d907f1b7db1cda,S01_MINI_DATAROOT=/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/fl_weather_project/data/nuscenes_mini,S01_OUTPUT_ROOT=/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s01_zip_review_fixes_54a48f9102fd \
  fl_v3/scripts/run_s01_nuscenes_zip_tests.sh
```

## Interpretation boundary

A pass would close the dependency-backed mini parity/lifecycle execution finding
and exercise the other five remediation regressions on GH200. It would not rerun or
retroactively attest job `332651`, produce a new full trainval `t1.v2` cache, prove
every shared payload's CRC, measure full-data/model-step throughput, establish
training readiness, or support any model, metric, FL, attack/defense, scientific,
or publication claim. Independent S01-R re-review of the new worker SHA remains
mandatory.

## Execution record — successful focused job 333206

- Submitted exactly once after the exact owner approval above; no concurrent S01
  job and no automatic resubmission.
- Job/node: `333206` / `n405` (`aarch64`).
- Submit/start/end: `2026-07-11T08:32:21` / `08:32:22` / `08:33:49`.
- State/elapsed/exit: `COMPLETED`, `00:01:27`, `0:0`.
- Actual allocation: approximately 0.0242 GPU-hours; cumulative S01 actual
  allocation approximately 0.1644 GPU-hours.
- In-job identity matched exact commit
  `54a48f9102fd0de9a9abe97701550740b547e769` and exact runtime source-state hash
  `260560ef3c5904825ad384825ec6755877748bbb403f65b5d5d907f1b7db1cda`.
- Pytest/JUnit: 56 tests passed in 13.18 seconds; zero failures, errors, or skips.
- Output root and logs are the exact paths declared above. Artifact/log hashes,
  focused acceptance details, resources, negative findings, and interpretation
  limits are recorded in `RESULTS.md`.
- No shared trainval archive was opened/scanned, no shared-dataset write or
  extraction occurred, and no model/scientific work ran.
