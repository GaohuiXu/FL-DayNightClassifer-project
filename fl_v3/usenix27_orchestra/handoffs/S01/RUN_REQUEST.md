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
