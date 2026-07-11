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
