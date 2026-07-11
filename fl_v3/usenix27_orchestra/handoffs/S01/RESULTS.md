# S01 RESULTS — bounded shared-ZIP reader smoke

## Verdict and scope

**Engineering smoke PASS; full S01 data gate NOT RUN.** Slurm job `330409` proved
that the exact uncommitted S01 runtime state can discover the licensed module on an
Arrhenius GH200 node, index one stored archive, and decode a bounded set of real
camera/keyframe-LiDAR/multi-sweep payloads deterministically with persistent
workers. It did not scan all ten archives, reconcile all train/val references, or
measure production loader throughput/data wait.

This result is bound to:

- HEAD/BASE_SHA `f262f6bea037580065a8505008773c04fdd259f5` in detached mode;
- S01 runtime source-state SHA-256
  `ee7c030911d5c2a99f7e60c73df4454d2f93d15de6881e0f081ceb04c1de0869`;
- the exact owner-approved command, data subset, output root, and stop conditions
  in `RUN_REQUEST.md` under owner-specific approval plus standing policy `O-009`.

## Job and command

- Job ID/name: `330409` / `flv3_s01_zip_smoke`.
- Submit/start/end: `2026-07-10T17:45:10` / `17:45:12` / `17:46:58`
  Europe/Stockholm.
- Node: `n408` (`aarch64`).
- Allocation: one node, one `nvidia_gh200_120gb`, eight CPUs, walltime limit
  `00:20:00`.
- Result: `COMPLETED`, elapsed `00:01:46`, exit `0:0`.
- Actual elapsed GPU allocation: approximately 0.0294 GPU-hours; no second job was
  submitted.

Exact submission:

```bash
sbatch --export=ALL,\
EXPECTED_S01_SHA=f262f6bea037580065a8505008773c04fdd259f5,\
EXPECTED_S01_STATE_HASH=ee7c030911d5c2a99f7e60c73df4454d2f93d15de6881e0f081ceb04c1de0869,\
S01_OUTPUT_ROOT=/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s01_zip_smoke_ee7c030911d5 \
fl_v3/scripts/run_s01_nuscenes_zip_smoke.sh
```

## Module and archive evidence

- `module load nuScenes-data/1.0-map-1.3-zip` exposed
  `/dataset/easybuild/data/nuScenes-data/1.0-map-1.3-zip` through
  `NUSCENES_DATA_DIR`.
- `v1.0-trainval` resolved to module table directory `trainval/`.
- Metadata contained 34,149 samples and passed the configured sentinel check.
- All ten expected archive names existed during lightweight dataset verification.
  Only `trainval01_blobs.zip` was opened/scanned for members.
- `trainval01_blobs.zip`: 41,307,560,463 bytes, 258,109 stored members.
- External SQLite manifest: 61,648,896 bytes, mode `0444`; logical manifest hash
  `0761493f3150aaa48e77ee93b4b27848cf3dd3537673800c920fcaec64c1734f`.
- No extraction and no output under the shared dataset. All generated files were
  written under the approved personal `/nobackup` output root.

## Bounded reference/decode coverage

Five metadata candidates were examined and four complete samples were selected.
For each sample the smoke required six cameras, one key `LIDAR_TOP`, and nine
previous `LIDAR_TOP` sweeps, all located in `trainval01`.

| Reference class | Selected unique references | Result |
|---|---:|---|
| Six cameras | 24 | all resolved, length/CRC checked, RGB decoded |
| Keyframe `LIDAR_TOP` | 4 | all resolved, length/CRC checked, float32 decoded |
| Previous `LIDAR_TOP` sweeps | 36 | all resolved, length/CRC checked, float32 decoded |
| Total | 64 | all passed |

The four concatenated 10-sweep point counts were 347,520; 347,488; 347,200; and
347,008. The zero-worker parent read 64 members / 31,086,681 bytes. This is selected
smoke coverage, not 100% train/val coverage.

## Determinism and handle lifecycle

- Decoded image and LiDAR SHA-256 digests matched between `num_workers=0` and
  `num_workers=2`, and across two persistent-worker epochs.
- Result field `deterministic_across_worker_counts_and_epochs` is `true`.
- Persistent worker PIDs stayed `417031` and `417032` across both epochs.
- Each worker advanced from 32 to 64 reads while `reopen_count` stayed at `1`.
  That single reopen is the expected post-fork ownership reset; there was no
  inter-epoch reopen.
- Each worker owned one open descriptor for `trainval01_blobs.zip` and its own
  read-only manifest connection. The parent had its own PID-owned handles.

## Scheduler/resource observations

The `sacct` batch-step values were:

- `MaxRSS=8709568K` (about 8.31 GiB), `MaxVMSize=29158784K`;
- `MaxDiskRead=3225.32M`, `MaxDiskWrite=522.20M`;
- `TotalCPU=00:37.847`.

These aggregate manifest-build plus decode activity and are not a loader throughput
measurement. The roughly 8.31-GiB peak RSS is a negative operational observation to
recheck during the ten-archive gate; it must not be projected linearly without a
measurement.

## Durable artifacts

Output root:
`/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s01_zip_smoke_ee7c030911d5`

| Artifact | Bytes | SHA-256 |
|---|---:|---|
| `trainval01_manifest.sqlite` | 61,648,896 | `d9aa3ada7261d9dea315f4fd8654cf559e773f01af2efc6f3ea796134d2d79c3` |
| `smoke_report.json` | 6,202 | `e882b490c8bd772c6addfdee20c2c369a2a83be7afddbf096bad9c51f2e1e830` |
| `sha256sums.txt` | 385 | `1d92fd30397301dd546b07e016565a67188e02e9cc615233fbeaf33c8c118c47` |

The JSON's canonical internal report hash is
`be476090e30d1a487239c72413aacbdfbfb3b53590d2223e8d28fc6e24680710`.

Logs:

- stdout:
  `/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/logs/s01_zip_smoke_330409.out`,
  7,471 bytes, SHA-256
  `c154fa96b040ca1acd970434530a8a8bd3847fee148aac8cf8d6ee77964a7e90`;
- stderr:
  `/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/logs/s01_zip_smoke_330409.err`,
  294 bytes, SHA-256
  `8db5d05b4abfa9c9cc1bd7028c410675c3e2d697af110ce6c6d9aa51f2e1e830`.

Stderr contained only normal module-purge and gated-dataset notices; no Python,
I/O, CRC, worker, or scheduler error was emitted.

## Negative and missing results

- Login-node payload access was denied because that process did not have the gated
  dataset group; the compute job had access. Login-node import failures are also
  non-definitive because the validated environment is aarch64-only.
- Full Python unit/pytest execution was not available on the login node with the
  GH200 environment. Static/synthetic checks are listed in `HANDOFF.md`.
- Archives `trainval02` through `trainval10` were not member-scanned.
- Full train/val six-camera/key-LiDAR/10-sweep coverage was not computed.
- Directory-versus-ZIP real mini parity tests are implemented but were not run in
  the GH200 smoke; the smoke compared ZIP decode across worker counts/epochs.
- No production DataLoader throughput or data-wait profile was produced.
- No model, epoch, evaluation, metric, attack, defense, or scientific run occurred.

## Interpretation boundary

Allowed: on this exact source state and bounded one-archive subset, the Arrhenius
module was discoverable/readable, real camera and 10-sweep LiDAR decoded with
length/CRC validation, and repeated persistent-worker outputs were deterministic.

Forbidden: full-data readiness, all-member coverage, directory/ZIP parity PASS,
production throughput, absence of random-read amplification, model readiness or
quality, and any scientific/attack/defense claim.
