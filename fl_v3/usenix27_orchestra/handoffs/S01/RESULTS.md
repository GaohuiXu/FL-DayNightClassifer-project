# S01 RESULTS — shared nuScenes ZIP backend

## Overall verdict

**Full-data ZIP coverage/determinism/loader-profile gates PASS on v2; focused S01-R
remediation tests PASS; final S01 remains pending independent re-review.** Job
`332651` completed all declared ten-archive/cache/coverage/
sentinel/profile stages. Job `332648` remains a preserved negative v1 result, and
job `330409` is the bounded lifecycle smoke. None is scientific/model evidence.

## Bounded smoke 330409 — PASS

This historical engineering smoke proved
that the exact uncommitted S01 runtime state can discover the licensed module on an
Arrhenius GH200 node, index one stored archive, and decode a bounded set of real
camera/keyframe-LiDAR/multi-sweep payloads deterministically with persistent
workers. By itself it did not scan all ten archives, reconcile all train/val
references, or measure production loader throughput/data wait; those later passed
in the v2 section below.

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
| `smoke_report.json` | 6,202 | `e882b490c8bd772c6addfdee20c2c369a2a83be7afddbf096bad9c51fbac79df` |
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

---

## Complete-gate attempt 332648 — FAILED

The exact owner-approved ten-archive job was submitted once on 2026-07-11 and was
not retried.

- Job/node: `332648` / `n569`.
- Submit/start/end: `04:57:36` / `04:57:37` / `04:58:47`.
- State/elapsed/exit: `FAILED`, `00:01:10`, `1:0`.
- Failing stage: `s01_nuscenes_zip_manifest.py` while scanning
  `trainval02_blobs.zip`.
- Exact error: SQLite `UNIQUE constraint failed: members.path`, surfaced as
  `ZipManifestError: duplicate ZIP member (including cross-archive duplicate)`.
- Result: no complete manifest, cache, coverage JSON, loader profile, or checksum
  file exists. The approved output root contains only an empty
  `info_cache_msweep10/` directory.

This proves at least one member path occurs in more than one shared archive. It
does **not** prove those payload copies differ or that either copy is corrupt; the
v1 schema stopped before comparing their central-directory size/CRC. Consequently
all full trainval coverage and throughput claims remain forbidden.

Resources: `MaxRSS=792M`, `MaxVMSize=6214208K`, `MaxDiskRead=140.03M`,
`MaxDiskWrite=0.28M`, `TotalCPU=00:11.949`; approximately 0.0194 actual GPU-hours.

Logs:

- `/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/logs/s01_zip_gate_332648.out`,
  SHA-256 `5f73f4b10fb5c4940fadcf375d9dbfbce4947da396fa1efd28f46a9772f6bc4b`;
- `/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/logs/s01_zip_gate_332648.err`,
  SHA-256 `7b232bac2b86f95c23709cc70fbb87fbeb6ebfcabb83b7b3324cb155de97ff47`.

Required correction before any follow-up: a manifest schema that retains every
archive occurrence, accepts a cross-archive duplicate only when path, size, and CRC
match, selects a deterministic runtime archive, keeps per-archive sentinels, and
still rejects conflicting copies and duplicates inside one archive. A follow-up is
not authorized by the failed job's approval.

---

## Complete v2 gate 332651 — PASS

### Identity and scheduler result

- Exact runtime commit: `1fe651700bd06a07707307c60ad4e31cc9d1e0ba`.
- Runtime source-state SHA-256:
  `64ba617eb2df8be49df89b83f691d6c91829c0cb91f85acbe665b499f5dab65c`.
- Job/node: `332651` / `n574` (`aarch64`).
- Submit/start/end: `2026-07-11T05:10:26` / `05:10:28` / `05:15:57`.
- State/elapsed/exit: `COMPLETED`, `00:05:29`, `0:0`.
- Allocation: one GH200, 32 CPUs, one node; approximately 0.0914 actual
  GPU-hours. No model/GPU computation, retry, array, or follow-on job occurred.

### Complete archive manifest

- Ten archives, total size 417,774,430,886 bytes.
- Member occurrences: 2,631,093.
- Unique members: 2,631,084.
- Duplicate occurrences: 9.
- The only repeated path was `LICENSE`, present once in every archive with identical
  size 25,319 and CRC `48f670e8`. No sensor payload path was duplicated across
  archives and no conflicting size/CRC pair occurred.
- Logical manifest hash:
  `023f72b4220bb0db587be00920308bf9074384740fe186d243be92f9a53119f6`.
- SQLite file: 633,106,432 bytes, mode `0444`, SHA-256
  `228e2f5bab30007acb06eb61393d1fbacc88979490668ff800f8f7f9752a47fb`.

### Cache and 100% reference coverage

- Train cache: 28,130 samples, 944,881 boxes, 246,840 previous-sweep records;
  canonical cache hash
  `7dcdc278bcf25ed5761d3d17d1b42ce206dd2de27d3758373fa610a1e86fdc2f`.
- Val cache: 6,019 samples, 187,528 boxes, 52,812 previous-sweep records;
  canonical cache hash
  `4acc83079f475005474fba411e60386f4bd1af4acad725eef6aff00ace1821ae`.
- Train and val sample counts matched the official split; token sets were disjoint
  and their union covered all 34,149 metadata samples.

| Reference class | Train | Val | Total | Missing |
|---|---:|---:|---:|---:|
| Six cameras | 168,780 | 36,114 | 204,894 | 0 |
| Key `LIDAR_TOP` | 28,130 | 6,019 | 34,149 | 0 |
| Previous LiDAR sweeps | 246,840 | 52,812 | 299,652 | 0 |
| All references | 443,750 | 94,945 | 538,695 | 0 |

All 538,695 references resolved to 534,532 unique members. Metadata traversal and
cache paths were identical, sensor prefixes had zero violations, cache sidecars
matched content hashes, and one real payload from every archive passed the
production `pread` length/CRC path. Coverage `gate_pass=true`, `gate_errors=[]`.

Coverage JSON: 18,250 bytes, file SHA-256
`773b8ea4513bd95363dcde0732bb9e836c7563e3d2e76c58d8b2c6a568ff579b`;
internal report hash
`cd0e298d31f98874ab398671d31e6d6a30e33a2ad599b7fe363761080bb55e50`.

### Decoded loader determinism and throughput

The benchmark used batch size 1 and 10 LiDAR frames per sample. For each worker
count it ran two persistent-worker repeats of 32 deterministic, 16 warm-up, and 256
measured batches. The same 304-token sequence was used for every profile; total
decoded sample reads were 2,432.

Every repeat and worker count produced digest
`4e46534f92c7979c04667a72f8a6dd0b9c61bfe0a14808b5debb85c34e0b54f7`.
`determinism_hash_identical_across_worker_counts=true`.

| Workers | Repeat | State | samples/s | wait p50 ms | wait p95 ms | wait max ms |
|---:|---:|---|---:|---:|---:|---:|
| 0 | 0 | cold start | 18.94 | 52.30 | 59.37 | 60.97 |
| 0 | 1 | warm | 24.05 | 40.99 | 47.83 | 49.36 |
| 2 | 0 | cold start | 46.81 | 22.53 | 46.60 | 49.67 |
| 2 | 1 | warm | 47.18 | 18.58 | 47.43 | 64.89 |
| 4 | 0 | cold start | 89.08 | 6.83 | 31.70 | 37.12 |
| 4 | 1 | warm | 89.25 | 3.12 | 43.22 | 53.78 |
| 8 | 0 | cold start | 154.36 | 0.06 | 32.68 | 54.53 |
| 8 | 1 | warm | 153.42 | 0.03 | 41.78 | 60.28 |

The near-zero 8-worker median reflects prefetch queue availability; p95 and total
samples/s remain the relevant tail/throughput measures. These are loader-only wait
measurements and cannot be converted to an end-to-end GPU-step data-wait percentage
without a model-step profile.

Profile JSON: 9,905 bytes, file SHA-256
`d34e7a90446dcf0bc3ec355d94ac6d984442e96583c85c0f599259faf987a108`;
internal report hash
`397bfeafbec5d07c1694031bef95d81a469e02dc8684e0750f93444b14e6847e`.

### Durable artifacts and resources

Output root (about 1.3 GiB):
`/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s01_zip_full_gate_v2_1fe651700bd0`

| Artifact | Bytes | SHA-256 |
|---|---:|---|
| `nuscenes_trainval_zip_manifest.sqlite` | 633,106,432 | `228e2f5bab30007acb06eb61393d1fbacc88979490668ff800f8f7f9752a47fb` |
| `coverage_train_val_msweep10.json` | 18,250 | `773b8ea4513bd95363dcde0732bb9e836c7563e3d2e76c58d8b2c6a568ff579b` |
| `loader_profile_train_msweep10.json` | 9,905 | `d34e7a90446dcf0bc3ec355d94ac6d984442e96583c85c0f599259faf987a108` |
| train cache pickle | 580,140,287 | `c45e7af2937120b128e85880625e379b67e5ddecce66a32d25c156810a9329c7` |
| val cache pickle | 117,994,540 | `52281790df9b2902002442fb7f4a02722c1118787f4cf8bac737828626582fdc` |
| train cache metadata | 271 | `b81867d660e7b97f8af3461e947c48ffd8b3584484cd66eee6ee8ccf2f697a1f` |
| val cache metadata | 268 | `9f668bdde4e0910d6880620d92acbf3425631589af29f4aa5553ce9b4e795114` |
| `sha256sums.txt` | 1,609 | `21c8780f51eac36a4867014c21bf7724573a4f4adb590b2c55c5149612da0e44` |

Logs:

- stdout: 29,390 bytes, SHA-256
  `5836dfe4ce50f67dca1adfb3d694531dcb35dc949f69fecdf219315aec4c727e`;
- stderr: 294 bytes, SHA-256
  `8db5d05b4abfa9c9cc1bd7028c410675c3e2d697af110ce6c6d9aa51f2e1e830`.

Stderr contained only the normal module-purge and gated-dataset notices. Scheduler
batch-step observations: `MaxRSS=11048512K` (about 10.54 GiB),
`MaxVMSize=71370624K`, `MaxDiskRead=21058.98M`, `MaxDiskWrite=6392.35M`,
`TotalCPU=05:40.946`.

### Interpretation boundary and remaining negative evidence

Allowed: every official train/val path requested by the 10-sweep pipeline exists in
the shared stored archives; all ten archives serve real CRC-checked bytes; the
declared loader is deterministic across 0/2/4/8 workers/repeats; and the table above
is measured loader-only performance for this exact GH200 runtime.

Still forbidden: every one of 2.63 million payloads received a CRC read;
trainval-scale directory/ZIP decoded parity; end-to-end model/GPU data-wait
percentage; training or scientific readiness; model quality; and any metric, FL,
attack/defense, or publication claim. Focused job 333206 establishes real-mini
parity only. Independent S01-R re-review remains required.

---

## Independent S01-R findings and remediation status

S01-R reviewed worker SHA `ce2e77284b290de4c9faa6b2f971c0bd52f98eff` and
returned **CHANGES-REQUESTED**. Its independent `REVIEW.md` SHA-256 is
`f69de33eec31e6d9e64c86f1fc30d3d76e17a1e482e65113b2c3ed5174551357`.
The review did not invalidate the job `332651` coverage, deterministic-loader, or
throughput observations. It identified six follow-ups:

1. Execute dependency-complete real-mini directory/ZIP decoded parity and spawn
   lifecycle tests.
2. Bind cache identity to `n_sweeps` so a shallow cache cannot silently satisfy a
   deeper request.
3. Add in-job Git/source attestation to future Slurm launchers; do not retrofit
   that claim to job `332651`.
4. Reject same-length local-header filename mutation.
5. Read duplicate-path sentinels from the exact archive occurrence.
6. Correct the bounded-smoke report hash to
   `e882b490c8bd772c6addfdee20c2c369a2a83be7afddbf096bad9c51fbac79df`.

Remediation commit `54a48f9102fd0de9a9abe97701550740b547e769` implements
items 2–6 and provides a bounded focused-test launcher for item 1. Cache format
`t1.v2` binds sweep depth in the filename,
metadata, every sample record, and canonical hash. The full-gate `t1.v1` cache
artifacts remain historical evidence and are deliberately rejected as remediated
production cache inputs. Exact focused job `333206` passed all 56 selected tests
with no failures, errors, or skips. Independent re-review remains required.

---

## S01-R remediation focused job 333206 — PASS

- Exact commit: `54a48f9102fd0de9a9abe97701550740b547e769`.
- Runtime source-state SHA-256:
  `260560ef3c5904825ad384825ec6755877748bbb403f65b5d5d907f1b7db1cda`.
- Job/node: `333206` / `n405` (`aarch64`).
- Submit/start/end: `2026-07-11T08:32:21` / `08:32:22` / `08:33:49`.
- State/elapsed/exit: `COMPLETED`, `00:01:27`, `0:0`.
- Allocation: one GH200, eight CPUs; actual approximately 0.0242 GPU-hours;
  cumulative S01 actual allocation approximately 0.1644 GPU-hours.
- Pytest: `56 passed in 13.18s`; JUnit reports 56 tests, zero failures, errors, or
  skips.

The exact real-mini parity test selected one scene-start `mini_train` keyframe and
one with nine previous LiDAR sweeps. Directory and generated stored-ZIP bytes,
decoded six-camera arrays, key-LiDAR, accumulated 10-sweep points, and GT arrays
matched. Both fork and spawn passed parent-open/child-handle ownership and repeated
persistent two-worker determinism tests. The `t1.v2` depth-binding, ambiguity,
sidecar, and 2-versus-10-sweep rejection tests passed, as did same-length
local-header mutation rejection and exact-archive duplicate-sentinel reads.

In-job identity attestation recorded the exact Git SHA and source-state hash above.
This is new provenance for job `333206`; it does not retroactively attest job
`332651`.

Output root:
`/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s01_zip_review_fixes_54a48f9102fd`.

| Artifact | Bytes | SHA-256 |
|---|---:|---|
| `execution_identity.json` | 442 | `a41970d8b84c575bfb90bde0f0e65d6801b5ef615068d13303addfd0998ef865` |
| `runtime_source_sha256s.txt` | 1,950 | `260560ef3c5904825ad384825ec6755877748bbb403f65b5d5d907f1b7db1cda` |
| `pytest.log` | 100 | `641ff631b532d33d50cdb8805d2ec88df1ed70e96f11a789e21a09218238ac3e` |
| `pytest.junit.xml` | 7,465 | `38b199b092bdcaecd49a2fde475a0aff276b9a344652aadd4f524e3d84fcd1bc` |
| `sha256sums.txt` | 787 | `e60dba68c7065a83c84bd6c5eeb02a535042e9416111cbdf672aa158ce5bf83a` |

Logs:

- stdout: 1,292 bytes, SHA-256
  `bc1547d4b400679915f1a022a025afc9e48ad1af6c7efc908e6d80d68e989544`;
- stderr: 123 bytes, SHA-256
  `ae6330855ac405b2e19691ca1681d7f9eeedc6216718d1516023d9376d891b57`.

Stderr contained only the normal module-purge notice. Batch-step resources were
`MaxRSS=540M`, `MaxVMSize=6279744K`, `MaxDiskRead=60.65M`,
`MaxDiskWrite=0.34M`, and `TotalCPU=00:19.221`.

Allowed: the six review remediations are implemented and the declared focused
mini parity/lifecycle/integrity regressions pass on this exact GH200 source state.
Forbidden: retroactive provenance for job 332651, a new full trainval t1.v2 cache,
all-payload CRC coverage, trainval/model-step readiness, model quality, metrics,
FL/attack/defense behavior, or any scientific/publication claim.
