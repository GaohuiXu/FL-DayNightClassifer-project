# S07-A RESULTS — reviewed data-foundation integration

## Overall result

The bounded O-009 focused data-foundation jobs **PASS**, including the
S07-A-R physical-cache provenance remediation gate at Job 335280. Full trainval `t1.v2`
cache materialization was **not submitted** and remains
`PENDING_OWNER_APPROVAL_DO_NOT_SUBMIT` in `RUN_REQUEST.md`. No model, training,
evaluation, profile, metric, matrix, seed, attack, defense, upload, or publication
action occurred.

## Job 333477 — focused data-foundation tests PASS

- Commit: `c1f4fbeade20975fd648e8d6c109f50d27f2bbf4`.
- Runtime source-state SHA-256:
  `dddca872e681a3616c279d9d41fb957f80ef1e780eb9a26705207bdf4269e544`.
- Job/name/node: `333477` / `flv3_s01_zip_tests` / `n430` (`aarch64`).
- State/exit/elapsed/timelimit: `COMPLETED`, `0:0`, `00:01:23`, `00:20:00`.
- Allocation: one node, one GH200, eight CPUs; approximately 0.0231 actual
  GPU-hours. No concurrent job, retry, or resubmission.
- Batch resources: `MaxRSS=36M`, `MaxVMSize=6017600K`,
  `MaxDiskRead=63.23M`, `MaxDiskWrite=0.43M`, `TotalCPU=00:19.544`.
- Pytest/JUnit: `62 passed in 12.83s`; zero failures, errors, or skips.

The job executed all five selected modules. New S07-A cases explicitly covered:

- GT-database explicit cache depth and frozen expected hash;
- historical `t1.v1` filename rejection;
- cache format mismatch, content-hash mutation, missing/mutated sidecars;
- ZIP logical manifest and SQLite-file SHA mismatch;
- directory backend preservation without ZIP relabeling;
- no direct GT-database `t1.v1` bypass.

The retained S01 cases also covered exact ten-archive synthetic manifests,
directory/ZIP bytes and decoded arrays, real-mini scene-start/full-history parity,
writable key/10-sweep LiDAR, fork/spawn/pickle/persistent-worker lifecycle,
local-header mutation, exact duplicate sentinels, cache depth ambiguity, and
extraction-free metadata/cache behavior.

## Runtime identity and dependencies

`execution_identity.json` records:

- Git SHA `c1f4fbeade20975fd648e8d6c109f50d27f2bbf4`;
- source-state hash `dddca872e681a3616c279d9d41fb957f80ef1e780eb9a26705207bdf4269e544`;
- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`, empty `PYTEST_ADDOPTS`;
- NumPy `1.26.4`, nuscenes-devkit `1.1.11`, Pillow `12.2.0`, pytest `9.1.1`,
  and Torch `2.11.0+cu128`;
- mini root
  `/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/fl_weather_project/data/nuscenes_mini`.

Output root:
`/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s07a_data_foundation_tests_c1f4fbeade20`.

| Artifact | SHA-256 |
|---|---|
| `execution_identity.json` | `acc448599638927cd8044a1f3e770c13bdb5a0bc189d099aaab66c810a3eb9c0` |
| `runtime_source_sha256s.txt` | `dddca872e681a3616c279d9d41fb957f80ef1e780eb9a26705207bdf4269e544` |
| `pytest.log` | `c35541e6e7d9cdfa32fac1f5895bb86c5d8d7592a4914402752af6a2dbfc49be` |
| `pytest.junit.xml` | `6b3b3253a78a978823feaa6bc225f774a5e076697528abd03665c9ed873523dc` |

`sha256sum -c sha256sums.txt` passed for every artifact. Logs:

- stdout SHA-256:
  `e839c06338865a3cd251d5094bdb423c240c2a34755ad2e675a073605ebc3902`;
- stderr SHA-256:
  `ae6330855ac405b2e19691ca1681d7f9eeedc6216718d1516023d9376d891b57`.

Stderr contains only the normal module-purge notice.

## Job 335280 — S07-A-R P1 focused provenance validation PASS

- Approval: exact one-time O-009/S00 approval under the owner's 2026-07-11
  validation-only delegation; consumed by this job.
- Executable commit: `44cefd06bc815e893919d95c754896711dba3402`.
- Runtime source-state SHA-256:
  `2710655b166a78e3af39d6537a5098c916463415d27dd9f5503bb79a533c1531`.
- C-locale-sorted 25-file list SHA-256:
  `90310705f1bac3bcdfba9128deea6aed60a270e811cc62759f1204612d61d913`.
- Job/name/node: `335280` / `flv3_s07a_provenance` / `n430` (`aarch64`).
- State/exit/elapsed/timelimit: `COMPLETED`, `0:0`, `00:01:16`, `00:15:00`.
- Allocation: one node, one GH200, eight CPUs; at most 0.25 requested GPU-hours
  and approximately 0.0211 elapsed GPU-hours. `AllocTRES` was
  `billing=1,cpu=8,gres/gpu:nvidia_gh200_120gb=1,gres/gpu=1,mem=11672M,node=1`.
- Batch resources: `MaxRSS=540M`, `MaxVMSize=6476352K`,
  `TotalCPU=00:08.591`.
- Scheduler `Restarts=0`; no requeue, retry, resubmission, or follow-on job.
- Pytest/JUnit: `7 passed in 1.52s`; seven tests, zero failures, errors, or skips.

The exact hostile regressions both executed and passed:

- `test_gt_database_rejects_derived_cache_mutation_before_blob_or_crop[gt_boxes]`;
- `test_gt_database_rejects_derived_cache_mutation_before_blob_or_crop[sweep2keylidar]`.

They preserve logically consistent raw-input canonical hash, pickle metadata, and
JSON sidecar while mutating only derived geometry, then require the GT caller to
reject the physical pickle mismatch before blob-store opening or point cropping.
The retained cases also cover exact depth/canonical/physical identity, historical
`t1.v1` rejection, directory backend preservation, ZIP manifest hashes, and the
absence of a direct legacy bypass.

Output root:
`/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s07a_provenance_tests_44cefd06bc81`.

| Artifact | SHA-256 |
|---|---|
| `sha256sums.txt` | `bae54e1d863523ec7662d16b6fada9b30651c2af08213a3faf09c14443f94c2f` |
| `runtime_source_sha256s.txt` | `2710655b166a78e3af39d6537a5098c916463415d27dd9f5503bb79a533c1531` |
| `pytest.junit.xml` | `c6f28882186bdd8403fa5a7e9d0a059193b34c89c32b4b9258c94aed755c62ba` |
| `pytest.log` | `40f696584529a148f334775fcbead4d33981049fbcc122f5b6a0562a17a2725d` |
| `execution_identity.json` | `56d4c10cf085085bd2ccd9ce8e31c0c4ef04989ca9597b64d40129fd7f50c770` |

`sha256sum -c sha256sums.txt` passed in-job. Independent post-job verification
also passed every one of the 25 per-source checks. Logs:

- stdout path
  `/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/logs/s07a_provenance_335280.out`, SHA-256:
  `9db6bc86997614d95fbd1a58a98641df44fc47cb0508667e0177abfd8fe35ac8`;
- stderr path
  `/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/logs/s07a_provenance_335280.err`, SHA-256:
  `ae6330855ac405b2e19691ca1681d7f9eeedc6216718d1516023d9376d891b57`.

Stderr contains only the normal module-purge notice. The execution identity records
CPython `3.11.15`, NumPy `1.26.4`, nuscenes-devkit `1.1.11`, pyquaternion `0.9.9`,
Pillow `12.2.0`, pytest `9.1.1`, and Torch `2.11.0+cu128`. It also records empty
`PYTEST_ADDOPTS` and disabled third-party pytest plugin autoload.

## Local/static verification

- `/usr/bin/python3 -m pytest ...` could not run because the login interpreter has
  no pytest, NumPy, or Torch. This is recorded as an environment limitation, not a
  code failure or PASS.
- `python3 -m py_compile` passed for changed Python source/scripts/tests.
- `bash -n` passed for all S01 ZIP launchers and the new S07-A cache launcher.
- `git diff --check` passed.

## Negative/missing evidence and interpretation limits

- No full trainval `t1.v2` cache exists from this session; its canonical/file hashes
  remain unknown until the pending exact request is approved and executed.
- Job 335280 is focused real-mini/synthetic provenance evidence only. It does not
  materialize the full trainval cache or establish trainval/model/scientific
  readiness.
- Job 333477 uses real mini plus synthetic ZIP/cache fixtures. It is not
  trainval-scale parity, all-payload CRC coverage, or model-step data-path evidence.
- Historical job 332651 remains the accepted full reference-coverage/loader-only
  evidence and its caches remain forbidden `t1.v1` production inputs. This job does
  not retrofit source attestation to it.
- Allowed: the exact S07-A provenance rejection and directory/ZIP regression suite
  passes on the declared GH200 runtime.
- Forbidden: model/full-data readiness, mAP/NDS, model quality, FL,
  attack/defense, generalization, or publication claims.
