# S07 RESULTS — data-foundation and integrated engineering gates

## Overall result

The bounded O-009 focused data-foundation jobs **PASS**, including the
S07-A-R physical-cache provenance remediation gate at Job 335280. Full trainval `t1.v2`
cache materialization was **not submitted** and remains
`PENDING_OWNER_APPROVAL_DO_NOT_SUBMIT` in `RUN_REQUEST.md`. No model, training,
evaluation, profile, metric, matrix, seed, attack, defense, upload, or publication
action occurred in those S07-A jobs.

The later S07-B one-time bounded integrated runtime Job 348557 **FAILED** by
internal timeout after recording three failure glyphs and four error glyphs. It
does not establish integrated-runtime PASS. O-052 is consumed and no retry or
follow-on is authorized.

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

---

## Job 348557 — S07-B bounded integrated runtime FAILED

### Immutable execution and scheduler record

- Canonical decision: O-054 at exact Orchestra commit
  `91526456ee4d4c9d63835868b055b537d0d6655c`.
- Exact executable/archive commit `L`:
  `05b733997968b8217e1fc6dd27c3a4add34f6c98`.
- Launcher SHA-256:
  `1b1c45d33b113d0c7d649e51b2ddf98a2d7822eab38d708d4bb0e223b8c334c0`.
- Exact 123-file source-list SHA-256:
  `be3b9157e213b942094d290d403306aa714e82157e36ba92847e32cfef71419a`.
- Exact aggregate source-state SHA-256:
  `d8c6cc0e20ed0c8ded5a4e13dd3ae52f32a62ebbcfafd2f9cbcd469fc5b87acd`.
- Literal mini root:
  `/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/fl_weather_project/data/nuscenes_mini`.
- Job/name/node: `348557` / `flv3_s07b_integrated` / `n30` (`aarch64`).
- Scheduler state/exit: `FAILED 1:0`.
- Submit/Start/End, Europe/Stockholm CEST:
  `2026-07-12T12:33:41+02:00` / `2026-07-12T12:33:42+02:00` /
  `2026-07-12T13:18:02+02:00`.
- Elapsed: `00:44:20`; scheduler restarts: `0`.
- Allocation: one node, one task, one NVIDIA GH200 120GB, eight CPUs and 64 GiB
  (`67108864K`) memory.
- Batch `MaxRSS=10573756K`; `TotalCPU=01:35.363`.
- O-052 authorized exactly one attempt and was consumed at submission. There is
  no retry, requeue, replacement or follow-on authority.

Independent post-job `sacct` inspection reproduced the scheduler rows above. The
execution identity reproduced the exact Git, launcher, list and aggregate hashes,
one visible `NVIDIA GH200 120GB`, mini root, `spconv==2.3.8` and `cumm==0.7.13`.
Thus launch/source/runtime attestation passed; test acceptance did not.

### Pytest outcome and failed acceptance

- `pytest.exitcode` contains exactly `124`, the GNU `timeout` status from the
  launcher's internal 42-minute guard.
- The shared stdout/pytest log contains progress through the displayed `[86%]`
  line and then ends without a newline at `.........F..EEEE............`.
  Across the complete log there are exactly three `F` glyphs and four `E`
  glyphs before the hang/timeout.
- There is no pytest summary, traceback, finalized per-test identity, JUnit XML
  or authoritative collected/pass/fail/error/skip count.
- Required `pytest.junit.xml`, `pytest_junit_counts.json` and final
  `sha256sums.txt` are absent. Because the launcher stops at the missing JUnit
  check, no final artifact checksum verification or JUnit acceptance executes.
- The exact gate therefore **FAILS**: pytest exit was nonzero, at least `3F+4E`
  were observed, the suite did not finish, required artifacts are missing, and
  zero-failure/error/skip acceptance cannot be established.

The output-local pytest basetemp contains
`test_repeated_persistent_multi0` and a
`test_repeated_persistent_multicurrent` symlink pointing to it; both are among
the latest timestamped basetemp entries. Exact `L` maps that truncated name to
`test_repeated_persistent_multiworker_reads_are_deterministic`. This is a
**high-confidence diagnostic inference** about the hang location only. It is
not a formal attribution or root-cause finding because `-q`, timeout, absent summary
and absent JUnit leave no finalized current-test record. The identities of all
three failures and four errors are likewise unknown from this run.

### Preserved artifacts and independently recomputed SHA-256

Output root:
`/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s07b_integrated_05b733997968`.

| Existing artifact | SHA-256 |
|---|---|
| `config_sha256s.txt` | `4a7b425b9078c8d16035aa501382787bb161f7d435d16e3d56d07b1221b671aa` |
| `execution_identity.json` | `af7a7ea953d56b1c909cb395c86075e5f23219411945490341dc804dcab4a69f` |
| `pytest.exitcode` | `ca2ebdf97d7469496b1f4b78958f9dc8447efdcb623953fee7b6996b762f6fff` |
| `pytest.log` | `eceba3ae66efdb901626eac108200bc9f50108229a290dad39dec64bd8abad2c` |
| `runtime_source_files.txt` | `be3b9157e213b942094d290d403306aa714e82157e36ba92847e32cfef71419a` |
| `runtime_source_sha256s.txt` | `d8c6cc0e20ed0c8ded5a4e13dd3ae52f32a62ebbcfafd2f9cbcd469fc5b87acd` |
| `selected_test_files.txt` | `c9305627d222dcaf0575e4006c81be797f2eb8e7cc21a13285fe59325840dfd5` |

All 123 records inside `runtime_source_sha256s.txt` independently verify against
the immutable snapshot. The following required artifacts are missing:

- `pytest.junit.xml`;
- `pytest_junit_counts.json`;
- `sha256sums.txt`.

Logs and independently recomputed hashes:

| Log | SHA-256 |
|---|---|
| `s07b_integrated_348557.out` | `eceba3ae66efdb901626eac108200bc9f50108229a290dad39dec64bd8abad2c` |
| `s07b_integrated_348557.err` | `ae6330855ac405b2e19691ca1681d7f9eeedc6216718d1516023d9376d891b57` |

Stdout is byte-identical to `pytest.log`. Stderr contains only the normal module
purge notice; it does not identify the failures, errors or hang.

### Negative evidence and interpretation boundary

Preserved negative result: exact source/runtime identity and environment setup
succeeded, but the selected integrated suite produced at least three failures,
four errors and a hang before timing out. This is S07-B integrated-runtime
**FAIL**, not an infrastructure-neutral PASS and not a partial module PASS.

Allowed interpretation: exact Job 348557 launched the attested 25-file mini plus
synthetic suite on one GH200 and exposed unresolved integrated test failures,
errors and a likely persistent-worker hang location requiring separate diagnosis.

Forbidden interpretation: which named tests caused the `3F+4E`; a proven root
cause for the hang; any S02-S07 component-level PASS inferred from progress dots;
production/full-trainval/cache readiness; throughput/memory performance; mAP/NDS,
model quality or fusion gain; 100/1000-step training; FL, attack/defense,
generalization, seed/matrix or publication evidence. O-052 is consumed and this
result authorizes no retry, diagnostic execution, merge, push or follow-on.

---

## Job 348818 — diagnostic harness COMPLETE, suite FAIL

### Scheduler and immutable identity

- Canonical O-058: `348f29c3c68243ae6010ea0d017e16850081c43c`.
- Scheduler state/exit: `COMPLETED 0:0`; this means harness completion only.
- Submit/Start/End: `2026-07-12T13:40:45+02:00` /
  `2026-07-12T13:40:46+02:00` / `2026-07-12T13:57:16+02:00`.
- Elapsed `00:16:30`; node `n412`; one GH200, eight CPUs, 64 GiB;
  `MaxRSS=10539927K`, `TotalCPU=05:05.306`, `Restarts=0`.
- Exact `L`: `fd142dc1c247ed527dbf5ddb823576c817dc415a`.
- Launcher/list/state SHA-256:
  `d8d7686eb727d4973591cf20186615f6bf2f3bc71ba020dec815c9b6d2d0dc1b` /
  `40c364201bda63386be614fca3710f62111e6964f9b7fdc1beffef69cb5f05d8` /
  `56ddfdc66045548899cdde1ad08f7e394c300a8fc27a6c0aaf6551a8178533b2`.
- Identity records the literal mini root, NVIDIA GH200 120GB,
  `spconv==2.3.8`, `cumm==0.7.13`, and parent Job 348557 negative commit
  `d7888a9fef615c83c8d36161bfa6d581a3dc4f0f`.

O-056 was consumed by this one attempt. No retry, requeue, replacement or
follow-on is authorized.

### Authoritative diagnostic summary

`diagnostic_complete=true` and `suite_pass=false`. All 25 isolated files were
attempted. Their JUnit aggregate is exactly 251 tests, 3 failures, 94 errors and
0 skips.

Ninety errors are diagnostic-launcher noise, not independently established code
failures: the unique basetemp was nested under a missing
`$JOB_TMP/isolated` parent, so pytest setup raised `FileNotFoundError` across
seven files (20+11+16+23+2+6+12 = 90 errors). Four additional real execution
errors in `test_model_task.py` are `PermissionError: [Errno 13]` for
`./fl_outputs` because the archive snapshot is read-only:

- `test_num_clients_iid_is_requested`;
- `test_client_data_materializes_dict_batch`;
- `test_generalized_loop_trains_detection_batch`;
- `test_loader_determinism_num_workers`.

The three genuine isolated failures are:

1. `test_multitask_loss_rejects_legacy_single_head_output`: expected regex
   `six task`, but actual message was
   `multi-task CenterHead must return 6 task dictionaries`.
2. `test_default_off_byte_identical`: expected the legacy 62-tensor trainable
   layout, but observed 230 (`OFF must keep the 62-tensor trainable layout`).
3. `test_dummy_regression_byte_identity_golden`: actual SHA-256
   `4fa46307bab67f2a836102b23b1ad2abc331702e83d16c65e11a09330c3d9edb`
   differed from golden
   `d2d819fee9a54fc302a9d6c9d0ac4e4d875629a0a16e75f2328f28b7f63cd7cc`.

The combined probe exited `124` and produced no JUnit. Its verbose log and
60-second faulthandler formally identify
`test_repeated_persistent_multiworker_reads_are_deterministic[fork]` as the hang:
the main thread waits in `queue.get` through Torch DataLoader `_try_get_data` /
`_get_data` / `_next_data`, while multiprocessing queue feeder and pin-memory
threads wait. This upgrades Job 348557's basetemp hint to a formal diagnostic
location, but does not by itself prove the underlying code root cause.

### Checksums and preserved artifacts

The sorted manifest contains 110 records and independent
`sha256sum -c sha256sums.txt` verification passed all 110. Key SHA-256 values:

| Artifact/log | SHA-256 |
|---|---|
| `diagnostic_summary.json` | `892d335d528c8ea29c671a5152bbf919398882a622b6ade17e2d25b6334de9ff` |
| `execution_identity.json` | `d1653ea9fa92504df2c2327a88db6003316f6a65af1401cf972adde74904cbe9` |
| `isolated_attempts.tsv` | `63d615c1dd0bb25e84b4f75ce117a69f405bc6752a44166e0189b0d45b6b8dd0` |
| `sha256sums.txt` | `b794336a825b7a44eb8d22033bf4684fa43a93b7999f24a597b90d8d5999c835` |
| `combined/pytest.log` | `ba4472f81a8f8b37f1e768f614c0a6d47f50271d7450a56e1bf18ebfbe0ec76d` |
| `combined/pytest.exitcode` | `ca2ebdf97d7469496b1f4b78958f9dc8447efdcb623953fee7b6996b762f6fff` |
| `config_sha256s.txt` | `4a7b425b9078c8d16035aa501382787bb161f7d435d16e3d56d07b1221b671aa` |
| `runtime_source_files.txt` | `40c364201bda63386be614fca3710f62111e6964f9b7fdc1beffef69cb5f05d8` |
| `runtime_source_sha256s.txt` | `56ddfdc66045548899cdde1ad08f7e394c300a8fc27a6c0aaf6551a8178533b2` |
| `selected_test_files.txt` | `c9305627d222dcaf0575e4006c81be797f2eb8e7cc21a13285fe59325840dfd5` |
| Slurm stdout | `39077092bfc314567c9ed2fc94e47ed412cfa0fbf525072171ae69f5d973bef8` |
| Slurm stderr | `ae6330855ac405b2e19691ca1681d7f9eeedc6216718d1516023d9376d891b57` |

### Interpretation boundary

Allowed: the diagnostic harness completed and attributed the observed failures,
launcher noise, read-only-output errors and fork DataLoader hang as stated above.
Forbidden: calling this suite PASS; treating 90 launcher-noise errors as product
failures; claiming the four output-path errors or hang root cause are remediated;
production/full-data/scientific readiness, metrics, training, FL, attack/defense
or publication claims. This result grants no remediation or compute authority.

---

## Job 349653 — two-snapshot dummy attribution PASS

### Immutable request and scheduler record

- Canonical terminal decision: O-063 at exact Orchestra commit
  `fe22ecca9bfc455c5d63ea3c9c2f4f00907a7609`.
- Exact detached executable `L`:
  `a9d657aebfb0f64d271fa74e312d6054eca57e1d`.
- Launcher Git blob/SHA-256:
  `295610fd422f3b371b8fd85e54785919903dc332` /
  `bbc1293a42034540327402a5df6c1f172b76afacca7906b4f0b71f5290b5968a`.
- Pre-S06/current snapshots:
  `968d81583c87ba76b7dbbb722760f8eb8eb6cd39` /
  `c69befe5e8dd6397059c4d3fe1cbf906a9646836`.
- Pre source list/state SHA-256:
  `0ec5e43e98ee6b98c949d6c3187c4484fb47842443fd3ffe54fcb61e4d777ae0` /
  `dc2144cc522d20035eeff81269e45973312c10f908ddc6808bc3c2611b38c93d`.
- Current source list/state SHA-256:
  `104a647441ce712e83c20d32372944e48777913cf746ae19ee124894ca927e41` /
  `0f2995fca7d323421e46326493b2f9cc5d0032ca3d794d422779fc10c626ee32`.
- Job/name/node: `349653` / `flv3_s07b_dummy_attr` / `n530` (`aarch64`).
- Submit/start: `2026-07-12T14:23:08+02:00` /
  `2026-07-12T14:23:09+02:00`.
- Scheduler state/exit/elapsed: `COMPLETED` / `0:0` / `00:01:26`.
- Allocation: one node/task/GH200, four CPUs, 32 GiB, ten-minute limit;
  batch `MaxRSS=540M`, `TotalCPU=00:14.815`.
- O-061 authorized and was consumed by this single attempt. No retry, requeue,
  replacement, alternate invocation, automatic resubmission or follow-on is
  authorized.

### Exact runtime and attribution result

The execution identity records host `n530`, Linux aarch64, CPython `3.11.15`,
NumPy `1.26.4`, SciPy `1.13.1`, Torch `2.11.0+cu128`, torchvision
`0.26.0+cu128`, spconv `2.3.8`, cumm `0.7.13`, nuscenes-devkit `1.1.11`,
pyquaternion `0.9.9`, and Pillow `12.2.0`. The exact interpreter is
`/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/envs/pt311-cu128-spconv/bin/python`.

All four independent subprocesses exited zero:

| Snapshot | Repetition | Checksum |
|---|---:|---|
| pre-S06 `968d815...` | 1 | `4fa46307bab67f2a836102b23b1ad2abc331702e83d16c65e11a09330c3d9edb` |
| pre-S06 `968d815...` | 2 | `4fa46307bab67f2a836102b23b1ad2abc331702e83d16c65e11a09330c3d9edb` |
| current `c69befe...` | 1 | `4fa46307bab67f2a836102b23b1ad2abc331702e83d16c65e11a09330c3d9edb` |
| current `c69befe...` | 2 | `4fa46307bab67f2a836102b23b1ad2abc331702e83d16c65e11a09330c3d9edb` |

The authoritative summary has `diagnostic_complete=true`, classification
`stable_equal_current`, and `automatic_code_or_golden_change_authorized=false`.
Thus the S06/current changes did not introduce the Job 348818 checksum drift in
this frozen runtime: the pre-S06 snapshot produces the same value. Historical
`d2d819fee9a54fc302a9d6c9d0ac4e4d875629a0a16e75f2328f28b7f63cd7cc`
is retained as old-environment evidence, not an Arrhenius-portable checksum.

### Raw artifact paths and SHA-256

Output root:
`/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s07b_dummy_attr_a9d657aebfb0`.

| Relative artifact | SHA-256 |
|---|---|
| `attempts.tsv` | `dfa41729753866671852071bddfc7539c44408b294c16058aeeb846fd3b15467` |
| `attribution_summary.json` | `806afbfd41eabad3d2181c7c829a74f4ded34cef91636b5bdb7018b5fbbc36fc` |
| `current_source_files.txt` | `104a647441ce712e83c20d32372944e48777913cf746ae19ee124894ca927e41` |
| `current_source_identity.txt` | `8d8a802f8fa76e7970649104167fa30afb3071f25d598741fcf1e6c166fa7b57` |
| `current_source_sha256s.txt` | `0f2995fca7d323421e46326493b2f9cc5d0032ca3d794d422779fc10c626ee32` |
| `execution_identity.json` | `b66bbc7400aa1acdfbaf059caea44faf3ac7bfb165b3172898a8e4d84462e9e9` |
| `pre_source_files.txt` | `0ec5e43e98ee6b98c949d6c3187c4484fb47842443fd3ffe54fcb61e4d777ae0` |
| `pre_source_identity.txt` | `a2cd149b5478b12f00b58ba603413878e83108dbc36992e9da20bdbdba4cacc0` |
| `pre_source_sha256s.txt` | `dc2144cc522d20035eeff81269e45973312c10f908ddc6808bc3c2611b38c93d` |
| `runs/current_1/exit.txt` | `9a271f2a916b0b6ee6cecb2426f0b3206ef074578be55d9bc94f6f3fe3ab86aa` |
| `runs/current_1/result.json` | `12441e87915f64ca6c278c4b7d1cc65394b3b3037c79d01e07a54e930ba62c50` |
| `runs/current_1/stderr.log` | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `runs/current_1/stdout.log` | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `runs/current_2/exit.txt` | `9a271f2a916b0b6ee6cecb2426f0b3206ef074578be55d9bc94f6f3fe3ab86aa` |
| `runs/current_2/result.json` | `418178d21e4251f1dd9cfef6559a7a5b31c2c78d05d4c60bf3da34d3c39ccfd4` |
| `runs/current_2/stderr.log` | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `runs/current_2/stdout.log` | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `runs/pre_1/exit.txt` | `9a271f2a916b0b6ee6cecb2426f0b3206ef074578be55d9bc94f6f3fe3ab86aa` |
| `runs/pre_1/result.json` | `2762770819182a81d4fba235a67610fed02b9ff817265196f14c2ff4accdeea6` |
| `runs/pre_1/stderr.log` | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `runs/pre_1/stdout.log` | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `runs/pre_2/exit.txt` | `9a271f2a916b0b6ee6cecb2426f0b3206ef074578be55d9bc94f6f3fe3ab86aa` |
| `runs/pre_2/result.json` | `11acaa2023f6d67f116c6b19e02727b19e4a706ad806b6ee8ede53a80905ef62` |
| `runs/pre_2/stderr.log` | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `runs/pre_2/stdout.log` | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |

The manifest itself is
`sha256sums.txt` =
`0c74aae4067bab74619269c16b38c8724ce38d56018d6dea035066e78528341c`.
All 25 entries passed in-job `sha256sum -c` and were independently read back.

Scheduler logs:

| Log | SHA-256 |
|---|---|
| `/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/logs/s07b_dummy_attr_349653.out` | `48f6a06c4ca6c84cd6c0d31ae5b4369c6b6f63e37dfaba1827e662e0ad3268fb` |
| `/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/logs/s07b_dummy_attr_349653.err` | `ae6330855ac405b2e19691ca1681d7f9eeedc6216718d1516023d9376d891b57` |

### Interpretation boundary and required follow-up

Allowed: on the exact frozen Arrhenius identity, both pre-S06 and current source
snapshots repeat stably at `4fa46307...`; the current drift from historical
`d2d819...` is therefore not attributable to S06/current source changes.

Forbidden: claiming a universal cross-platform checksum; silently replacing or
deleting the historical value; changing `training/loop.py`; calling the prior
integrated suite PASS; production/full-cache/trainval readiness; throughput or
memory performance; mAP/NDS, model quality, fusion gain, FL, attack/defense,
generalization, matrix/seed or publication evidence.

O-063 authorizes only a runtime-aware test contract plus this evidence update.
That test change still requires independent review and focused execution before
it can contribute to S07-B runtime readiness. Job 349653 grants no retry, merge,
push, upload or additional compute authority.
