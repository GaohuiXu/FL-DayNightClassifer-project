# S07-A RUN_REQUEST — focused data-foundation tests and pending t1.v2 cache

## A. O-009 focused data-foundation test — standing authorization

### Approval and interpretation

- **Status:** `EXECUTED_ONCE_COMPLETED_PASS` under exact O-009 scope.
- This is one dependency-complete engineering test job, not a full-data gate or
  scientific run. It is within O-009: one node, one GH200, eight CPUs, 20 minutes,
  one concurrent S07-A job, maximum 0.333 GPU-hours, and cumulative S07-A usage
  below two GPU-hours.
- It does not authorize retry, array, DDP, model steps, 100/1000-step work,
  trainval scan/cache/profile, metric/evaluation, matrix, seed, or scientific claim.

### Immutable preflight

- Historical submission branch: `codex/s07-a-data-foundation`. This completed
  Job 333477 record is not the execution-worktree contract for the pending cache
  request and must not be reused to submit it.
- Exact implementation HEAD:
  `c1f4fbeade20975fd648e8d6c109f50d27f2bbf4`.
- Working source diff before this audit record: empty; SHA-256 of
  `git diff --binary HEAD` was
  `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`.
  This request/results/handoff prose is not in the launcher source-state set.
- Focused runtime source-state SHA-256:
  `dddca872e681a3616c279d9d41fb957f80ef1e780eb9a26705207bdf4269e544`.
- The source-state set includes every regular file under
  `fl_v3/src/fl_v3/data/nuscenes/`, `build_gt_database.py`, all five selected test
  modules, `fl_v3/tests/conftest.py`, the focused launcher, `fl_v3/pyproject.toml`,
  `requirements.txt`, and `requirements.lock.txt`.
- The launcher disables third-party pytest plugin autoload, clears
  `PYTEST_ADDOPTS`, and records installed NumPy, nuscenes-devkit, Pillow, pytest,
  and Torch versions in `execution_identity.json`.
- Input is the existing extracted mini root only:
  `/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/fl_weather_project/data/nuscenes_mini`.
  Synthetic ZIPs/manifests/caches are created only under the job's temporary
  directory. No shared trainval archive is opened.
- Exact unique output root, verified absent before submission:
  `/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s07a_data_foundation_tests_c1f4fbeade20`.
- Preflight found no active S07-A job.

### Exact command, acceptance, and stop conditions

```bash
test "$(git rev-parse HEAD)" = "c1f4fbeade20975fd648e8d6c109f50d27f2bbf4" && \
test ! -e /nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s07a_data_foundation_tests_c1f4fbeade20 && \
test -z "$(squeue -u "$USER" -h -o '%i %j' | awk '$2 ~ /flv3_s0(1_zip_tests|7a)/ {print}')" && \
sbatch --time=00:20:00 --cpus-per-task=8 \
  --export=ALL,EXPECTED_S01_SHA=c1f4fbeade20975fd648e8d6c109f50d27f2bbf4,EXPECTED_S01_STATE_HASH=dddca872e681a3616c279d9d41fb957f80ef1e780eb9a26705207bdf4269e544,S01_MINI_DATAROOT=/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/fl_weather_project/data/nuscenes_mini,S01_OUTPUT_ROOT=/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s07a_data_foundation_tests_c1f4fbeade20 \
  fl_v3/scripts/run_s01_nuscenes_zip_tests.sh
```

The launcher runs exactly these files:

1. `fl_v3/tests/test_build_gt_database.py`;
2. `fl_v3/tests/test_nuscenes_zip_backend.py`;
3. `fl_v3/tests/test_nuscenes_zip_dataset.py`;
4. `fl_v3/tests/test_nuscenes_zip_info_cache.py`;
5. `fl_v3/tests/test_nuscenes_info_cache.py`.

Pass requires zero failures/errors/skips, exact Git/source identity, directory/ZIP
preservation, explicit depth binding, t1.v1/format/content/sidecar rejection, cache
and manifest mismatch rejection, GT caller provenance behavior, fork/spawn
lifecycle, and checksummed JUnit/log/identity/source-list artifacts. The job stops
on any mismatch, output collision, skip/failure/error, mini-root failure, or
walltime. There is no automatic retry.

### Execution record

- Submitted exactly once as Slurm job `333477`; no concurrent S07-A job, retry,
  or follow-on was submitted.
- Node/architecture: `n430` / `aarch64` GH200.
- State/exit/elapsed: `COMPLETED`, `0:0`, `00:01:23`; approximately 0.0231 actual
  GPU-hours, within the 0.333 requested upper bound.
- Exact identity matched commit
  `c1f4fbeade20975fd648e8d6c109f50d27f2bbf4` and source hash
  `dddca872e681a3616c279d9d41fb957f80ef1e780eb9a26705207bdf4269e544`.
- Pytest/JUnit: `62 passed in 12.83s`, zero failures/errors/skips. Checksums and
  detailed artifact/resource evidence are in `RESULTS.md`.

---

## B. Full trainval t1.v2 cache materialization — PENDING OWNER APPROVAL

### Approval state and immutable identity

- **Status:** `PENDING_OWNER_APPROVAL_DO_NOT_SUBMIT`.
- This full trainval cache generation is outside O-009. Preparing this request is
  not permission to execute it.
- Durable S07-A-R review:
  `976206405ccf7d2c864d318f5ee27302bdf59059` (`CHANGES-REQUESTED`).
- Exact P1-remediated implementation candidate (`NEW_IMPL_SHA`):
  `44cefd06bc815e893919d95c754896711dba3402`.
- Exact cache-launcher source-state SHA-256:
  `1322c87255bc350323de108e347eea1e54daeb12b59fe1889cb15006f79c3884`.
- The 23-file set was recomputed identically from the clean worktree and immutable
  Git blobs. It includes every tracked Python file under
  `fl_v3/src/fl_v3/data/nuscenes/`, package initializers,
  `fl_v3/src/fl_v3/data/partition.py`, `fl_v3/src/fl_v3/utils/runtime.py`, the
  cache builder/launcher/environment bootstrap, and dependency/config manifests.
- Source-list sorting alone is locked with `LC_ALL=C`. The exact full-cache file-
  list SHA-256 is
  `eebaaf9528a56004b63cc2cb37fe6d312b75a52df450f374307e8e559cb1cbb5`;
  this list and aggregate hash matched under ambient `C.UTF-8`, `en_US.UTF-8`, and
  `sv_SE.UTF-8`. No global Python/cache runtime locale is changed.
- Eventual executor contract: after separate owner approval, the owner/Codex task
  UI provisions a fresh isolated worktree at
  `detached@44cefd06bc815e893919d95c754896711dba3402`. The executor verifies a
  clean worktree, empty branch name, and exact HEAD before submission. This
  request does not name, reuse, or depend on the current worker worktree. No
  execution worktree has been created by this remediation.
- Environment: Arrhenius persistent prefix
  `/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/envs/pt311-cu128-spconv`,
  activated through `fl_v3/scripts/arrhenius_env.sh` after
  `arrhenius_load_modules build`; dataset module
  `nuScenes-data/1.0-map-1.3-zip` is loaded afterward.
- Dataset/version/splits/depth: module `NUSCENES_DATA_DIR`, official
  `v1.0-trainval`, splits `train val`, `n_sweeps=10` total including keyframe.
- Read-only prefix metadata currently records CPython `3.11.15` (`aarch64`),
  NumPy `1.26.4`, nuscenes-devkit `1.1.11`, and pyquaternion `0.9.9`. These are
  expected values only. The in-job `execution_identity.json` records the actual
  interpreter path/implementation/version, platform, and installed versions of
  NumPy, nuscenes-devkit, pyquaternion, Torch, and Pillow; an approved execution
  must retain and review that actual identity. Source/requirements hashes do not
  substitute for this runtime record.
- The obsolete tuple `ed31f23` / `7ddb06...` /
  `s07a_cache_t1v2_ed31f23b2ee1` is permanently superseded and unapproved.
- The later c8dd/6a4 tuple is also superseded before submission: its ambient-locale
  source-list sorting was not reproducible across executor locale. No cache job
  was submitted from either superseded request.

### Accepted immutable manifest input

- Path:
  `/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s01_zip_full_gate_v2_1fe651700bd0/nuscenes_trainval_zip_manifest.sqlite`.
- Format: `s01.nuscenes-zip.v2`.
- Logical manifest hash:
  `023f72b4220bb0db587be00920308bf9074384740fe186d243be92f9a53119f6`.
- SQLite file SHA-256:
  `228e2f5bab30007acb06eb61393d1fbacc88979490668ff800f8f7f9752a47fb`.
- Exact archive set: `trainval01_blobs.zip` through `trainval10_blobs.zip`;
  2,631,093 occurrences, 2,631,084 unique members, nine identical `LICENSE`
  duplicate occurrences. The launcher validates the exact logical/file hashes and
  archive names before creating the output root. It never rebuilds or modifies the
  accepted manifest.

### Requested resources, command, and outputs

- One node, one GH200, eight CPUs, walltime `00:30:00`; maximum 0.5 GPU-hours.
  GPU allocation is for the validated aarch64 environment; cache construction is
  metadata/CPU/I/O work. One job only, no array/DDP/retry/follow-on.
- Exact unique output root:
  `/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s07a_cache_t1v2_44cefd06bc81`.
  It was confirmed absent during this local remediation; this is a proposed path,
  not a created output.
- Logs:
  `/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/logs/s07a_cache_t1v2_%j.{out,err}`.

```bash
test -z "$(git branch --show-current)" && \
test "$(git rev-parse HEAD)" = "44cefd06bc815e893919d95c754896711dba3402" && \
test -z "$(git status --short)" && \
test ! -e /nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s07a_cache_t1v2_44cefd06bc81 && \
test -z "$(squeue -u "$USER" -h -o '%i %j' | awk '$2 ~ /flv3_s07a_cache_t1v2/ {print}')" && \
sbatch --time=00:30:00 --cpus-per-task=8 \
  --export=ALL,EXPECTED_S07A_SHA=44cefd06bc815e893919d95c754896711dba3402,EXPECTED_S07A_STATE_HASH=1322c87255bc350323de108e347eea1e54daeb12b59fe1889cb15006f79c3884,S07A_ACCEPTED_MANIFEST=/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s01_zip_full_gate_v2_1fe651700bd0/nuscenes_trainval_zip_manifest.sqlite,S07A_ACCEPTED_MANIFEST_HASH=023f72b4220bb0db587be00920308bf9074384740fe186d243be92f9a53119f6,S07A_ACCEPTED_MANIFEST_FILE_SHA256=228e2f5bab30007acb06eb61393d1fbacc88979490668ff800f8f7f9752a47fb,S07A_OUTPUT_ROOT=/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s07a_cache_t1v2_44cefd06bc81 \
  fl_v3/scripts/run_s07a_nuscenes_cache_t1v2.sh
```

Expected outputs whose identities/checksums must be frozen:

- `info_cache_msweep10/nuscenes_info_v1.0-trainval_train_t1.v2_nsweeps10.pkl`;
- matching train sidecar `.meta.json`;
- `info_cache_msweep10/nuscenes_info_v1.0-trainval_val_t1.v2_nsweeps10.pkl`;
- matching val sidecar `.meta.json`;
- `execution_identity.json`, `runtime_source_sha256s.txt`,
  `cache_identity.json`, and `sha256sums.txt`.

`cache_identity.json` must freeze, for both train and val, format `t1.v2`, depth
10, predeclared/actual/metadata sample and box counts, canonical cache hash,
absolute output path, byte size, pickle SHA-256, and sidecar SHA-256. The exact
locked counts are train `n_samples=28130`, `n_boxes=944881` and val
`n_samples=6019`, `n_boxes=187528`; both actual records and metadata must equal
these values. The launcher first generates `sha256sums.txt`, then separately runs
`sha256sum -c "$S07A_OUTPUT_ROOT/sha256sums.txt"`; checksum generation alone is
not acceptance.
The accepted manifest logical/file hashes are copied into the execution identity;
the cache is not silently relabeled from historical `t1.v1` output.

### Acceptance and stop conditions

Pass requires exact commit/source identity; accepted manifest logical/file/archive
identity; external fresh output; successful explicit `t1.v2` train and val builds;
explicit `load_cache(..., n_sweeps=10)` validation of every record, sidecar, and
canonical content hash; exact train `28130/944881` and val `6019/187528`
sample/box counts in both actual records and metadata; captured actual runtime
identity; generated checksums; and successful in-job `sha256sum -c` verification.
Stop on any identity/hash/archive/output mismatch, missing/ambiguous cache,
format/depth/record/sidecar/content mismatch, unexpected sample count, exception,
or walltime. No retry or additional coverage/profile/model job is implied.

Allowed after an independently reviewed pass: the two exact `t1.v2` cache artifacts
may be proposed as production inputs with their frozen cache and accepted manifest
hashes. Forbidden: retroactive attestation of job 332651, trainval-scale decoded
directory/ZIP parity, all-payload CRC coverage, model-step readiness, model quality,
metrics, FL/attack/defense, generalization, or publication claims.

---

## C. S07-A-R P1 focused provenance validation — EXECUTED ONCE, COMPLETED PASS

### Approval state and immutable scope

- **Status:** `EXECUTED_ONCE_COMPLETED_PASS`; the exact one-time approval
  `APPROVED_ONCE_BY_S00_UNDER_O-009_AND_OWNER_DELEGATION_2026-07-11` is consumed.
- **Locale-stable replacement approval:** after independently verifying the actual
  diff, three-locale hashes, fresh output absence, empty queue, and unchanged
  RESULTS blob, S00 reapproved exactly one submission. This approval is strictly
  bound to:
  - implementation `44cefd06bc815e893919d95c754896711dba3402`;
  - focused aggregate source hash
    `2710655b166a78e3af39d6537a5098c916463415d27dd9f5503bb79a533c1531`;
  - C-locale-sorted file-list hash
    `90310705f1bac3bcdfba9128deea6aed60a270e811cc62759f1204612d61d913`;
  - mini root
    `/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/fl_weather_project/data/nuscenes_mini`;
  - fresh output root
    `/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s07a_provenance_tests_44cefd06bc81`;
  - the exact command below;
  - one node, one GH200, eight CPUs, walltime at most `00:15:00`, and at most
    0.25 GPU-hours;
  - no array, DDP, model, full cache, metric, retry, resubmit, or follow-on.
- Any bound-field change invalidates this approval. Section B remains
  `PENDING_OWNER_APPROVAL_DO_NOT_SUBMIT`.
- **Preserved preflight rejection:** the earlier status
  `APPROVED_ONCE_BY_S00_UNDER_O-009_AND_OWNER_DELEGATION_2026-07-11` was bound to
  executable `c8dd920...`, focused hash `357da487...`, and output
  `s07a_provenance_tests_c8dd920cf3f8`. A clean detached executor at c8dd under
  `LANG=en_US.UTF-8` computed
  `8de319b624519ae9582be70699eafa6d9ebb8964bc8e8ba548bc67372201475c`
  because ambient-locale `sort -u` collated the source list differently from
  S00's `C.UTF-8` preflight. The exact hash guard rejected before `sbatch`; there
  was no output and the queue was empty. Implementation change invalidates the
  old one-time approval, which must not be reused or relabeled.
- Section B remains `PENDING_OWNER_APPROVAL_DO_NOT_SUBMIT`; this replacement
  focused request does not approve it.
- This request validates only the remediated GT-cache physical-identity contract
  and its existing directory/ZIP provenance neighbors on real mini plus synthetic
  mutated caches. It does not open shared trainval archives, build full caches,
  run a model, profile, evaluate metrics, or make scientific claims.
- Exact implementation: `44cefd06bc815e893919d95c754896711dba3402`.
- Exact 25-file focused runtime source-state SHA-256:
  `2710655b166a78e3af39d6537a5098c916463415d27dd9f5503bb79a533c1531`.
- Exact C-locale-sorted file-list SHA-256:
  `90310705f1bac3bcdfba9128deea6aed60a270e811cc62759f1204612d61d913`.
- Locale regression evidence: file-list and aggregate hashes above matched exactly
  under ambient `C.UTF-8`, `en_US.UTF-8`, and `sv_SE.UTF-8`. Only the source-list
  `sort -u` runs with `LC_ALL=C`; the launcher does not change global test/runtime
  locale.
- Input mini root:
  `/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/fl_weather_project/data/nuscenes_mini`.
- Exact output root, confirmed absent before submission and created only by Job
  335280:
  `/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s07a_provenance_tests_44cefd06bc81`.
- Resources: one submission, one node, one GH200, eight CPUs, walltime at most
  `00:15:00`, maximum 0.25 GPU-hours. No array, DDP, model, full cache, metric,
  retry, resubmit, or follow-on job was authorized or executed.
- Executor: a fresh owner/Codex-UI-provisioned clean
  `detached@44cefd06bc815e893919d95c754896711dba3402` worktree.

### Exact command

```bash
test -z "$(git branch --show-current)" && \
test "$(git rev-parse HEAD)" = "44cefd06bc815e893919d95c754896711dba3402" && \
test -z "$(git status --short)" && \
test ! -e /nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s07a_provenance_tests_44cefd06bc81 && \
test -z "$(squeue -u "$USER" -h -o '%i %j' | awk '$2 ~ /flv3_s07a_provenance/ {print}')" && \
sbatch --time=00:15:00 --cpus-per-task=8 \
  --export=ALL,EXPECTED_S07A_SHA=44cefd06bc815e893919d95c754896711dba3402,EXPECTED_S07A_PROVENANCE_STATE_HASH=2710655b166a78e3af39d6537a5098c916463415d27dd9f5503bb79a533c1531,S07A_MINI_DATAROOT=/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/fl_weather_project/data/nuscenes_mini,S07A_PROVENANCE_OUTPUT_ROOT=/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s07a_provenance_tests_44cefd06bc81 \
  fl_v3/scripts/run_s07a_provenance_tests.sh
```

### Acceptance and stop conditions

The launcher itself rechecks clean detached SHA, source hash, mini dataset, and
fresh output before executing exactly
`fl_v3/tests/test_build_gt_database.py`. Pass requires at least one collected test,
zero failures, zero errors, zero skips, and specifically both hostile variants:

- derived `gt_boxes` changed while raw canonical inputs/meta/sidecar remain
  logically consistent;
- derived `lidar_sweeps[*].sweep2keylidar` changed under the same conditions.

Both must be rejected by physical pickle SHA mismatch before blob-store opening or
point cropping. The job records Python/platform and installed NumPy,
nuscenes-devkit, pyquaternion, Pillow, pytest, and Torch versions, emits JUnit/log/
source/identity artifacts, generates checksums, and runs `sha256sum -c`. Stop on
any SHA/source/output/data mismatch, any failure/error/skip, exception, or walltime.
No retry or downstream action follows automatically.

### Execution record — Job 335280

- Submitted exactly once under the bound command. Job `335280`
  (`flv3_s07a_provenance`) completed `0:0` on `n430` (`aarch64`) in `00:01:16`.
  It used one node, one GH200 and eight CPUs; `Restarts=0`, with no retry,
  requeue, resubmit, or follow-on. Batch resources were `MaxRSS=540M`,
  `MaxVMSize=6476352K`, and `TotalCPU=00:08.591`.
- Exact execution identity matched implementation `44cefd06...`, focused source
  hash `2710655b...`, declared mini root, aarch64 Python `3.11.15`, and the
  approved output. The clean-detached SHA/source/output guards passed before test
  execution.
- Pytest/JUnit recorded seven tests, zero failures/errors/skips in `1.52s`;
  both `gt_boxes` and `sweep2keylidar` hostile mutation cases were present and
  passed.
- In-job `sha256sum -c` passed for all output artifacts. Independent post-job
  checking passed all 25 source entries, reproduced the exact list/aggregate
  identities, and matched scheduler/log/artifact records. Complete hashes are in
  `RESULTS.md`.
- Section C approval is consumed. No rerun is authorized. Section B remains
  `PENDING_OWNER_APPROVAL_DO_NOT_SUBMIT` and was not submitted.

---

## D. S07-B bounded integrated GH200 validation — EXECUTED ONCE / FAILED / APPROVAL CONSUMED

### Approval state and immutable scope

- **Status:**
  `EXECUTED_ONCE_JOB_348557_FAILED_TIMEOUT_APPROVAL_CONSUMED_NO_RETRY`.
- Canonical approval is O-052 at exact Orchestra commit
  `e71274b1a169c1af92fe638608785a6e479d2b3a`. S00 independently audited the
  complete tuple below under the owner's delegated S07-B validation authority.
  Canonical O-053 at exact Orchestra commit
  `e8ee6461ff543b258ebaf588ff36ca5591277909` records that the exact command was
  submitted once as Slurm Job `348557`; O-052 was consumed immediately. This is
  not generic O-009 authority. There is no retry, resubmission, rerun, requeue,
  automatic follow-on, replacement or spare-job authority.
- Independent S07-B-R8 code-level review is **PASS** at
  `8a144ddaa624f3fd0605c7464eb30c1dcf6a51d9`; exact REVIEW blob
  `384a4a531f7967f25c75fc1282e1a7767bd4f97c`, size 145,973 bytes, SHA-256
  `bdb4093a526efa22fc3f32bf99e97c5f6264b03e95b5985ee35eacc795f5876f`.
- Immutable executable/archive source commit (`L`):
  `05b733997968b8217e1fc6dd27c3a4add34f6c98`.
- Launcher path and exact SHA-256:
  `fl_v3/scripts/run_s07_b_runtime_tests.sh` /
  `1b1c45d33b113d0c7d649e51b2ddf98a2d7822eab38d708d4bb0e223b8c334c0`.
- Exact C-locale source file count: 123.
- Exact source-list SHA-256:
  `be3b9157e213b942094d290d403306aa714e82157e36ba92847e32cfef71419a`.
- Exact aggregate source-state SHA-256 (SHA-256 of the ordered per-file
  `sha256sum` records):
  `d8c6cc0e20ed0c8ded5a4e13dd3ae52f32a62ebbcfafd2f9cbcd469fc5b87acd`.
- Both hashes were reproduced independently from a `git archive` extraction of
  exact `L` and directly from the same ordered immutable Git blobs.

### O-052 exact approval and O-053 consumed submission record

O-052 at canonical commit `e71274b1a169c1af92fe638608785a6e479d2b3a`
approves only this immutable tuple:

- executable/archive commit `L`:
  `05b733997968b8217e1fc6dd27c3a4add34f6c98`;
- launcher SHA-256:
  `1b1c45d33b113d0c7d649e51b2ddf98a2d7822eab38d708d4bb0e223b8c334c0`;
- exact 123-file source-list SHA-256:
  `be3b9157e213b942094d290d403306aa714e82157e36ba92847e32cfef71419a`;
- exact aggregate source-state SHA-256:
  `d8c6cc0e20ed0c8ded5a4e13dd3ae52f32a62ebbcfafd2f9cbcd469fc5b87acd`;
- exactly the 25 named test files enumerated below, with no whole-tree or extra
  collection;
- literal mini input
  `/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/fl_weather_project/data/nuscenes_mini`;
- fresh output
  `/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s07b_integrated_05b733997968`;
- fresh immutable snapshot
  `/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/execution_snapshots/s07b_integrated_05b733997968`;
- one job, one node, one task, one GH200, eight CPUs, 64 GiB and at most 45
  minutes; and
- the exact preflight plus `sbatch --no-requeue` command and exact `--export`
  values recorded in the command block below.

That exact approval was consumed by one submission attempt as Job `348557` under
O-053 at canonical commit `e8ee6461ff543b258ebaf588ff36ca5591277909`.
Scheduler SubmitTime was exactly `2026-07-12T12:33:41+02:00` and StartTime was
exactly `2026-07-12T12:33:42+02:00` on Arrhenius node `n30` (Europe/Stockholm,
CEST). The submitted command tuple is byte-for-byte the preflight, resources,
exports and launcher command recorded below. The single attempt consumed approval
regardless of whether it passes, fails, times out or is cancelled. No retry,
follow-on or changed tuple is authorized.

Exact config file hashes inside `L` are:

| Config | SHA-256 |
|---|---|
| `s07_b_c_str8.json` | `d2eaa46c800ebea5927359398acd88b38d90219c2f1f3841a4b1897ed05f8cc6` |
| `s07_b_f_cbgs.json` | `bd8c57e84b34f835f3eaafe71f259a0c4131748bb27a62edf83bcd7f44bb54f0` |
| `s07_b_f_u.json` | `df7f36fe28e0d0c6c8275b293318cf7fae2e3c71fe3c60b7a7b81c26af69fa2e` |
| `s07_b_l_p020.json` | `625242234a03314010860e6026b0fbb88b774a9aeec12c7f7fe870203da07421` |
| `s07_b_l_s075.json` | `1658cd5ec0e9c1b8945646d2e23a8db4419d16c2f644ca5a99b94c3477dcce1d` |

They remain fail-closed, non-runnable templates. Hashing them here does not fill
their unresolved cache/build/seed/budget fields or authorize a training run.

The 123-file source state contains every tracked Python file under
`fl_v3/src/fl_v3/`, the launcher and Arrhenius bootstrap, centralized trainer,
mini-matrix, T4 readiness, T5 attack and T5 mini-smoke scripts, `tests/conftest.py`,
all 25 selected test files below, all five exact `s07_b_*.json` templates,
`pyproject.toml`, `requirements.txt`, and `requirements.lock.txt`. List generation
uses `LC_ALL=C sort -u`; the job writes both the list and per-file hashes and
reproduces both expected identities before environment import or tests.

### Exact resources, immutable snapshot, data and outputs

- One job, one node, one task, one GH200, eight CPUs, 64 GiB RAM, walltime at most
  `00:45:00`; maximum 0.75 allocated GPU-hours. No array, DDP, requeue, retry,
  automatic follow-on or spare-GPU job.
- The launcher creates a fresh snapshot only at
  `/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/execution_snapshots/s07b_integrated_05b733997968`.
  It extracts `git archive 05b7339...`, validates the exact list/hash tuple, and
  recursively removes write permission before imports/tests. A collision stops
  the job; the snapshot is never reused.
- Existing mini input only:
  `/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/fl_weather_project/data/nuscenes_mini`.
  The launcher requires this literal path and equal realpath, exports only its
  mini dataroot overrides, and clears shared-trainval and ZIP-manifest overrides.
  Synthetic ZIP/cache/config/test data may exist only under the job's output-local
  pytest temp directory. It does not load the shared trainval module, scan/build a
  full cache, or run trainval metrics.
- Fresh output root only:
  `/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s07b_integrated_05b733997968`.
  Any pre-existing output or snapshot is a hard stop.
- Slurm logs:
  `/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/logs/s07b_integrated_%j.{out,err}`.
- Environment activation is exactly `source fl_v3/scripts/arrhenius_env.sh`,
  `arrhenius_load_modules build`, `arrhenius_activate_env`. The job requires
  aarch64, one node/task and eight allocated CPUs.
- Third-party pytest plugin autoload is disabled, `PYTEST_ADDOPTS` is cleared,
  cacheprovider is disabled, `WORLD_SIZE=1`, and the pytest subprocess has a
  42-minute timeout inside the 45-minute allocation.

### Exact selected tests

The launcher passes these 25 files explicitly; it does not run the whole test
tree or collect files outside this list:

1. `fl_v3/tests/test_s02_p0_correctness.py`
2. `fl_v3/tests/test_s02_gpu_forward_backward.py`
3. `fl_v3/tests/test_s03_camera_contract.py`
4. `fl_v3/tests/test_s04_second_contract.py`
5. `fl_v3/tests/test_s04_second_smoke.py`
6. `fl_v3/tests/test_s04_fp16_eval_dispatch.py`
7. `fl_v3/tests/test_s05_centerhead_decode.py`
8. `fl_v3/tests/test_s05_eval_roundtrip.py`
9. `fl_v3/tests/test_s05_nms.py`
10. `fl_v3/tests/test_s06_checkpoint_resume.py`
11. `fl_v3/tests/test_s06_loader_eval.py`
12. `fl_v3/tests/test_s06_model_modes.py`
13. `fl_v3/tests/test_s06_resolved_config.py`
14. `fl_v3/tests/test_s06_training_runtime.py`
15. `fl_v3/tests/test_s07_b_data_lifecycle.py`
16. `fl_v3/tests/test_s07_b_integration.py`
17. `fl_v3/tests/test_sparse_voxel_encoder.py`
18. `fl_v3/tests/test_lidar_backbone.py`
19. `fl_v3/tests/test_head_capacity.py`
20. `fl_v3/tests/test_eval_box_to_global.py`
21. `fl_v3/tests/test_eval_detection_eval.py`
22. `fl_v3/tests/test_eval_provenance.py`
23. `fl_v3/tests/test_model_task.py`
24. `fl_v3/tests/test_profiling_neutral.py`
25. `fl_v3/tests/test_nuscenes_zip_dataset.py`

Static AST inventory finds 177 test functions and estimates 249 collected cases
after the visible parametrizations; the future JUnit count is authoritative.
Bounded forward/backward coverage is concentrated in S02 GPU forward/backward,
S03 camera contract, S04 SECOND smoke/fp16 dispatch, S07 integration gradient
reachability, sparse voxel encoder and model-task cases. Bounded optimizer-state or
optimizer-step validation occurs only inside the focused S06 training/checkpoint
unit cases and profiling-neutrality unit case. S04 smoke explicitly performs no
optimizer update. These are small correctness fixtures, not a training campaign:
the launcher invokes no trainer, requests no 100/1000-step run, and performs zero
optimizer steps as an experimental training campaign.

### Exact command submitted once as Job 348557

```bash
test "$(git hash-object fl_v3/scripts/run_s07_b_runtime_tests.sh)" = "1e182ebc1fe883ad59702bfeb1b3db110bbf54c1" && \
git cat-file -e 05b733997968b8217e1fc6dd27c3a4add34f6c98^{commit} && \
test ! -e /nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s07b_integrated_05b733997968 && \
test ! -e /nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/execution_snapshots/s07b_integrated_05b733997968 && \
test -z "$(squeue -u "$USER" -h -o '%i %j' | awk '$2 == "flv3_s07b_integrated" {print}')" && \
sbatch --nodes=1 --ntasks=1 --gpus-per-node=nvidia_gh200_120gb:1 \
  --cpus-per-task=8 --mem=64G --time=00:45:00 --no-requeue \
  --export=ALL,EXPECTED_S07B_EXECUTABLE_SHA=05b733997968b8217e1fc6dd27c3a4add34f6c98,EXPECTED_S07B_LAUNCHER_SHA256=1b1c45d33b113d0c7d649e51b2ddf98a2d7822eab38d708d4bb0e223b8c334c0,EXPECTED_S07B_SOURCE_SHA256=d8c6cc0e20ed0c8ded5a4e13dd3ae52f32a62ebbcfafd2f9cbcd469fc5b87acd,EXPECTED_S07B_SOURCE_LIST_SHA256=be3b9157e213b942094d290d403306aa714e82157e36ba92847e32cfef71419a,S07B_APPROVAL_SCOPE=owner-delegated-s07b-integrated-validation,S07B_MINI_DATAROOT=/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/fl_weather_project/data/nuscenes_mini,S07B_OUTPUT_ROOT=/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s07b_integrated_05b733997968 \
  fl_v3/scripts/run_s07_b_runtime_tests.sh
```

This exact command was submitted once as Job `348557` at
`2026-07-12T12:33:41+02:00`; O-052 is consumed. It must not be submitted again.
No alternate invocation, retry, replacement or follow-on is approved.

### Executed result — FAILED / approval consumed

Canonical O-054 at exact Orchestra commit
`91526456ee4d4c9d63835868b055b537d0d6655c` records the terminal result. Job
`348557` was `FAILED 1:0`: Submit/Start/End
`2026-07-12T12:33:41+02:00` / `2026-07-12T12:33:42+02:00` /
`2026-07-12T13:18:02+02:00`, elapsed `00:44:20`, node `n30`, one GH200/eight
CPUs/64 GiB, batch `MaxRSS=10573756K`, `TotalCPU=01:35.363`, `Restarts=0`.

Exact Git/launcher/list/state/mini/runtime identities passed. Pytest then emitted
exactly three `F` and four `E` progress glyphs, hung without a summary and was
terminated by the internal timeout with exit `124`. Required JUnit,
`pytest_junit_counts.json` and final `sha256sums.txt` were not produced; therefore
the zero-failure/error/skip and checksum acceptance gate is **FAIL**. The latest
basetemp symlink/directory points to the truncated name for
`test_repeated_persistent_multiworker_reads_are_deterministic`, which is only a
high-confidence hang-location inference, not a formal attribution without JUnit
or summary. Exact raw-artifact hashes and interpretation limits are preserved in
`RESULTS.md`.

O-052 is consumed. No retry, requeue, replacement, diagnostic execution or
follow-on is approved by this record.

### Runtime identity, artifacts, acceptance and stop conditions

`execution_identity.json` records exact Git/source/list/scope identity; Slurm job,
nodes/tasks/CPUs/memory; host/machine/platform; Python executable,
implementation/version; Torch version/Git/CUDA/build config; visible CUDA device,
GH200 name/capability/cuDNN; spconv/cumm module version/path; installed NumPy,
SciPy, pytest, Torch, torchvision, spconv, cumm, nuscenes-devkit, pyquaternion and
Pillow versions; mini/output/snapshot paths; plugin state; and the 25 test files.

Required artifacts are `execution_identity.json`, `runtime_source_files.txt`,
`runtime_source_sha256s.txt`, `config_sha256s.txt`, `selected_test_files.txt`,
`pytest.log`, `pytest.junit.xml`, `pytest_junit_counts.json`, `pytest.exitcode`, and
`sha256sums.txt`. The launcher generates checksums for every preceding artifact
and runs `sha256sum -c` before accepting the result.

PASS requires exact `L`, source-list/source-state/config identities; fresh output
and snapshot; exact mini-only inputs; aarch64/one-GH200 runtime identity; pytest and
tee exit zero; a positive JUnit test count with **zero failures, zero errors and
zero skips**; complete artifacts; and successful in-job checksum verification.
Any identity/hash/path/resource/output/snapshot mismatch, package/CUDA failure,
pytest failure/error/skip, timeout/walltime, missing artifact or checksum failure
stops the single job. There is no retry or next cell.

Allowed interpretation after an independently audited PASS: bounded engineering
evidence that the integrated S02-S07 selected contracts execute together on one
Arrhenius GH200 with real mini plus synthetic temporary fixtures. Forbidden:
production/full-trainval readiness, cache readiness, throughput/memory claims,
mAP/NDS or other scientific metrics, model-quality/fusion-gain, 100/1000-step
training, FL, attack/defense, generalization, seed/matrix or publication evidence.
Job 348557 did not meet the conditional PASS gate; none of the allowed PASS
interpretation is available from this execution.

---

## E. S07-B failure/hang diagnostic attribution — EXECUTED ONCE / DIAGNOSTIC COMPLETE / SUITE FAIL

### Approval state and immutable identity

- **Status:**
  `EXECUTED_ONCE_JOB_348818_DIAGNOSTIC_COMPLETE_SUITE_FAIL_APPROVAL_CONSUMED_NO_RETRY`.
- Canonical O-055 at exact Orchestra commit
  `d56e01d3b80a7dae41f90211c0be9ff565861b85` authorized preparation only.
  Canonical O-056 at exact Orchestra commit
  `07ec16f37cbe0816be6ce102350036e8c7511e1e` records S00's audit and approves
  exactly one submission of the immutable tuple below under the owner's delegated
  S07-B diagnostic scheduling authority. Canonical O-057 at exact Orchestra
  commit `da9dbb4f643f9ab92f6e979e605e9ef24722963a` records that the exact command
  was submitted once as Job `348818` and O-056 was consumed immediately.
- This is a distinct diagnostic proposal after the durable Job 348557 negative
  record at `d7888a9fef615c83c8d36161bfa6d581a3dc4f0f`. It is not an O-052 retry and
  cannot change or overwrite the old output.
- Immutable diagnostic launcher/archive commit (`L`):
  `fd142dc1c247ed527dbf5ddb823576c817dc415a`.
- Launcher path, Git blob and SHA-256:
  `fl_v3/scripts/run_s07_b_diagnostic_tests.sh` /
  `e41e97d31ff0a4e5555a548a63ac04d656565538` /
  `d8d7686eb727d4973591cf20186615f6bf2f3bc71ba020dec815c9b6d2d0dc1b`.
- Exact C-locale source file count: 124.
- Exact source-list SHA-256:
  `40c364201bda63386be614fca3710f62111e6964f9b7fdc1beffef69cb5f05d8`.
- Exact aggregate source-state SHA-256:
  `56ddfdc66045548899cdde1ad08f7e394c300a8fc27a6c0aaf6551a8178533b2`.
- Exact parent runtime launcher SHA-256, checked separately in-job:
  `1b1c45d33b113d0c7d649e51b2ddf98a2d7822eab38d708d4bb0e223b8c334c0`.

### O-056 exact one-time approval record

O-056 at canonical commit `07ec16f37cbe0816be6ce102350036e8c7511e1e`
approves only this exact tuple:

- executable/archive `fd142dc1c247ed527dbf5ddb823576c817dc415a`;
- diagnostic launcher SHA-256
  `d8d7686eb727d4973591cf20186615f6bf2f3bc71ba020dec815c9b6d2d0dc1b`;
- exact 124-file list/state SHA-256
  `40c364201bda63386be614fca3710f62111e6964f9b7fdc1beffef69cb5f05d8` /
  `56ddfdc66045548899cdde1ad08f7e394c300a8fc27a6c0aaf6551a8178533b2`;
- the exact ordered 25 test files below, isolated 120-second verbose attempts and
  the 600-second verbose combined probe;
- literal mini root
  `/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/fl_weather_project/data/nuscenes_mini`;
- fresh snapshot/output
  `/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/execution_snapshots/s07b_diagnostic_fd142dc1c247` /
  `/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s07b_diagnostic_fd142dc1c247`;
- one job/node/task/GH200, eight CPUs, 64 GiB and 30 minutes; and
- the exact preflight, `sbatch --no-requeue`, exports and launcher command in the
  command block below.

This authority was consumed by Job `348818`, submitted exactly at
`2026-07-12T13:40:45+02:00` and started at `2026-07-12T13:40:46+02:00` on
Arrhenius node `n412` (Europe/Stockholm, CEST). The submitted tuple is exactly the
preflight/resources/exports/launcher command below. The one attempt consumed
approval regardless of outcome.
Any Git/hash/test/order/timeout/data/path/resource/environment/command/summary/
acceptance/stop-rule drift invalidates approval. There is no retry, requeue,
replacement, alternate invocation, automatic follow-on or spare job. A completed
diagnostic harness is not suite PASS: `diagnostic_complete` never substitutes for
the separately computed `suite_pass` result.

The 124-file state is the exact prior 123-file runtime set plus only the new
diagnostic launcher. It contains all tracked `fl_v3/src/fl_v3/**/*.py`, both
S07-B launchers, the Arrhenius bootstrap, centralized/mini/T4/T5 scripts,
`tests/conftest.py`, the 25 tests below, five S07-B configs, pyproject and both
requirements manifests. Both list/state hashes were independently reproduced
from `git archive L` and ordered immutable Git blobs.

Exact config SHA-256 identities remain:

| Config | SHA-256 |
|---|---|
| `s07_b_c_str8.json` | `d2eaa46c800ebea5927359398acd88b38d90219c2f1f3841a4b1897ed05f8cc6` |
| `s07_b_f_cbgs.json` | `bd8c57e84b34f835f3eaafe71f259a0c4131748bb27a62edf83bcd7f44bb54f0` |
| `s07_b_f_u.json` | `df7f36fe28e0d0c6c8275b293318cf7fae2e3c71fe3c60b7a7b81c26af69fa2e` |
| `s07_b_l_p020.json` | `625242234a03314010860e6026b0fbb88b774a9aeec12c7f7fe870203da07421` |
| `s07_b_l_s075.json` | `1658cd5ec0e9c1b8945646d2e23a8db4419d16c2f644ca5a99b94c3477dcce1d` |

### Resources, data, snapshot and outputs

- One job, one node, one task, one NVIDIA GH200 120GB, eight CPUs, 64 GiB and
  at most `00:30:00`; no array, DDP, requeue, retry, automatic follow-on or
  spare-GPU job.
- Fresh immutable archive snapshot only:
  `/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/execution_snapshots/s07b_diagnostic_fd142dc1c247`.
- Fresh output only:
  `/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s07b_diagnostic_fd142dc1c247`.
- Slurm logs:
  `/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/logs/s07b_diagnostic_%j.{out,err}`.
- Existing literal mini input only:
  `/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/fl_weather_project/data/nuscenes_mini`.
  The launcher requires equal literal path and realpath, clears
  `NUSCENES_DATA_DIR` and both ZIP-manifest overrides, and confines synthetic
  pytest scratch state to the new output-local `tmp/`.
- The Job 348557 output
  `/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s07b_integrated_05b733997968`
  is read-only negative evidence and is never reused or modified by this request.

The launcher archives exact `L`, validates launcher/runtime/config/list/state
identities, removes snapshot write permission, activates the persistent Arrhenius
environment, requires one visible CUDA device and records full Python/Torch/CUDA/
GH200/spconv/cumm/dependency identity. The identity embeds Job 348557's exact
negative evidence and `d7888a9...` RESULTS parent.

### Exact tests and two diagnostic stages

The exact ordered 25-file list is unchanged:

1. `fl_v3/tests/test_s02_p0_correctness.py`
2. `fl_v3/tests/test_s02_gpu_forward_backward.py`
3. `fl_v3/tests/test_s03_camera_contract.py`
4. `fl_v3/tests/test_s04_second_contract.py`
5. `fl_v3/tests/test_s04_second_smoke.py`
6. `fl_v3/tests/test_s04_fp16_eval_dispatch.py`
7. `fl_v3/tests/test_s05_centerhead_decode.py`
8. `fl_v3/tests/test_s05_eval_roundtrip.py`
9. `fl_v3/tests/test_s05_nms.py`
10. `fl_v3/tests/test_s06_checkpoint_resume.py`
11. `fl_v3/tests/test_s06_loader_eval.py`
12. `fl_v3/tests/test_s06_model_modes.py`
13. `fl_v3/tests/test_s06_resolved_config.py`
14. `fl_v3/tests/test_s06_training_runtime.py`
15. `fl_v3/tests/test_s07_b_data_lifecycle.py`
16. `fl_v3/tests/test_s07_b_integration.py`
17. `fl_v3/tests/test_sparse_voxel_encoder.py`
18. `fl_v3/tests/test_lidar_backbone.py`
19. `fl_v3/tests/test_head_capacity.py`
20. `fl_v3/tests/test_eval_box_to_global.py`
21. `fl_v3/tests/test_eval_detection_eval.py`
22. `fl_v3/tests/test_eval_provenance.py`
23. `fl_v3/tests/test_model_task.py`
24. `fl_v3/tests/test_profiling_neutral.py`
25. `fl_v3/tests/test_nuscenes_zip_dataset.py`

Stage 1 runs each file, in that order, in a distinct subprocess with a unique
output directory, basetemp, verbose log, optional JUnit and explicit pytest/tee
exit files. The exact command form is:

```bash
timeout --signal=TERM --kill-after=15s 120s \
  python -m pytest -vv -ra --tb=long -p no:cacheprovider \
  -o faulthandler_timeout=60 \
  --basetemp="UNIQUE_BASETEMP" --junitxml="UNIQUE_JUNIT" "ONE_TEST_FILE"
```

Every isolated failure, error or timeout is recorded and then execution continues
to the next file. A nonzero pytest exit is diagnostic evidence, not a harness
failure.

Stage 2 performs one separate verbose combined probe over the same ordered list:

```bash
timeout --signal=TERM --kill-after=15s 600s \
  python -m pytest -vv -ra --tb=short -p no:cacheprovider \
  -o faulthandler_timeout=60 \
  --basetemp="COMBINED_BASETEMP" --junitxml="COMBINED_JUNIT" \
  "THE_SAME_25_TEST_FILES_IN_ORDER"
```

Its independent log/JUnit/exit files retain verbose current-test and prior
FAILED/ERROR names even if the probe times out. The 30-minute Slurm walltime is
absolute: if the 25 isolated attempts plus combined probe do not finish, the
diagnostic harness is incomplete and fails; no retry follows.

### Summary, artifacts and diagnostic acceptance

`diagnostic_summary.json` lists all 25 isolated attempts with exact test path,
pytest and tee exits, log/JUnit presence, JUnit counts or parse error, plus the
combined probe's equivalent status. It contains separate booleans:

- `diagnostic_complete`: identities passed, all 25 isolated subprocesses and the
  combined subprocess were attempted, logs/exits were captured and both tee
  exits were zero;
- `suite_pass`: every isolated and combined pytest exit was zero and every JUnit
  existed, parsed, had a positive test count and zero failures/errors/skips.

`diagnostic_complete` does **not** imply `suite_pass`. Expected diagnostic
failures/timeouts may yield a Slurm `COMPLETED 0:0` only when the diagnostic
harness and artifacts complete; the summary must still report
`suite_pass=false`. Harness nonzero is reserved for identity/setup failure,
failure to attempt the full required probes, summary/artifact-capture failure or
checksum failure—not for captured pytest failures/timeouts themselves.

Formal output artifacts include execution/source/config/test identities,
`isolated_attempts.tsv`, every existing isolated and combined log/JUnit/exit
file, and `diagnostic_summary.json`. After both stages, the launcher C-locale
sorts every preceding existing formal output file outside scratch `tmp/` into
`sha256sums.txt` and requires `sha256sum -c` success; the checksum manifest
cannot recursively contain itself. Scratch fixtures are not formal evidence.
Scheduler stdout/stderr remain separately preserved for S00 to hash after the
job closes.

### Exact command submitted once as Job 348818

```bash
test "$(git hash-object fl_v3/scripts/run_s07_b_diagnostic_tests.sh)" = "e41e97d31ff0a4e5555a548a63ac04d656565538" && \
git cat-file -e fd142dc1c247ed527dbf5ddb823576c817dc415a^{commit} && \
test ! -e /nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s07b_diagnostic_fd142dc1c247 && \
test ! -e /nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/execution_snapshots/s07b_diagnostic_fd142dc1c247 && \
test -z "$(squeue -u "$USER" -h -o '%i %j' | awk '$2 == "flv3_s07b_diagnostic" {print}')" && \
sbatch --nodes=1 --ntasks=1 --gpus-per-node=nvidia_gh200_120gb:1 \
  --cpus-per-task=8 --mem=64G --time=00:30:00 --no-requeue \
  --export=ALL,EXPECTED_S07B_DIAGNOSTIC_SHA=fd142dc1c247ed527dbf5ddb823576c817dc415a,EXPECTED_S07B_DIAGNOSTIC_LAUNCHER_SHA256=d8d7686eb727d4973591cf20186615f6bf2f3bc71ba020dec815c9b6d2d0dc1b,EXPECTED_S07B_DIAGNOSTIC_SOURCE_SHA256=56ddfdc66045548899cdde1ad08f7e394c300a8fc27a6c0aaf6551a8178533b2,EXPECTED_S07B_DIAGNOSTIC_SOURCE_LIST_SHA256=40c364201bda63386be614fca3710f62111e6964f9b7fdc1beffef69cb5f05d8,S07B_DIAGNOSTIC_APPROVAL_SCOPE=s07b-diagnostic-attribution-only,S07B_DIAGNOSTIC_MINI_DATAROOT=/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/fl_weather_project/data/nuscenes_mini,S07B_DIAGNOSTIC_OUTPUT_ROOT=/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s07b_diagnostic_fd142dc1c247 \
  fl_v3/scripts/run_s07_b_diagnostic_tests.sh
```

This exact command was submitted once as Job `348818` at
`2026-07-12T13:40:45+02:00`; O-056 is consumed. It must not be altered, repeated
or followed automatically. Any Git,
hash, file list/order, test command/timeout, data, resource, path, environment,
summary, acceptance or stop-rule drift invalidates approval.

### Executed result — harness COMPLETE, suite FAIL

Canonical O-058 at `348f29c3c68243ae6010ea0d017e16850081c43c`
records Job `348818` as scheduler `COMPLETED 0:0` at
`2026-07-12T13:57:16+02:00` after `00:16:30` on `n412` with one GH200/eight
CPUs/64 GiB, `MaxRSS=10539927K`, `TotalCPU=05:05.306`, `Restarts=0`.

This is harness completion only: `diagnostic_complete=true` and
`suite_pass=false`. Isolated JUnit aggregate is 251 tests / 3 failures / 94
errors / 0 skips. Ninety errors are the missing `$JOB_TMP/isolated` parent;
four are read-only `./fl_outputs` PermissionErrors; three exact failures and the
combined fork DataLoader hang are recorded in `RESULTS.md`. Combined pytest exit
is `124`, JUnit is absent, and faulthandler formally locates the queue wait at
`test_repeated_persistent_multiworker_reads_are_deterministic[fork]`.

All 110 sorted artifact checksum records verified. O-056 remains consumed; no
retry, replacement, remediation execution or follow-on is approved.

### Interpretation limits

The only purpose is to recover exact failure/error names, long per-file
tracebacks and a bounded high-confidence hang location for Job 348557 follow-up.
It is engineering diagnostics only: no production/full-data/cache readiness,
throughput/memory, mAP/NDS, model-quality/fusion-gain, 100/1000-step training,
FL, attack/defense, generalization, matrix/seed or publication claim is allowed.
No trainer, full trainval/cache scan, profile, metric, DDP or scientific cell is
invoked. Preparation authorizes no compute, retry, merge, push or upload.

---

## F. S07-B two-snapshot dummy checksum attribution — EXECUTED ONCE / JOB 349653 ACTIVE / APPROVAL CONSUMED

### Approval state and exact purpose

- **Status:** `EXECUTED_ONCE_JOB_349653_ACTIVE_APPROVAL_CONSUMED_NO_RETRY`.
- Canonical O-060 is
  `34ee1f9672df5c907881b5c6335b6be6e204c156`. It authorizes preparation only;
  it did not authorize `sbatch`, `srun`, a retry, or any other compute.
- Canonical O-061 at exact Orchestra commit
  `7ee0b040787bef1c26c1c3d15b0983824d42770e` independently audits Section F
  and approves exactly one submission of the unchanged exact tuple below. The
  approval is bound to executable `L`, both snapshot SHAs and source identities,
  launcher/bootstrap/dependency identities, exact workload/repetitions, command,
  resources, fresh paths, artifacts, classification and stop conditions already
  frozen in this section. Any drift voids O-061 before submission. The single
  attempt consumes approval regardless of scheduler or harness outcome; no
  retry, requeue, alternate invocation, replacement, automatic resubmission or
  follow-on is approved.
- Canonical O-062 at exact Orchestra commit
  `85798992da9837b86b731eb5b2b11ff71c7aa674` records that the unchanged exact
  Section F command was submitted once as Slurm Job `349653` at
  `2026-07-12T14:23:08+02:00`, started at `2026-07-12T14:23:09+02:00`, and is
  active on node `n530`. This single submission consumed O-061 immediately.
  Terminal scheduler state and runtime artifacts have not yet been read or
  interpreted in this record. No retry, requeue, replacement, alternate
  invocation, automatic resubmission or follow-on is authorized.
- This is a bounded engineering attribution of the dummy-regression checksum,
  not a retry of Jobs 348557/348818 and not a scientific matrix. It compares
  exactly the pre-S06 snapshot
  `968d81583c87ba76b7dbbb722760f8eb8eb6cd39` with the current remediated
  snapshot `c69befe5e8dd6397059c4d3fe1cbf906a9646836` in the same GH200 job and
  persistent environment. Each snapshot runs twice in a fresh independent
  Python process.
- No outcome automatically authorizes changing the committed golden, training
  loop, production code, tests, or scientific interpretation. S00 must inspect
  the exact artifacts and decide the next scoped action.

### Immutable executable, source, and environment identity

- Exact detached executable commit `L`:
  `a9d657aebfb0f64d271fa74e312d6054eca57e1d`.
- Launcher Git blob:
  `295610fd422f3b371b8fd85e54785919903dc332`.
- Launcher SHA-256:
  `bbc1293a42034540327402a5df6c1f172b76afacca7906b4f0b71f5290b5968a`.
- Environment bootstrap `fl_v3/scripts/arrhenius_env.sh` SHA-256:
  `f57befbb5082aaf4d4bb186958a88420ea873e0fdee5c65da1091b73f566c2bf`.
- Pre-S06 snapshot source identity: 78 files; C-locale source-list SHA-256
  `0ec5e43e98ee6b98c949d6c3187c4484fb47842443fd3ffe54fcb61e4d777ae0`;
  ordered source-state SHA-256
  `dc2144cc522d20035eeff81269e45973312c10f908ddc6808bc3c2611b38c93d`.
- Current snapshot source identity: 85 files; C-locale source-list SHA-256
  `104a647441ce712e83c20d32372944e48777913cf746ae19ee124894ca927e41`;
  ordered source-state SHA-256
  `0f2995fca7d323421e46326493b2f9cc5d0032ca3d794d422779fc10c626ee32`.
- Each source set contains every tracked Python file under
  `fl_v3/src/fl_v3/` plus `pyproject.toml`, `requirements.txt`, and
  `requirements.lock.txt`. The launcher separately binds itself and the
  environment bootstrap.
- The exact validated dependency identity is locked, not merely logged:
  CPython `3.11.15`, NumPy `1.26.4`, SciPy `1.13.1`, Torch
  `2.11.0+cu128`, torchvision `0.26.0+cu128`, spconv `2.3.8`, cumm
  `0.7.13`, nuscenes-devkit `1.1.11`, pyquaternion `0.9.9`, and Pillow
  `12.2.0`. The launcher stops on any mismatch and records the actual identity.

### Exact workload, resources, snapshots, and outputs

- One node, one task, one GH200, four CPUs, 32 GiB, walltime `00:10:00`;
  at most one sixth GPU-hour. One job only, `--no-requeue`; no array, DDP,
  retry, automatic resubmission, or follow-on.
- The GPU allocation is only for the validated aarch64 environment. The exact
  dummy workload runs on `device=cpu`; it opens no nuScenes data, cache, ZIP,
  trainval, model metric, trainer campaign, profile, or scientific cell.
- The two Git commits are independently archived into fresh immutable snapshots:
  `/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/execution_snapshots/s07b_dummy_attr_a9d657aebfb0_pre`
  and
  `/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/execution_snapshots/s07b_dummy_attr_a9d657aebfb0_current`.
- Exact fresh output root:
  `/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s07b_dummy_attr_a9d657aebfb0`.
- Scheduler logs:
  `/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/logs/s07b_dummy_attr_%j.{out,err}`.
- For each snapshot and repetition 1/2, a fresh Python subprocess uses exactly:
  `task-type=dummy_regression`, `seed=42`, `device=cpu`, `num-clients=4`,
  `num-local-epochs=1`, `batch-size=8`, `learning-rate=0.01`,
  `weight-decay=0.0`, `num-workers=0`, `loss=mse`, `defense=none`, and
  `server_round=1`.

### Exact command — submitted once as Job 349653; O-061 consumed

```bash
test -z "$(git branch --show-current)" && \
test "$(git rev-parse HEAD)" = "a9d657aebfb0f64d271fa74e312d6054eca57e1d" && \
test -z "$(git status --short)" && \
test "$(git hash-object fl_v3/scripts/run_s07_b_dummy_attribution.sh)" = "295610fd422f3b371b8fd85e54785919903dc332" && \
test ! -e /nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s07b_dummy_attr_a9d657aebfb0 && \
test ! -e /nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/execution_snapshots/s07b_dummy_attr_a9d657aebfb0_pre && \
test ! -e /nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/execution_snapshots/s07b_dummy_attr_a9d657aebfb0_current && \
test -z "$(squeue -u "$USER" -h -o '%i %j' | awk '$2 == "flv3_s07b_dummy_attr" {print}')" && \
sbatch --nodes=1 --ntasks=1 --gpus-per-node=nvidia_gh200_120gb:1 \
  --cpus-per-task=4 --mem=32G --time=00:10:00 --no-requeue \
  --export=ALL,EXPECTED_S07B_ATTR_EXECUTABLE_SHA=a9d657aebfb0f64d271fa74e312d6054eca57e1d,EXPECTED_S07B_ATTR_LAUNCHER_SHA256=bbc1293a42034540327402a5df6c1f172b76afacca7906b4f0b71f5290b5968a,EXPECTED_S07B_ATTR_ENV_SHA256=f57befbb5082aaf4d4bb186958a88420ea873e0fdee5c65da1091b73f566c2bf,EXPECTED_S07B_ATTR_PRE_SOURCE_LIST_SHA256=0ec5e43e98ee6b98c949d6c3187c4484fb47842443fd3ffe54fcb61e4d777ae0,EXPECTED_S07B_ATTR_PRE_SOURCE_SHA256=dc2144cc522d20035eeff81269e45973312c10f908ddc6808bc3c2611b38c93d,EXPECTED_S07B_ATTR_CURRENT_SOURCE_LIST_SHA256=104a647441ce712e83c20d32372944e48777913cf746ae19ee124894ca927e41,EXPECTED_S07B_ATTR_CURRENT_SOURCE_SHA256=0f2995fca7d323421e46326493b2f9cc5d0032ca3d794d422779fc10c626ee32,S07B_ATTR_APPROVAL_SCOPE=s07b-dummy-two-snapshot-attribution-only,S07B_ATTR_OUTPUT_ROOT=/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s07b_dummy_attr_a9d657aebfb0 \
  fl_v3/scripts/run_s07_b_dummy_attribution.sh
```

Any change to executable/snapshot SHA, launcher/bootstrap/source/dependency
identity, workload, repetitions, command, resource, path, summary, or stop rule
invalidates a future approval. This request does not authorize an alternate
invocation.

### Required artifacts, classification, acceptance, and stop rules

The launcher records both source lists/states/identities, exact execution and
dependency identity, four per-process JSON results plus stdout/stderr/exit,
`attempts.tsv`, `attribution_summary.json`, and a C-locale-sorted
`sha256sums.txt`. It then requires `sha256sum -c` for every prior formal artifact.

The summary must classify exactly one of:

- `stable_equal_current`: both snapshots repeat stably and both equal Job
  348818's observed `4fa46307...` checksum;
- `pre_historical_current_new`: pre-S06 repeats equal historical `d2d819...`
  while current repeats equal `4fa46307...`;
- `unstable`: either snapshot differs across its two fresh subprocesses;
- `unexpected_stable_pair`: all other stable checksum pairs.

Harness completion requires exact identities, four successful independent
subprocesses, two attempts per snapshot, valid 64-hex checksums, one declared
classification, complete artifacts, and checksum verification. `unstable` or
`unexpected_stable_pair` is a preserved diagnostic outcome, not permission to
hide or rerun it. Stop on identity/dependency/output/snapshot collision,
archive/source mismatch, subprocess or artifact failure, malformed checksum,
checksum verification failure, or walltime. No retry or automatic code/golden
change follows any outcome.
