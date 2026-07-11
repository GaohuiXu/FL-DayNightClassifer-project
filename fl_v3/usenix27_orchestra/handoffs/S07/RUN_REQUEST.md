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
  `c8dd920cf3f8007c3b2ec03f48bcc3f83144ebbe`.
- Exact cache-launcher source-state SHA-256:
  `6a4ad312b41ff161aa07f7628176ab74f550768f8b15c335314c5d262cbec1c2`.
- The 23-file set was recomputed identically from the clean worktree and immutable
  Git blobs. It includes every tracked Python file under
  `fl_v3/src/fl_v3/data/nuscenes/`, package initializers,
  `fl_v3/src/fl_v3/data/partition.py`, `fl_v3/src/fl_v3/utils/runtime.py`, the
  cache builder/launcher/environment bootstrap, and dependency/config manifests.
- Eventual executor contract: after separate owner approval, the owner/Codex task
  UI provisions a fresh isolated worktree at
  `detached@c8dd920cf3f8007c3b2ec03f48bcc3f83144ebbe`. The executor verifies a
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
  `/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s07a_cache_t1v2_c8dd920cf3f8`.
  It was confirmed absent during this local remediation; this is a proposed path,
  not a created output.
- Logs:
  `/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/logs/s07a_cache_t1v2_%j.{out,err}`.

```bash
test -z "$(git branch --show-current)" && \
test "$(git rev-parse HEAD)" = "c8dd920cf3f8007c3b2ec03f48bcc3f83144ebbe" && \
test -z "$(git status --short)" && \
test ! -e /nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s07a_cache_t1v2_c8dd920cf3f8 && \
test -z "$(squeue -u "$USER" -h -o '%i %j' | awk '$2 ~ /flv3_s07a_cache_t1v2/ {print}')" && \
sbatch --time=00:30:00 --cpus-per-task=8 \
  --export=ALL,EXPECTED_S07A_SHA=c8dd920cf3f8007c3b2ec03f48bcc3f83144ebbe,EXPECTED_S07A_STATE_HASH=6a4ad312b41ff161aa07f7628176ab74f550768f8b15c335314c5d262cbec1c2,S07A_ACCEPTED_MANIFEST=/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s01_zip_full_gate_v2_1fe651700bd0/nuscenes_trainval_zip_manifest.sqlite,S07A_ACCEPTED_MANIFEST_HASH=023f72b4220bb0db587be00920308bf9074384740fe186d243be92f9a53119f6,S07A_ACCEPTED_MANIFEST_FILE_SHA256=228e2f5bab30007acb06eb61393d1fbacc88979490668ff800f8f7f9752a47fb,S07A_OUTPUT_ROOT=/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s07a_cache_t1v2_c8dd920cf3f8 \
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

## C. S07-A-R P1 focused provenance validation — PENDING OWNER APPROVAL

### Approval state and immutable scope

- **Status:** `PENDING_OWNER_APPROVAL_DO_NOT_SUBMIT`.
- This request validates only the remediated GT-cache physical-identity contract
  and its existing directory/ZIP provenance neighbors on real mini plus synthetic
  mutated caches. It does not open shared trainval archives, build full caches,
  run a model, profile, evaluate metrics, or make scientific claims.
- Exact implementation: `c8dd920cf3f8007c3b2ec03f48bcc3f83144ebbe`.
- Exact 25-file focused runtime source-state SHA-256:
  `357da48780436aaba3cbc6735e350d446763acc9f6cb8a0bf424728e55a32d0e`.
- Input mini root:
  `/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/fl_weather_project/data/nuscenes_mini`.
- Exact fresh output root, confirmed absent and not created:
  `/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s07a_provenance_tests_c8dd920cf3f8`.
- Resources: one submission, one node, one GH200, eight CPUs, walltime at most
  `00:15:00`, maximum 0.25 GPU-hours. No array, DDP, model, full cache, metric,
  retry, resubmit, or follow-on job is authorized by this request.
- Executor: a fresh owner/Codex-UI-provisioned clean
  `detached@c8dd920cf3f8007c3b2ec03f48bcc3f83144ebbe` worktree.

### Exact command

```bash
test -z "$(git branch --show-current)" && \
test "$(git rev-parse HEAD)" = "c8dd920cf3f8007c3b2ec03f48bcc3f83144ebbe" && \
test -z "$(git status --short)" && \
test ! -e /nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s07a_provenance_tests_c8dd920cf3f8 && \
test -z "$(squeue -u "$USER" -h -o '%i %j' | awk '$2 ~ /flv3_s07a_provenance/ {print}')" && \
sbatch --time=00:15:00 --cpus-per-task=8 \
  --export=ALL,EXPECTED_S07A_SHA=c8dd920cf3f8007c3b2ec03f48bcc3f83144ebbe,EXPECTED_S07A_PROVENANCE_STATE_HASH=357da48780436aaba3cbc6735e350d446763acc9f6cb8a0bf424728e55a32d0e,S07A_MINI_DATAROOT=/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/fl_weather_project/data/nuscenes_mini,S07A_PROVENANCE_OUTPUT_ROOT=/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s07a_provenance_tests_c8dd920cf3f8 \
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
