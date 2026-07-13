# S07-B-COMPLETE RUN_REQUEST — consumed gate and draft diagnostic continuation

## Approval and immutable-materialization state

```text
SESSION_ID: S07-B-COMPLETE
APPROVAL_STATE: SUBMITTED_ONCE / TERMINAL_FAILED / AUTHORIZATION_CONSUMED
APPROVED_COMPUTE: one exact bounded GH200 engineering validation job (consumed)
APPROVAL_DATE: 2026-07-13
APPROVAL_SOURCE: owner message "批准执行" in the canonical S00 task
REQUEST_SEAL: 6802f34fdafdf33bd31157ed15537b8f7955d1ad
BASE_SHA: 4aa2b133d1d33382bf1514f7a3c86fcb03cf83e5
WORKER_SHA: 34cbe02b7b72114e3a2d61f6f797c8dec022798c
EXECUTABLE_SHA: 34cbe02b7b72114e3a2d61f6f797c8dec022798c
EXECUTABLE_TREE: ed2d4091f0098f6b2144028afd87e20d023b1da2
DELIVERY_REF: 6802f34fdafdf33bd31157ed15537b8f7955d1ad
APPROVAL_SEAL: e5f8dcf9f8608b40d49ad72c62b3557769b780fb
COMMAND_BINDING_SEAL: 8d087f6d43a668c92dd540ccae7f80ac57f44def
SLURM_JOB_ID: 372819
TERMINAL_STATE: FAILED / ExitCode=1:0 / Restarts=0
SUBMIT_START_END: 2026-07-13T11:25:44 / 11:25:45 / 11:25:53 Europe/Stockholm
OUTPUT_ROOT: /nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s07b_complete_34cbe02b7b72
RETRY_STATE: forbidden / not requested / not approved / not submitted
DIAGNOSTIC_REQUEST_STATE: OWNER_APPROVED_FOR_ONE_EXACT_SUBMISSION / NOT YET SUBMITTED
DIAGNOSTIC_SCOPE: same W/tree and 205-case gate with durable bootstrap observations
DIAGNOSTIC_FIX: replace reserved GIT_COMMON_DIR export with PROJECT_GIT_DIR and unset Git repository-selection overrides
DIAGNOSTIC_OUTPUT_ROOT: /nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s07b_complete_diag1_34cbe02b7b72
PRIOR_COMMAND_SHA256: 888982814f0033e54528ab9f6e01de02c3596dfbf2fef7078f9e73b3ca540f99
DIAGNOSTIC_COMMAND_SHA256: b5a98f0a09b79d9c64a474b1449f4e144c58e10a3b497d2c427c704e275d6596
DIAGNOSTIC_JOB_BODY_SHA256: 8c99f9026cdc09af3ffc17e91bcc490bc95f010cbfdec9e0511fec241d829e3e
PYTEST_ARGS_SHA256: 2b9f312535632b7ec17a72ec5fbf0b300b5b690a4fd9d8a81ae94aea21028a67 (unchanged)
DIAGNOSTIC_REQUEST_SEAL: 261b2a9eef2a49afe14a04f16da87bb38a02b274
DIAGNOSTIC_APPROVAL_DATE: 2026-07-13
DIAGNOSTIC_APPROVAL_SOURCE: owner message "批准创建并且compute也批准，立刻开始执行"
DIAGNOSTIC_APPROVED_COMPUTE: one exact instrumented GH200 validation job
DIAGNOSTIC_RETRY_STATE: forbidden / not approved
```

The owner approved exactly one submission of the frozen command and resources at
command-binding seal `8d087f6d...` after S00's audit. That consumed approval was
bound to the request seal,
executable commit/tree, literal source closure and hashes, dependency identities,
mini-data scope, output root, resource ceiling, stop conditions, and no-retry rule
recorded at that Git version. The exact historical command remains preserved in
Git and job artifacts. The draft diagnostic command below is different and has no
compute authority. No repository launcher, compatibility wrapper, retry,
replacement, or expanded cell is authorized.

## Terminal execution record

S00 submitted the exact command below once. Slurm accepted job `372819` with the
requested one GH200, eight CPUs, 96 GiB, one node, 60-minute limit, and
`--no-requeue`; it failed after eight seconds with exit `1:0` and zero restarts.
The immutable Git archive, literal/Git-selected 100-path manifests, all 100 source
records, and executable-patch hash were created and checksum-verified. Execution
then stopped before dependency baseline capture, environment recording, JUnit, or
pytest output. Both Slurm streams are empty, so the exact failing silent bootstrap
assertion was not recoverable from the run artifacts alone. S00 subsequently
reproduced the exact failure as documented below. This remains a preserved
negative result, not a runtime PASS. The one-submission authorization is consumed;
any diagnostic change or rerun requires a new exact request and owner approval.

## Root-cause diagnosis after job 372819

The consumed command passed the project Git path to Slurm as environment variable
`GIT_COMMON_DIR`. That name is a Git repository-selection override, not a neutral
project variable. It therefore contaminated later `git -C` commands intended for
the independent cumm/spconv repositories.

Exact login-side reproduction with the same exported value proves the failure:

```text
GIT_COMMON_DIR=<project .git> git -C <cumm> rev-parse HEAD = 4dedaf43... (passes)
GIT_COMMON_DIR=<project .git> git -C <cumm> rev-parse --git-common-dir = <project .git>
GIT_COMMON_DIR=<project .git> git -C <cumm> diff --cached --name-only = 267 paths
GIT_COMMON_DIR=<project .git> git -C <cumm> diff --cached --quiet rc = 1
GIT_COMMON_DIR=<project .git> git -C <spconv> diff --cached --name-only = 160 paths
```

A local harness using the consumed environment passed mini path, environment
Python, and both dependency HEAD checks, then failed exactly at
`cumm-staged-paths`; the observed output was 7,793 bytes. This matches job
`372819` stopping after executable validation but before dependency baseline.
With the reserved variable removed, all thirteen instrumented gates pass locally
and the generated artifact manifest re-verifies. No worker/source/test/config byte
was changed by this diagnosis.

## Frozen durable executable identities

The executable commit is a direct child of BASE and differs only in
`fl_v3/configs/flwr_config.toml` and
`fl_v3/tests/test_s07_b_clean_completion.py`; handoff Markdown is not part of the
100-file runtime source closure.

```text
executable patch sha256 = 98c0521973ab9963cbf3447618efbedcba7a2fc6807804da222976e5b90f1002
source file count = 100
source path-list sha256 = ce5c38764b43efa027b88b0b37de3a63407fb71ee9b0c5ad5bcd0671a0323ac4
source-record aggregate sha256 = acb80014ff8dd3ef123e689b3be34efae219c95c95ea63f64c36e28f6d546a9e
completion test sha256 = 71d461eb3eb80a7e945ff4ae9e3fc8b07d7a99ed2b55b26a56d4e3c7ada4eef2
Flower config sha256 = 2f459f816ad1bfcc9d1f9c1c2de9cc6491f5ea564eee633290e47665ff2003ab
```

The executable patch digest is reproduced from immutable Git objects with:

```bash
{
  git diff --binary 4aa2b133d1d33382bf1514f7a3c86fcb03cf83e5 \
    34cbe02b7b72114e3a2d61f6f797c8dec022798c -- \
    fl_v3/configs/flwr_config.toml
  printf '\0%s\0' fl_v3/tests/test_s07_b_clean_completion.py
  git show 34cbe02b7b72114e3a2d61f6f797c8dec022798c:\
fl_v3/tests/test_s07_b_clean_completion.py
} | sha256sum
```

The path-list digest is SHA-256 of the following exact UTF-8 bytes, including
one LF after every path. This is the literal locale-sorted 100-path manifest:

```text
fl_v3/configs/flwr_config.toml
fl_v3/configs/s06_synthetic_camera.json
fl_v3/configs/s07_b_c_str8.json
fl_v3/configs/s07_b_f_cbgs.json
fl_v3/configs/s07_b_f_u.json
fl_v3/configs/s07_b_l_p020.json
fl_v3/configs/s07_b_l_s075.json
fl_v3/pyproject.toml
fl_v3/requirements.lock.txt
fl_v3/requirements.txt
fl_v3/scripts/arrhenius_env.sh
fl_v3/scripts/centralized_train.py
fl_v3/src/fl_v3/__init__.py
fl_v3/src/fl_v3/client_app.py
fl_v3/src/fl_v3/config/__init__.py
fl_v3/src/fl_v3/config/resolved.py
fl_v3/src/fl_v3/data/__init__.py
fl_v3/src/fl_v3/data/nuscenes/__init__.py
fl_v3/src/fl_v3/data/nuscenes/augment.py
fl_v3/src/fl_v3/data/nuscenes/cbgs.py
fl_v3/src/fl_v3/data/nuscenes/class_map.py
fl_v3/src/fl_v3/data/nuscenes/dataset.py
fl_v3/src/fl_v3/data/nuscenes/gt_database.py
fl_v3/src/fl_v3/data/nuscenes/gt_paste.py
fl_v3/src/fl_v3/data/nuscenes/info_cache.py
fl_v3/src/fl_v3/data/nuscenes/partition.py
fl_v3/src/fl_v3/data/nuscenes/paths.py
fl_v3/src/fl_v3/data/nuscenes/transforms.py
fl_v3/src/fl_v3/data/nuscenes/zip_backend.py
fl_v3/src/fl_v3/data/partition.py
fl_v3/src/fl_v3/engine/__init__.py
fl_v3/src/fl_v3/engine/local_runner.py
fl_v3/src/fl_v3/eval/__init__.py
fl_v3/src/fl_v3/eval/box_to_global.py
fl_v3/src/fl_v3/eval/detection_eval.py
fl_v3/src/fl_v3/eval/provenance.py
fl_v3/src/fl_v3/models/__init__.py
fl_v3/src/fl_v3/models/dummy.py
fl_v3/src/fl_v3/models/fusion/__init__.py
fl_v3/src/fl_v3/models/fusion/bev_grid.py
fl_v3/src/fl_v3/models/fusion/bev_neck.py
fl_v3/src/fl_v3/models/fusion/camera_backbone.py
fl_v3/src/fl_v3/models/fusion/camera_neck.py
fl_v3/src/fl_v3/models/fusion/centerhead_decode.py
fl_v3/src/fl_v3/models/fusion/collate.py
fl_v3/src/fl_v3/models/fusion/detector.py
fl_v3/src/fl_v3/models/fusion/fusion.py
fl_v3/src/fl_v3/models/fusion/head.py
fl_v3/src/fl_v3/models/fusion/lidar_backbone.py
fl_v3/src/fl_v3/models/fusion/lidar_encoder.py
fl_v3/src/fl_v3/models/fusion/losses.py
fl_v3/src/fl_v3/models/fusion/nms_deterministic.py
fl_v3/src/fl_v3/models/fusion/preprocess.py
fl_v3/src/fl_v3/models/fusion/second_sparse_backbone.py
fl_v3/src/fl_v3/models/fusion/sparse_voxel_encoder.py
fl_v3/src/fl_v3/models/fusion/swin_sdpa.py
fl_v3/src/fl_v3/models/fusion/view_transform.py
fl_v3/src/fl_v3/server_app.py
fl_v3/src/fl_v3/strategy/__init__.py
fl_v3/src/fl_v3/strategy/aggregation_core.py
fl_v3/src/fl_v3/strategy/flower_strategies.py
fl_v3/src/fl_v3/strategy/sampling.py
fl_v3/src/fl_v3/strategy/server_opt.py
fl_v3/src/fl_v3/training/__init__.py
fl_v3/src/fl_v3/training/checkpoint.py
fl_v3/src/fl_v3/training/loop.py
fl_v3/src/fl_v3/training/runtime_state.py
fl_v3/src/fl_v3/training/tasks.py
fl_v3/src/fl_v3/utils/__init__.py
fl_v3/src/fl_v3/utils/profiling.py
fl_v3/src/fl_v3/utils/runtime.py
fl_v3/src/fl_v3/viz/__init__.py
fl_v3/src/fl_v3/viz/calibration.py
fl_v3/src/fl_v3/viz/encoder.py
fl_v3/src/fl_v3/viz/fusion.py
fl_v3/src/fl_v3/viz/writer.py
fl_v3/tests/conftest.py
fl_v3/tests/test_eval_box_to_global.py
fl_v3/tests/test_eval_detection_eval.py
fl_v3/tests/test_eval_provenance.py
fl_v3/tests/test_fl_local_runner_multiround.py
fl_v3/tests/test_fl_round_smoke.py
fl_v3/tests/test_fl_sampling.py
fl_v3/tests/test_fl_server_opt_integration.py
fl_v3/tests/test_fl_trainable_only.py
fl_v3/tests/test_flower_fp32_parity.py
fl_v3/tests/test_flower_strategies_construct.py
fl_v3/tests/test_model_task.py
fl_v3/tests/test_nuscenes_info_cache.py
fl_v3/tests/test_nuscenes_partition.py
fl_v3/tests/test_nuscenes_zip_backend.py
fl_v3/tests/test_nuscenes_zip_dataset.py
fl_v3/tests/test_s06_checkpoint_resume.py
fl_v3/tests/test_s06_loader_eval.py
fl_v3/tests/test_s06_model_modes.py
fl_v3/tests/test_s06_resolved_config.py
fl_v3/tests/test_s06_training_runtime.py
fl_v3/tests/test_s07_b_clean_completion.py
fl_v3/tests/test_s07_b_data_lifecycle.py
fl_v3/tests/test_s07_b_integration.py
```

For a checkout or immutable snapshot root `$ROOT`, the source records are
constructed without filesystem discovery from that exact manifest:

```bash
export LC_ALL=C
while IFS= read -r path; do
  test -f "$ROOT/$path"
  printf '%s  %s\n' "$(sha256sum "$ROOT/$path" | cut -d' ' -f1)" "$path"
done < source-paths.txt > source-records.sha256
sha256sum source-paths.txt
sha256sum source-records.sha256
```

The future body additionally reconstructs the selected set from the immutable
Git tree: all `*.py` blobs under `fl_v3/src/fl_v3` plus the 36 literal
config/dependency/script/test inputs shown in its `git ls-tree` command. It
requires an exact `cmp` with the manifest, exactly 100 sorted unique records, no
missing path, the list digest above, and every file record/aggregate digest above.

## Exact resource, data, environment, and output envelope

- exactly one job, one node, one task, one `nvidia_gh200_120gb`, one concurrent
  job, eight CPUs, 96 GiB, `01:00:00`, `--no-requeue`, no retry/follow-on;
- account `naiss2025-22-1113-gpu`, partition `gpu`;
- persistent read-only environment prefix:
  `/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/envs/pt311-cu128-spconv`;
- exact read-only mini directory only:
  `/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/fl_weather_project/data/nuscenes_mini`;
- Git common directory is read-only input:
  `/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/fl_weather_project/.git`;
- fresh writable job root, deterministically
  `/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s07b_complete_34cbe02b7b72`;
- the immutable source archive/snapshot, Slurm logs, pytest log/JUnit/status,
  source records, `$HOME`, cwd, temp, bytecode, XDG, compiler/framework caches,
  and generated mini info-cache are all children of that single fresh job root;
- `./fl_outputs` therefore resolves under the writable job root. The generated
  mini info-cache is preserved and hashed as engineering-test output; it is not a
  production cache. Nothing is written to the checkout, snapshot, persistent
  environment prefix, dependency source checkouts, or shared mini dataset.

There is no trainval module/cache/manifest, no full-data scan, and no persistent
cache outside the job root. Expected versions fail closed: CPython `3.11.15`,
Torch `2.11.0+cu128`, Flower `1.27.0`, Ray `2.51.1`, NumPy `1.26.4`,
nuscenes-devkit `1.1.11`, pyquaternion `0.9.9`, Pillow `12.2.0`, pytest `9.1.1`,
spconv `2.3.8`, cumm `0.7.13`, and scikit-learn `1.8.0`. Accepted source HEADs
are cumm `4dedaf43ff801e417c60c6bd7536a29d83d29ee0` and spconv
`263d6b47425ef843c82f997b12d8b714013d216c`. Cumm must have no tracked change
and exactly the existing untracked `cumm/core_cc/common.pyi`. Spconv must have no
staged or untracked change and exactly one tracked path, `pyproject.toml`, whose
known build-contract patch and resulting file are pinned below. Any other source
state is fatal before imports. The job records platform/GPU/CUDA, module origins,
Torch Git/build config, and full production-algorithm executable artifact records
and aggregate hashes for Torch/spconv/cumm.

## Exact pytest inventory

Complete files:

```text
fl_v3/tests/test_s07_b_clean_completion.py
fl_v3/tests/test_s07_b_integration.py
fl_v3/tests/test_s07_b_data_lifecycle.py
fl_v3/tests/test_s06_resolved_config.py
fl_v3/tests/test_s06_model_modes.py
fl_v3/tests/test_s06_training_runtime.py
fl_v3/tests/test_s06_checkpoint_resume.py
fl_v3/tests/test_s06_loader_eval.py
fl_v3/tests/test_eval_provenance.py
fl_v3/tests/test_flower_fp32_parity.py
fl_v3/tests/test_flower_strategies_construct.py
fl_v3/tests/test_fl_sampling.py
fl_v3/tests/test_fl_round_smoke.py
fl_v3/tests/test_fl_local_runner_multiround.py
fl_v3/tests/test_fl_server_opt_integration.py
fl_v3/tests/test_fl_trainable_only.py
fl_v3/tests/test_nuscenes_info_cache.py
```

Complete files with exact deselections:

```text
fl_v3/tests/test_nuscenes_zip_backend.py
  --deselect=fl_v3/tests/test_nuscenes_zip_backend.py::test_parent_open_then_child_gets_process_owned_handles
fl_v3/tests/test_nuscenes_zip_dataset.py
  --deselect=fl_v3/tests/test_nuscenes_zip_dataset.py::test_repeated_persistent_multiworker_reads_are_deterministic
```

Exact individual nodes:

```text
fl_v3/tests/test_nuscenes_partition.py::test_partition_seed_coercion
fl_v3/tests/test_nuscenes_partition.py::test_stable_shards_same_inputs
fl_v3/tests/test_nuscenes_partition.py::test_mini_is_degenerate_smoke
fl_v3/tests/test_nuscenes_partition.py::test_iid_baseline_partition
fl_v3/tests/test_nuscenes_partition.py::test_iid_partition_deterministic_and_seed_sensitive
fl_v3/tests/test_nuscenes_partition.py::test_sub_floor_client_is_surfaced_not_silent
fl_v3/tests/test_model_task.py::test_dummy_regression_byte_identity_golden
fl_v3/tests/test_model_task.py::test_detection_task_registered
fl_v3/tests/test_model_task.py::test_detection_config_rejects_legacy_model_mode_alias
fl_v3/tests/test_model_task.py::test_num_clients_iid_is_requested
fl_v3/tests/test_model_task.py::test_client_data_materializes_dict_batch
fl_v3/tests/test_eval_box_to_global.py::test_rotmat_to_quaternion_matches_pyquaternion_and_roundtrips
fl_v3/tests/test_eval_box_to_global.py::test_yaw_about_z_is_pure_yaw
fl_v3/tests/test_eval_box_to_global.py::test_box_to_global_matches_raw_devkit_annotation
fl_v3/tests/test_eval_detection_eval.py::test_assert_version_split
fl_v3/tests/test_eval_detection_eval.py::test_submission_meta_uses_actual_mode
fl_v3/tests/test_eval_detection_eval.py::test_results_dict_has_all_tokens_as_keys
fl_v3/tests/test_eval_detection_eval.py::test_gt_as_pred_per_class_ap_near_one
```

The source-level expansion count is exactly 205 cases. JUnit must report exactly
205 tests, zero failure/error/skip, and the process must emit no warning under
`-W error`. The two deselections remove the old fork/spawn matrices. Exact node
selection excludes the trainval partition nodes, catch-and-skip legacy worker
test, extra legacy detector update, `test_model_overfit.py`, and extra official
permutation metric. `test_gt_as_pred_per_class_ap_near_one` is the only named
official-devkit metric identity case; its AP is test output, not science.

## Exact owner-approved instrumented command — one submission only

This draft keeps the executable commit/tree, source closure, 205-case inventory,
mini scope, resources, timeout, and no-retry behavior unchanged. It adds durable
labels plus expected/observed files around the thirteen formerly silent bootstrap
gates, renames the project Git path to neutral `PROJECT_GIT_DIR`, explicitly
clears Git repository-selection overrides, and uses a fresh diagnostic job root.
It does not depend on a mutable or temporary Codex worktree. The owner approved
exactly one submission of this command; any byte, resource, data, test, output,
or stop-condition change invalidates that approval.

```bash
#!/bin/bash
set -euo pipefail
export LC_ALL=C
export LANG=C
unset GIT_DIR GIT_WORK_TREE GIT_INDEX_FILE GIT_OBJECT_DIRECTORY \
  GIT_COMMON_DIR GIT_ALTERNATE_OBJECT_DIRECTORIES

PROJECT_GIT_DIR=/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/fl_weather_project/.git
BASE_SHA=4aa2b133d1d33382bf1514f7a3c86fcb03cf83e5
EXPECTED_SHA=34cbe02b7b72114e3a2d61f6f797c8dec022798c
EXPECTED_TREE=ed2d4091f0098f6b2144028afd87e20d023b1da2
EXPECTED_SOURCE_AGG=acb80014ff8dd3ef123e689b3be34efae219c95c95ea63f64c36e28f6d546a9e
OUTPUT_PARENT=/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs

[[ "$EXPECTED_SHA" =~ ^[0-9a-f]{40}$ ]]
[[ "$EXPECTED_TREE" =~ ^[0-9a-f]{40}$ ]]
test "$(git --git-dir="$PROJECT_GIT_DIR" rev-parse "$EXPECTED_SHA")" = "$EXPECTED_SHA"
test "$(git --git-dir="$PROJECT_GIT_DIR" rev-parse "$EXPECTED_SHA^{tree}")" = "$EXPECTED_TREE"
test "$(git --git-dir="$PROJECT_GIT_DIR" rev-parse "$EXPECTED_SHA^")" = "$BASE_SHA"
test "$(git --git-dir="$PROJECT_GIT_DIR" diff-tree --no-commit-id --name-only \
  -r "$EXPECTED_SHA" | LC_ALL=C sort)" = "$(printf '%s\n' \
  fl_v3/configs/flwr_config.toml \
  fl_v3/tests/test_s07_b_clean_completion.py)"

JOB_ROOT="$OUTPUT_PARENT/s07b_complete_diag1_${EXPECTED_SHA:0:12}"
test ! -e "$JOB_ROOT"
install -d -m 0700 "$JOB_ROOT"

JOB_BODY=$(cat <<'S07B_JOB'
set -euo pipefail
umask 077
export LC_ALL=C
export LANG=C
unset GIT_DIR GIT_WORK_TREE GIT_INDEX_FILE GIT_OBJECT_DIRECTORY \
  GIT_COMMON_DIR GIT_ALTERNATE_OBJECT_DIRECTORIES

: "${SLURM_JOB_ID:?}"
: "${EXPECTED_SHA:?}"
: "${EXPECTED_TREE:?}"
: "${EXPECTED_SOURCE_AGG:?}"
: "${JOB_ROOT:?}"
: "${PROJECT_GIT_DIR:?}"
test -d "$JOB_ROOT"

ARTIFACTS="$JOB_ROOT/artifacts"
STATUS="$JOB_ROOT/status"
SOURCE_ROOT="$JOB_ROOT/source"
SOURCE_ARCHIVE="$SOURCE_ROOT/source.tar"
SNAPSHOT="$SOURCE_ROOT/snapshot"
PYTEST_TMP="$JOB_ROOT/pytest_tmp"
PYTEST_LOG="$ARTIFACTS/pytest.log"
JUNIT="$ARTIFACTS/pytest.junit.xml"
install -d -m 0700 \
  "$ARTIFACTS" "$STATUS" "$SOURCE_ROOT" "$SNAPSHOT" "$PYTEST_TMP" \
  "$JOB_ROOT/home" "$JOB_ROOT/tmp" "$JOB_ROOT/xdg" "$JOB_ROOT/pycache" \
  "$JOB_ROOT/matplotlib" "$JOB_ROOT/ray" "$JOB_ROOT/wandb" \
  "$JOB_ROOT/runtime" "$JOB_ROOT/cuda_cache" "$JOB_ROOT/numba_cache" \
  "$JOB_ROOT/triton_cache" "$JOB_ROOT/fl_outputs/nuscenes/info_cache"
DEPENDENCY_STATE_BASELINE_READY=0
BOOTSTRAP_DIR="$ARTIFACTS/bootstrap"
BOOTSTRAP_GATES="$ARTIFACTS/bootstrap-gates.tsv"
BOOTSTRAP_STAGE="$STATUS/bootstrap-stage.txt"
install -d -m 0700 "$BOOTSTRAP_DIR"
printf 'sequence\tfinished_utc\tlabel\tstate\tcommand_rc\texpected_sha256\tobserved_sha256\n' \
  > "$BOOTSTRAP_GATES"
printf '%s\n' '000|bootstrap|READY' > "$BOOTSTRAP_STAGE"
BOOTSTRAP_SEQUENCE=0

bootstrap_expect_eq() {
  local label=$1
  local expected=$2
  shift 2
  BOOTSTRAP_SEQUENCE=$((BOOTSTRAP_SEQUENCE + 1))
  local prefix
  local expected_file
  local observed_file
  local observed
  local command_rc
  local state
  local result_rc
  local expected_sha
  local observed_sha
  local finished_utc
  prefix=$(printf '%03d-%s' "$BOOTSTRAP_SEQUENCE" "$label")
  expected_file="$BOOTSTRAP_DIR/$prefix.expected.txt"
  observed_file="$BOOTSTRAP_DIR/$prefix.observed.txt"
  printf '%s' "$expected" > "$expected_file"
  printf '%03d|%s|BEGIN\n' "$BOOTSTRAP_SEQUENCE" "$label" \
    > "$BOOTSTRAP_STAGE"
  set +e
  observed=$("$@" 2>&1)
  command_rc=$?
  set -e
  printf '%s' "$observed" > "$observed_file"
  if test "$command_rc" != 0; then
    state=COMMAND_FAIL
    result_rc=$command_rc
  elif test "$observed" != "$expected"; then
    state=MISMATCH
    result_rc=1
  else
    state=PASS
    result_rc=0
  fi
  expected_sha=$(sha256sum "$expected_file" | cut -d' ' -f1)
  observed_sha=$(sha256sum "$observed_file" | cut -d' ' -f1)
  finished_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  printf '%03d\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$BOOTSTRAP_SEQUENCE" "$finished_utc" "$label" "$state" \
    "$command_rc" "$expected_sha" "$observed_sha" >> "$BOOTSTRAP_GATES"
  printf '%03d|%s|%s|command_rc=%s|result_rc=%s\n' \
    "$BOOTSTRAP_SEQUENCE" "$label" "$state" "$command_rc" "$result_rc" \
    > "$BOOTSTRAP_STAGE"
  return "$result_rc"
}

bootstrap_expect_success() {
  local label=$1
  shift
  BOOTSTRAP_SEQUENCE=$((BOOTSTRAP_SEQUENCE + 1))
  local prefix
  local expected_file
  local observed_file
  local command_rc
  local state
  local expected_sha
  local observed_sha
  local finished_utc
  prefix=$(printf '%03d-%s' "$BOOTSTRAP_SEQUENCE" "$label")
  expected_file="$BOOTSTRAP_DIR/$prefix.expected.txt"
  observed_file="$BOOTSTRAP_DIR/$prefix.observed.txt"
  printf '%s' 'exit=0' > "$expected_file"
  printf '%03d|%s|BEGIN\n' "$BOOTSTRAP_SEQUENCE" "$label" \
    > "$BOOTSTRAP_STAGE"
  set +e
  "$@" > "$observed_file" 2>&1
  command_rc=$?
  set -e
  if test "$command_rc" = 0; then state=PASS; else state=COMMAND_FAIL; fi
  expected_sha=$(sha256sum "$expected_file" | cut -d' ' -f1)
  observed_sha=$(sha256sum "$observed_file" | cut -d' ' -f1)
  finished_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  printf '%03d\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$BOOTSTRAP_SEQUENCE" "$finished_utc" "$label" "$state" \
    "$command_rc" "$expected_sha" "$observed_sha" >> "$BOOTSTRAP_GATES"
  printf '%03d|%s|%s|command_rc=%s|result_rc=%s\n' \
    "$BOOTSTRAP_SEQUENCE" "$label" "$state" "$command_rc" "$command_rc" \
    > "$BOOTSTRAP_STAGE"
  return "$command_rc"
}

finalize_s07b() {
  original_rc=$?
  trap - EXIT
  set +e
  printf '%s\n' "$original_rc" > "$STATUS/original-exit.txt"
  date -u +%Y-%m-%dT%H:%M:%SZ > "$STATUS/finished-utc.txt"
  dependency_finalize_rc=0
  if test "$DEPENDENCY_STATE_BASELINE_READY" = 1 && \
      { test ! -f "$ARTIFACTS/cumm-source-state-after.txt" || \
        test ! -f "$ARTIFACTS/spconv-source-state-after.txt"; }; then
    (
      set -e
      dependency_source_state "$DEPENDENCY_SRC/cumm" \
        "$ARTIFACTS/cumm-source-state-after.txt"
      dependency_source_state "$DEPENDENCY_SRC/spconv" \
        "$ARTIFACTS/spconv-source-state-after.txt"
    )
    dependency_capture_rc=$?
    cmp "$ARTIFACTS/cumm-source-state-before.txt" \
      "$ARTIFACTS/cumm-source-state-after.txt"
    dependency_cumm_cmp_rc=$?
    cmp "$ARTIFACTS/spconv-source-state-before.txt" \
      "$ARTIFACTS/spconv-source-state-after.txt"
    dependency_spconv_cmp_rc=$?
    printf '%s\n' "$dependency_capture_rc" \
      > "$STATUS/finalizer-dependency-capture-exit.txt"
    printf '%s\n' "$dependency_cumm_cmp_rc" \
      > "$STATUS/finalizer-cumm-state-cmp-exit.txt"
    printf '%s\n' "$dependency_spconv_cmp_rc" \
      > "$STATUS/finalizer-spconv-state-cmp-exit.txt"
    if test "$dependency_capture_rc" != 0 || \
        test "$dependency_cumm_cmp_rc" != 0 || \
        test "$dependency_spconv_cmp_rc" != 0; then
      dependency_finalize_rc=78
    fi
  fi
  if test -d "$JOB_ROOT/fl_outputs/nuscenes/info_cache"; then
    (
      cd "$JOB_ROOT"
      find fl_outputs/nuscenes/info_cache -type f -print0 \
        | LC_ALL=C sort -z \
        | xargs -0 -r sha256sum
    ) > "$ARTIFACTS/mini-cache-records.sha256"
  fi
  (
    cd "$JOB_ROOT"
    find artifacts status fl_outputs source/source.tar \
      "slurm-$SLURM_JOB_ID.out" "slurm-$SLURM_JOB_ID.err" -type f \
      ! -path 'artifacts/sha256sums.txt' \
      ! -path 'status/final-exit.txt' \
      ! -path 'status/final-exit.sha256' \
      -print0 \
      | LC_ALL=C sort -z \
      | xargs -0 -r sha256sum
  ) > "$ARTIFACTS/sha256sums.txt"
  (
    cd "$JOB_ROOT"
    sha256sum -c artifacts/sha256sums.txt
  ) > "$ARTIFACTS/sha256sums.verify.log" 2>&1
  checksum_rc=$?
  final_rc=$original_rc
  if test "$final_rc" = 0 && test "$dependency_finalize_rc" != 0; then
    final_rc=$dependency_finalize_rc
  fi
  if test "$final_rc" = 0 && test "$checksum_rc" != 0; then
    final_rc=97
  fi
  printf '%s\n' "$final_rc" > "$STATUS/final-exit.txt"
  sha256sum "$STATUS/final-exit.txt" > "$STATUS/final-exit.sha256"
  exit "$final_rc"
}
trap finalize_s07b EXIT

test -f "$JOB_ROOT/slurm-$SLURM_JOB_ID.out"
test -f "$JOB_ROOT/slurm-$SLURM_JOB_ID.err"
test "$SLURM_JOB_NUM_NODES" = 1
test "$SLURM_NTASKS" = 1
test "$SLURM_CPUS_PER_TASK" = 8
test "${SLURM_RESTART_COUNT:-0}" = 0

git --git-dir="$PROJECT_GIT_DIR" cat-file -e "$EXPECTED_SHA^{commit}"
test "$(git --git-dir="$PROJECT_GIT_DIR" rev-parse "$EXPECTED_SHA^{tree}")" = "$EXPECTED_TREE"
git --git-dir="$PROJECT_GIT_DIR" archive \
  --format=tar --output="$SOURCE_ARCHIVE" "$EXPECTED_SHA"
tar -xf "$SOURCE_ARCHIVE" -C "$SNAPSHOT"
chmod -R a-w "$SNAPSHOT"
sha256sum "$SOURCE_ARCHIVE" > "$ARTIFACTS/source-archive.sha256"

cat > "$ARTIFACTS/source-paths.txt" <<'S07B_PATHS'
fl_v3/configs/flwr_config.toml
fl_v3/configs/s06_synthetic_camera.json
fl_v3/configs/s07_b_c_str8.json
fl_v3/configs/s07_b_f_cbgs.json
fl_v3/configs/s07_b_f_u.json
fl_v3/configs/s07_b_l_p020.json
fl_v3/configs/s07_b_l_s075.json
fl_v3/pyproject.toml
fl_v3/requirements.lock.txt
fl_v3/requirements.txt
fl_v3/scripts/arrhenius_env.sh
fl_v3/scripts/centralized_train.py
fl_v3/src/fl_v3/__init__.py
fl_v3/src/fl_v3/client_app.py
fl_v3/src/fl_v3/config/__init__.py
fl_v3/src/fl_v3/config/resolved.py
fl_v3/src/fl_v3/data/__init__.py
fl_v3/src/fl_v3/data/nuscenes/__init__.py
fl_v3/src/fl_v3/data/nuscenes/augment.py
fl_v3/src/fl_v3/data/nuscenes/cbgs.py
fl_v3/src/fl_v3/data/nuscenes/class_map.py
fl_v3/src/fl_v3/data/nuscenes/dataset.py
fl_v3/src/fl_v3/data/nuscenes/gt_database.py
fl_v3/src/fl_v3/data/nuscenes/gt_paste.py
fl_v3/src/fl_v3/data/nuscenes/info_cache.py
fl_v3/src/fl_v3/data/nuscenes/partition.py
fl_v3/src/fl_v3/data/nuscenes/paths.py
fl_v3/src/fl_v3/data/nuscenes/transforms.py
fl_v3/src/fl_v3/data/nuscenes/zip_backend.py
fl_v3/src/fl_v3/data/partition.py
fl_v3/src/fl_v3/engine/__init__.py
fl_v3/src/fl_v3/engine/local_runner.py
fl_v3/src/fl_v3/eval/__init__.py
fl_v3/src/fl_v3/eval/box_to_global.py
fl_v3/src/fl_v3/eval/detection_eval.py
fl_v3/src/fl_v3/eval/provenance.py
fl_v3/src/fl_v3/models/__init__.py
fl_v3/src/fl_v3/models/dummy.py
fl_v3/src/fl_v3/models/fusion/__init__.py
fl_v3/src/fl_v3/models/fusion/bev_grid.py
fl_v3/src/fl_v3/models/fusion/bev_neck.py
fl_v3/src/fl_v3/models/fusion/camera_backbone.py
fl_v3/src/fl_v3/models/fusion/camera_neck.py
fl_v3/src/fl_v3/models/fusion/centerhead_decode.py
fl_v3/src/fl_v3/models/fusion/collate.py
fl_v3/src/fl_v3/models/fusion/detector.py
fl_v3/src/fl_v3/models/fusion/fusion.py
fl_v3/src/fl_v3/models/fusion/head.py
fl_v3/src/fl_v3/models/fusion/lidar_backbone.py
fl_v3/src/fl_v3/models/fusion/lidar_encoder.py
fl_v3/src/fl_v3/models/fusion/losses.py
fl_v3/src/fl_v3/models/fusion/nms_deterministic.py
fl_v3/src/fl_v3/models/fusion/preprocess.py
fl_v3/src/fl_v3/models/fusion/second_sparse_backbone.py
fl_v3/src/fl_v3/models/fusion/sparse_voxel_encoder.py
fl_v3/src/fl_v3/models/fusion/swin_sdpa.py
fl_v3/src/fl_v3/models/fusion/view_transform.py
fl_v3/src/fl_v3/server_app.py
fl_v3/src/fl_v3/strategy/__init__.py
fl_v3/src/fl_v3/strategy/aggregation_core.py
fl_v3/src/fl_v3/strategy/flower_strategies.py
fl_v3/src/fl_v3/strategy/sampling.py
fl_v3/src/fl_v3/strategy/server_opt.py
fl_v3/src/fl_v3/training/__init__.py
fl_v3/src/fl_v3/training/checkpoint.py
fl_v3/src/fl_v3/training/loop.py
fl_v3/src/fl_v3/training/runtime_state.py
fl_v3/src/fl_v3/training/tasks.py
fl_v3/src/fl_v3/utils/__init__.py
fl_v3/src/fl_v3/utils/profiling.py
fl_v3/src/fl_v3/utils/runtime.py
fl_v3/src/fl_v3/viz/__init__.py
fl_v3/src/fl_v3/viz/calibration.py
fl_v3/src/fl_v3/viz/encoder.py
fl_v3/src/fl_v3/viz/fusion.py
fl_v3/src/fl_v3/viz/writer.py
fl_v3/tests/conftest.py
fl_v3/tests/test_eval_box_to_global.py
fl_v3/tests/test_eval_detection_eval.py
fl_v3/tests/test_eval_provenance.py
fl_v3/tests/test_fl_local_runner_multiround.py
fl_v3/tests/test_fl_round_smoke.py
fl_v3/tests/test_fl_sampling.py
fl_v3/tests/test_fl_server_opt_integration.py
fl_v3/tests/test_fl_trainable_only.py
fl_v3/tests/test_flower_fp32_parity.py
fl_v3/tests/test_flower_strategies_construct.py
fl_v3/tests/test_model_task.py
fl_v3/tests/test_nuscenes_info_cache.py
fl_v3/tests/test_nuscenes_partition.py
fl_v3/tests/test_nuscenes_zip_backend.py
fl_v3/tests/test_nuscenes_zip_dataset.py
fl_v3/tests/test_s06_checkpoint_resume.py
fl_v3/tests/test_s06_loader_eval.py
fl_v3/tests/test_s06_model_modes.py
fl_v3/tests/test_s06_resolved_config.py
fl_v3/tests/test_s06_training_runtime.py
fl_v3/tests/test_s07_b_clean_completion.py
fl_v3/tests/test_s07_b_data_lifecycle.py
fl_v3/tests/test_s07_b_integration.py
S07B_PATHS

test "$(wc -l < "$ARTIFACTS/source-paths.txt")" = 100
LC_ALL=C sort -c "$ARTIFACTS/source-paths.txt"
test -z "$(LC_ALL=C sort "$ARTIFACTS/source-paths.txt" | uniq -d)"
test "$(sha256sum "$ARTIFACTS/source-paths.txt" | cut -d' ' -f1)" = \
  ce5c38764b43efa027b88b0b37de3a63407fb71ee9b0c5ad5bcd0671a0323ac4

{
  git --git-dir="$PROJECT_GIT_DIR" ls-tree -r --name-only "$EXPECTED_SHA" -- \
    fl_v3/src/fl_v3 | awk '/[.]py$/'
  git --git-dir="$PROJECT_GIT_DIR" ls-tree -r --name-only "$EXPECTED_SHA" -- \
    fl_v3/configs/flwr_config.toml \
    fl_v3/configs/s06_synthetic_camera.json \
    fl_v3/configs/s07_b_c_str8.json \
    fl_v3/configs/s07_b_f_cbgs.json \
    fl_v3/configs/s07_b_f_u.json \
    fl_v3/configs/s07_b_l_p020.json \
    fl_v3/configs/s07_b_l_s075.json \
    fl_v3/pyproject.toml fl_v3/requirements.lock.txt fl_v3/requirements.txt \
    fl_v3/scripts/arrhenius_env.sh fl_v3/scripts/centralized_train.py \
    fl_v3/tests/conftest.py \
    fl_v3/tests/test_eval_box_to_global.py \
    fl_v3/tests/test_eval_detection_eval.py \
    fl_v3/tests/test_eval_provenance.py \
    fl_v3/tests/test_fl_local_runner_multiround.py \
    fl_v3/tests/test_fl_round_smoke.py fl_v3/tests/test_fl_sampling.py \
    fl_v3/tests/test_fl_server_opt_integration.py \
    fl_v3/tests/test_fl_trainable_only.py \
    fl_v3/tests/test_flower_fp32_parity.py \
    fl_v3/tests/test_flower_strategies_construct.py \
    fl_v3/tests/test_model_task.py fl_v3/tests/test_nuscenes_info_cache.py \
    fl_v3/tests/test_nuscenes_partition.py \
    fl_v3/tests/test_nuscenes_zip_backend.py \
    fl_v3/tests/test_nuscenes_zip_dataset.py \
    fl_v3/tests/test_s06_checkpoint_resume.py \
    fl_v3/tests/test_s06_loader_eval.py fl_v3/tests/test_s06_model_modes.py \
    fl_v3/tests/test_s06_resolved_config.py \
    fl_v3/tests/test_s06_training_runtime.py \
    fl_v3/tests/test_s07_b_clean_completion.py \
    fl_v3/tests/test_s07_b_data_lifecycle.py \
    fl_v3/tests/test_s07_b_integration.py
} | LC_ALL=C sort > "$ARTIFACTS/git-source-paths.txt"
test "$(wc -l < "$ARTIFACTS/git-source-paths.txt")" = 100
test -z "$(uniq -d "$ARTIFACTS/git-source-paths.txt")"
cmp "$ARTIFACTS/source-paths.txt" "$ARTIFACTS/git-source-paths.txt"

while IFS= read -r path; do
  test -f "$SNAPSHOT/$path"
  printf '%s  %s\n' "$(sha256sum "$SNAPSHOT/$path" | cut -d' ' -f1)" "$path"
done < "$ARTIFACTS/source-paths.txt" > "$ARTIFACTS/source-records.sha256"
test "$(sha256sum "$ARTIFACTS/source-records.sha256" | cut -d' ' -f1)" = \
  "$EXPECTED_SOURCE_AGG"
test "$(sha256sum "$SNAPSHOT/fl_v3/tests/test_s07_b_clean_completion.py" | cut -d' ' -f1)" = \
  71d461eb3eb80a7e945ff4ae9e3fc8b07d7a99ed2b55b26a56d4e3c7ada4eef2
test "$(sha256sum "$SNAPSHOT/fl_v3/configs/flwr_config.toml" | cut -d' ' -f1)" = \
  2f459f816ad1bfcc9d1f9c1c2de9cc6491f5ea564eee633290e47665ff2003ab
{
  git --git-dir="$PROJECT_GIT_DIR" diff --binary \
    4aa2b133d1d33382bf1514f7a3c86fcb03cf83e5 "$EXPECTED_SHA" -- \
    fl_v3/configs/flwr_config.toml
  printf '\0%s\0' fl_v3/tests/test_s07_b_clean_completion.py
  sed -n '1,$p' "$SNAPSHOT/fl_v3/tests/test_s07_b_clean_completion.py"
} | sha256sum > "$ARTIFACTS/executable-patch.sha256"
test "$(cut -d' ' -f1 "$ARTIFACTS/executable-patch.sha256")" = \
  98c0521973ab9963cbf3447618efbedcba7a2fc6807804da222976e5b90f1002

MINI_ROOT=/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/fl_weather_project/data/nuscenes_mini
PERSISTENT_VENV=/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/envs/pt311-cu128-spconv
export DEPENDENCY_SRC=/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/src

spconv_patch_hash_observed() {
  git -C "$DEPENDENCY_SRC/spconv" diff --no-color --binary --full-index \
    --no-ext-diff --no-textconv HEAD -- pyproject.toml \
    | sha256sum | cut -d' ' -f1
}

spconv_file_hash_observed() {
  sha256sum "$DEPENDENCY_SRC/spconv/pyproject.toml" | cut -d' ' -f1
}

(
  set +e
  printf 'hostname=%s\n' "$(hostname)"
  printf 'uname_machine=%s\n' "$(uname -m)"
  printf 'mini_declared=%s\n' "$MINI_ROOT"
  printf 'mini_realpath=%s\n' "$(realpath "$MINI_ROOT" 2>&1)"
  printf 'venv_python_declared=%s\n' "$PERSISTENT_VENV/bin/python"
  printf 'venv_python_realpath=%s\n' \
    "$(realpath "$PERSISTENT_VENV/bin/python" 2>&1)"
  printf 'dependency_src_declared=%s\n' "$DEPENDENCY_SRC"
  printf 'dependency_src_realpath=%s\n' \
    "$(realpath "$DEPENDENCY_SRC" 2>&1)"
  command -v git realpath sha256sum findmnt file
  git --version
  findmnt -T "$MINI_ROOT" -o TARGET,SOURCE,FSTYPE,OPTIONS -n
  findmnt -T "$DEPENDENCY_SRC" -o TARGET,SOURCE,FSTYPE,OPTIONS -n
  stat -Lc 'mini_dev=%d inode=%i mode=%a uid=%u gid=%g' "$MINI_ROOT"
  stat -Lc 'venv_python_dev=%d inode=%i mode=%a uid=%u gid=%g' \
    "$PERSISTENT_VENV/bin/python"
  file "$PERSISTENT_VENV/bin/python" \
    "$(realpath "$PERSISTENT_VENV/bin/python" 2>/dev/null)"
  exit 0
) > "$ARTIFACTS/bootstrap-context.txt" 2>&1

bootstrap_expect_eq mini-root-realpath "$MINI_ROOT" realpath "$MINI_ROOT"
bootstrap_expect_success mini-v1.0-mini-directory test -d "$MINI_ROOT/v1.0-mini"
bootstrap_expect_success persistent-python-executable \
  test -x "$PERSISTENT_VENV/bin/python"
bootstrap_expect_eq cumm-head \
  4dedaf43ff801e417c60c6bd7536a29d83d29ee0 \
  git -C "$DEPENDENCY_SRC/cumm" rev-parse HEAD
bootstrap_expect_eq spconv-head \
  263d6b47425ef843c82f997b12d8b714013d216c \
  git -C "$DEPENDENCY_SRC/spconv" rev-parse HEAD
bootstrap_expect_eq cumm-staged-paths '' \
  git -C "$DEPENDENCY_SRC/cumm" diff --cached --name-only
bootstrap_expect_eq cumm-tracked-paths '' \
  git -C "$DEPENDENCY_SRC/cumm" diff --name-only HEAD --
bootstrap_expect_eq cumm-untracked-paths cumm/core_cc/common.pyi \
  git -C "$DEPENDENCY_SRC/cumm" ls-files --others --exclude-standard
bootstrap_expect_eq spconv-staged-paths '' \
  git -C "$DEPENDENCY_SRC/spconv" diff --cached --name-only
bootstrap_expect_eq spconv-tracked-paths pyproject.toml \
  git -C "$DEPENDENCY_SRC/spconv" diff --name-only HEAD --
bootstrap_expect_eq spconv-untracked-paths '' \
  git -C "$DEPENDENCY_SRC/spconv" ls-files --others --exclude-standard
bootstrap_expect_eq spconv-patch-hash \
  6d398e709e73d770d17fdb6dce3c80aed4c56b7fb173ee1c5ba9029c01639cf3 \
  spconv_patch_hash_observed
bootstrap_expect_eq spconv-file-hash \
  e2c84544b5b5d6fd8e149d88539c3a6e989a1824637fd6b0006891955cb7a7e9 \
  spconv_file_hash_observed
test "$BOOTSTRAP_SEQUENCE" = 13
test "$(awk -F '\t' 'NR > 1 && $4 == "PASS" {n += 1} END {print n + 0}' \
  "$BOOTSTRAP_GATES")" = 13
printf '%s\n' '013|bootstrap|COMPLETE' > "$BOOTSTRAP_STAGE"

dependency_source_state() {
  repo=$1
  output=$2
  (
    cd "$repo"
    printf 'HEAD %s\n' "$(git rev-parse HEAD)"
    printf '%s\n' STATUS
    git status --porcelain --untracked-files=all
    printf '%s\n' TRACKED_DIFF
    git diff --no-color --binary --full-index --no-ext-diff --no-textconv HEAD --
    printf '%s\n' CHANGED_AND_UNTRACKED_RECORDS
    while IFS= read -r -d '' path; do
      test -f "$path"
      printf '%s  %s\n' "$(sha256sum "$path" | cut -d' ' -f1)" "$path"
    done < <(
      {
        git diff --name-only -z HEAD --
        git ls-files --others --exclude-standard -z
      } | LC_ALL=C sort -zu
    )
  ) > "$output"
}
dependency_source_state "$DEPENDENCY_SRC/cumm" \
  "$ARTIFACTS/cumm-source-state-before.txt"
dependency_source_state "$DEPENDENCY_SRC/spconv" \
  "$ARTIFACTS/spconv-source-state-before.txt"
DEPENDENCY_STATE_BASELINE_READY=1

export HOME="$JOB_ROOT/home"
export TMPDIR="$JOB_ROOT/tmp"
export XDG_CACHE_HOME="$JOB_ROOT/xdg"
export PYTHONPYCACHEPREFIX="$JOB_ROOT/pycache"
export MPLCONFIGDIR="$JOB_ROOT/matplotlib"
export RAY_TMPDIR="$JOB_ROOT/ray"
export WANDB_DIR="$JOB_ROOT/wandb"
export WANDB_MODE=disabled
export HF_HOME="$JOB_ROOT/xdg/huggingface"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export CUDA_CACHE_PATH="$JOB_ROOT/cuda_cache"
export NUMBA_CACHE_DIR="$JOB_ROOT/numba_cache"
export TRITON_CACHE_DIR="$JOB_ROOT/triton_cache"
export PYTHONNOUSERSITE=1
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
export PYTEST_ADDOPTS=
export PYTHONWARNINGS=error
export NUSCENES_DATAROOT="$MINI_ROOT"
unset ARRHENIUS_NUSCENES_DATAROOT NUSCENES_DATA_DIR
unset NUSCENES_ZIP_MANIFEST ARRHENIUS_NUSCENES_ZIP_MANIFEST
unset PYTHONHOME PYTHONPATH
export ARRHENIUS_ENV_ROOT="$JOB_ROOT/runtime"
export ARRHENIUS_VENV="$PERSISTENT_VENV"

source "$SNAPSHOT/fl_v3/scripts/arrhenius_env.sh"
arrhenius_load_modules build
arrhenius_activate_env
export PYTHONPATH="$SNAPSHOT/fl_v3/src"
export TMPDIR="$JOB_ROOT/tmp"
export XDG_CACHE_HOME="$JOB_ROOT/xdg"
export PYTHONPYCACHEPREFIX="$JOB_ROOT/pycache"
export TORCH_HOME="$JOB_ROOT/runtime/torch_home"
export TORCHINDUCTOR_CACHE_DIR="$JOB_ROOT/runtime/torchinductor_cache"
export CCACHE_DIR="$JOB_ROOT/runtime/ccache"
export SPCONV_DEBUG_SAVE_PATH="$JOB_ROOT/runtime/spconv_debug"
test "$(command -v python)" = "$PERSISTENT_VENV/bin/python"

scontrol show job "$SLURM_JOB_ID" -o > "$ARTIFACTS/scontrol.txt"
python - <<'PY' > "$ARTIFACTS/execution-identity.json"
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import platform
import sys
from urllib.parse import unquote, urlparse

expected = {
    "cumm": "0.7.13",
    "flwr": "1.27.0",
    "nuscenes-devkit": "1.1.11",
    "numpy": "1.26.4",
    "pillow": "12.2.0",
    "pyquaternion": "0.9.9",
    "pytest": "9.1.1",
    "ray": "2.51.1",
    "scikit-learn": "1.8.0",
    "spconv": "2.3.8",
    "torch": "2.11.0+cu128",
}
actual = {name: importlib.metadata.version(name) for name in expected}
if actual != expected:
    raise SystemExit(f"dependency version drift: {actual!r}")
if platform.machine() != "aarch64" or platform.python_version() != "3.11.15":
    raise SystemExit("platform/Python identity drift")

import torch
import fl_v3
from fl_v3.utils import runtime as runtime_identity

snapshot = Path(os.environ["PYTHONPATH"]).resolve()
fl_origin = Path(fl_v3.__file__).resolve()
if snapshot not in fl_origin.parents:
    raise SystemExit(f"fl_v3 import escaped immutable snapshot: {fl_origin}")
if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
    raise SystemExit("exactly one visible CUDA device is required")
torch.cuda.init()

torch_source = str(torch.version.git_version or "")
torch_meta = {
    "version": str(torch.__version__),
    "git_version": torch_source,
    "cuda": str(torch.version.cuda),
    "config": str(torch.__config__.show()),
}
torch_build, torch_origins = runtime_identity._runtime_build_identity(
    "torch", "torch", ("torch",), torch_meta
)
records = {
    "torch": runtime_identity._executable_artifact_records("torch", "torch")
}
sparse = {}
for label, targets, source_sha in (
    ("spconv", ("spconv", "spconv.pytorch"), "263d6b47425ef843c82f997b12d8b714013d216c"),
    ("cumm", ("cumm", "cumm.tensorview"), "4dedaf43ff801e417c60c6bd7536a29d83d29ee0"),
):
    if label == "cumm":
        head, origin = runtime_identity._source_checkout_identity(label, label)
    else:
        head = source_sha
        source_root = (Path(os.environ["DEPENDENCY_SRC"]) / label).resolve()
        distribution = importlib.metadata.distribution(label)
        direct = json.loads(distribution.read_text("direct_url.json") or "")
        parsed = urlparse(str(direct.get("url", "")))
        direct_root = Path(unquote(parsed.path)).resolve() if parsed.scheme == "file" else None
        if direct_root != source_root:
            raise SystemExit(
                f"spconv editable source drift: direct_url={direct_root}, expected={source_root}"
            )
        spec = importlib.util.find_spec(label)
        if spec is None or not spec.origin:
            raise SystemExit("cannot resolve installed spconv import origin")
        origin_path = Path(spec.origin).resolve()
        if source_root not in origin_path.parents:
            raise SystemExit(
                f"spconv import escaped accepted patched source: {origin_path}"
            )
        origin = str(origin_path)
    if head != source_sha:
        raise SystemExit(f"{label} source SHA drift: {head}")
    build, origins = runtime_identity._runtime_build_identity(
        label, label, targets, {"version": actual[label], "source_sha": head}
    )
    records[label] = runtime_identity._executable_artifact_records(label, label)
    sparse[label] = {
        "build_sha256": build,
        "import_origin": origin,
        "loaded_origins": origins,
        "source_sha": head,
    }

identity = {
    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
    "device_name": torch.cuda.get_device_name(0),
    "fl_v3_origin": str(fl_origin),
    "machine": platform.machine(),
    "packages": actual,
    "platform": platform.platform(),
    "python": platform.python_version(),
    "python_executable": sys.executable,
    "source_sha": os.environ["EXPECTED_SHA"],
    "source_tree": os.environ["EXPECTED_TREE"],
    "sparse": sparse,
    "torch_build_config": torch_meta["config"],
    "torch_build_config_sha256": hashlib.sha256(
        torch_meta["config"].encode("utf-8")
    ).hexdigest(),
    "torch_build_sha256": torch_build,
    "torch_cuda": torch.version.cuda,
    "torch_executable_artifacts": records["torch"],
    "torch_loaded_origins": torch_origins,
    "torch_source_sha": torch_source,
    "torch_version": torch.__version__,
}
for label in ("spconv", "cumm"):
    identity[f"{label}_executable_artifacts"] = records[label]
print(json.dumps(identity, indent=2, sort_keys=True))
PY

cd "$JOB_ROOT"
PYTEST_ARGS=(
  -c "$SNAPSHOT/fl_v3/pyproject.toml"
  --rootdir="$SNAPSHOT"
  --basetemp="$PYTEST_TMP"
  -p no:cacheprovider
  --strict-config
  --strict-markers
  -W error
  -s
  --junitxml="$JUNIT"
  "$SNAPSHOT/fl_v3/tests/test_s07_b_clean_completion.py"
  "$SNAPSHOT/fl_v3/tests/test_s07_b_integration.py"
  "$SNAPSHOT/fl_v3/tests/test_s07_b_data_lifecycle.py"
  "$SNAPSHOT/fl_v3/tests/test_s06_resolved_config.py"
  "$SNAPSHOT/fl_v3/tests/test_s06_model_modes.py"
  "$SNAPSHOT/fl_v3/tests/test_s06_training_runtime.py"
  "$SNAPSHOT/fl_v3/tests/test_s06_checkpoint_resume.py"
  "$SNAPSHOT/fl_v3/tests/test_s06_loader_eval.py"
  "$SNAPSHOT/fl_v3/tests/test_eval_provenance.py"
  "$SNAPSHOT/fl_v3/tests/test_flower_fp32_parity.py"
  "$SNAPSHOT/fl_v3/tests/test_flower_strategies_construct.py"
  "$SNAPSHOT/fl_v3/tests/test_fl_sampling.py"
  "$SNAPSHOT/fl_v3/tests/test_fl_round_smoke.py"
  "$SNAPSHOT/fl_v3/tests/test_fl_local_runner_multiround.py"
  "$SNAPSHOT/fl_v3/tests/test_fl_server_opt_integration.py"
  "$SNAPSHOT/fl_v3/tests/test_fl_trainable_only.py"
  "$SNAPSHOT/fl_v3/tests/test_nuscenes_info_cache.py"
  "$SNAPSHOT/fl_v3/tests/test_nuscenes_zip_backend.py"
  "$SNAPSHOT/fl_v3/tests/test_nuscenes_zip_dataset.py"
  "$SNAPSHOT/fl_v3/tests/test_nuscenes_partition.py::test_partition_seed_coercion"
  "$SNAPSHOT/fl_v3/tests/test_nuscenes_partition.py::test_stable_shards_same_inputs"
  "$SNAPSHOT/fl_v3/tests/test_nuscenes_partition.py::test_mini_is_degenerate_smoke"
  "$SNAPSHOT/fl_v3/tests/test_nuscenes_partition.py::test_iid_baseline_partition"
  "$SNAPSHOT/fl_v3/tests/test_nuscenes_partition.py::test_iid_partition_deterministic_and_seed_sensitive"
  "$SNAPSHOT/fl_v3/tests/test_nuscenes_partition.py::test_sub_floor_client_is_surfaced_not_silent"
  "$SNAPSHOT/fl_v3/tests/test_model_task.py::test_dummy_regression_byte_identity_golden"
  "$SNAPSHOT/fl_v3/tests/test_model_task.py::test_detection_task_registered"
  "$SNAPSHOT/fl_v3/tests/test_model_task.py::test_detection_config_rejects_legacy_model_mode_alias"
  "$SNAPSHOT/fl_v3/tests/test_model_task.py::test_num_clients_iid_is_requested"
  "$SNAPSHOT/fl_v3/tests/test_model_task.py::test_client_data_materializes_dict_batch"
  "$SNAPSHOT/fl_v3/tests/test_eval_box_to_global.py::test_rotmat_to_quaternion_matches_pyquaternion_and_roundtrips"
  "$SNAPSHOT/fl_v3/tests/test_eval_box_to_global.py::test_yaw_about_z_is_pure_yaw"
  "$SNAPSHOT/fl_v3/tests/test_eval_box_to_global.py::test_box_to_global_matches_raw_devkit_annotation"
  "$SNAPSHOT/fl_v3/tests/test_eval_detection_eval.py::test_assert_version_split"
  "$SNAPSHOT/fl_v3/tests/test_eval_detection_eval.py::test_submission_meta_uses_actual_mode"
  "$SNAPSHOT/fl_v3/tests/test_eval_detection_eval.py::test_results_dict_has_all_tokens_as_keys"
  "$SNAPSHOT/fl_v3/tests/test_eval_detection_eval.py::test_gt_as_pred_per_class_ap_near_one"
  --deselect=fl_v3/tests/test_nuscenes_zip_backend.py::test_parent_open_then_child_gets_process_owned_handles
  --deselect=fl_v3/tests/test_nuscenes_zip_dataset.py::test_repeated_persistent_multiworker_reads_are_deterministic
)

set +e
timeout --signal=TERM --kill-after=60s 50m \
  python -m pytest "${PYTEST_ARGS[@]}" 2>&1 | tee "$PYTEST_LOG"
pytest_rc=${PIPESTATUS[0]}
tee_rc=${PIPESTATUS[1]}
set -e
printf '%s\n' "$pytest_rc" > "$STATUS/pytest-exit.txt"
printf '%s\n' "$tee_rc" > "$STATUS/pytest-tee-exit.txt"

set +e
(
  set -e
  dependency_source_state "$DEPENDENCY_SRC/cumm" \
    "$ARTIFACTS/cumm-source-state-after.txt"
  dependency_source_state "$DEPENDENCY_SRC/spconv" \
    "$ARTIFACTS/spconv-source-state-after.txt"
)
source_state_capture_rc=$?
cmp "$ARTIFACTS/cumm-source-state-before.txt" \
  "$ARTIFACTS/cumm-source-state-after.txt"
cumm_state_cmp_rc=$?
cmp "$ARTIFACTS/spconv-source-state-before.txt" \
  "$ARTIFACTS/spconv-source-state-after.txt"
spconv_state_cmp_rc=$?
set -e
printf '%s\n' "$source_state_capture_rc" > "$STATUS/dependency-state-capture-exit.txt"
printf '%s\n' "$cumm_state_cmp_rc" > "$STATUS/cumm-state-cmp-exit.txt"
printf '%s\n' "$spconv_state_cmp_rc" > "$STATUS/spconv-state-cmp-exit.txt"

test "$pytest_rc" = 0 || exit "$pytest_rc"
test "$tee_rc" = 0 || exit 74
test "$source_state_capture_rc" = 0 || exit 75
test "$cumm_state_cmp_rc" = 0 || exit 76
test "$spconv_state_cmp_rc" = 0 || exit 77

python - "$JUNIT" "$PYTEST_LOG" > "$ARTIFACTS/acceptance-summary.json" <<'PY'
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

junit = Path(sys.argv[1])
log = Path(sys.argv[2])
root = ET.parse(junit).getroot()
suites = [root] if root.tag == "testsuite" else list(root.findall("testsuite"))
totals = {
    key: sum(int(suite.attrib.get(key, "0")) for suite in suites)
    for key in ("tests", "failures", "errors", "skipped")
}
if totals != {"tests": 205, "failures": 0, "errors": 0, "skipped": 0}:
    raise SystemExit(f"unexpected JUnit totals: {totals!r}")
prefix = "S07_B_CLEAN_MODE_EVIDENCE="
records = [
    json.loads(line.split(prefix, 1)[1])
    for line in log.read_text(encoding="utf-8").splitlines()
    if prefix in line
]
if [record["tag"] for record in records] != ["C-STR8", "L-S075", "F-U"]:
    raise SystemExit(f"unexpected mode evidence order/content: {records!r}")
for record in records:
    if not (
        record["batch_size"] == 1
        and record["num_workers"] == 0
        and record["precision"] == "fp16"
        and record["optimizer_steps"] == 1
        and record["exposure_samples"] == 1
        and record["last_grad_norm"] > 0
        and record["telemetry_interval"] == 1
        and record["grad_scaler_enabled"] is True
        and record["grad_scaler_skips"] == 0
        and record["nonfinite_loss_steps"] == 0
    ):
        raise SystemExit(f"invalid mode evidence: {record!r}")
print(json.dumps({"junit": totals, "mode_evidence": records}, indent=2, sort_keys=True))
PY

test -n "$(find "$JOB_ROOT/fl_outputs/nuscenes/info_cache" -type f -print -quit)"
PYTHONPATH="$SNAPSHOT/fl_v3/src" python - <<'PY'
from pathlib import Path
from fl_v3.data.nuscenes.paths import resolve_writable

actual = Path(resolve_writable("./fl_outputs/nuscenes/info_cache"))
expected = Path.cwd() / "fl_outputs" / "nuscenes" / "info_cache"
if actual != expected:
    raise SystemExit(f"mini cache escaped writable job root: {actual} != {expected}")
PY
S07B_JOB
)

WRAPPED="bash -lc $(printf '%q' "$JOB_BODY")"
set +e
sbatch \
  -A naiss2025-22-1113-gpu \
  -p gpu \
  --nodes=1 \
  --ntasks=1 \
  --gpus-per-node=nvidia_gh200_120gb:1 \
  --cpus-per-task=8 \
  --mem=96G \
  --time=01:00:00 \
  --no-requeue \
  --job-name=flv3_s07b_diag1 \
  --output="$JOB_ROOT/slurm-%j.out" \
  --error="$JOB_ROOT/slurm-%j.err" \
  --export=EXPECTED_SHA="$EXPECTED_SHA",EXPECTED_TREE="$EXPECTED_TREE",EXPECTED_SOURCE_AGG="$EXPECTED_SOURCE_AGG",JOB_ROOT="$JOB_ROOT",PROJECT_GIT_DIR="$PROJECT_GIT_DIR" \
  --wrap="$WRAPPED"
submit_rc=$?
set -e
if test "$submit_rc" != 0; then
  rmdir "$JOB_ROOT"
fi
exit "$submit_rc"
```

The pre-submission guard requires the SHA-derived job root not to exist, creates
it once, and the job body requires both expected Slurm log files. All later
writes are explicitly rooted there. `timeout` preserves pytest status through
`PIPESTATUS`; a nonzero pytest status is returned unchanged, a tee failure returns
74, and a successful pipeline with checksum failure returns 97. Finalization
always records original/final status and constructs/verifies hashes. The snapshot
is extracted from the exact Git commit then made non-writable before imports.

## Accepted intentional spconv build-patch provenance

Read-only login inspection on 2026-07-13 proves that the retained repository
builder `fl_v3/scripts/build_arrhenius_env.sh:102-117` installs pinned local cumm,
removes only `cumm>=0.7.11` from spconv v2.3.8's build-system requirements, then
installs spconv with `--no-build-isolation --no-deps`. The current checkout
matches that contract exactly:

```text
spconv tracked path set = pyproject.toml only
spconv full-index binary diff sha256 = 6d398e709e73d770d17fdb6dce3c80aed4c56b7fb173ee1c5ba9029c01639cf3
patched spconv pyproject.toml sha256 = e2c84544b5b5d6fd8e149d88539c3a6e989a1824637fd6b0006891955cb7a7e9
cumm tracked path set = empty
cumm untracked path set = cumm/core_cc/common.pyi only
cumm/core_cc/common.pyi observed sha256 = 656f8279c81e83f17f350be158c840d71ab973d7a7d893ec9d7b28a2a1847bfa
```

This known patch is accepted environment provenance, not runtime PASS, not an
environment mutation authorization, and not permission for any future tracked or
untracked source dirt. Before imports the command requires the exact heads,
staging state, path sets, diff bytes and patched-file hash. Its complete Git-state
records bind HEAD, porcelain status, full tracked diff, and SHA-256 for every
changed/untracked path; byte-identical records are required after pytest.

## Acceptance and stop conditions

Accept only one `COMPLETED 0:0` job with `Restarts=0`, the exact resources, exact
commit/tree/100 source records, exact dependencies/build identity, exact 205 cases,
and zero failure/error/skip/timeout/warning. C-STR8, L-S075, and F-U must each emit
one B=1/fp16 update record with finite positive production `last_grad_norm`, one
optimizer step, enabled GradScaler, zero scaler skip/nonfinite loss, and clean
TrainingState boundary. S06 checkpoint/save/load/resume/CUDA rollback; Flower
1.27 FP32 parity/deterministic sampling/plain FedAvg/no EMA; current DataLoader
workers 0 versus 2; S01 mini directory/ZIP/cache/partition; and exact official
evaluation identities must all pass. JUnit/log/status/source/archive/cache and
artifact checksums must exist and verify in-job.

On any dependency source-identity mismatch, nonzero status, timeout, warning-as-error,
missing/extra/skip case, worker abort, OOM, or walltime, preserve the negative
evidence and stop. No retry, resubmit, extra test, source edit, or replacement job
is implied.

Forbidden even later: trainval/full cache, 100/1000-step/tiny-overfit,
capability/mAP/NDS, profile/throughput, Ray live federation, DDP/multi-GPU,
actor/process/seed matrix, old S07-B harness, retry, attack/defense/ASR/Protocol,
upload, or publication.
