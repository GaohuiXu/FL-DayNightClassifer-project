# S07-B-COMPLETE RUN_REQUEST — draft bounded clean engineering gate

## Approval and immutable-materialization state

```text
SESSION_ID: S07-B-COMPLETE
APPROVAL_STATE: DRAFT_FOR_S00_OWNER_AUDIT / NOT REQUESTED / NOT APPROVED
APPROVED_COMPUTE: none
BASE_SHA: 4aa2b133d1d33382bf1514f7a3c86fcb03cf83e5
WORKER_SHA: 34cbe02b7b72114e3a2d61f6f797c8dec022798c
EXECUTABLE_SHA: 34cbe02b7b72114e3a2d61f6f797c8dec022798c
EXECUTABLE_TREE: ed2d4091f0098f6b2144028afd87e20d023b1da2
DELIVERY_REF: this docs-only handoff seal; full SHA supplied externally after commit
```

This file is text only and grants no `sbatch`/`srun` authority. The owner
authorized local durable materialization without changing candidate source bytes;
the exact executable commit and tree above are now Git objects. The command below
contains no materialization placeholder and archives that exact commit directly
from the Git common directory. S00 must recheck all identities and obtain a new
exact owner approval before submitting it. No repository launcher or compatibility
wrapper is authorized.

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

## Fully specified future submission template — text only, do not run

The template pins the durable executable commit/tree directly. It validates the
commit in the Git common directory, requires its sole parent to be BASE and its
exact two-path diff, creates one fresh SHA-derived job root, then submits exactly
one job. It does not depend on a mutable or temporary Codex worktree. A submission
failure removes the still-empty job root; any accepted job retains all evidence
and never retries.

```bash
#!/bin/bash
set -euo pipefail
export LC_ALL=C
export LANG=C

GIT_COMMON_DIR=/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/fl_weather_project/.git
BASE_SHA=4aa2b133d1d33382bf1514f7a3c86fcb03cf83e5
EXPECTED_SHA=34cbe02b7b72114e3a2d61f6f797c8dec022798c
EXPECTED_TREE=ed2d4091f0098f6b2144028afd87e20d023b1da2
EXPECTED_SOURCE_AGG=acb80014ff8dd3ef123e689b3be34efae219c95c95ea63f64c36e28f6d546a9e
OUTPUT_PARENT=/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs

[[ "$EXPECTED_SHA" =~ ^[0-9a-f]{40}$ ]]
[[ "$EXPECTED_TREE" =~ ^[0-9a-f]{40}$ ]]
test "$(git --git-dir="$GIT_COMMON_DIR" rev-parse "$EXPECTED_SHA")" = "$EXPECTED_SHA"
test "$(git --git-dir="$GIT_COMMON_DIR" rev-parse "$EXPECTED_SHA^{tree}")" = "$EXPECTED_TREE"
test "$(git --git-dir="$GIT_COMMON_DIR" rev-parse "$EXPECTED_SHA^")" = "$BASE_SHA"
test "$(git --git-dir="$GIT_COMMON_DIR" diff-tree --no-commit-id --name-only \
  -r "$EXPECTED_SHA" | LC_ALL=C sort)" = "$(printf '%s\n' \
  fl_v3/configs/flwr_config.toml \
  fl_v3/tests/test_s07_b_clean_completion.py)"

JOB_ROOT="$OUTPUT_PARENT/s07b_complete_${EXPECTED_SHA:0:12}"
test ! -e "$JOB_ROOT"
install -d -m 0700 "$JOB_ROOT"

JOB_BODY=$(cat <<'S07B_JOB'
set -euo pipefail
umask 077
export LC_ALL=C
export LANG=C

: "${SLURM_JOB_ID:?}"
: "${EXPECTED_SHA:?}"
: "${EXPECTED_TREE:?}"
: "${EXPECTED_SOURCE_AGG:?}"
: "${JOB_ROOT:?}"
: "${GIT_COMMON_DIR:?}"
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

git --git-dir="$GIT_COMMON_DIR" cat-file -e "$EXPECTED_SHA^{commit}"
test "$(git --git-dir="$GIT_COMMON_DIR" rev-parse "$EXPECTED_SHA^{tree}")" = "$EXPECTED_TREE"
git --git-dir="$GIT_COMMON_DIR" archive \
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
  git --git-dir="$GIT_COMMON_DIR" ls-tree -r --name-only "$EXPECTED_SHA" -- \
    fl_v3/src/fl_v3 | awk '/[.]py$/'
  git --git-dir="$GIT_COMMON_DIR" ls-tree -r --name-only "$EXPECTED_SHA" -- \
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
  git --git-dir="$GIT_COMMON_DIR" diff --binary \
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
test "$(realpath "$MINI_ROOT")" = "$MINI_ROOT"
test -d "$MINI_ROOT/v1.0-mini"
test -x "$PERSISTENT_VENV/bin/python"
test "$(git -C "$DEPENDENCY_SRC/cumm" rev-parse HEAD)" = \
  4dedaf43ff801e417c60c6bd7536a29d83d29ee0
test "$(git -C "$DEPENDENCY_SRC/spconv" rev-parse HEAD)" = \
  263d6b47425ef843c82f997b12d8b714013d216c

git -C "$DEPENDENCY_SRC/cumm" diff --cached --quiet
git -C "$DEPENDENCY_SRC/cumm" diff --quiet HEAD --
test "$(git -C "$DEPENDENCY_SRC/cumm" ls-files --others --exclude-standard)" = \
  cumm/core_cc/common.pyi

git -C "$DEPENDENCY_SRC/spconv" diff --cached --quiet
test "$(git -C "$DEPENDENCY_SRC/spconv" diff --name-only HEAD --)" = \
  pyproject.toml
test -z "$(git -C "$DEPENDENCY_SRC/spconv" ls-files --others --exclude-standard)"
test "$(git -C "$DEPENDENCY_SRC/spconv" diff --no-color --binary --full-index \
  --no-ext-diff --no-textconv HEAD -- pyproject.toml | sha256sum | cut -d' ' -f1)" = \
  6d398e709e73d770d17fdb6dce3c80aed4c56b7fb173ee1c5ba9029c01639cf3
test "$(sha256sum "$DEPENDENCY_SRC/spconv/pyproject.toml" | cut -d' ' -f1)" = \
  e2c84544b5b5d6fd8e149d88539c3a6e989a1824637fd6b0006891955cb7a7e9

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
  --job-name=flv3_s07b_complete \
  --output="$JOB_ROOT/slurm-%j.out" \
  --error="$JOB_ROOT/slurm-%j.err" \
  --export=EXPECTED_SHA="$EXPECTED_SHA",EXPECTED_TREE="$EXPECTED_TREE",EXPECTED_SOURCE_AGG="$EXPECTED_SOURCE_AGG",JOB_ROOT="$JOB_ROOT",GIT_COMMON_DIR="$GIT_COMMON_DIR" \
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
