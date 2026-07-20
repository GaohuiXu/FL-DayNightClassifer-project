#!/bin/bash
# O-146 Envelope-A Job B: exact D_fit GTDB and LiDAR calibration only.
set -euo pipefail
umask 077

: "${S10_P1_ROOT:?required}"
: "${S10_P1_OUTPUT:?required}"
: "${S10_P1_EXPECTED_SOURCE_SHA:?required}"
: "${S10_P1_EXPECTED_TREE:?required}"
: "${S10_P1_EXPECTED_RUNNER_SHA256:?required}"
: "${S10_P1_EXPECTED_ENTRY_SHA256:?required}"
: "${S10_P1_EXPECTED_GTDB_ENTRY_SHA256:?required}"
: "${S10_P1_EXPECTED_CONFIG_FILE_SHA256:?required}"
: "${S10_P1_EXPECTED_RESOLVED_SHA256:?required}"

RUNNER_REL="fl_v3/scripts/run_s10_phase1_job_b.sh"
ENTRY_REL="fl_v3/scripts/s10_phase1_calibrate.py"
GTDB_ENTRY_REL="fl_v3/scripts/build_gt_database.py"
CONFIG_REL="fl_v3/configs/s10_phase1_lidar.json"
ZIP_MANIFEST="/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s01_zip_full_gate_v2_1fe651700bd0/nuscenes_trainval_zip_manifest.sqlite"
GTDB_ROOT="/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s10_phase1_envelope_a_data_e321aed749fd/gtdb_keyframe_d_fit"
GTDB_MANIFEST="${GTDB_ROOT}/manifest.json"
WORK="${S10_P1_OUTPUT}.control"

preflight_note() {
  printf '[s10-p1-lidar preflight] %s\n' "$1" >&2
}

preflight_fail() {
  printf '[s10-p1-lidar preflight] FAIL: %s\n' "$1" >&2
  exit 2
}

require_equal() {
  local label="$1" actual="$2" expected="$3"
  if [[ "${actual}" != "${expected}" ]]; then
    preflight_fail "${label}: actual=${actual@Q} expected=${expected@Q}"
  fi
}

require_absent() {
  local label="$1" path="$2"
  [[ ! -e "${path}" ]] || preflight_fail "${label} already exists: ${path}"
}

preflight_note "validate source, hashes, and fresh roots"
[[ -d "${S10_P1_ROOT}" ]] || preflight_fail "source root missing: ${S10_P1_ROOT}"
require_absent "output root" "${S10_P1_OUTPUT}"
require_absent "control root" "${WORK}"
actual_source_sha="$(git -C "${S10_P1_ROOT}" rev-parse HEAD)" || preflight_fail "cannot resolve source HEAD"
actual_source_tree="$(git -C "${S10_P1_ROOT}" rev-parse 'HEAD^{tree}')" || preflight_fail "cannot resolve source tree"
actual_branch="$(git -C "${S10_P1_ROOT}" branch --show-current)" || preflight_fail "cannot resolve source branch"
require_equal "source SHA" "${actual_source_sha}" "${S10_P1_EXPECTED_SOURCE_SHA}"
require_equal "source tree" "${actual_source_tree}" "${S10_P1_EXPECTED_TREE}"
require_equal "source branch" "${actual_branch}" "codex/s10-phase1-branch-qualification"
for hash_binding in \
  "runner|${RUNNER_REL}|${S10_P1_EXPECTED_RUNNER_SHA256}" \
  "calibration entry|${ENTRY_REL}|${S10_P1_EXPECTED_ENTRY_SHA256}" \
  "GTDB entry|${GTDB_ENTRY_REL}|${S10_P1_EXPECTED_GTDB_ENTRY_SHA256}" \
  "config|${CONFIG_REL}|${S10_P1_EXPECTED_CONFIG_FILE_SHA256}"; do
  IFS='|' read -r hash_label hash_relative hash_expected <<< "${hash_binding}"
  hash_actual="$(sha256sum "${S10_P1_ROOT}/${hash_relative}" | cut -d' ' -f1)" \
    || preflight_fail "cannot hash ${hash_label}"
  require_equal "${hash_label} SHA-256" "${hash_actual}" "${hash_expected}"
done
untracked="$(git -C "${S10_P1_ROOT}" ls-files --others --exclude-standard)" \
  || preflight_fail "cannot inspect untracked files"
[[ -z "${untracked}" ]] || preflight_fail "untracked source files: ${untracked}"
if ! git -C "${S10_P1_ROOT}" diff --quiet HEAD -- \
  fl_v3/src fl_v3/scripts fl_v3/configs fl_v3/tests pyproject.toml; then
  preflight_fail "tracked executable source differs from HEAD"
fi

preflight_note "source Arrhenius environment bootstrap"
# shellcheck disable=SC1091
source "${S10_P1_ROOT}/fl_v3/scripts/arrhenius_env.sh"
preflight_note "load Arrhenius build modules"
arrhenius_load_modules build
preflight_note "load nuScenes dataset module"
module load nuScenes-data/1.0-map-1.3-zip
preflight_note "activate persistent environment"
arrhenius_activate_env

export PYTHONPATH="${S10_P1_ROOT}/fl_v3/src"
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export PYTHONHASHSEED=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export WORLD_SIZE=1
export TORCH_HOME="/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/torch_home"
export NUSCENES_DATAROOT="${NUSCENES_DATA_DIR}"
export NUSCENES_ZIP_MANIFEST="${ZIP_MANIFEST}"

preflight_note "validate dataset and Slurm resource bindings"
require_equal "nuScenes dataroot" "${NUSCENES_DATAROOT}" "/dataset/easybuild/data/nuScenes-data/1.0-map-1.3-zip"
require_equal "Slurm partition" "${SLURM_JOB_PARTITION:-}" "gpu"
require_equal "Slurm CPUs per task" "${SLURM_CPUS_PER_TASK:-}" "16"
require_equal "Slurm memory per node" "${SLURM_MEM_PER_NODE:-}" "98304"
require_equal "Slurm GPUs on node" "${SLURM_GPUS_ON_NODE:-0}" "1"
preflight_note "PASS"
mkdir -p "${WORK}"
runner_complete=0

finalize() {
  local status="${1:-1}"
  local target="${S10_P1_OUTPUT}"
  if [[ ! -d "${target}" ]]; then
    mv "${WORK}" "${target}"
  fi
  printf '%s\n' "${status}" > "${target}/final.exit"
  local temporary="${target}/.runner_artifact_sha256s.tmp"
  find "${target}" -type f ! -name runner_artifact_sha256s.txt \
    ! -name .runner_artifact_sha256s.tmp -printf '%P\0' | sort -z | \
    while IFS= read -r -d '' relative; do
      sha256sum "${target}/${relative}" | sed "s#  ${target}/#  #"
    done > "${temporary}"
  mv "${temporary}" "${target}/runner_artifact_sha256s.txt"
  find "${target}" -type f -exec chmod 0444 {} +
  find "${target}" -type d -exec chmod 0555 {} +
}

handle_exit() {
  local status=$?
  trap - EXIT TERM INT HUP QUIT
  if (( status == 0 && runner_complete != 1 )); then status=125; fi
  finalize "${status}"
  exit "${status}"
}
trap 'exit 143' TERM
trap 'exit 130' INT
trap 'exit 129' HUP
trap 'exit 131' QUIT
trap handle_exit EXIT

cd "${S10_P1_ROOT}"
set +e
python -m pytest -q -p no:cacheprovider \
  fl_v3/tests/test_s10_phase1_config.py \
  fl_v3/tests/test_s10_phase1_shared_recipe.py \
  fl_v3/tests/test_s10_phase1_lidar.py \
  fl_v3/tests/test_build_gt_database.py \
  > "${WORK}/focused_tests.stdout" 2> "${WORK}/focused_tests.stderr"
test_status=$?
set -e
printf '%s\n' "${test_status}" > "${WORK}/focused_tests.exit"
if (( test_status != 0 )); then exit "${test_status}"; fi

if [[ ! -e "${GTDB_ROOT}" ]]; then
  python "${GTDB_ENTRY_REL}" --phase1-config "${CONFIG_REL}" \
    > "${WORK}/gtdb.stdout" 2> "${WORK}/gtdb.stderr"
elif [[ -f "${GTDB_MANIFEST}" ]]; then
  printf '%s\n' "reuse_existing_sealed_manifest" > "${WORK}/gtdb.stdout"
  : > "${WORK}/gtdb.stderr"
else
  echo "GTDB root exists without a sealed manifest" >&2
  exit 2
fi
test -s "${GTDB_MANIFEST}"

python "${ENTRY_REL}" \
  --branch lidar \
  --config "${CONFIG_REL}" \
  --output-dir "${WORK}/evidence" \
  --source-sha "${S10_P1_EXPECTED_SOURCE_SHA}" \
  --gtdb-manifest "${GTDB_MANIFEST}" \
  > "${WORK}/calibration.stdout" 2> "${WORK}/calibration.stderr"

RESULT="${WORK}/evidence/result.json"
jq -e --arg source "${S10_P1_EXPECTED_SOURCE_SHA}" --arg resolved "${S10_P1_EXPECTED_RESOLVED_SHA256}" '
  .schema == "s10.phase1.envelope-a-calibration.v1" and
  .status == "PASS" and .branch == "lidar" and
  .source.git_sha == $source and .source_config.sha256 == $resolved and
  .scope.optimizer_updates == 0 and
  .scope.capability_metrics_executed == false and
  .scope.D_select_executed == false and
  .scope.D_audit_executed == false and
  .scope.official_validation_executed == false and
  .branch_evidence.calibration.optimizer_updates == 0 and
  .evaluator_schema_preflight.precision_policy.sparse_collapse_dtype == "torch.float32" and
  .evaluator_schema_preflight.precision_policy.group_norm_modules == []
' "${RESULT}" >/dev/null
test -s "${WORK}/evidence/resolved_config.qualified.json"
runner_complete=1
