#!/usr/bin/env bash
# Prepared S06 remediation-2 bounded synthetic validation. Approval is mandatory.
#SBATCH -A naiss2025-22-1113-gpu
#SBATCH -p gpu
#SBATCH --gpus-per-node=nvidia_gh200_120gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH -J flv3_s06_runtime_r2
#SBATCH -o /nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/logs/s06_runtime_r2_%j.out
#SBATCH -e /nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/logs/s06_runtime_r2_%j.err

set -euo pipefail

: "${EXPECTED_S06_EXECUTABLE_SHA:?required}"
: "${EXPECTED_S06_SOURCE_SHA256:?required}"
: "${S06_OUTPUT_ROOT:?required}"
: "${S06_REQUEST_GENERATION:?required}"
test "${S06_REQUEST_GENERATION}" = "remediation-2"

REPO=/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/fl_weather_project
SNAPSHOT_ROOT=/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/execution_snapshots
SNAPSHOT="${SNAPSHOT_ROOT}/s06_runtime_remediation2_${EXPECTED_S06_EXECUTABLE_SHA:0:12}"
test ! -e "${S06_OUTPUT_ROOT}"
test ! -e "${SNAPSHOT}"
mkdir -p "${SNAPSHOT_ROOT}" "${S06_OUTPUT_ROOT}" "${SNAPSHOT}"
git --git-dir="${REPO}/.git" archive "${EXPECTED_S06_EXECUTABLE_SHA}" | tar -x -C "${SNAPSHOT}"
chmod -R a-w "${SNAPSHOT}"
cd "${SNAPSHOT}"

source fl_v3/scripts/arrhenius_env.sh
arrhenius_load_modules build
arrhenius_activate_env

SOURCE_LIST="${S06_OUTPUT_ROOT}/runtime_source_files.txt"
SOURCE_HASHES="${S06_OUTPUT_ROOT}/runtime_source_sha256s.txt"
{
find fl_v3/src/fl_v3/config fl_v3/src/fl_v3/training -type f -name '*.py' -print
printf '%s\n' \
  fl_v3/src/fl_v3/models/fusion/detector.py \
  fl_v3/src/fl_v3/eval/detection_eval.py \
  fl_v3/src/fl_v3/eval/provenance.py \
  fl_v3/src/fl_v3/utils/runtime.py \
  fl_v3/scripts/centralized_train.py \
  fl_v3/scripts/run_s06_runtime_tests.sh \
  fl_v3/tests/conftest.py \
  fl_v3/tests/test_s06_resolved_config.py \
  fl_v3/tests/test_s06_model_modes.py \
  fl_v3/tests/test_s06_training_runtime.py \
  fl_v3/tests/test_s06_checkpoint_resume.py \
  fl_v3/tests/test_s06_loader_eval.py \
  fl_v3/tests/test_model_task.py \
  fl_v3/tests/test_eval_detection_eval.py \
  fl_v3/tests/test_eval_provenance.py \
  fl_v3/tests/test_profiling_neutral.py \
  fl_v3/requirements.txt \
  fl_v3/requirements.lock.txt
} | LC_ALL=C sort -u > "${SOURCE_LIST}"
(while IFS= read -r path; do sha256sum "${path}"; done < "${SOURCE_LIST}") > "${SOURCE_HASHES}"
ACTUAL_SOURCE_SHA256=$(sha256sum "${SOURCE_HASHES}" | awk '{print $1}')
test "${ACTUAL_SOURCE_SHA256}" = "${EXPECTED_S06_SOURCE_SHA256}"

python - <<'PY' > "${S06_OUTPUT_ROOT}/execution_identity.json"
import importlib.metadata, json, os, platform, torch
print(json.dumps({
  "git_sha": os.environ["EXPECTED_S06_EXECUTABLE_SHA"],
  "runtime_source_sha256": os.environ["EXPECTED_S06_SOURCE_SHA256"],
  "request_generation": os.environ["S06_REQUEST_GENERATION"],
  "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
  "host": platform.node(), "machine": platform.machine(),
  "python": platform.python_version(), "torch": torch.__version__,
  "spconv": importlib.metadata.version("spconv"),
  "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
  "synthetic_only": True,
}, sort_keys=True))
PY

set +e
python -m pytest -q -p no:cacheprovider \
  fl_v3/tests/test_s06_resolved_config.py \
  fl_v3/tests/test_s06_model_modes.py \
  fl_v3/tests/test_s06_training_runtime.py \
  fl_v3/tests/test_s06_checkpoint_resume.py \
  fl_v3/tests/test_s06_loader_eval.py \
  fl_v3/tests/test_model_task.py::test_detection_config_rejects_legacy_model_mode_alias \
  fl_v3/tests/test_eval_detection_eval.py::test_submission_meta_uses_actual_mode \
  fl_v3/tests/test_eval_provenance.py::test_s06_provenance_binds_mode_config_checkpoint_and_data \
  fl_v3/tests/test_profiling_neutral.py \
  --junitxml="${S06_OUTPUT_ROOT}/pytest.junit.xml" \
  | tee "${S06_OUTPUT_ROOT}/pytest.log"
PIPE_STATUS=("${PIPESTATUS[@]}")
set -e
PYTEST_EXIT="${PIPE_STATUS[0]}"
TEE_EXIT="${PIPE_STATUS[1]}"
printf '%s\n' "${PYTEST_EXIT}" > "${S06_OUTPUT_ROOT}/pytest.exitcode"

sha256sum "${S06_OUTPUT_ROOT}/execution_identity.json" \
          "${SOURCE_LIST}" "${SOURCE_HASHES}" \
          "${S06_OUTPUT_ROOT}/pytest.log" "${S06_OUTPUT_ROOT}/pytest.junit.xml" \
          "${S06_OUTPUT_ROOT}/pytest.exitcode" \
          > "${S06_OUTPUT_ROOT}/sha256sums.txt"
sha256sum -c "${S06_OUTPUT_ROOT}/sha256sums.txt"
test "${TEE_EXIT}" -eq 0
exit "${PYTEST_EXIT}"
