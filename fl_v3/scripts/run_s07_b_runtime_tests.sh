#!/bin/bash
# Prepared S07-B integrated GH200 validation.  This launcher is NOT executable
# authority: submission requires the separately audited exact RUN_REQUEST tuple.
# It is owner-delegated S07-B validation scope, not generic O-009 authorization.
#SBATCH -A naiss2025-22-1113-gpu
#SBATCH -p gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=nvidia_gh200_120gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:45:00
#SBATCH --no-requeue
#SBATCH --job-name=flv3_s07b_integrated
#SBATCH --output=/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/logs/s07b_integrated_%j.out
#SBATCH --error=/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/logs/s07b_integrated_%j.err

set -euo pipefail

readonly MINI_ROOT=/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/fl_weather_project/data/nuscenes_mini
readonly REPO=/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/fl_weather_project
readonly SNAPSHOT_BASE=/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/execution_snapshots
readonly APPROVAL_SCOPE=owner-delegated-s07b-integrated-validation

readonly -a SELECTED_TESTS=(
  fl_v3/tests/test_s02_p0_correctness.py
  fl_v3/tests/test_s02_gpu_forward_backward.py
  fl_v3/tests/test_s03_camera_contract.py
  fl_v3/tests/test_s04_second_contract.py
  fl_v3/tests/test_s04_second_smoke.py
  fl_v3/tests/test_s04_fp16_eval_dispatch.py
  fl_v3/tests/test_s05_centerhead_decode.py
  fl_v3/tests/test_s05_eval_roundtrip.py
  fl_v3/tests/test_s05_nms.py
  fl_v3/tests/test_s06_checkpoint_resume.py
  fl_v3/tests/test_s06_loader_eval.py
  fl_v3/tests/test_s06_model_modes.py
  fl_v3/tests/test_s06_resolved_config.py
  fl_v3/tests/test_s06_training_runtime.py
  fl_v3/tests/test_s07_b_data_lifecycle.py
  fl_v3/tests/test_s07_b_integration.py
  fl_v3/tests/test_sparse_voxel_encoder.py
  fl_v3/tests/test_lidar_backbone.py
  fl_v3/tests/test_head_capacity.py
  fl_v3/tests/test_eval_box_to_global.py
  fl_v3/tests/test_eval_detection_eval.py
  fl_v3/tests/test_eval_provenance.py
  fl_v3/tests/test_model_task.py
  fl_v3/tests/test_profiling_neutral.py
  fl_v3/tests/test_nuscenes_zip_dataset.py
)

readonly -a S07_B_CONFIGS=(
  fl_v3/configs/s07_b_c_str8.json
  fl_v3/configs/s07_b_f_cbgs.json
  fl_v3/configs/s07_b_f_u.json
  fl_v3/configs/s07_b_l_p020.json
  fl_v3/configs/s07_b_l_s075.json
)

runtime_source_files() {
  {
    find fl_v3/src/fl_v3 -type f -name '*.py' -print
    printf '%s\n' \
      fl_v3/scripts/arrhenius_env.sh \
      fl_v3/scripts/centralized_train.py \
      fl_v3/scripts/arrhenius_mini_matrix.py \
      fl_v3/scripts/t4_readiness_eval.py \
      fl_v3/scripts/t5_attack_eval.py \
      fl_v3/scripts/t5_mini_smoke.py \
      fl_v3/scripts/run_s07_b_runtime_tests.sh \
      fl_v3/tests/conftest.py \
      fl_v3/pyproject.toml \
      fl_v3/requirements.txt \
      fl_v3/requirements.lock.txt
    printf '%s\n' "${SELECTED_TESTS[@]}"
    printf '%s\n' "${S07_B_CONFIGS[@]}"
  } | LC_ALL=C sort -u
}

if [ "${1:-}" = "--print-source-files" ]; then
  runtime_source_files
  exit 0
fi

required=(
  EXPECTED_S07B_EXECUTABLE_SHA
  EXPECTED_S07B_LAUNCHER_SHA256
  EXPECTED_S07B_SOURCE_SHA256
  EXPECTED_S07B_SOURCE_LIST_SHA256
  S07B_MINI_DATAROOT
  S07B_OUTPUT_ROOT
  S07B_APPROVAL_SCOPE
)
for name in "${required[@]}"; do
  if [ -z "${!name:-}" ]; then
    echo "${name} is required" >&2
    exit 2
  fi
done
test "$(sha256sum "${BASH_SOURCE[0]}" | awk '{print $1}')" = "${EXPECTED_S07B_LAUNCHER_SHA256}"
test "${S07B_APPROVAL_SCOPE}" = "${APPROVAL_SCOPE}"
test "${S07B_MINI_DATAROOT}" = "${MINI_ROOT}"
test -d "${MINI_ROOT}"
test "$(readlink -f "${S07B_MINI_DATAROOT}")" = "$(readlink -f "${MINI_ROOT}")"
test "${SLURM_NNODES:-1}" = "1"
test "${SLURM_NTASKS:-1}" = "1"
test "${SLURM_CPUS_PER_TASK:-8}" = "8"
test "$(uname -m)" = "aarch64"

readonly SHORT_EXECUTABLE="${EXPECTED_S07B_EXECUTABLE_SHA:0:12}"
readonly SNAPSHOT="${SNAPSHOT_BASE}/s07b_integrated_${SHORT_EXECUTABLE}"
readonly EXPECTED_OUTPUT_ROOT="/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s07b_integrated_${SHORT_EXECUTABLE}"
test "${S07B_OUTPUT_ROOT}" = "${EXPECTED_OUTPUT_ROOT}"
test ! -e "${S07B_OUTPUT_ROOT}"
test ! -e "${SNAPSHOT}"
git -C "${REPO}" cat-file -e "${EXPECTED_S07B_EXECUTABLE_SHA}^{commit}"
mkdir -p "${SNAPSHOT_BASE}"
mkdir "${SNAPSHOT}"
git -C "${REPO}" archive "${EXPECTED_S07B_EXECUTABLE_SHA}" | tar -x -C "${SNAPSHOT}"
cd "${SNAPSHOT}"

mapfile -t SOURCE_FILES < <(runtime_source_files)
test "${#SELECTED_TESTS[@]}" -eq 25
test "${#S07_B_CONFIGS[@]}" -eq 5
test "${#SOURCE_FILES[@]}" -gt 25
for path in "${SOURCE_FILES[@]}"; do
  test -f "${path}"
done
ACTUAL_SOURCE_LIST_SHA256="$(printf '%s\n' "${SOURCE_FILES[@]}" | sha256sum | awk '{print $1}')"
ACTUAL_SOURCE_SHA256="$(printf '%s\n' "${SOURCE_FILES[@]}" | while IFS= read -r path; do
  sha256sum "${path}"
done | sha256sum | awk '{print $1}')"
test "${ACTUAL_SOURCE_LIST_SHA256}" = "${EXPECTED_S07B_SOURCE_LIST_SHA256}"
test "${ACTUAL_SOURCE_SHA256}" = "${EXPECTED_S07B_SOURCE_SHA256}"

chmod -R a-w "${SNAPSHOT}"
mkdir "${S07B_OUTPUT_ROOT}"
[[ "${SLURM_JOB_ID:-}" =~ ^[0-9]+$ ]]
[[ "${SHORT_EXECUTABLE}" =~ ^[0-9a-f]{12}$ ]]
JOB_TMP_PREVIOUS_UMASK="$(umask)"
readonly JOB_TMP_PREVIOUS_UMASK
umask 077
JOB_TMP="$(mktemp -d -p /tmp "flv3-s07b-${SLURM_JOB_ID}-${SHORT_EXECUTABLE}.XXXXXX")"
umask "${JOB_TMP_PREVIOUS_UMASK}"
readonly JOB_TMP
JOB_TMP_IDENTITY="$(stat -c '%d:%i' "${JOB_TMP}")"
readonly JOB_TMP_IDENTITY
readonly JOB_TMP_PATTERN="^/tmp/flv3-s07b-${SLURM_JOB_ID}-${SHORT_EXECUTABLE}\\.[A-Za-z0-9]{6}$"
readonly JOB_TMP_CLEANUP_TAG="S07B_TMP_CLEANUP_FAILURE:run_s07_b_runtime_tests"
cleanup_job_tmp() {
  local status=$?
  local cleanup_status=0
  local current_identity=""
  trap - EXIT
  if ! [[ "${JOB_TMP}" =~ ${JOB_TMP_PATTERN} ]]; then
    printf '%s\n' "${JOB_TMP_CLEANUP_TAG} reason=path_pattern" >&2
    cleanup_status=1
  elif [ "$(dirname -- "${JOB_TMP}")" != "/tmp" ]; then
    printf '%s\n' "${JOB_TMP_CLEANUP_TAG} reason=dirname" >&2
    cleanup_status=1
  elif test -L "${JOB_TMP}"; then
    printf '%s\n' "${JOB_TMP_CLEANUP_TAG} reason=symlink" >&2
    cleanup_status=1
  elif ! test -d "${JOB_TMP}"; then
    printf '%s\n' "${JOB_TMP_CLEANUP_TAG} reason=directory" >&2
    cleanup_status=1
  elif ! current_identity="$(stat -c '%d:%i' "${JOB_TMP}" 2>/dev/null)"; then
    printf '%s\n' "${JOB_TMP_CLEANUP_TAG} reason=stat" >&2
    cleanup_status=1
  elif [ "${current_identity}" != "${JOB_TMP_IDENTITY}" ]; then
    printf '%s\n' "${JOB_TMP_CLEANUP_TAG} reason=device_inode" >&2
    cleanup_status=1
  elif ! rm -rf -- "${JOB_TMP}" 2>/dev/null; then
    printf '%s\n' "${JOB_TMP_CLEANUP_TAG} reason=rm" >&2
    cleanup_status=1
  else
    cleanup_status=0
  fi
  if [ "${status}" -eq 0 ] && [ "${cleanup_status}" -ne 0 ]; then
    status="${cleanup_status}"
  fi
  exit "${status}"
}
trap cleanup_job_tmp EXIT
test "${#JOB_TMP}" -le 48
[[ "${JOB_TMP}" =~ ${JOB_TMP_PATTERN} ]]
test "$(dirname -- "${JOB_TMP}")" = "/tmp"
test -d "${JOB_TMP}"
test ! -L "${JOB_TMP}"
test "$(stat -c '%a' "${JOB_TMP}")" = "700"

# shellcheck disable=SC1091
source fl_v3/scripts/arrhenius_env.sh
arrhenius_load_modules build
arrhenius_activate_env

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export WORLD_SIZE=1
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
unset PYTEST_ADDOPTS
export NUSCENES_DATAROOT="${MINI_ROOT}"
export ARRHENIUS_NUSCENES_DATAROOT="${MINI_ROOT}"
unset NUSCENES_DATA_DIR NUSCENES_ZIP_MANIFEST ARRHENIUS_NUSCENES_ZIP_MANIFEST
export TMPDIR="${JOB_TMP}"
export TMP="${JOB_TMP}"
export TEMP="${JOB_TMP}"

readonly SOURCE_LIST="${S07B_OUTPUT_ROOT}/runtime_source_files.txt"
readonly SOURCE_HASHES="${S07B_OUTPUT_ROOT}/runtime_source_sha256s.txt"
readonly CONFIG_HASHES="${S07B_OUTPUT_ROOT}/config_sha256s.txt"
readonly TEST_LIST="${S07B_OUTPUT_ROOT}/selected_test_files.txt"
readonly EXECUTION_JSON="${S07B_OUTPUT_ROOT}/execution_identity.json"
readonly PYTEST_LOG="${S07B_OUTPUT_ROOT}/pytest.log"
readonly JUNIT_XML="${S07B_OUTPUT_ROOT}/pytest.junit.xml"
readonly JUNIT_COUNTS="${S07B_OUTPUT_ROOT}/pytest_junit_counts.json"
readonly EXITCODE="${S07B_OUTPUT_ROOT}/pytest.exitcode"

printf '%s\n' "${SOURCE_FILES[@]}" > "${SOURCE_LIST}"
while IFS= read -r path; do sha256sum "${path}"; done < "${SOURCE_LIST}" > "${SOURCE_HASHES}"
printf '%s\n' "${SELECTED_TESTS[@]}" > "${TEST_LIST}"
sha256sum "${S07_B_CONFIGS[@]}" > "${CONFIG_HASHES}"
test "$(sha256sum "${SOURCE_LIST}" | awk '{print $1}')" = "${EXPECTED_S07B_SOURCE_LIST_SHA256}"
test "$(sha256sum "${SOURCE_HASHES}" | awk '{print $1}')" = "${EXPECTED_S07B_SOURCE_SHA256}"
test "$(wc -l < "${CONFIG_HASHES}")" -eq 5

python - "${EXECUTION_JSON}" "${TEST_LIST}" <<'PY'
import importlib.metadata
import json
import os
import platform
import socket
import sys

import cumm
import spconv
import torch

output, test_list = sys.argv[1:]
if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
    raise SystemExit("S07-B bounded gate requires exactly one visible CUDA device")
dependencies = (
    "numpy", "scipy", "pytest", "torch", "torchvision", "spconv", "cumm",
    "nuscenes-devkit", "pyquaternion", "Pillow",
)
with open(test_list, encoding="utf-8") as stream:
    selected_tests = [line.strip() for line in stream if line.strip()]
record = {
    "schema": "s07b.integrated-bounded-runtime.v1",
    "approval_scope": os.environ["S07B_APPROVAL_SCOPE"],
    "git_sha": os.environ["EXPECTED_S07B_EXECUTABLE_SHA"],
    "launcher_sha256": os.environ["EXPECTED_S07B_LAUNCHER_SHA256"],
    "runtime_source_sha256": os.environ["EXPECTED_S07B_SOURCE_SHA256"],
    "runtime_source_list_sha256": os.environ["EXPECTED_S07B_SOURCE_LIST_SHA256"],
    "tmpdir": os.environ["TMPDIR"],
    "tmpdir_bytes": len(os.fsencode(os.environ["TMPDIR"])),
    "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
    "slurm_job_name": os.environ.get("SLURM_JOB_NAME", ""),
    "slurm_nnodes": os.environ.get("SLURM_NNODES", ""),
    "slurm_ntasks": os.environ.get("SLURM_NTASKS", ""),
    "slurm_cpus_per_task": os.environ.get("SLURM_CPUS_PER_TASK", ""),
    "slurm_mem_per_node": os.environ.get("SLURM_MEM_PER_NODE", ""),
    "host": socket.gethostname(),
    "machine": platform.machine(),
    "platform": platform.platform(),
    "python_executable": sys.executable,
    "python_implementation": platform.python_implementation(),
    "python_version": platform.python_version(),
    "torch_version": torch.__version__,
    "torch_git_version": torch.version.git_version,
    "torch_cuda_version": torch.version.cuda,
    "torch_build_config": torch.__config__.show(),
    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
    "cuda_device_count": torch.cuda.device_count(),
    "cuda_device_name": torch.cuda.get_device_name(0),
    "cuda_capability": list(torch.cuda.get_device_capability(0)),
    "cudnn_version": torch.backends.cudnn.version(),
    "spconv_module_version": getattr(spconv, "__version__", ""),
    "spconv_module_file": getattr(spconv, "__file__", ""),
    "cumm_module_version": getattr(cumm, "__version__", ""),
    "cumm_module_file": getattr(cumm, "__file__", ""),
    "dependency_versions": {
        name: importlib.metadata.version(name) for name in dependencies
    },
    "mini_dataroot": os.path.realpath(os.environ["S07B_MINI_DATAROOT"]),
    "output_root": os.path.realpath(os.environ["S07B_OUTPUT_ROOT"]),
    "snapshot": os.getcwd(),
    "selected_test_files": selected_tests,
    "pytest_disable_plugin_autoload": os.environ.get("PYTEST_DISABLE_PLUGIN_AUTOLOAD", ""),
    "pytest_addopts": os.environ.get("PYTEST_ADDOPTS", ""),
    "bounded_validation_only": True,
    "optimizer_training_campaign_steps": 0,
}
with open(output, "w", encoding="utf-8") as stream:
    json.dump(record, stream, indent=2, sort_keys=True)
    stream.write("\n")
PY

set +e
timeout --signal=TERM --kill-after=30s 42m \
  python -m pytest -q -ra -p no:cacheprovider \
  --basetemp="${JOB_TMP}/pytest" \
  "${SELECTED_TESTS[@]}" \
  --junitxml="${JUNIT_XML}" 2>&1 | tee "${PYTEST_LOG}"
pipeline_status=("${PIPESTATUS[@]}")
set -e
pytest_status="${pipeline_status[0]}"
tee_status="${pipeline_status[1]}"
printf '%s\n' "${pytest_status}" > "${EXITCODE}"

test -f "${JUNIT_XML}"
python - "${JUNIT_XML}" "${JUNIT_COUNTS}" <<'PY'
import json
import sys
import xml.etree.ElementTree as ET

root = ET.parse(sys.argv[1]).getroot()
suites = [root] if root.tag == "testsuite" else list(root.findall("testsuite"))
counts = {
    key: sum(int(suite.attrib.get(key, "0")) for suite in suites)
    for key in ("tests", "failures", "errors", "skipped")
}
with open(sys.argv[2], "w", encoding="utf-8") as stream:
    json.dump(counts, stream, sort_keys=True)
    stream.write("\n")
print(f"[S07-B] JUnit counts: {counts}")
PY

sha256sum \
  "${EXECUTION_JSON}" "${SOURCE_LIST}" "${SOURCE_HASHES}" "${CONFIG_HASHES}" \
  "${TEST_LIST}" "${PYTEST_LOG}" "${JUNIT_XML}" "${JUNIT_COUNTS}" "${EXITCODE}" \
  > "${S07B_OUTPUT_ROOT}/sha256sums.txt"
sha256sum -c "${S07B_OUTPUT_ROOT}/sha256sums.txt"
python - "${JUNIT_COUNTS}" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as stream:
    counts = json.load(stream)
if counts["tests"] <= 0 or any(counts[key] for key in ("failures", "errors", "skipped")):
    raise SystemExit(f"S07-B JUnit acceptance failed: {counts}")
print(f"[S07-B] JUnit acceptance passed: {counts}")
PY
test "${tee_status}" -eq 0
test "${pytest_status}" -eq 0

echo "[S07-B] bounded integrated validation PASS"
echo "[S07-B] sha=${EXPECTED_S07B_EXECUTABLE_SHA} source=${EXPECTED_S07B_SOURCE_SHA256}"
