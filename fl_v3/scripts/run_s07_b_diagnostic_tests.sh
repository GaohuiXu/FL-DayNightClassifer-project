#!/bin/bash
# Prepared S07-B diagnostic attribution launcher. This file is not execution
# authority: a separately audited exact RUN_REQUEST tuple requires approval.
# This is a distinct diagnostic after Job 348557, never an O-052 retry.
#SBATCH -A naiss2025-22-1113-gpu
#SBATCH -p gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=nvidia_gh200_120gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --no-requeue
#SBATCH --job-name=flv3_s07b_diagnostic
#SBATCH --output=/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/logs/s07b_diagnostic_%j.out
#SBATCH --error=/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/logs/s07b_diagnostic_%j.err

set -euo pipefail

readonly MINI_ROOT=/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/fl_weather_project/data/nuscenes_mini
readonly REPO=/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/fl_weather_project
readonly SNAPSHOT_BASE=/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/execution_snapshots
readonly APPROVAL_SCOPE=s07b-diagnostic-attribution-only
readonly PARENT_RUNTIME_LAUNCHER_SHA256=1b1c45d33b113d0c7d649e51b2ddf98a2d7822eab38d708d4bb0e223b8c334c0
readonly PARENT_NEGATIVE_RESULTS_COMMIT=d7888a9fef615c83c8d36161bfa6d581a3dc4f0f

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

readonly -a EXPECTED_CONFIG_SHA256S=(
  "d2eaa46c800ebea5927359398acd88b38d90219c2f1f3841a4b1897ed05f8cc6  fl_v3/configs/s07_b_c_str8.json"
  "bd8c57e84b34f835f3eaafe71f259a0c4131748bb27a62edf83bcd7f44bb54f0  fl_v3/configs/s07_b_f_cbgs.json"
  "df7f36fe28e0d0c6c8275b293318cf7fae2e3c71fe3c60b7a7b81c26af69fa2e  fl_v3/configs/s07_b_f_u.json"
  "625242234a03314010860e6026b0fbb88b774a9aeec12c7f7fe870203da07421  fl_v3/configs/s07_b_l_p020.json"
  "1658cd5ec0e9c1b8945646d2e23a8db4419d16c2f644ca5a99b94c3477dcce1d  fl_v3/configs/s07_b_l_s075.json"
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
      fl_v3/scripts/run_s07_b_diagnostic_tests.sh \
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
if [ "${1:-}" = "--print-tests" ]; then
  printf '%s\n' "${SELECTED_TESTS[@]}"
  exit 0
fi

required=(
  EXPECTED_S07B_DIAGNOSTIC_SHA
  EXPECTED_S07B_DIAGNOSTIC_LAUNCHER_SHA256
  EXPECTED_S07B_DIAGNOSTIC_SOURCE_SHA256
  EXPECTED_S07B_DIAGNOSTIC_SOURCE_LIST_SHA256
  S07B_DIAGNOSTIC_MINI_DATAROOT
  S07B_DIAGNOSTIC_OUTPUT_ROOT
  S07B_DIAGNOSTIC_APPROVAL_SCOPE
)
for name in "${required[@]}"; do
  if [ -z "${!name:-}" ]; then
    echo "${name} is required" >&2
    exit 2
  fi
done

test "$(sha256sum "${BASH_SOURCE[0]}" | awk '{print $1}')" = "${EXPECTED_S07B_DIAGNOSTIC_LAUNCHER_SHA256}"
test "${S07B_DIAGNOSTIC_APPROVAL_SCOPE}" = "${APPROVAL_SCOPE}"
test "${S07B_DIAGNOSTIC_MINI_DATAROOT}" = "${MINI_ROOT}"
test -d "${MINI_ROOT}"
test "$(readlink -f "${S07B_DIAGNOSTIC_MINI_DATAROOT}")" = "$(readlink -f "${MINI_ROOT}")"
test "${SLURM_NNODES:-1}" = "1"
test "${SLURM_NTASKS:-1}" = "1"
test "${SLURM_CPUS_PER_TASK:-8}" = "8"
test "$(uname -m)" = "aarch64"

readonly SHORT_EXECUTABLE="${EXPECTED_S07B_DIAGNOSTIC_SHA:0:12}"
readonly SNAPSHOT="${SNAPSHOT_BASE}/s07b_diagnostic_${SHORT_EXECUTABLE}"
readonly EXPECTED_OUTPUT_ROOT="/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s07b_diagnostic_${SHORT_EXECUTABLE}"
test "${S07B_DIAGNOSTIC_OUTPUT_ROOT}" = "${EXPECTED_OUTPUT_ROOT}"
test ! -e "${S07B_DIAGNOSTIC_OUTPUT_ROOT}"
test ! -e "${SNAPSHOT}"
git -C "${REPO}" cat-file -e "${EXPECTED_S07B_DIAGNOSTIC_SHA}^{commit}"
mkdir -p "${SNAPSHOT_BASE}"
mkdir "${SNAPSHOT}"
git -C "${REPO}" archive "${EXPECTED_S07B_DIAGNOSTIC_SHA}" | tar -x -C "${SNAPSHOT}"
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
test "${ACTUAL_SOURCE_LIST_SHA256}" = "${EXPECTED_S07B_DIAGNOSTIC_SOURCE_LIST_SHA256}"
test "${ACTUAL_SOURCE_SHA256}" = "${EXPECTED_S07B_DIAGNOSTIC_SOURCE_SHA256}"
test "$(sha256sum fl_v3/scripts/run_s07_b_diagnostic_tests.sh | awk '{print $1}')" = "${EXPECTED_S07B_DIAGNOSTIC_LAUNCHER_SHA256}"
test "$(sha256sum fl_v3/scripts/run_s07_b_runtime_tests.sh | awk '{print $1}')" = "${PARENT_RUNTIME_LAUNCHER_SHA256}"

chmod -R a-w "${SNAPSHOT}"
mkdir "${S07B_DIAGNOSTIC_OUTPUT_ROOT}"
readonly ISOLATED_ROOT="${S07B_DIAGNOSTIC_OUTPUT_ROOT}/isolated"
readonly COMBINED_ROOT="${S07B_DIAGNOSTIC_OUTPUT_ROOT}/combined"
mkdir "${ISOLATED_ROOT}" "${COMBINED_ROOT}"
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
readonly JOB_TMP_CLEANUP_TAG="S07B_TMP_CLEANUP_FAILURE:run_s07_b_diagnostic_tests"
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
mkdir "${JOB_TMP}/isolated"
test -d "${JOB_TMP}/isolated" -a -w "${JOB_TMP}/isolated"

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

readonly SOURCE_LIST="${S07B_DIAGNOSTIC_OUTPUT_ROOT}/runtime_source_files.txt"
readonly SOURCE_HASHES="${S07B_DIAGNOSTIC_OUTPUT_ROOT}/runtime_source_sha256s.txt"
readonly CONFIG_HASHES="${S07B_DIAGNOSTIC_OUTPUT_ROOT}/config_sha256s.txt"
readonly TEST_LIST="${S07B_DIAGNOSTIC_OUTPUT_ROOT}/selected_test_files.txt"
readonly EXECUTION_JSON="${S07B_DIAGNOSTIC_OUTPUT_ROOT}/execution_identity.json"
readonly ATTEMPT_MANIFEST="${S07B_DIAGNOSTIC_OUTPUT_ROOT}/isolated_attempts.tsv"
readonly SUMMARY_JSON="${S07B_DIAGNOSTIC_OUTPUT_ROOT}/diagnostic_summary.json"
readonly CHECKSUMS="${S07B_DIAGNOSTIC_OUTPUT_ROOT}/sha256sums.txt"

printf '%s\n' "${SOURCE_FILES[@]}" > "${SOURCE_LIST}"
while IFS= read -r path; do sha256sum "${path}"; done < "${SOURCE_LIST}" > "${SOURCE_HASHES}"
printf '%s\n' "${SELECTED_TESTS[@]}" > "${TEST_LIST}"
printf '%s\n' "${EXPECTED_CONFIG_SHA256S[@]}" > "${CONFIG_HASHES}"
test "$(sha256sum "${SOURCE_LIST}" | awk '{print $1}')" = "${EXPECTED_S07B_DIAGNOSTIC_SOURCE_LIST_SHA256}"
test "$(sha256sum "${SOURCE_HASHES}" | awk '{print $1}')" = "${EXPECTED_S07B_DIAGNOSTIC_SOURCE_SHA256}"
test "$(wc -l < "${CONFIG_HASHES}")" -eq 5
sha256sum -c "${CONFIG_HASHES}"

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
    raise SystemExit("S07-B diagnostic requires exactly one visible CUDA device")
dependencies = (
    "numpy", "scipy", "pytest", "torch", "torchvision", "spconv", "cumm",
    "nuscenes-devkit", "pyquaternion", "Pillow",
)
with open(test_list, encoding="utf-8") as stream:
    selected_tests = [line.strip() for line in stream if line.strip()]
record = {
    "schema": "s07b.diagnostic-attribution.v1",
    "approval_scope": os.environ["S07B_DIAGNOSTIC_APPROVAL_SCOPE"],
    "git_sha": os.environ["EXPECTED_S07B_DIAGNOSTIC_SHA"],
    "diagnostic_launcher_sha256": os.environ["EXPECTED_S07B_DIAGNOSTIC_LAUNCHER_SHA256"],
    "runtime_source_sha256": os.environ["EXPECTED_S07B_DIAGNOSTIC_SOURCE_SHA256"],
    "runtime_source_list_sha256": os.environ["EXPECTED_S07B_DIAGNOSTIC_SOURCE_LIST_SHA256"],
    "tmpdir": os.environ["TMPDIR"],
    "tmpdir_bytes": len(os.fsencode(os.environ["TMPDIR"])),
    "parent_runtime_launcher_sha256": "1b1c45d33b113d0c7d649e51b2ddf98a2d7822eab38d708d4bb0e223b8c334c0",
    "parent_negative_evidence": {
        "job_id": "348557",
        "result": "FAILED 1:0 / timeout 124 / 3F+4E / no JUnit",
        "results_commit": "d7888a9fef615c83c8d36161bfa6d581a3dc4f0f",
        "executable_sha": "05b733997968b8217e1fc6dd27c3a4add34f6c98",
        "source_list_sha256": "be3b9157e213b942094d290d403306aa714e82157e36ba92847e32cfef71419a",
        "source_state_sha256": "d8c6cc0e20ed0c8ded5a4e13dd3ae52f32a62ebbcfafd2f9cbcd469fc5b87acd",
        "output_root": "/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s07b_integrated_05b733997968",
    },
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
    "mini_dataroot": os.path.realpath(os.environ["S07B_DIAGNOSTIC_MINI_DATAROOT"]),
    "output_root": os.path.realpath(os.environ["S07B_DIAGNOSTIC_OUTPUT_ROOT"]),
    "snapshot": os.getcwd(),
    "selected_test_files": selected_tests,
    "isolated_timeout_seconds": 120,
    "combined_timeout_seconds": 600,
    "pytest_faulthandler_timeout_seconds": 60,
    "pytest_disable_plugin_autoload": os.environ.get("PYTEST_DISABLE_PLUGIN_AUTOLOAD", ""),
    "pytest_addopts": os.environ.get("PYTEST_ADDOPTS", ""),
    "diagnostic_only": True,
    "diagnostic_complete_does_not_imply_suite_pass": True,
    "optimizer_training_campaign_steps": 0,
}
with open(output, "w", encoding="utf-8") as stream:
    json.dump(record, stream, indent=2, sort_keys=True)
    stream.write("\n")
PY

printf 'index\ttest_file\tattempt_dir\tpytest_exitcode\ttee_exitcode\tlog\tjunit\n' > "${ATTEMPT_MANIFEST}"
isolated_index=0
for test_file in "${SELECTED_TESTS[@]}"; do
  isolated_index=$((isolated_index + 1))
  printf -v ordinal '%02d' "${isolated_index}"
  stem="$(basename "${test_file}" .py)"
  attempt_dir="${ISOLATED_ROOT}/${ordinal}_${stem}"
  basetemp="${JOB_TMP}/isolated/${ordinal}_${stem}"
  test -d "${JOB_TMP}/isolated" -a -w "${JOB_TMP}/isolated"
  log="${attempt_dir}/pytest.log"
  junit="${attempt_dir}/pytest.junit.xml"
  exitcode="${attempt_dir}/pytest.exitcode"
  tee_exitcode="${attempt_dir}/tee.exitcode"
  mkdir "${attempt_dir}"
  set +e
  timeout --signal=TERM --kill-after=15s 120s \
    python -m pytest -vv -ra --tb=long -p no:cacheprovider \
    -o faulthandler_timeout=60 \
    --basetemp="${basetemp}" --junitxml="${junit}" "${test_file}" \
    2>&1 | tee "${log}"
  pipeline_status=("${PIPESTATUS[@]}")
  set -e
  pytest_status="${pipeline_status[0]}"
  tee_status="${pipeline_status[1]}"
  printf '%s\n' "${pytest_status}" > "${exitcode}"
  printf '%s\n' "${tee_status}" > "${tee_exitcode}"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "${ordinal}" "${test_file}" "${attempt_dir}" "${pytest_status}" \
    "${tee_status}" "${log}" "${junit}" >> "${ATTEMPT_MANIFEST}"
done

readonly COMBINED_LOG="${COMBINED_ROOT}/pytest.log"
readonly COMBINED_JUNIT="${COMBINED_ROOT}/pytest.junit.xml"
readonly COMBINED_EXITCODE="${COMBINED_ROOT}/pytest.exitcode"
readonly COMBINED_TEE_EXITCODE="${COMBINED_ROOT}/tee.exitcode"
set +e
timeout --signal=TERM --kill-after=15s 600s \
  python -m pytest -vv -ra --tb=short -p no:cacheprovider \
  -o faulthandler_timeout=60 \
  --basetemp="${JOB_TMP}/combined" --junitxml="${COMBINED_JUNIT}" \
  "${SELECTED_TESTS[@]}" 2>&1 | tee "${COMBINED_LOG}"
combined_pipeline_status=("${PIPESTATUS[@]}")
set -e
combined_pytest_status="${combined_pipeline_status[0]}"
combined_tee_status="${combined_pipeline_status[1]}"
printf '%s\n' "${combined_pytest_status}" > "${COMBINED_EXITCODE}"
printf '%s\n' "${combined_tee_status}" > "${COMBINED_TEE_EXITCODE}"

python - \
  "${ATTEMPT_MANIFEST}" "${TEST_LIST}" "${COMBINED_LOG}" \
  "${COMBINED_JUNIT}" "${COMBINED_EXITCODE}" "${COMBINED_TEE_EXITCODE}" \
  "${SUMMARY_JSON}" <<'PY'
import csv
import json
import os
import sys
import xml.etree.ElementTree as ET

(
    manifest_path,
    test_list_path,
    combined_log,
    combined_junit,
    combined_exitcode,
    combined_tee_exitcode,
    output_path,
) = sys.argv[1:]


def read_status(path):
    try:
        with open(path, encoding="utf-8") as stream:
            return int(stream.read().strip())
    except (OSError, ValueError):
        return None


def junit_record(path):
    record = {"present": os.path.isfile(path), "counts": None, "parse_error": None}
    if not record["present"]:
        return record
    try:
        root = ET.parse(path).getroot()
        suites = [root] if root.tag == "testsuite" else list(root.findall("testsuite"))
        record["counts"] = {
            key: sum(int(suite.attrib.get(key, "0")) for suite in suites)
            for key in ("tests", "failures", "errors", "skipped")
        }
    except (OSError, ET.ParseError, TypeError, ValueError) as exc:
        record["parse_error"] = f"{type(exc).__name__}: {exc}"
    return record


with open(test_list_path, encoding="utf-8") as stream:
    expected_tests = [line.strip() for line in stream if line.strip()]
with open(manifest_path, encoding="utf-8", newline="") as stream:
    manifest_rows = list(csv.DictReader(stream, delimiter="\t"))

isolated = []
for row in manifest_rows:
    entry = {
        "index": row["index"],
        "test_file": row["test_file"],
        "attempt_dir": row["attempt_dir"],
        "pytest_exitcode": read_status(os.path.join(row["attempt_dir"], "pytest.exitcode")),
        "tee_exitcode": read_status(os.path.join(row["attempt_dir"], "tee.exitcode")),
        "log": row["log"],
        "log_present": os.path.isfile(row["log"]),
        "junit": row["junit"],
        "junit_record": junit_record(row["junit"]),
    }
    isolated.append(entry)

combined = {
    "pytest_exitcode": read_status(combined_exitcode),
    "tee_exitcode": read_status(combined_tee_exitcode),
    "log": combined_log,
    "log_present": os.path.isfile(combined_log),
    "junit": combined_junit,
    "junit_record": junit_record(combined_junit),
}

attempted_complete = (
    len(expected_tests) == 25
    and len(isolated) == 25
    and [entry["test_file"] for entry in isolated] == expected_tests
    and all(entry["pytest_exitcode"] is not None for entry in isolated)
    and combined["pytest_exitcode"] is not None
)
artifact_capture_complete = (
    all(entry["log_present"] and entry["tee_exitcode"] == 0 for entry in isolated)
    and combined["log_present"]
    and combined["tee_exitcode"] == 0
)


def attempt_pass(entry):
    junit = entry["junit_record"]
    counts = junit["counts"]
    return (
        entry["pytest_exitcode"] == 0
        and junit["present"]
        and junit["parse_error"] is None
        and counts is not None
        and counts["tests"] > 0
        and not any(counts[key] for key in ("failures", "errors", "skipped"))
    )


diagnostic_complete = attempted_complete and artifact_capture_complete
suite_pass = (
    diagnostic_complete
    and all(attempt_pass(entry) for entry in isolated)
    and attempt_pass(combined)
)
summary = {
    "schema": "s07b.diagnostic-summary.v1",
    "parent_negative_job": "348557",
    "isolated_timeout_seconds": 120,
    "combined_timeout_seconds": 600,
    "expected_isolated_attempts": 25,
    "isolated_attempts": isolated,
    "combined": combined,
    "attempted_complete": attempted_complete,
    "artifact_capture_complete": artifact_capture_complete,
    "diagnostic_complete": diagnostic_complete,
    "suite_pass": suite_pass,
    "semantic_guard": "diagnostic_complete does not imply suite_pass",
}
with open(output_path, "w", encoding="utf-8") as stream:
    json.dump(summary, stream, indent=2, sort_keys=True)
    stream.write("\n")
print(
    "[S07-B diagnostic] "
    f"attempted_complete={attempted_complete} "
    f"diagnostic_complete={diagnostic_complete} suite_pass={suite_pass}"
)
PY

cd "${S07B_DIAGNOSTIC_OUTPUT_ROOT}"
find . -path ./tmp -prune -o -type f ! -name sha256sums.txt -print0 \
  | LC_ALL=C sort -z | xargs -0 sha256sum > "${CHECKSUMS}"
test -s "${CHECKSUMS}"
sha256sum -c "${CHECKSUMS}"

python - "${SUMMARY_JSON}" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as stream:
    summary = json.load(stream)
if not summary.get("diagnostic_complete"):
    raise SystemExit("S07-B diagnostic harness incomplete")
print("[S07-B diagnostic] diagnostic_complete=true")
print(f"[S07-B diagnostic] suite_pass={str(summary['suite_pass']).lower()}")
print("[S07-B diagnostic] diagnostic_complete does not imply suite_pass")
PY

echo "[S07-B diagnostic] harness COMPLETE; inspect summary before any conclusion"
