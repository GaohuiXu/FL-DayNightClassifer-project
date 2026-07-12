#!/bin/bash
# Prepared S07-B post-remediation focused GH200 validation. This launcher is
# not execution authority: the exact tuple in RUN_REQUEST.md requires a
# separate S00 audit and one-time approval before submission.
#SBATCH -A naiss2025-22-1113-gpu
#SBATCH -p gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=nvidia_gh200_120gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:25:00
#SBATCH --no-requeue
#SBATCH --job-name=flv3_s07b_postrem_focus
#SBATCH --output=/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/logs/s07b_postrem_focus_%j.out
#SBATCH --error=/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/logs/s07b_postrem_focus_%j.err

set -euo pipefail

readonly MINI_ROOT=/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/fl_weather_project/data/nuscenes_mini
readonly REPO=/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/fl_weather_project
readonly SNAPSHOT_BASE=/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/execution_snapshots
readonly APPROVAL_SCOPE=s07b-postremediation-focused-runtime-only
readonly INTERNAL_TIMEOUT_SECONDS=180

readonly -a SELECTION_LABELS=(
  nuscenes_zip_dataset
  model_task
  lidar_backbone
  model_viz
  legacy_multitask_loss
)
readonly -a SELECTED_TESTS=(
  fl_v3/tests/test_nuscenes_zip_dataset.py
  fl_v3/tests/test_model_task.py
  fl_v3/tests/test_lidar_backbone.py
  fl_v3/tests/test_model_viz.py
  fl_v3/tests/test_s07_b_integration.py::test_multitask_loss_rejects_legacy_single_head_output
)

runtime_source_files() {
  {
    find fl_v3/src/fl_v3 -type f -name '*.py' -print
    printf '%s\n' \
      fl_v3/scripts/arrhenius_env.sh \
      fl_v3/scripts/run_s07_b_postremediation_focused.sh \
      fl_v3/tests/conftest.py \
      fl_v3/tests/test_nuscenes_zip_dataset.py \
      fl_v3/tests/test_model_task.py \
      fl_v3/tests/test_lidar_backbone.py \
      fl_v3/tests/test_model_viz.py \
      fl_v3/tests/test_s07_b_integration.py \
      fl_v3/pyproject.toml \
      fl_v3/requirements.txt \
      fl_v3/requirements.lock.txt
  } | LC_ALL=C sort -u
}

if [ "${1:-}" = "--print-source-files" ]; then
  runtime_source_files
  exit 0
fi
if [ "${1:-}" = "--print-tests" ]; then
  paste <(printf '%s\n' "${SELECTION_LABELS[@]}") <(printf '%s\n' "${SELECTED_TESTS[@]}")
  exit 0
fi

required=(
  EXPECTED_S07B_FOCUSED_CANDIDATE_SHA
  EXPECTED_S07B_FOCUSED_EXECUTABLE_SHA
  EXPECTED_S07B_FOCUSED_LAUNCHER_SHA256
  EXPECTED_S07B_FOCUSED_SOURCE_SHA256
  EXPECTED_S07B_FOCUSED_SOURCE_LIST_SHA256
  S07B_FOCUSED_MINI_DATAROOT
  S07B_FOCUSED_OUTPUT_ROOT
  S07B_FOCUSED_APPROVAL_SCOPE
)
for name in "${required[@]}"; do
  if [ -z "${!name:-}" ]; then
    echo "${name} is required" >&2
    exit 2
  fi
done

test "${S07B_FOCUSED_APPROVAL_SCOPE}" = "${APPROVAL_SCOPE}"
test "${S07B_FOCUSED_MINI_DATAROOT}" = "${MINI_ROOT}"
test -d "${MINI_ROOT}"
test "$(readlink -f "${S07B_FOCUSED_MINI_DATAROOT}")" = "$(readlink -f "${MINI_ROOT}")"
test "${SLURM_NNODES:-1}" = "1"
test "${SLURM_NTASKS:-1}" = "1"
test "${SLURM_CPUS_PER_TASK:-8}" = "8"
test "$(uname -m)" = "aarch64"
test "$(sha256sum "${BASH_SOURCE[0]}" | awk '{print $1}')" = "${EXPECTED_S07B_FOCUSED_LAUNCHER_SHA256}"

git -C "${REPO}" cat-file -e "${EXPECTED_S07B_FOCUSED_CANDIDATE_SHA}^{commit}"
git -C "${REPO}" cat-file -e "${EXPECTED_S07B_FOCUSED_EXECUTABLE_SHA}^{commit}"
test "$(git -C "${REPO}" rev-parse "${EXPECTED_S07B_FOCUSED_EXECUTABLE_SHA}^")" = \
  "${EXPECTED_S07B_FOCUSED_CANDIDATE_SHA}"
mapfile -t TRANSPORT_DIFF < <(
  git -C "${REPO}" diff --name-only \
    "${EXPECTED_S07B_FOCUSED_CANDIDATE_SHA}" \
    "${EXPECTED_S07B_FOCUSED_EXECUTABLE_SHA}"
)
test "${#TRANSPORT_DIFF[@]}" -eq 1
test "${TRANSPORT_DIFF[0]}" = "fl_v3/scripts/run_s07_b_postremediation_focused.sh"

readonly SHORT_EXECUTABLE="${EXPECTED_S07B_FOCUSED_EXECUTABLE_SHA:0:12}"
readonly SNAPSHOT="${SNAPSHOT_BASE}/s07b_postrem_focus_${SHORT_EXECUTABLE}"
readonly EXPECTED_OUTPUT_ROOT="/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s07b_postrem_focus_${SHORT_EXECUTABLE}"
test "${S07B_FOCUSED_OUTPUT_ROOT}" = "${EXPECTED_OUTPUT_ROOT}"
test ! -e "${S07B_FOCUSED_OUTPUT_ROOT}"
test ! -e "${SNAPSHOT}"
mkdir -p "${SNAPSHOT_BASE}"
mkdir "${SNAPSHOT}"
git -C "${REPO}" archive "${EXPECTED_S07B_FOCUSED_EXECUTABLE_SHA}" | tar -x -C "${SNAPSHOT}"
cd "${SNAPSHOT}"

mapfile -t SOURCE_FILES < <(runtime_source_files)
test "${#SELECTED_TESTS[@]}" -eq 5
test "${#SELECTION_LABELS[@]}" -eq 5
test "${#SOURCE_FILES[@]}" -gt 5
for path in "${SOURCE_FILES[@]}"; do
  test -f "${path}"
done
ACTUAL_SOURCE_LIST_SHA256="$(printf '%s\n' "${SOURCE_FILES[@]}" | sha256sum | awk '{print $1}')"
ACTUAL_SOURCE_SHA256="$(
  printf '%s\n' "${SOURCE_FILES[@]}" |
    while IFS= read -r path; do sha256sum "${path}"; done |
    sha256sum | awk '{print $1}'
)"
test "${ACTUAL_SOURCE_LIST_SHA256}" = "${EXPECTED_S07B_FOCUSED_SOURCE_LIST_SHA256}"
test "${ACTUAL_SOURCE_SHA256}" = "${EXPECTED_S07B_FOCUSED_SOURCE_SHA256}"
test "$(sha256sum fl_v3/scripts/run_s07_b_postremediation_focused.sh | awk '{print $1}')" = \
  "${EXPECTED_S07B_FOCUSED_LAUNCHER_SHA256}"

chmod -R a-w "${SNAPSHOT}"
mkdir "${S07B_FOCUSED_OUTPUT_ROOT}"
readonly TEST_ROOT="${S07B_FOCUSED_OUTPUT_ROOT}/tests"
readonly WORK_CWD="${S07B_FOCUSED_OUTPUT_ROOT}/work"
mkdir "${TEST_ROOT}" "${WORK_CWD}"
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
cleanup_job_tmp() {
  local status=$?
  local cleanup_status=0
  local current_identity=""
  trap - EXIT
  if [[ "${JOB_TMP}" =~ ${JOB_TMP_PATTERN} ]] &&
    [ "$(dirname -- "${JOB_TMP}")" = "/tmp" ] &&
    test -d "${JOB_TMP}" && test ! -L "${JOB_TMP}"; then
    current_identity="$(stat -c '%d:%i' "${JOB_TMP}")" || cleanup_status=$?
    if [ "${cleanup_status}" -eq 0 ] &&
      [ "${current_identity}" = "${JOB_TMP_IDENTITY}" ]; then
      rm -rf -- "${JOB_TMP}" || cleanup_status=$?
    else
      cleanup_status=1
    fi
  else
    cleanup_status=1
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
source "${SNAPSHOT}/fl_v3/scripts/arrhenius_env.sh"
arrhenius_load_modules build
arrhenius_activate_env

export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
unset PYTEST_ADDOPTS
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export WORLD_SIZE=1
export NUSCENES_DATAROOT="${MINI_ROOT}"
export ARRHENIUS_NUSCENES_DATAROOT="${MINI_ROOT}"
unset NUSCENES_DATA_DIR NUSCENES_ZIP_MANIFEST ARRHENIUS_NUSCENES_ZIP_MANIFEST
export TMPDIR="${JOB_TMP}"
export TMP="${JOB_TMP}"
export TEMP="${JOB_TMP}"
export PYTHONPATH="${SNAPSHOT}/fl_v3/src"

readonly SOURCE_LIST="${S07B_FOCUSED_OUTPUT_ROOT}/runtime_source_files.txt"
readonly SOURCE_HASHES="${S07B_FOCUSED_OUTPUT_ROOT}/runtime_source_sha256s.txt"
readonly SELECTIONS_TSV="${S07B_FOCUSED_OUTPUT_ROOT}/selected_pytest_entries.tsv"
readonly EXECUTION_JSON="${S07B_FOCUSED_OUTPUT_ROOT}/execution_identity.json"
readonly RUN_CONFIG="${S07B_FOCUSED_OUTPUT_ROOT}/focused_run_config.json"
readonly CONFIG_HASH="${S07B_FOCUSED_OUTPUT_ROOT}/focused_config_sha256.txt"
readonly SUMMARY_JSON="${S07B_FOCUSED_OUTPUT_ROOT}/focused_summary.json"
readonly CHECKSUMS="${S07B_FOCUSED_OUTPUT_ROOT}/sha256sums.txt"

printf '%s\n' "${SOURCE_FILES[@]}" > "${SOURCE_LIST}"
while IFS= read -r path; do sha256sum "${path}"; done < "${SOURCE_LIST}" > "${SOURCE_HASHES}"
paste <(printf '%s\n' "${SELECTION_LABELS[@]}") \
  <(printf '%s\n' "${SELECTED_TESTS[@]}") > "${SELECTIONS_TSV}"
test "$(sha256sum "${SOURCE_LIST}" | awk '{print $1}')" = "${EXPECTED_S07B_FOCUSED_SOURCE_LIST_SHA256}"
test "$(sha256sum "${SOURCE_HASHES}" | awk '{print $1}')" = "${EXPECTED_S07B_FOCUSED_SOURCE_SHA256}"

python - "${EXECUTION_JSON}" "${SELECTIONS_TSV}" <<'PY'
import importlib.metadata
import json
import os
import platform
import socket
import sys

import cumm
import spconv
import torch

output, selections_path = sys.argv[1:]
expected = {
    "numpy": "1.26.4",
    "scipy": "1.13.1",
    "pytest": "9.1.1",
    "torch": "2.11.0+cu128",
    "torchvision": "0.26.0+cu128",
    "spconv": "2.3.8",
    "cumm": "0.7.13",
    "nuscenes-devkit": "1.1.11",
    "pyquaternion": "0.9.9",
    "Pillow": "12.2.0",
}
actual = {name: importlib.metadata.version(name) for name in expected}
if actual != expected:
    raise SystemExit(f"dependency identity mismatch: expected={expected!r}, actual={actual!r}")
if platform.python_version() != "3.11.15":
    raise SystemExit(f"Python identity mismatch: {platform.python_version()}")
if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
    raise SystemExit("focused gate requires exactly one visible CUDA device")
with open(selections_path, encoding="utf-8") as stream:
    selections = [line.rstrip("\n").split("\t", 1) for line in stream]
if len(selections) != 5 or any(len(item) != 2 for item in selections):
    raise SystemExit("exactly five labelled pytest entries are required")
record = {
    "schema": "s07b.postremediation-focused.v1",
    "approval_scope": os.environ["S07B_FOCUSED_APPROVAL_SCOPE"],
    "candidate_sha": os.environ["EXPECTED_S07B_FOCUSED_CANDIDATE_SHA"],
    "executable_sha": os.environ["EXPECTED_S07B_FOCUSED_EXECUTABLE_SHA"],
    "launcher_sha256": os.environ["EXPECTED_S07B_FOCUSED_LAUNCHER_SHA256"],
    "runtime_source_sha256": os.environ["EXPECTED_S07B_FOCUSED_SOURCE_SHA256"],
    "runtime_source_list_sha256": os.environ["EXPECTED_S07B_FOCUSED_SOURCE_LIST_SHA256"],
    "host": socket.gethostname(),
    "machine": platform.machine(),
    "platform": platform.platform(),
    "python_executable": sys.executable,
    "python_implementation": platform.python_implementation(),
    "python_version": platform.python_version(),
    "torch_cuda_version": torch.version.cuda,
    "dependencies": actual,
    "selected_pytest_entries": selections,
    "mini_dataroot": os.environ["S07B_FOCUSED_MINI_DATAROOT"],
    "output_root": os.environ["S07B_FOCUSED_OUTPUT_ROOT"],
    "tmpdir": os.environ["TMPDIR"],
    "tmpdir_bytes": len(os.fsencode(os.environ["TMPDIR"])),
    "slurm": {
        "job_id": os.environ.get("SLURM_JOB_ID", ""),
        "job_name": os.environ.get("SLURM_JOB_NAME", ""),
        "nnodes": os.environ.get("SLURM_NNODES", ""),
        "ntasks": os.environ.get("SLURM_NTASKS", ""),
        "cpus_per_task": os.environ.get("SLURM_CPUS_PER_TASK", ""),
        "mem_per_node": os.environ.get("SLURM_MEM_PER_NODE", ""),
    },
    "pytest_disable_plugin_autoload": os.environ.get("PYTEST_DISABLE_PLUGIN_AUTOLOAD"),
    "pytest_addopts": os.environ.get("PYTEST_ADDOPTS", ""),
    "zip_and_full_data_overrides_cleared": all(
        not os.environ.get(name)
        for name in (
            "NUSCENES_DATA_DIR",
            "NUSCENES_ZIP_MANIFEST",
            "ARRHENIUS_NUSCENES_ZIP_MANIFEST",
        )
    ),
}
with open(output, "w", encoding="utf-8") as stream:
    json.dump(record, stream, indent=2, sort_keys=True)
    stream.write("\n")
PY

python - "${RUN_CONFIG}" "${SELECTIONS_TSV}" "${INTERNAL_TIMEOUT_SECONDS}" <<'PY'
import json
import sys

output, selections_path, timeout_seconds = sys.argv[1:]
with open(selections_path, encoding="utf-8") as stream:
    selections = [line.rstrip("\n").split("\t", 1) for line in stream]
record = {
    "schema": "s07b.postremediation-focused-config.v1",
    "selected_pytest_entries": selections,
    "pytest_flags": [
        "-vv", "--tb=long", "-p", "no:cacheprovider", "-o", "cache_dir=/dev/null"
    ],
    "per_entry_internal_timeout_seconds": int(timeout_seconds),
    "pytest_faulthandler": "python -X faulthandler",
    "snapshot_read_only": True,
    "output_cwd_writable": True,
    "mini_only": True,
    "forbidden": [
        "test_model_overfit.py",
        "full-25-file-suite",
        "full-cache-or-trainval",
        "100-or-1000-step",
        "metrics-or-profile",
        "DDP-or-matrix",
        "retry-or-requeue",
    ],
}
with open(output, "w", encoding="utf-8") as stream:
    json.dump(record, stream, indent=2, sort_keys=True)
    stream.write("\n")
PY
printf '%s  %s\n' \
  "$(sha256sum "${RUN_CONFIG}" | awk '{print $1}')" \
  "focused_run_config.json" > "${CONFIG_HASH}"

suite_failed=0
for index in "${!SELECTED_TESTS[@]}"; do
  label="${SELECTION_LABELS[${index}]}"
  selection="${SELECTED_TESTS[${index}]}"
  node_root="${TEST_ROOT}/${label}"
  mkdir "${node_root}" "${node_root}/basetemp"
  log="${node_root}/pytest.log"
  junit="${node_root}/pytest.junit.xml"
  exitcode_file="${node_root}/pytest.exitcode"
  set +e
  (
    cd "${WORK_CWD}"
    timeout --signal=TERM --kill-after=30s "${INTERNAL_TIMEOUT_SECONDS}s" \
      python -X faulthandler -m pytest \
        -vv --tb=long -p no:cacheprovider -o cache_dir=/dev/null \
        --basetemp="${node_root}/basetemp" \
        --junitxml="${junit}" \
        "${SNAPSHOT}/${selection}"
  ) > "${log}" 2>&1
  exitcode=$?
  set -e
  printf '%s\n' "${exitcode}" > "${exitcode_file}"
  if [ "${exitcode}" -ne 0 ]; then
    suite_failed=1
  fi
  (
    cd "${node_root}"
    for artifact in pytest.exitcode pytest.junit.xml pytest.log; do
      if [ -f "${artifact}" ]; then
        sha256sum "${artifact}"
      fi
    done > sha256sums.txt
    sha256sum -c sha256sums.txt
  )
done

set +e
python - "${SUMMARY_JSON}" "${SELECTIONS_TSV}" "${TEST_ROOT}" <<'PY'
import json
import pathlib
import sys
import xml.etree.ElementTree as ET

output, selections_path, test_root_text = sys.argv[1:]
test_root = pathlib.Path(test_root_text)
with open(selections_path, encoding="utf-8") as stream:
    selections = [line.rstrip("\n").split("\t", 1) for line in stream]
entries = []
for label, selection in selections:
    node_root = test_root / label
    exitcode = int((node_root / "pytest.exitcode").read_text(encoding="utf-8").strip())
    junit_path = node_root / "pytest.junit.xml"
    counts = None
    if junit_path.is_file():
        root = ET.parse(junit_path).getroot()
        suites = [root] if root.tag == "testsuite" else list(root.findall("testsuite"))
        counts = {
            key: sum(int(suite.attrib.get(key, "0")) for suite in suites)
            for key in ("tests", "failures", "errors", "skipped")
        }
    entries.append({
        "label": label,
        "selection": selection,
        "exitcode": exitcode,
        "timed_out": exitcode in (124, 137),
        "junit_present": junit_path.is_file(),
        "junit_counts": counts,
    })

def entry_passed(entry):
    counts = entry["junit_counts"]
    return (
        entry["exitcode"] == 0
        and not entry["timed_out"]
        and entry["junit_present"]
        and counts is not None
        and counts["tests"] > 0
        and counts["failures"] == 0
        and counts["errors"] == 0
        and counts["skipped"] == 0
    )

suite_pass = len(entries) == 5 and all(entry_passed(entry) for entry in entries)
record = {
    "schema": "s07b.postremediation-focused-summary.v1",
    "expected_selection_entries": 5,
    "observed_selection_entries": len(entries),
    "all_entries_collected_and_executed_positive": all(
        entry["junit_counts"] is not None and entry["junit_counts"]["tests"] > 0
        for entry in entries
    ),
    "suite_pass": suite_pass,
    "scheduler_success_is_not_suite_pass": True,
    "entries": entries,
    "totals": {
        key: sum((entry["junit_counts"] or {}).get(key, 0) for entry in entries)
        for key in ("tests", "failures", "errors", "skipped")
    },
}
with open(output, "w", encoding="utf-8") as stream:
    json.dump(record, stream, indent=2, sort_keys=True)
    stream.write("\n")
if not suite_pass:
    raise SystemExit(1)
PY
summary_exit=$?
set -e

find "${S07B_FOCUSED_OUTPUT_ROOT}" -type f \
  ! -path "${JOB_TMP}/*" \
  ! -path '*/basetemp/*' \
  ! -path "${CHECKSUMS}" \
  -printf '%P\n' | LC_ALL=C sort | while IFS= read -r path; do
    sha256sum "${S07B_FOCUSED_OUTPUT_ROOT}/${path}"
  done > "${CHECKSUMS}"
(
  cd "${S07B_FOCUSED_OUTPUT_ROOT}"
  sed "s#  ${S07B_FOCUSED_OUTPUT_ROOT}/#  #" "${CHECKSUMS}" > "${CHECKSUMS}.relative"
  mv "${CHECKSUMS}.relative" "${CHECKSUMS}"
  sha256sum -c "${CHECKSUMS}"
)

test "${suite_failed}" -eq 0
test "${summary_exit}" -eq 0
