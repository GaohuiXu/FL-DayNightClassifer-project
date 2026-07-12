#!/bin/bash
# Prepared S07-B bounded dummy-regression attribution launcher. This file is not
# execution authority: the exact immutable tuple in RUN_REQUEST.md requires a
# separate S00 approval. It is not a retry of Jobs 348557 or 348818.
#SBATCH -A naiss2025-22-1113-gpu
#SBATCH -p gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=nvidia_gh200_120gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:10:00
#SBATCH --no-requeue
#SBATCH --job-name=flv3_s07b_dummy_attr
#SBATCH --output=/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/logs/s07b_dummy_attr_%j.out
#SBATCH --error=/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/logs/s07b_dummy_attr_%j.err

set -euo pipefail

readonly REPO=/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/fl_weather_project
readonly SNAPSHOT_BASE=/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/execution_snapshots
readonly PRE_SHA=968d81583c87ba76b7dbbb722760f8eb8eb6cd39
readonly CURRENT_SHA=c69befe5e8dd6397059c4d3fe1cbf906a9646836
readonly APPROVAL_SCOPE=s07b-dummy-two-snapshot-attribution-only
readonly HISTORICAL_GOLDEN=d2d819fee9a54fc302a9d6c9d0ac4e4d875629a0a16e75f2328f28b7f63cd7cc
readonly JOB348818_CURRENT=4fa46307bab67f2a836102b23b1ad2abc331702e83d16c65e11a09330c3d9edb

snapshot_source_files() {
  {
    find fl_v3/src/fl_v3 -type f -name '*.py' -print
    printf '%s\n' \
      fl_v3/pyproject.toml \
      fl_v3/requirements.txt \
      fl_v3/requirements.lock.txt
  } | LC_ALL=C sort -u
}

if [ "${1:-}" = "--print-source-files" ]; then
  snapshot_source_files
  exit 0
fi

required=(
  EXPECTED_S07B_ATTR_EXECUTABLE_SHA
  EXPECTED_S07B_ATTR_LAUNCHER_SHA256
  EXPECTED_S07B_ATTR_ENV_SHA256
  EXPECTED_S07B_ATTR_PRE_SOURCE_LIST_SHA256
  EXPECTED_S07B_ATTR_PRE_SOURCE_SHA256
  EXPECTED_S07B_ATTR_CURRENT_SOURCE_LIST_SHA256
  EXPECTED_S07B_ATTR_CURRENT_SOURCE_SHA256
  S07B_ATTR_OUTPUT_ROOT
  S07B_ATTR_APPROVAL_SCOPE
)
for name in "${required[@]}"; do
  if [ -z "${!name:-}" ]; then
    echo "${name} is required" >&2
    exit 2
  fi
done

test "$(git rev-parse HEAD)" = "${EXPECTED_S07B_ATTR_EXECUTABLE_SHA}"
test -z "$(git branch --show-current)"
test -z "$(git status --short)"
test "$(sha256sum "${BASH_SOURCE[0]}" | awk '{print $1}')" = "${EXPECTED_S07B_ATTR_LAUNCHER_SHA256}"
test "$(sha256sum fl_v3/scripts/arrhenius_env.sh | awk '{print $1}')" = "${EXPECTED_S07B_ATTR_ENV_SHA256}"
test "${S07B_ATTR_APPROVAL_SCOPE}" = "${APPROVAL_SCOPE}"
test "${SLURM_NNODES:-1}" = "1"
test "${SLURM_NTASKS:-1}" = "1"
test "${SLURM_CPUS_PER_TASK:-4}" = "4"
test "$(uname -m)" = "aarch64"

readonly SHORT_EXECUTABLE="${EXPECTED_S07B_ATTR_EXECUTABLE_SHA:0:12}"
readonly EXPECTED_OUTPUT_ROOT="/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s07b_dummy_attr_${SHORT_EXECUTABLE}"
readonly PRE_SNAPSHOT="${SNAPSHOT_BASE}/s07b_dummy_attr_${SHORT_EXECUTABLE}_pre"
readonly CURRENT_SNAPSHOT="${SNAPSHOT_BASE}/s07b_dummy_attr_${SHORT_EXECUTABLE}_current"
test "${S07B_ATTR_OUTPUT_ROOT}" = "${EXPECTED_OUTPUT_ROOT}"
test ! -e "${S07B_ATTR_OUTPUT_ROOT}"
test ! -e "${PRE_SNAPSHOT}"
test ! -e "${CURRENT_SNAPSHOT}"
git -C "${REPO}" cat-file -e "${PRE_SHA}^{commit}"
git -C "${REPO}" cat-file -e "${CURRENT_SHA}^{commit}"

mkdir -p "${SNAPSHOT_BASE}"
mkdir "${PRE_SNAPSHOT}" "${CURRENT_SNAPSHOT}"
git -C "${REPO}" archive "${PRE_SHA}" | tar -x -C "${PRE_SNAPSHOT}"
git -C "${REPO}" archive "${CURRENT_SHA}" | tar -x -C "${CURRENT_SNAPSHOT}"
mkdir "${S07B_ATTR_OUTPUT_ROOT}"
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
readonly JOB_TMP_CLEANUP_TAG="S07B_TMP_CLEANUP_FAILURE:run_s07_b_dummy_attribution"
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

verify_snapshot() {
  local label="$1"
  local snapshot="$2"
  local expected_sha="$3"
  local expected_list_hash="$4"
  local expected_state_hash="$5"
  local list_file="${S07B_ATTR_OUTPUT_ROOT}/${label}_source_files.txt"
  local state_file="${S07B_ATTR_OUTPUT_ROOT}/${label}_source_sha256s.txt"
  local actual_list_hash actual_state_hash

  cd "${snapshot}"
  mapfile -t source_files < <(snapshot_source_files)
  test "${#source_files[@]}" -gt 3
  for path in "${source_files[@]}"; do
    test -f "${path}"
  done
  printf '%s\n' "${source_files[@]}" > "${list_file}"
  while IFS= read -r path; do sha256sum "${path}"; done < "${list_file}" > "${state_file}"
  actual_list_hash="$(sha256sum "${list_file}" | awk '{print $1}')"
  actual_state_hash="$(sha256sum "${state_file}" | awk '{print $1}')"
  test "${actual_list_hash}" = "${expected_list_hash}"
  test "${actual_state_hash}" = "${expected_state_hash}"
  printf '%s\n' \
    "label=${label}" \
    "commit=${expected_sha}" \
    "source_list_sha256=${actual_list_hash}" \
    "source_state_sha256=${actual_state_hash}" \
    "source_count=${#source_files[@]}" \
    > "${S07B_ATTR_OUTPUT_ROOT}/${label}_source_identity.txt"
}

verify_snapshot pre "${PRE_SNAPSHOT}" "${PRE_SHA}" \
  "${EXPECTED_S07B_ATTR_PRE_SOURCE_LIST_SHA256}" \
  "${EXPECTED_S07B_ATTR_PRE_SOURCE_SHA256}"
verify_snapshot current "${CURRENT_SNAPSHOT}" "${CURRENT_SHA}" \
  "${EXPECTED_S07B_ATTR_CURRENT_SOURCE_LIST_SHA256}" \
  "${EXPECTED_S07B_ATTR_CURRENT_SOURCE_SHA256}"

chmod -R a-w "${PRE_SNAPSHOT}" "${CURRENT_SNAPSHOT}"

# shellcheck disable=SC1091
source fl_v3/scripts/arrhenius_env.sh
arrhenius_load_modules build
arrhenius_activate_env

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export WORLD_SIZE=1
export TMPDIR="${JOB_TMP}"
export TMP="${JOB_TMP}"
export TEMP="${JOB_TMP}"
unset NUSCENES_DATAROOT ARRHENIUS_NUSCENES_DATAROOT NUSCENES_DATA_DIR
unset NUSCENES_ZIP_MANIFEST ARRHENIUS_NUSCENES_ZIP_MANIFEST

readonly IDENTITY_JSON="${S07B_ATTR_OUTPUT_ROOT}/execution_identity.json"
readonly ATTEMPTS_TSV="${S07B_ATTR_OUTPUT_ROOT}/attempts.tsv"
readonly SUMMARY_JSON="${S07B_ATTR_OUTPUT_ROOT}/attribution_summary.json"
readonly CHECKSUMS="${S07B_ATTR_OUTPUT_ROOT}/sha256sums.txt"

python - "${IDENTITY_JSON}" <<'PY'
import importlib.metadata
import json
import os
import platform
import socket
import sys

dependencies = (
    "numpy", "scipy", "torch", "torchvision", "spconv", "cumm",
    "nuscenes-devkit", "pyquaternion", "Pillow",
)
expected_versions = {
    "numpy": "1.26.4",
    "scipy": "1.13.1",
    "torch": "2.11.0+cu128",
    "torchvision": "0.26.0+cu128",
    "spconv": "2.3.8",
    "cumm": "0.7.13",
    "nuscenes-devkit": "1.1.11",
    "pyquaternion": "0.9.9",
    "Pillow": "12.2.0",
}
versions = {}
for name in dependencies:
    try:
        versions[name] = importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        versions[name] = None
if platform.python_version() != "3.11.15":
    raise SystemExit(f"unexpected Python version: {platform.python_version()}")
if versions != expected_versions:
    raise SystemExit(f"dependency identity mismatch: {versions!r}")
record = {
    "hostname": socket.gethostname(),
    "architecture": platform.machine(),
    "platform": platform.platform(),
    "python_executable": sys.executable,
    "python_implementation": platform.python_implementation(),
    "python_version": platform.python_version(),
    "dependencies": versions,
    "tmpdir": os.environ["TMPDIR"],
    "tmpdir_bytes": len(os.fsencode(os.environ["TMPDIR"])),
    "slurm": {
        key: os.environ.get(key)
        for key in (
            "SLURM_JOB_ID", "SLURM_JOB_NAME", "SLURM_JOB_NODELIST",
            "SLURM_NNODES", "SLURM_NTASKS", "SLURM_CPUS_PER_TASK",
        )
    },
}
with open(sys.argv[1], "w", encoding="utf-8") as stream:
    json.dump(record, stream, indent=2, sort_keys=True)
    stream.write("\n")
PY

printf 'label\tcommit\trepetition\texit\tchecksum\tjson\n' > "${ATTEMPTS_TSV}"
run_attempt() {
  local label="$1"
  local commit="$2"
  local snapshot="$3"
  local repetition="$4"
  local run_dir="${S07B_ATTR_OUTPUT_ROOT}/runs/${label}_${repetition}"
  local output_json="${run_dir}/result.json"
  local exit_file="${run_dir}/exit.txt"
  local checksum

  mkdir -p "${run_dir}"
  set +e
  (
    cd "${run_dir}"
    PYTHONPATH="${snapshot}/fl_v3/src" python - "${output_json}" "${label}" "${commit}" "${repetition}" <<'PY'
import importlib.metadata
import json
import platform
import sys

from fl_v3.engine.local_runner import run_clean_round

output, label, commit, repetition = sys.argv[1:]
config = {
    "task-type": "dummy_regression",
    "seed": 42,
    "device": "cpu",
    "num-clients": 4,
    "num-local-epochs": 1,
    "batch-size": 8,
    "learning-rate": 0.01,
    "weight-decay": 0.0,
    "num-workers": 0,
    "loss": "mse",
}
result = run_clean_round(config, defense="none", server_round=1)
dependencies = {}
for name in ("numpy", "torch"):
    try:
        dependencies[name] = importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        dependencies[name] = None
record = {
    "label": label,
    "commit": commit,
    "repetition": int(repetition),
    "config": config,
    "defense": "none",
    "server_round": 1,
    "agg_checksum": result["agg_checksum"],
    "decision_valid": bool(result["decision_valid"]),
    "n_clients": int(result["n_clients"]),
    "python_executable": sys.executable,
    "python_implementation": platform.python_implementation(),
    "python_version": platform.python_version(),
    "dependencies": dependencies,
}
with open(output, "w", encoding="utf-8") as stream:
    json.dump(record, stream, indent=2, sort_keys=True)
    stream.write("\n")
PY
  ) > "${run_dir}/stdout.log" 2> "${run_dir}/stderr.log"
  local rc=$?
  set -e
  printf '%s\n' "${rc}" > "${exit_file}"
  test "${rc}" -eq 0
  test -s "${output_json}"
  checksum="$(python - "${output_json}" <<'PY'
import json
import sys
with open(sys.argv[1], encoding="utf-8") as stream:
    record = json.load(stream)
value = record.get("agg_checksum")
if not isinstance(value, str) or len(value) != 64:
    raise SystemExit("invalid agg_checksum")
print(value)
PY
)"
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
    "${label}" "${commit}" "${repetition}" "${rc}" "${checksum}" \
    "runs/${label}_${repetition}/result.json" >> "${ATTEMPTS_TSV}"
}

for repetition in 1 2; do
  run_attempt pre "${PRE_SHA}" "${PRE_SNAPSHOT}" "${repetition}"
done
for repetition in 1 2; do
  run_attempt current "${CURRENT_SHA}" "${CURRENT_SNAPSHOT}" "${repetition}"
done

python - "${ATTEMPTS_TSV}" "${SUMMARY_JSON}" "${HISTORICAL_GOLDEN}" "${JOB348818_CURRENT}" <<'PY'
import csv
import json
import sys

attempts_path, output, historical, job_current = sys.argv[1:]
with open(attempts_path, newline="", encoding="utf-8") as stream:
    attempts = list(csv.DictReader(stream, delimiter="\t"))
if len(attempts) != 4:
    raise SystemExit(f"expected 4 attempts, got {len(attempts)}")
by_label = {"pre": [], "current": []}
for attempt in attempts:
    label = attempt["label"]
    if label not in by_label or attempt["exit"] != "0":
        raise SystemExit("invalid attempt record")
    by_label[label].append(attempt["checksum"])
if any(len(values) != 2 for values in by_label.values()):
    raise SystemExit("expected two attempts per snapshot")
stable = all(len(set(values)) == 1 for values in by_label.values())
pre = by_label["pre"][0]
current = by_label["current"][0]
if not stable:
    classification = "unstable"
elif pre == current == job_current:
    classification = "stable_equal_current"
elif pre == historical and current == job_current:
    classification = "pre_historical_current_new"
else:
    classification = "unexpected_stable_pair"
record = {
    "diagnostic_complete": True,
    "classification": classification,
    "pre_checksums": by_label["pre"],
    "current_checksums": by_label["current"],
    "historical_golden": historical,
    "job_348818_current_checksum": job_current,
    "automatic_code_or_golden_change_authorized": False,
}
with open(output, "w", encoding="utf-8") as stream:
    json.dump(record, stream, indent=2, sort_keys=True)
    stream.write("\n")
PY

find "${S07B_ATTR_OUTPUT_ROOT}" -type f ! -name 'sha256sums.txt' -printf '%P\n' \
  | LC_ALL=C sort \
  | while IFS= read -r path; do
      sha256sum "${S07B_ATTR_OUTPUT_ROOT}/${path}"
    done > "${CHECKSUMS}"
test "$(wc -l < "${CHECKSUMS}")" -gt 10
sha256sum -c "${CHECKSUMS}"

echo "S07-B dummy attribution complete; no code/golden decision is automatic"
