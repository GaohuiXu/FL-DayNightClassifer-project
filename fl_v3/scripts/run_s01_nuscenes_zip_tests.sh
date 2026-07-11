#!/bin/bash
# Dependency-complete S01 directory/ZIP parity + lifecycle regression gate.
# Submission requires an exact owner-approved RUN_REQUEST bound to SHA/state/output.
#SBATCH -A naiss2025-22-1113-gpu
#SBATCH -p gpu
#SBATCH --job-name=flv3_s01_zip_tests
#SBATCH --gpus-per-node=nvidia_gh200_120gb:1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:20:00
#SBATCH --output=/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/logs/s01_zip_tests_%j.out
#SBATCH --error=/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/logs/s01_zip_tests_%j.err
set -euo pipefail

if [ -z "${EXPECTED_S01_SHA:-}" ] || [ -z "${EXPECTED_S01_STATE_HASH:-}" ] || \
   [ -z "${S01_OUTPUT_ROOT:-}" ] || [ -z "${S01_MINI_DATAROOT:-}" ]; then
  echo "EXPECTED_S01_SHA, EXPECTED_S01_STATE_HASH, S01_OUTPUT_ROOT, and S01_MINI_DATAROOT are required" >&2
  exit 2
fi
if [ -e "${S01_OUTPUT_ROOT}" ]; then
  echo "Refusing to reuse/overwrite S01_OUTPUT_ROOT=${S01_OUTPUT_ROOT}" >&2
  exit 2
fi

REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "${REPO}"
ACTUAL_SHA="$(git rev-parse HEAD)"
if [ "${ACTUAL_SHA}" != "${EXPECTED_S01_SHA}" ]; then
  echo "SHA mismatch: expected=${EXPECTED_S01_SHA} actual=${ACTUAL_SHA}" >&2
  exit 2
fi
runtime_source_files() {
  {
    find fl_v3/src/fl_v3/data/nuscenes -type f ! -path '*/__pycache__/*'
    printf '%s\n' \
      fl_v3/scripts/build_gt_database.py \
      fl_v3/tests/conftest.py \
      fl_v3/tests/test_build_gt_database.py \
      fl_v3/tests/test_nuscenes_zip_backend.py \
      fl_v3/tests/test_nuscenes_zip_dataset.py \
      fl_v3/tests/test_nuscenes_zip_info_cache.py \
      fl_v3/tests/test_nuscenes_info_cache.py \
      fl_v3/pyproject.toml \
      fl_v3/requirements.txt \
      fl_v3/requirements.lock.txt \
      fl_v3/scripts/run_s01_nuscenes_zip_tests.sh
  } | sort -u
}
S01_STATE_HASH="$(runtime_source_files | while IFS= read -r path; do
  sha256sum "${path}"
done | sha256sum | awk '{print $1}')"
if [ "${S01_STATE_HASH}" != "${EXPECTED_S01_STATE_HASH}" ]; then
  echo "S01 execution-state hash mismatch: expected=${EXPECTED_S01_STATE_HASH} actual=${S01_STATE_HASH}" >&2
  exit 2
fi

# shellcheck disable=SC1091
source fl_v3/scripts/arrhenius_env.sh
arrhenius_load_modules build
arrhenius_activate_env

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
unset PYTEST_ADDOPTS
export NUSCENES_DATAROOT="${S01_MINI_DATAROOT}"
export ARRHENIUS_NUSCENES_DATAROOT="${S01_MINI_DATAROOT}"
unset NUSCENES_ZIP_MANIFEST ARRHENIUS_NUSCENES_ZIP_MANIFEST NUSCENES_DATA_DIR

python - "${S01_OUTPUT_ROOT}" "${S01_MINI_DATAROOT}" <<'PY'
import sys
from fl_v3.data.nuscenes import paths as P
P.resolve_writable(sys.argv[1], sys.argv[2])
P.verify_dataset("v1.0-mini", sys.argv[2])
PY
mkdir -p "${S01_OUTPUT_ROOT}"
SOURCE_HASHES="${S01_OUTPUT_ROOT}/runtime_source_sha256s.txt"
EXECUTION_JSON="${S01_OUTPUT_ROOT}/execution_identity.json"
PYTEST_LOG="${S01_OUTPUT_ROOT}/pytest.log"
JUNIT_XML="${S01_OUTPUT_ROOT}/pytest.junit.xml"
runtime_source_files | while IFS= read -r path; do
  sha256sum "${path}"
done > "${SOURCE_HASHES}"
test "$(sha256sum "${SOURCE_HASHES}" | awk '{print $1}')" = "${S01_STATE_HASH}"
python - "${EXECUTION_JSON}" "${ACTUAL_SHA}" "${S01_STATE_HASH}" "${S01_MINI_DATAROOT}" <<'PY'
import json
import importlib.metadata
import os
import platform
import socket
import sys

output, git_sha, source_hash, dataroot = sys.argv[1:]
record = {
    "schema": "s07a.nuscenes-data-foundation-focused-tests.v1",
    "git_sha": git_sha,
    "runtime_source_sha256": source_hash,
    "runtime_source_list": "runtime_source_sha256s.txt",
    "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
    "host": socket.gethostname(),
    "machine": platform.machine(),
    "mini_dataroot": os.path.abspath(dataroot),
    "pytest_disable_plugin_autoload": os.environ.get("PYTEST_DISABLE_PLUGIN_AUTOLOAD", ""),
    "pytest_addopts": os.environ.get("PYTEST_ADDOPTS", ""),
    "dependency_versions": {
        name: importlib.metadata.version(name)
        for name in ("numpy", "nuscenes-devkit", "pillow", "pytest", "torch")
    },
}
with open(output, "w", encoding="utf-8") as stream:
    json.dump(record, stream, indent=2, sort_keys=True)
    stream.write("\n")
PY

echo "[S01 tests] host=$(hostname) arch=$(uname -m) job=${SLURM_JOB_ID:-unset}"
echo "[S01 tests] sha=${ACTUAL_SHA} state_hash=${S01_STATE_HASH}"
echo "[S01 tests] mini=${S01_MINI_DATAROOT} output=${S01_OUTPUT_ROOT}"

set +e
python -m pytest -q -ra \
  fl_v3/tests/test_build_gt_database.py \
  fl_v3/tests/test_nuscenes_zip_backend.py \
  fl_v3/tests/test_nuscenes_zip_dataset.py \
  fl_v3/tests/test_nuscenes_zip_info_cache.py \
  fl_v3/tests/test_nuscenes_info_cache.py \
  --junitxml="${JUNIT_XML}" 2>&1 | tee "${PYTEST_LOG}"
pytest_status=${PIPESTATUS[0]}
set -e

artifacts=("${EXECUTION_JSON}" "${SOURCE_HASHES}" "${PYTEST_LOG}")
if [ -f "${JUNIT_XML}" ]; then
  artifacts+=("${JUNIT_XML}")
fi
sha256sum "${artifacts[@]}" > "${S01_OUTPUT_ROOT}/sha256sums.txt"
cat "${S01_OUTPUT_ROOT}/sha256sums.txt"
exit "${pytest_status}"
