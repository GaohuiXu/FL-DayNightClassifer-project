#!/bin/bash
# Focused S07-A physical-cache provenance regression gate.
# Submission requires the exact immutable request in handoffs/S07/RUN_REQUEST.md.
# It does not build/scan trainval, run a model, or emit scientific metrics.
#SBATCH -A naiss2025-22-1113-gpu
#SBATCH -p gpu
#SBATCH --job-name=flv3_s07a_provenance
#SBATCH --gpus-per-node=nvidia_gh200_120gb:1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:15:00
#SBATCH --output=/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/logs/s07a_provenance_%j.out
#SBATCH --error=/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/logs/s07a_provenance_%j.err
set -euo pipefail

required=(
  EXPECTED_S07A_SHA
  EXPECTED_S07A_PROVENANCE_STATE_HASH
  S07A_MINI_DATAROOT
  S07A_PROVENANCE_OUTPUT_ROOT
)
for name in "${required[@]}"; do
  if [ -z "${!name:-}" ]; then
    echo "${name} is required" >&2
    exit 2
  fi
done
if [ -e "${S07A_PROVENANCE_OUTPUT_ROOT}" ]; then
  echo "Refusing to reuse/overwrite S07A_PROVENANCE_OUTPUT_ROOT=${S07A_PROVENANCE_OUTPUT_ROOT}" >&2
  exit 2
fi

REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "${REPO}"
ACTUAL_SHA="$(git rev-parse HEAD)"
if [ "${ACTUAL_SHA}" != "${EXPECTED_S07A_SHA}" ]; then
  echo "SHA mismatch: expected=${EXPECTED_S07A_SHA} actual=${ACTUAL_SHA}" >&2
  exit 2
fi
if [ -n "$(git branch --show-current)" ] || [ -n "$(git status --short)" ]; then
  echo "Focused provenance execution requires a clean detached worktree" >&2
  exit 2
fi

runtime_source_files() {
  {
    git ls-files -- 'fl_v3/src/fl_v3/data/nuscenes/*.py'
    printf '%s\n' \
      fl_v3/src/fl_v3/__init__.py \
      fl_v3/src/fl_v3/data/__init__.py \
      fl_v3/src/fl_v3/data/partition.py \
      fl_v3/src/fl_v3/utils/__init__.py \
      fl_v3/src/fl_v3/utils/runtime.py \
      fl_v3/scripts/build_gt_database.py \
      fl_v3/scripts/run_s07a_provenance_tests.sh \
      fl_v3/tests/conftest.py \
      fl_v3/tests/test_build_gt_database.py \
      fl_v3/pyproject.toml \
      fl_v3/requirements.txt \
      fl_v3/requirements.lock.txt \
      fl_v3/scripts/arrhenius_env.sh
  } | LC_ALL=C sort -u
}
S07A_STATE_HASH="$(runtime_source_files | while IFS= read -r path; do
  sha256sum "${path}"
done | sha256sum | awk '{print $1}')"
if [ "${S07A_STATE_HASH}" != "${EXPECTED_S07A_PROVENANCE_STATE_HASH}" ]; then
  echo "S07-A provenance state hash mismatch: expected=${EXPECTED_S07A_PROVENANCE_STATE_HASH} actual=${S07A_STATE_HASH}" >&2
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
export NUSCENES_DATAROOT="${S07A_MINI_DATAROOT}"
export ARRHENIUS_NUSCENES_DATAROOT="${S07A_MINI_DATAROOT}"
unset NUSCENES_ZIP_MANIFEST ARRHENIUS_NUSCENES_ZIP_MANIFEST NUSCENES_DATA_DIR

python - "${S07A_PROVENANCE_OUTPUT_ROOT}" "${S07A_MINI_DATAROOT}" <<'PY'
import sys
from fl_v3.data.nuscenes import paths as P

P.resolve_writable(sys.argv[1], sys.argv[2])
P.verify_dataset("v1.0-mini", sys.argv[2])
PY

mkdir -p "${S07A_PROVENANCE_OUTPUT_ROOT}"
SOURCE_HASHES="${S07A_PROVENANCE_OUTPUT_ROOT}/runtime_source_sha256s.txt"
EXECUTION_JSON="${S07A_PROVENANCE_OUTPUT_ROOT}/execution_identity.json"
PYTEST_LOG="${S07A_PROVENANCE_OUTPUT_ROOT}/pytest.log"
JUNIT_XML="${S07A_PROVENANCE_OUTPUT_ROOT}/pytest.junit.xml"
runtime_source_files | while IFS= read -r path; do
  sha256sum "${path}"
done > "${SOURCE_HASHES}"
test "$(sha256sum "${SOURCE_HASHES}" | awk '{print $1}')" = "${S07A_STATE_HASH}"

python - "${EXECUTION_JSON}" "${ACTUAL_SHA}" "${S07A_STATE_HASH}" \
  "${S07A_MINI_DATAROOT}" <<'PY'
import importlib.metadata
import json
import os
import platform
import socket
import sys

output, git_sha, source_hash, dataroot = sys.argv[1:]
record = {
    "schema": "s07a.physical-cache-provenance-focused-tests.v1",
    "git_sha": git_sha,
    "runtime_source_sha256": source_hash,
    "runtime_source_list": "runtime_source_sha256s.txt",
    "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
    "host": socket.gethostname(),
    "machine": platform.machine(),
    "platform": platform.platform(),
    "python_executable": sys.executable,
    "python_implementation": platform.python_implementation(),
    "python_version": platform.python_version(),
    "mini_dataroot": os.path.abspath(dataroot),
    "pytest_disable_plugin_autoload": os.environ.get(
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD", ""
    ),
    "pytest_addopts": os.environ.get("PYTEST_ADDOPTS", ""),
    "dependency_versions": {
        name: importlib.metadata.version(name)
        for name in (
            "numpy",
            "nuscenes-devkit",
            "pyquaternion",
            "Pillow",
            "pytest",
            "torch",
        )
    },
}
with open(output, "w", encoding="utf-8") as stream:
    json.dump(record, stream, indent=2, sort_keys=True)
    stream.write("\n")
PY

echo "[S07-A provenance] host=$(hostname) arch=$(uname -m) job=${SLURM_JOB_ID:-unset}"
echo "[S07-A provenance] sha=${ACTUAL_SHA} state_hash=${S07A_STATE_HASH}"
echo "[S07-A provenance] mini=${S07A_MINI_DATAROOT} output=${S07A_PROVENANCE_OUTPUT_ROOT}"

set +e
python -m pytest -q -ra \
  fl_v3/tests/test_build_gt_database.py \
  --junitxml="${JUNIT_XML}" 2>&1 | tee "${PYTEST_LOG}"
pytest_status=${PIPESTATUS[0]}
set -e

if [ "${pytest_status}" -eq 0 ]; then
  set +e
  python - "${JUNIT_XML}" <<'PY'
import sys
import xml.etree.ElementTree as ET

root = ET.parse(sys.argv[1]).getroot()
suites = [root] if root.tag == "testsuite" else list(root.findall("testsuite"))
counts = {
    key: sum(int(suite.attrib.get(key, "0")) for suite in suites)
    for key in ("tests", "failures", "errors", "skipped")
}
if counts["tests"] <= 0 or any(counts[key] for key in ("failures", "errors", "skipped")):
    raise SystemExit(f"focused pytest JUnit acceptance failed: {counts}")
print(f"[S07-A provenance] JUnit acceptance passed: {counts}")
PY
  junit_status=$?
  set -e
  if [ "${junit_status}" -ne 0 ]; then
    pytest_status="${junit_status}"
  fi
fi

artifacts=("${EXECUTION_JSON}" "${SOURCE_HASHES}" "${PYTEST_LOG}")
if [ -f "${JUNIT_XML}" ]; then
  artifacts+=("${JUNIT_XML}")
fi
sha256sum "${artifacts[@]}" > "${S07A_PROVENANCE_OUTPUT_ROOT}/sha256sums.txt"
sha256sum -c "${S07A_PROVENANCE_OUTPUT_ROOT}/sha256sums.txt"
exit "${pytest_status}"
