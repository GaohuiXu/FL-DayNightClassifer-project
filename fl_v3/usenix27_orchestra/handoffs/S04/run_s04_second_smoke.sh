#!/bin/bash
# Exact S04 synthetic SECOND/spconv validation. Submission requires S00 approval
# of the immutable tuple recorded in RUN_REQUEST.md.
#SBATCH -A naiss2025-22-1113-gpu
#SBATCH -p gpu
#SBATCH --job-name=flv3_s04_second
#SBATCH --gpus-per-node=nvidia_gh200_120gb:1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:20:00
#SBATCH --output=/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/logs/s04_second_%j.out
#SBATCH --error=/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/logs/s04_second_%j.err
set -euo pipefail

required=(
  EXPECTED_S04_SHA
  EXPECTED_S04_REF
  EXPECTED_S04_SOURCE_HASH
  EXPECTED_S04_REQUEST_HASH
  S04_OUTPUT_ROOT
)
for name in "${required[@]}"; do
  if [ -z "${!name:-}" ]; then
    echo "${name} is required" >&2
    exit 2
  fi
done
if [ -e "${S04_OUTPUT_ROOT}" ]; then
  echo "Refusing to reuse S04_OUTPUT_ROOT=${S04_OUTPUT_ROOT}" >&2
  exit 2
fi

REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)}"
cd "${REPO}"
ACTUAL_SHA="$(git rev-parse HEAD)"
ACTUAL_REF="$(git branch --show-current)"
if [ "${ACTUAL_SHA}" != "${EXPECTED_S04_SHA}" ]; then
  echo "SHA mismatch: expected=${EXPECTED_S04_SHA} actual=${ACTUAL_SHA}" >&2
  exit 2
fi
if [ "${ACTUAL_REF:-detached}" != "${EXPECTED_S04_REF}" ]; then
  echo "ref mismatch: expected=${EXPECTED_S04_REF} actual=${ACTUAL_REF:-detached}" >&2
  exit 2
fi
REQUEST_PATH="fl_v3/usenix27_orchestra/handoffs/S04/RUN_REQUEST.md"
EXPECTED_STATUS="?? ${REQUEST_PATH}"
ACTUAL_STATUS="$(git status --short)"
if [ "${ACTUAL_STATUS}" != "${EXPECTED_STATUS}" ]; then
  echo "worktree mismatch: expected only '${EXPECTED_STATUS}', got '${ACTUAL_STATUS}'" >&2
  exit 2
fi
ACTUAL_REQUEST_HASH="$(sha256sum "${REQUEST_PATH}" | awk '{print $1}')"
if [ "${ACTUAL_REQUEST_HASH}" != "${EXPECTED_S04_REQUEST_HASH}" ]; then
  echo "request hash mismatch: expected=${EXPECTED_S04_REQUEST_HASH} actual=${ACTUAL_REQUEST_HASH}" >&2
  exit 2
fi

runtime_source_files() {
  printf '%s\n' \
    fl_v3/src/fl_v3/__init__.py \
    fl_v3/src/fl_v3/models/__init__.py \
    fl_v3/src/fl_v3/models/fusion/__init__.py \
    fl_v3/src/fl_v3/models/fusion/bev_grid.py \
    fl_v3/src/fl_v3/models/fusion/second_sparse_backbone.py \
    fl_v3/src/fl_v3/models/fusion/sparse_voxel_encoder.py \
    fl_v3/src/fl_v3/utils/__init__.py \
    fl_v3/src/fl_v3/utils/runtime.py \
    fl_v3/tests/conftest.py \
    fl_v3/tests/test_s04_second_contract.py \
    fl_v3/tests/test_s04_second_smoke.py \
    fl_v3/tests/test_sparse_voxel_encoder.py \
    fl_v3/pyproject.toml \
    fl_v3/requirements.txt \
    fl_v3/requirements.lock.txt \
    fl_v3/scripts/arrhenius_env.sh \
    fl_v3/usenix27_orchestra/handoffs/S04/run_s04_second_smoke.sh \
    | LC_ALL=C sort -u
}
S04_SOURCE_HASH="$(runtime_source_files | while IFS= read -r path; do
  sha256sum "${path}"
done | sha256sum | awk '{print $1}')"
if [ "${S04_SOURCE_HASH}" != "${EXPECTED_S04_SOURCE_HASH}" ]; then
  echo "source mismatch: expected=${EXPECTED_S04_SOURCE_HASH} actual=${S04_SOURCE_HASH}" >&2
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

mkdir -p "${S04_OUTPUT_ROOT}"
SOURCE_HASHES="${S04_OUTPUT_ROOT}/runtime_source_sha256s.txt"
EXECUTION_JSON="${S04_OUTPUT_ROOT}/execution_identity.json"
PYTEST_LOG="${S04_OUTPUT_ROOT}/pytest.log"
JUNIT_XML="${S04_OUTPUT_ROOT}/pytest.junit.xml"
runtime_source_files | while IFS= read -r path; do sha256sum "${path}"; done > "${SOURCE_HASHES}"
test "$(sha256sum "${SOURCE_HASHES}" | awk '{print $1}')" = "${S04_SOURCE_HASH}"

python - "${EXECUTION_JSON}" "${ACTUAL_SHA}" "${ACTUAL_REF:-detached}" \
  "${S04_SOURCE_HASH}" "${ACTUAL_REQUEST_HASH}" <<'PY'
import importlib.metadata
import json
import os
import platform
import socket
import sys

output, git_sha, git_ref, source_hash, request_hash = sys.argv[1:]
record = {
    "schema": "s04.second-synthetic-gh200.v1",
    "git_sha": git_sha,
    "git_ref": git_ref,
    "runtime_source_sha256": source_hash,
    "run_request_sha256": request_hash,
    "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
    "host": socket.gethostname(),
    "machine": platform.machine(),
    "platform": platform.platform(),
    "python_executable": sys.executable,
    "python_version": platform.python_version(),
    "synthetic_only": True,
    "dependency_versions": {
        name: importlib.metadata.version(name)
        for name in ("numpy", "pytest", "torch", "spconv", "cumm")
    },
}
with open(output, "w", encoding="utf-8") as stream:
    json.dump(record, stream, indent=2, sort_keys=True)
    stream.write("\n")
PY

echo "[S04] host=$(hostname) arch=$(uname -m) job=${SLURM_JOB_ID:-unset}"
echo "[S04] sha=${ACTUAL_SHA} ref=${ACTUAL_REF:-detached} source=${S04_SOURCE_HASH} request=${ACTUAL_REQUEST_HASH}"
echo "[S04] output=${S04_OUTPUT_ROOT} data=synthetic-only"

set +e
python -m pytest -q -ra -s \
  fl_v3/tests/test_s04_second_contract.py \
  fl_v3/tests/test_sparse_voxel_encoder.py \
  fl_v3/tests/test_s04_second_smoke.py \
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
    raise SystemExit(f"S04 JUnit acceptance failed: {counts}")
print(f"[S04] JUnit acceptance passed: {counts}")
PY
  junit_status=$?
  set -e
  if [ "${junit_status}" -ne 0 ]; then pytest_status="${junit_status}"; fi
fi

sha256sum \
  "${EXECUTION_JSON}" "${SOURCE_HASHES}" "${PYTEST_LOG}" "${JUNIT_XML}" \
  > "${S04_OUTPUT_ROOT}/sha256sums.txt"
sha256sum -c "${S04_OUTPUT_ROOT}/sha256sums.txt"
exit "${pytest_status}"
