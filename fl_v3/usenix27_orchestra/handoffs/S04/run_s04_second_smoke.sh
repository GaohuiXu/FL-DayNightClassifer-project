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
  EXPECTED_S04_SOURCE_HASH
  EXPECTED_S04_REQUEST_HASH
  S04_SNAPSHOT_ROOT
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

REPO="$(realpath "${S04_SNAPSHOT_ROOT}")"
case "${REPO}" in
  /nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/snapshots/s04_*) ;;
  *) echo "invalid S04 snapshot root: ${REPO}" >&2; exit 2 ;;
esac
if [ "$(realpath "${SLURM_SUBMIT_DIR:-.}")" != "${REPO}" ]; then
  echo "submit-dir/snapshot mismatch: submit=${SLURM_SUBMIT_DIR:-unset} snapshot=${REPO}" >&2
  exit 2
fi
cd "${REPO}"
if find "${REPO}" -xdev \( -type f -o -type d \) -perm /0222 -print -quit | grep -q .; then
  echo "snapshot is not immutable (a path has write bits): ${REPO}" >&2
  exit 2
fi
REQUEST_PATH="fl_v3/usenix27_orchestra/handoffs/S04/RUN_REQUEST.md"
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
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
unset PYTEST_ADDOPTS

mkdir -p "${S04_OUTPUT_ROOT}"
export TMPDIR="${S04_OUTPUT_ROOT}/tmp"
mkdir -p "${TMPDIR}"
SOURCE_HASHES="${S04_OUTPUT_ROOT}/runtime_source_sha256s.txt"
EXECUTION_JSON="${S04_OUTPUT_ROOT}/execution_identity.json"
PYTEST_LOG="${S04_OUTPUT_ROOT}/pytest.log"
JUNIT_XML="${S04_OUTPUT_ROOT}/pytest.junit.xml"
runtime_source_files | while IFS= read -r path; do sha256sum "${path}"; done > "${SOURCE_HASHES}"
test "$(sha256sum "${SOURCE_HASHES}" | awk '{print $1}')" = "${S04_SOURCE_HASH}"

JOB_DESC="$(scontrol show job -o "${SLURM_JOB_ID:?SLURM_JOB_ID is required}")"
python - "${JOB_DESC}" <<'PY'
import re
import sys

desc = sys.argv[1]
required = ("NumNodes=1 ", "NumCPUs=8 ", "TresPerNode=gres/gpu:nvidia_gh200_120gb:1")
missing = [item for item in required if item not in desc]
match = re.search(r"AllocTRES=([^ ]+)", desc)
if missing or match is None:
    raise SystemExit(f"S04 allocation identity failed: missing={missing} desc={desc}")
tres = dict(item.split("=", 1) for item in match.group(1).split(",") if "=" in item)
if tres.get("gres/gpu") != "1" or tres.get("gres/gpu:nvidia_gh200_120gb") != "1":
    raise SystemExit(f"S04 requires exactly one allocated GH200, got AllocTRES={match.group(1)}")
PY

python - <<'PY'
import torch

if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
    raise SystemExit(
        f"S04 requires exactly one visible CUDA GPU, available={torch.cuda.is_available()} "
        f"count={torch.cuda.device_count()}"
    )
PY

python - "${EXECUTION_JSON}" "${EXPECTED_S04_SHA}" "${REPO}" \
  "${S04_SOURCE_HASH}" "${ACTUAL_REQUEST_HASH}" "${JOB_DESC}" <<'PY'
import importlib.metadata
import json
import os
import platform
import socket
import sys

output, git_sha, snapshot_root, source_hash, request_hash, job_desc = sys.argv[1:]
record = {
    "schema": "s04.second-synthetic-gh200.v2",
    "git_sha": git_sha,
    "snapshot_root": snapshot_root,
    "runtime_source_sha256": source_hash,
    "run_request_sha256": request_hash,
    "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
    "host": socket.gethostname(),
    "machine": platform.machine(),
    "platform": platform.platform(),
    "python_executable": sys.executable,
    "python_version": platform.python_version(),
    "synthetic_only": True,
    "allocation_fail_closed": True,
    "slurm_job_description": job_desc,
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
echo "[S04] sha=${EXPECTED_S04_SHA} snapshot=${REPO} source=${S04_SOURCE_HASH} request=${ACTUAL_REQUEST_HASH}"
echo "[S04] output=${S04_OUTPUT_ROOT} data=synthetic-only"

set +e
python -m pytest -q -ra -s \
  -p no:cacheprovider \
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
expected_tests = 10
if counts["tests"] != expected_tests or any(
    counts[key] for key in ("failures", "errors", "skipped")
):
    raise SystemExit(f"S04 JUnit acceptance failed: {counts}")
print(f"[S04] JUnit acceptance passed: expected={expected_tests} actual={counts}")
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
