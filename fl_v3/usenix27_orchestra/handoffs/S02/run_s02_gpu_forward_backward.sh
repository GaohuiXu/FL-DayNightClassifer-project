#!/bin/bash
# Exact synthetic CUDA forward/backward remediation for the missing S02 gate.
# Submission requires the immutable request and explicit S00 approval.
#SBATCH -A naiss2025-22-1113-gpu
#SBATCH -p gpu
#SBATCH --job-name=flv3_s02_gpu_fb
#SBATCH --gpus-per-node=nvidia_gh200_120gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --output=/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/logs/s02_gpu_fb_%j.out
#SBATCH --error=/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/logs/s02_gpu_fb_%j.err
set -euo pipefail

required=(
  EXPECTED_S02_SHA
  EXPECTED_S02_TREE
  EXPECTED_S02_SOURCE_HASH
  EXPECTED_S02_REQUEST_HASH
  EXPECTED_S02_LAUNCHER_HASH
  S02_SNAPSHOT_ROOT
  S02_OUTPUT_ROOT
)
for name in "${required[@]}"; do
  if [ -z "${!name:-}" ]; then
    echo "${name} is required" >&2
    exit 2
  fi
done

case "${S02_SNAPSHOT_ROOT}" in
  /nobackup/*) ;;
  *) echo "S02_SNAPSHOT_ROOT must be under /nobackup" >&2; exit 2 ;;
esac
case "${S02_OUTPUT_ROOT}" in
  /nobackup/*) ;;
  *) echo "S02_OUTPUT_ROOT must be under /nobackup" >&2; exit 2 ;;
esac
if [ ! -d "${S02_SNAPSHOT_ROOT}" ] || [ -e "${S02_OUTPUT_ROOT}" ]; then
  echo "snapshot must exist and output must be absent" >&2
  exit 2
fi

cd "${S02_SNAPSHOT_ROOT}"
EXPECTED_IDENTITY="git_sha=${EXPECTED_S02_SHA}
git_tree=${EXPECTED_S02_TREE}"
if [ "$(cat .s02_snapshot_identity)" != "${EXPECTED_IDENTITY}" ]; then
  echo "snapshot SHA/tree identity mismatch" >&2
  exit 2
fi

REQUEST_PATH="fl_v3/usenix27_orchestra/handoffs/S02/RUN_REQUEST.md"
LAUNCHER_PATH="fl_v3/usenix27_orchestra/handoffs/S02/run_s02_gpu_forward_backward.sh"
test "$(sha256sum "${REQUEST_PATH}" | awk '{print $1}')" = "${EXPECTED_S02_REQUEST_HASH}"
test "$(sha256sum "${LAUNCHER_PATH}" | awk '{print $1}')" = "${EXPECTED_S02_LAUNCHER_HASH}"

runtime_source_files() {
  printf '%s\n' \
    fl_v3/pyproject.toml \
    fl_v3/requirements.lock.txt \
    fl_v3/requirements.txt \
    fl_v3/scripts/arrhenius_env.sh \
    fl_v3/src/fl_v3/__init__.py \
    fl_v3/src/fl_v3/models/__init__.py \
    fl_v3/src/fl_v3/models/fusion/__init__.py \
    fl_v3/src/fl_v3/models/fusion/bev_grid.py \
    fl_v3/src/fl_v3/models/fusion/lidar_encoder.py \
    fl_v3/src/fl_v3/utils/__init__.py \
    fl_v3/src/fl_v3/utils/runtime.py \
    fl_v3/tests/conftest.py \
    fl_v3/tests/test_s02_gpu_forward_backward.py \
    "${LAUNCHER_PATH}" \
    "${REQUEST_PATH}" \
    | LC_ALL=C sort -u
}

mkdir -p "${S02_OUTPUT_ROOT}"
SOURCE_HASHES="${S02_OUTPUT_ROOT}/runtime_source_sha256s.txt"
EXECUTION_JSON="${S02_OUTPUT_ROOT}/execution_identity.json"
PYTEST_LOG="${S02_OUTPUT_ROOT}/pytest.log"
JUNIT_XML="${S02_OUTPUT_ROOT}/pytest.junit.xml"

runtime_source_files | while IFS= read -r path; do sha256sum "${path}"; done > "${SOURCE_HASHES}"
sha256sum -c "${SOURCE_HASHES}"
ACTUAL_SOURCE_HASH="$(sha256sum "${SOURCE_HASHES}" | awk '{print $1}')"
if [ "${ACTUAL_SOURCE_HASH}" != "${EXPECTED_S02_SOURCE_HASH}" ]; then
  echo "source mismatch: expected=${EXPECTED_S02_SOURCE_HASH} actual=${ACTUAL_SOURCE_HASH}" >&2
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

python - "${EXECUTION_JSON}" <<'PY'
import importlib.metadata
import json
import os
import platform
import socket
import sys

import torch

if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable")
if torch.cuda.device_count() != 1:
    raise SystemExit(f"expected exactly one visible GPU, got {torch.cuda.device_count()}")
if os.environ.get("SLURM_CPUS_PER_TASK") != "4":
    raise SystemExit(f"expected four CPUs, got {os.environ.get('SLURM_CPUS_PER_TASK')!r}")
gpus_on_node = os.environ.get("SLURM_GPUS_ON_NODE", "")
if gpus_on_node and gpus_on_node != "1":
    raise SystemExit(f"expected one allocated GPU, SLURM_GPUS_ON_NODE={gpus_on_node!r}")

record = {
    "schema": "s02.synthetic-cuda-forward-backward.v1",
    "git_sha": os.environ["EXPECTED_S02_SHA"],
    "git_tree": os.environ["EXPECTED_S02_TREE"],
    "runtime_source_sha256": os.environ["EXPECTED_S02_SOURCE_HASH"],
    "run_request_sha256": os.environ["EXPECTED_S02_REQUEST_HASH"],
    "launcher_sha256": os.environ["EXPECTED_S02_LAUNCHER_HASH"],
    "snapshot_root": os.environ["S02_SNAPSHOT_ROOT"],
    "output_root": os.environ["S02_OUTPUT_ROOT"],
    "synthetic_only": True,
    "optimizer_or_scaler_step": False,
    "slurm": {
        name: os.environ.get(name, "")
        for name in (
            "SLURM_JOB_ID",
            "SLURM_JOB_NAME",
            "SLURM_JOB_NODELIST",
            "SLURM_JOB_NUM_NODES",
            "SLURM_CPUS_PER_TASK",
            "SLURM_GPUS_ON_NODE",
            "SLURM_JOB_GPUS",
            "CUDA_VISIBLE_DEVICES",
        )
    },
    "host": socket.gethostname(),
    "machine": platform.machine(),
    "python_executable": sys.executable,
    "python_version": platform.python_version(),
    "torch_cuda_device_count": torch.cuda.device_count(),
    "torch_cuda_device_name": torch.cuda.get_device_name(0),
    "torch_cuda_capability": list(torch.cuda.get_device_capability(0)),
    "dependency_versions": {
        name: importlib.metadata.version(name)
        for name in ("numpy", "pytest", "torch")
    },
}
with open(sys.argv[1], "w", encoding="utf-8") as stream:
    json.dump(record, stream, indent=2, sort_keys=True)
    stream.write("\n")
PY

echo "[S02-GPU] snapshot=${S02_SNAPSHOT_ROOT} sha=${EXPECTED_S02_SHA} tree=${EXPECTED_S02_TREE}"
echo "[S02-GPU] source=${EXPECTED_S02_SOURCE_HASH} request=${EXPECTED_S02_REQUEST_HASH}"
echo "[S02-GPU] job=${SLURM_JOB_ID:-unset} nodes=${SLURM_JOB_NUM_NODES:-unset} gpus=${SLURM_GPUS_ON_NODE:-unset} visible=${CUDA_VISIBLE_DEVICES:-unset}"

python -m pytest -q -ra -s \
  fl_v3/tests/test_s02_gpu_forward_backward.py::test_s02_cuda_b3_overcap_empty_isolation_forward_backward \
  --junitxml="${JUNIT_XML}" 2>&1 | tee "${PYTEST_LOG}"

python - "${JUNIT_XML}" <<'PY'
import sys
import xml.etree.ElementTree as ET

root = ET.parse(sys.argv[1]).getroot()
suites = [root] if root.tag == "testsuite" else list(root.iter("testsuite"))
if not suites:
    raise SystemExit(f"JUnit contains no testsuite: root={root.tag}")
counts = {
    key: sum(int(suite.attrib.get(key, "0")) for suite in suites)
    for key in ("tests", "failures", "errors", "skipped")
}
if tuple(counts[key] for key in ("tests", "failures", "errors", "skipped")) != (1, 0, 0, 0):
    raise SystemExit(f"S02 GPU JUnit acceptance failed: {counts}")
print(f"[S02-GPU] JUnit acceptance passed: {counts}")
PY

# Verify immutable runtime sources again after execution, then verify all final artifacts.
sha256sum -c "${SOURCE_HASHES}"
sha256sum \
  "${EXECUTION_JSON}" \
  "${SOURCE_HASHES}" \
  "${PYTEST_LOG}" \
  "${JUNIT_XML}" \
  > "${S02_OUTPUT_ROOT}/sha256sums.txt"
sha256sum -c "${S02_OUTPUT_ROOT}/sha256sums.txt"
