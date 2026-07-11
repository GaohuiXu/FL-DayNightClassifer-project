#!/bin/bash
# Bounded synthetic/CPU-only S02 correctness gate. Submission requires the exact
# pending request in RUN_REQUEST.md and explicit S00 approval under O-017/O-009.
#SBATCH -A naiss2025-22-1113-gpu
#SBATCH -p gpu
#SBATCH --job-name=flv3_s02_cpu_tests
#SBATCH --gpus-per-node=nvidia_gh200_120gb:1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:10:00
#SBATCH --output=/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/logs/s02_cpu_tests_%j.out
#SBATCH --error=/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/logs/s02_cpu_tests_%j.err
set -euo pipefail

required=(EXPECTED_S02_SHA EXPECTED_S02_STATE_HASH S02_OUTPUT_ROOT)
for name in "${required[@]}"; do
  if [ -z "${!name:-}" ]; then
    echo "${name} is required" >&2
    exit 2
  fi
done
if [ -e "${S02_OUTPUT_ROOT}" ]; then
  echo "Refusing to reuse/overwrite S02_OUTPUT_ROOT=${S02_OUTPUT_ROOT}" >&2
  exit 2
fi

REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)}"
cd "${REPO}"
ACTUAL_SHA="$(git rev-parse HEAD)"
if [ "${ACTUAL_SHA}" != "${EXPECTED_S02_SHA}" ]; then
  echo "SHA mismatch: expected=${EXPECTED_S02_SHA} actual=${ACTUAL_SHA}" >&2
  exit 2
fi
if [ "$(git branch --show-current)" != "codex/s02-cl-p0-correctness" ]; then
  echo "S02 gate requires the authorized codex/s02-cl-p0-correctness branch" >&2
  exit 2
fi

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
    fl_v3/src/fl_v3/models/fusion/losses.py \
    fl_v3/src/fl_v3/utils/__init__.py \
    fl_v3/src/fl_v3/utils/runtime.py \
    fl_v3/tests/conftest.py \
    fl_v3/tests/test_model_determinism.py \
    fl_v3/tests/test_s02_p0_correctness.py \
    fl_v3/usenix27_orchestra/handoffs/S02/run_s02_cpu_tests.sh \
    | LC_ALL=C sort -u
}

if ! git diff --quiet -- $(runtime_source_files); then
  echo "Tracked S02 runtime source differs from ${EXPECTED_S02_SHA}" >&2
  exit 2
fi
S02_STATE_HASH="$(runtime_source_files | while IFS= read -r path; do
  sha256sum "${path}"
done | sha256sum | awk '{print $1}')"
if [ "${S02_STATE_HASH}" != "${EXPECTED_S02_STATE_HASH}" ]; then
  echo "S02 source hash mismatch: expected=${EXPECTED_S02_STATE_HASH} actual=${S02_STATE_HASH}" >&2
  exit 2
fi

# shellcheck disable=SC1091
source fl_v3/scripts/arrhenius_env.sh
arrhenius_load_modules build
arrhenius_activate_env

# This gate intentionally exercises only CPU tensor paths. The GH200 allocation is
# required solely because the validated aarch64 environment is compute-node-only.
export CUDA_VISIBLE_DEVICES=""
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
unset PYTEST_ADDOPTS

mkdir -p "${S02_OUTPUT_ROOT}"
SOURCE_HASHES="${S02_OUTPUT_ROOT}/runtime_source_sha256s.txt"
EXECUTION_JSON="${S02_OUTPUT_ROOT}/execution_identity.json"
PYTEST_LOG="${S02_OUTPUT_ROOT}/pytest.log"
JUNIT_XML="${S02_OUTPUT_ROOT}/pytest.junit.xml"

runtime_source_files | while IFS= read -r path; do
  sha256sum "${path}"
done > "${SOURCE_HASHES}"
test "$(sha256sum "${SOURCE_HASHES}" | awk '{print $1}')" = "${S02_STATE_HASH}"

python - "${EXECUTION_JSON}" "${ACTUAL_SHA}" "${S02_STATE_HASH}" <<'PY'
import importlib.metadata
import json
import os
import platform
import socket
import sys

output, git_sha, source_hash = sys.argv[1:]
record = {
    "schema": "s02.cpu-correctness-focused-tests.v1",
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
    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>"),
    "dependency_versions": {
        name: importlib.metadata.version(name)
        for name in ("numpy", "pytest", "torch")
    },
}
with open(output, "w", encoding="utf-8") as handle:
    json.dump(record, handle, indent=2, sort_keys=True)
    handle.write("\n")
PY

python -m pytest -q \
  fl_v3/tests/test_s02_p0_correctness.py \
  fl_v3/tests/test_model_determinism.py::test_pillar_scatter_permutation_invariant \
  fl_v3/tests/test_model_determinism.py::test_pillar_scatter_permutation_invariant_OVERCAP \
  --junitxml="${JUNIT_XML}" 2>&1 | tee "${PYTEST_LOG}"

python - "${JUNIT_XML}" <<'PY'
import sys
import xml.etree.ElementTree as ET

root = ET.parse(sys.argv[1]).getroot()
tests = int(root.attrib.get("tests", 0))
failures = int(root.attrib.get("failures", 0))
errors = int(root.attrib.get("errors", 0))
skipped = int(root.attrib.get("skipped", 0))
if (tests, failures, errors, skipped) != (12, 0, 0, 0):
    raise SystemExit(
        f"unexpected JUnit counts: tests={tests} failures={failures} "
        f"errors={errors} skipped={skipped}"
    )
PY

sha256sum \
  "${EXECUTION_JSON}" \
  "${SOURCE_HASHES}" \
  "${PYTEST_LOG}" \
  "${JUNIT_XML}" \
  > "${S02_OUTPUT_ROOT}/sha256sums.txt"
sha256sum -c "${S02_OUTPUT_ROOT}/sha256sums.txt"
