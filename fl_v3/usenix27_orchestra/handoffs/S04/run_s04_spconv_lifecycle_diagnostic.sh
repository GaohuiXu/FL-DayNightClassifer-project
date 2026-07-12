#!/bin/bash
# Exact subprocess-isolated S04 spconv lifecycle diagnostic. This launcher does
# not make S04 pass: it only requires a complete structured matrix.
#SBATCH -A naiss2025-22-1113-gpu
#SBATCH -p gpu
#SBATCH --job-name=flv3_s04_lifecycle
#SBATCH --gpus-per-node=nvidia_gh200_120gb:1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:20:00
#SBATCH --output=/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/logs/s04_lifecycle_%j.out
#SBATCH --error=/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/logs/s04_lifecycle_%j.err
set -euo pipefail

required=(
  EXPECTED_S04_DIAG_SHA
  EXPECTED_S04_DIAG_TREE
  EXPECTED_S04_DIAG_SOURCE_HASH
  EXPECTED_S04_DIAG_DEP_SOURCE_HASH
  EXPECTED_S04_DIAG_REQUEST_HASH
  EXPECTED_S04_DIAG_IDENTITY_HASH
  S04_DIAG_SNAPSHOT_ROOT
  S04_DIAG_OUTPUT_ROOT
)
for name in "${required[@]}"; do
  if [ -z "${!name:-}" ]; then
    echo "${name} is required" >&2
    exit 2
  fi
done
if [ -e "${S04_DIAG_OUTPUT_ROOT}" ]; then
  echo "Refusing to reuse S04_DIAG_OUTPUT_ROOT=${S04_DIAG_OUTPUT_ROOT}" >&2
  exit 2
fi

REPO="$(realpath "${S04_DIAG_SNAPSHOT_ROOT}")"
case "${REPO}" in
  /nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/snapshots/s04_lifecycle_*) ;;
  *) echo "invalid S04 diagnostic snapshot root: ${REPO}" >&2; exit 2 ;;
esac
if [ "$(pwd -P)" != "${REPO}" ]; then
  echo "working-directory/snapshot mismatch: pwd=$(pwd -P) snapshot=${REPO}" >&2
  exit 2
fi
if [ "$(realpath "${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is required}")" != "${REPO}" ]; then
  echo "submit-dir/snapshot mismatch: submit=${SLURM_SUBMIT_DIR:-unset} snapshot=${REPO}" >&2
  exit 2
fi
if find "${REPO}" -xdev \( -type f -o -type d \) -perm /0222 -print -quit | grep -q .; then
  echo "diagnostic snapshot is not immutable: ${REPO}" >&2
  exit 2
fi

IDENTITY_PATH=".s04_lifecycle_snapshot_identity"
ACTUAL_IDENTITY_HASH="$(sha256sum "${IDENTITY_PATH}" | awk '{print $1}')"
if [ "${ACTUAL_IDENTITY_HASH}" != "${EXPECTED_S04_DIAG_IDENTITY_HASH}" ]; then
  echo "snapshot identity hash mismatch" >&2
  exit 2
fi
EXPECTED_IDENTITY="$(printf '%s\n' \
  'schema=s04.lifecycle-snapshot.v1' \
  "exec_sha=${EXPECTED_S04_DIAG_SHA}" \
  "exec_tree=${EXPECTED_S04_DIAG_TREE}" \
  "source_sha256=${EXPECTED_S04_DIAG_SOURCE_HASH}" \
  "dependency_source_sha256=${EXPECTED_S04_DIAG_DEP_SOURCE_HASH}" \
  "request_sha256=${EXPECTED_S04_DIAG_REQUEST_HASH}")"
if [ "$(cat "${IDENTITY_PATH}")" != "${EXPECTED_IDENTITY}" ]; then
  echo "snapshot identity content mismatch" >&2
  exit 2
fi

REQUEST_PATH="fl_v3/usenix27_orchestra/handoffs/S04/RUN_REQUEST.md"
ACTUAL_REQUEST_HASH="$(sha256sum "${REQUEST_PATH}" | awk '{print $1}')"
if [ "${ACTUAL_REQUEST_HASH}" != "${EXPECTED_S04_DIAG_REQUEST_HASH}" ]; then
  echo "request hash mismatch" >&2
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
    fl_v3/tests/s04_spconv_lifecycle_diagnostic.py \
    fl_v3/pyproject.toml \
    fl_v3/requirements.txt \
    fl_v3/requirements.lock.txt \
    fl_v3/scripts/arrhenius_env.sh \
    fl_v3/usenix27_orchestra/handoffs/S04/run_s04_spconv_lifecycle_diagnostic.sh \
    | LC_ALL=C sort -u
}
S04_DIAG_SOURCE_HASH="$(runtime_source_files | while IFS= read -r path; do
  sha256sum "${path}"
done | sha256sum | awk '{print $1}')"
if [ "${S04_DIAG_SOURCE_HASH}" != "${EXPECTED_S04_DIAG_SOURCE_HASH}" ]; then
  echo "repo source mismatch" >&2
  exit 2
fi

# shellcheck disable=SC1091
source fl_v3/scripts/arrhenius_env.sh
arrhenius_load_modules build
arrhenius_activate_env

SPCONV_SRC=/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/src/spconv/spconv
dependency_source_files() {
  printf '%s\n' \
    "${SPCONV_SRC}/pytorch/conv.py" \
    "${SPCONV_SRC}/pytorch/functional.py" \
    "${SPCONV_SRC}/pytorch/ops.py" \
    "${SPCONV_SRC}/build/core_cc/src/csrc/sparse/convops/convops/ConvTunerSimple/ConvTunerSimple_get_all_available.cc" \
    "${SPCONV_SRC}/build/core_cc/src/csrc/sparse/convops/convops/ConvTunerSimple/ConvTunerSimple_get_tuned_algo.cc" \
    "${SPCONV_SRC}/build/core_cc/src/csrc/sparse/convops/convops/ConvTunerSimple/ConvTunerSimple_tune_and_cache.cc" \
    | LC_ALL=C sort -u
}
S04_DIAG_DEP_SOURCE_HASH="$(dependency_source_files | while IFS= read -r path; do
  sha256sum "${path}"
done | sha256sum | awk '{print $1}')"
if [ "${S04_DIAG_DEP_SOURCE_HASH}" != "${EXPECTED_S04_DIAG_DEP_SOURCE_HASH}" ]; then
  echo "spconv dependency source mismatch" >&2
  exit 2
fi

export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${REPO}/fl_v3/src${PYTHONPATH:+:${PYTHONPATH}}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
unset PYTEST_ADDOPTS

mkdir -p "${S04_DIAG_OUTPUT_ROOT}"
export TMPDIR="${S04_DIAG_OUTPUT_ROOT}/tmp"
mkdir -p "${TMPDIR}"
SOURCE_HASHES="${S04_DIAG_OUTPUT_ROOT}/runtime_source_sha256s.txt"
DEP_SOURCE_HASHES="${S04_DIAG_OUTPUT_ROOT}/dependency_source_sha256s.txt"
EXECUTION_JSON="${S04_DIAG_OUTPUT_ROOT}/execution_identity.json"
DIAGNOSTIC_LOG="${S04_DIAG_OUTPUT_ROOT}/diagnostic.log"
MATRIX_JSON="${S04_DIAG_OUTPUT_ROOT}/lifecycle_matrix.json"
runtime_source_files | while IFS= read -r path; do sha256sum "${path}"; done > "${SOURCE_HASHES}"
dependency_source_files | while IFS= read -r path; do sha256sum "${path}"; done > "${DEP_SOURCE_HASHES}"
test "$(sha256sum "${SOURCE_HASHES}" | awk '{print $1}')" = "${S04_DIAG_SOURCE_HASH}"
test "$(sha256sum "${DEP_SOURCE_HASHES}" | awk '{print $1}')" = "${S04_DIAG_DEP_SOURCE_HASH}"

JOB_DESC="$(scontrol show job -o "${SLURM_JOB_ID:?SLURM_JOB_ID is required}")"
python - "${JOB_DESC}" <<'PY'
import re
import sys

desc = sys.argv[1]
required = ("NumNodes=1 ", "NumCPUs=8 ", "TresPerNode=gres/gpu:nvidia_gh200_120gb:1")
missing = [item for item in required if item not in desc]
match = re.search(r"AllocTRES=([^ ]+)", desc)
if missing or match is None:
    raise SystemExit(f"S04 diagnostic allocation identity failed: missing={missing}")
tres = dict(item.split("=", 1) for item in match.group(1).split(",") if "=" in item)
if tres.get("gres/gpu") != "1" or tres.get("gres/gpu:nvidia_gh200_120gb") != "1":
    raise SystemExit(f"S04 diagnostic requires exactly one GH200: {match.group(1)}")
PY
python - <<'PY'
import torch
if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
    raise SystemExit(f"expected exactly one visible GPU, got {torch.cuda.device_count()}")
PY

python - "${EXECUTION_JSON}" "${ACTUAL_IDENTITY_HASH}" "${JOB_DESC}" <<'PY'
import importlib.metadata
import json
import os
import platform
import socket
import sys

output, identity_hash, job_desc = sys.argv[1:]
record = {
    "schema": "s04.spconv-lifecycle-execution.v1",
    "git_sha": os.environ["EXPECTED_S04_DIAG_SHA"],
    "git_tree": os.environ["EXPECTED_S04_DIAG_TREE"],
    "runtime_source_sha256": os.environ["EXPECTED_S04_DIAG_SOURCE_HASH"],
    "dependency_source_sha256": os.environ["EXPECTED_S04_DIAG_DEP_SOURCE_HASH"],
    "run_request_sha256": os.environ["EXPECTED_S04_DIAG_REQUEST_HASH"],
    "snapshot_identity_sha256": identity_hash,
    "snapshot_root": os.environ["S04_DIAG_SNAPSHOT_ROOT"],
    "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
    "slurm_job_description": job_desc,
    "host": socket.gethostname(),
    "machine": platform.machine(),
    "python_version": platform.python_version(),
    "dependency_versions": {
        name: importlib.metadata.version(name) for name in ("torch", "spconv", "cumm")
    },
    "synthetic_only": True,
    "diagnostic_only": True,
}
with open(output, "w", encoding="utf-8") as stream:
    json.dump(record, stream, indent=2, sort_keys=True)
    stream.write("\n")
PY

echo "[S04-DIAG] job=${SLURM_JOB_ID} host=$(hostname) source=${S04_DIAG_SOURCE_HASH} dep_source=${S04_DIAG_DEP_SOURCE_HASH}"
python fl_v3/tests/s04_spconv_lifecycle_diagnostic.py \
  --matrix "${MATRIX_JSON}" 2>&1 | tee "${DIAGNOSTIC_LOG}"

python - "${MATRIX_JSON}" <<'PY'
import json
import sys

expected = [
    "fresh_fp16_eval_6",
    "train_to_eval_no_backward_6",
    "train_to_eval_after_backward_6",
    "fresh_fp16_eval_large",
    "fresh_fp32_eval_6",
    "fp32_then_fp16_eval_6",
    "fp16_train_then_fresh_fp16_eval_6",
]
with open(sys.argv[1], encoding="utf-8") as stream:
    matrix = json.load(stream)
records = matrix.get("cells", [])
assert [record.get("cell") for record in records] == expected
assert all(record.get("process_returncode") == 0 for record in records)
assert all(
    record.get("result", {}).get("cell") == record.get("cell")
    and record["result"].get("status") in {"success", "error"}
    for record in records
)
assert matrix.get("scientific_metric") is False
assert matrix.get("optimizer_or_parameter_update") is False
print("[S04-DIAG] structured matrix complete; cell errors are observations, not launcher failures")
PY

sha256sum \
  "${EXECUTION_JSON}" "${SOURCE_HASHES}" "${DEP_SOURCE_HASHES}" \
  "${DIAGNOSTIC_LOG}" "${MATRIX_JSON}" \
  > "${S04_DIAG_OUTPUT_ROOT}/sha256sums.txt"
sha256sum -c "${S04_DIAG_OUTPUT_ROOT}/sha256sums.txt"
