#!/bin/bash
# Prepared S01 full-data gate. DO NOT SUBMIT without an owner-approved
# handoffs/S01/RUN_REQUEST.md bound to the exact worker commit and command.
#
# Required submission variables:
#   EXPECTED_S01_SHA=<approved-commit> \
#   EXPECTED_S01_STATE_HASH=<approved-runtime-source-hash> \
#   S01_OUTPUT_ROOT=/nobackup/.../immutable_s01_output \
#     sbatch fl_v3/scripts/run_s01_nuscenes_zip_full_gate.sh
#
# The job performs one exhaustive central-directory manifest scan, builds the
# train/val 10-sweep info cache from metadata without extracting blobs, audits
# 100% referenced-member coverage, then profiles decoded DataLoader wait.
#SBATCH -A naiss2025-22-1113-gpu
#SBATCH -p gpu
#SBATCH --job-name=flv3_s01_zip_gate
#SBATCH --gpus-per-node=nvidia_gh200_120gb:1
#SBATCH --cpus-per-task=32
#SBATCH --time=02:00:00
#SBATCH --output=/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/logs/s01_zip_gate_%j.out
#SBATCH --error=/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/logs/s01_zip_gate_%j.err
set -euo pipefail

if [ -z "${EXPECTED_S01_SHA:-}" ] || [ -z "${EXPECTED_S01_STATE_HASH:-}" ] || [ -z "${S01_OUTPUT_ROOT:-}" ]; then
  echo "EXPECTED_S01_SHA, EXPECTED_S01_STATE_HASH, and exact S01_OUTPUT_ROOT are required" >&2
  exit 2
fi
if [ -e "${S01_OUTPUT_ROOT}" ]; then
  echo "Refusing to reuse/overwrite existing S01_OUTPUT_ROOT=${S01_OUTPUT_ROOT}" >&2
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
      fl_v3/scripts/build_nuscenes_cache.py \
      fl_v3/scripts/s01_nuscenes_zip_manifest.py \
      fl_v3/scripts/s01_nuscenes_zip_audit.py \
      fl_v3/scripts/s01_nuscenes_zip_benchmark.py \
      fl_v3/scripts/run_s01_nuscenes_zip_full_gate.sh
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
# arrhenius_load_modules purges modules, so load the licensed dataset afterwards.
module load nuScenes-data/1.0-map-1.3-zip
arrhenius_activate_env

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUSCENES_DATAROOT="${NUSCENES_DATA_DIR}"
export NUSCENES_ZIP_MANIFEST="${S01_OUTPUT_ROOT}/nuscenes_trainval_zip_manifest.sqlite"
CACHE_DIR="${S01_OUTPUT_ROOT}/info_cache_msweep10"
COVERAGE_JSON="${S01_OUTPUT_ROOT}/coverage_train_val_msweep10.json"
PROFILE_JSON="${S01_OUTPUT_ROOT}/loader_profile_train_msweep10.json"
EXECUTION_JSON="${S01_OUTPUT_ROOT}/execution_identity.json"
SOURCE_HASHES="${S01_OUTPUT_ROOT}/runtime_source_sha256s.txt"

# Fail closed before the first mkdir if the requested output resolves under the
# immutable shared dataset.
python - "${S01_OUTPUT_ROOT}" "${NUSCENES_DATAROOT}" <<'PY'
import sys
from fl_v3.data.nuscenes import paths as P
P.resolve_writable(sys.argv[1], sys.argv[2])
PY
mkdir -p "${S01_OUTPUT_ROOT}" "${CACHE_DIR}"
runtime_source_files | while IFS= read -r path; do
  sha256sum "${path}"
done > "${SOURCE_HASHES}"
test "$(sha256sum "${SOURCE_HASHES}" | awk '{print $1}')" = "${S01_STATE_HASH}"
python - "${EXECUTION_JSON}" "${ACTUAL_SHA}" "${S01_STATE_HASH}" "${NUSCENES_DATAROOT}" <<'PY'
import json
import os
import platform
import socket
import sys

output, git_sha, source_hash, dataroot = sys.argv[1:]
record = {
    "schema": "s01.nuscenes-zip-execution-identity.v1",
    "git_sha": git_sha,
    "runtime_source_sha256": source_hash,
    "runtime_source_list": "runtime_source_sha256s.txt",
    "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
    "host": socket.gethostname(),
    "machine": platform.machine(),
    "dataroot": os.path.abspath(dataroot),
}
with open(output, "w", encoding="utf-8") as stream:
    json.dump(record, stream, indent=2, sort_keys=True)
    stream.write("\n")
PY
echo "[S01] host=$(hostname) arch=$(uname -m) job=${SLURM_JOB_ID:-unset}"
echo "[S01] sha=${ACTUAL_SHA} state_hash=${S01_STATE_HASH}"
echo "[S01] repo=${REPO}"
echo "[S01] dataroot=${NUSCENES_DATAROOT}"
echo "[S01] manifest=${NUSCENES_ZIP_MANIFEST}"
echo "[S01] output=${S01_OUTPUT_ROOT}"

python fl_v3/scripts/s01_nuscenes_zip_manifest.py \
  --dataroot "${NUSCENES_DATAROOT}" \
  --manifest "${NUSCENES_ZIP_MANIFEST}"

python fl_v3/scripts/build_nuscenes_cache.py \
  --dataroot "${NUSCENES_DATAROOT}" \
  --version v1.0-trainval \
  --splits train val \
  --n-sweeps 10 \
  --cache-dir "${CACHE_DIR}" \
  --rebuild

python fl_v3/scripts/s01_nuscenes_zip_audit.py \
  --dataroot "${NUSCENES_DATAROOT}" \
  --manifest "${NUSCENES_ZIP_MANIFEST}" \
  --cache-dir "${CACHE_DIR}" \
  --version v1.0-trainval \
  --splits train val \
  --n-sweeps 10 \
  --output "${COVERAGE_JSON}"

python fl_v3/scripts/s01_nuscenes_zip_benchmark.py \
  --dataroot "${NUSCENES_DATAROOT}" \
  --manifest "${NUSCENES_ZIP_MANIFEST}" \
  --cache-dir "${CACHE_DIR}" \
  --version v1.0-trainval \
  --split train \
  --n-sweeps 10 \
  --workers 0,2,4,8 \
  --batch-size 1 \
  --determinism-batches 32 \
  --warmup-batches 16 \
  --measured-batches 256 \
  --repeats 2 \
  --output "${PROFILE_JSON}"

sha256sum \
  "${EXECUTION_JSON}" \
  "${SOURCE_HASHES}" \
  "${NUSCENES_ZIP_MANIFEST}" \
  "${COVERAGE_JSON}" \
  "${PROFILE_JSON}" \
  "${CACHE_DIR}"/*.pkl \
  "${CACHE_DIR}"/*.meta.json \
  > "${S01_OUTPUT_ROOT}/sha256sums.txt"
echo "[S01] completed; checksums:"
cat "${S01_OUTPUT_ROOT}/sha256sums.txt"
