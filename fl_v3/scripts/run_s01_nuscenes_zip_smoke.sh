#!/bin/bash
# Bounded one-archive real-data S01 smoke. Submission still requires the exact
# immutable SHA/output in handoffs/S01/RUN_REQUEST.md to be owner-approved.
#SBATCH -A naiss2025-22-1113-gpu
#SBATCH -p gpu
#SBATCH --job-name=flv3_s01_zip_smoke
#SBATCH --gpus-per-node=nvidia_gh200_120gb:1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:20:00
#SBATCH --output=/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/logs/s01_zip_smoke_%j.out
#SBATCH --error=/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/logs/s01_zip_smoke_%j.err
set -euo pipefail

if [ -z "${EXPECTED_S01_SHA:-}" ] || [ -z "${EXPECTED_S01_STATE_HASH:-}" ] || [ -z "${S01_OUTPUT_ROOT:-}" ]; then
  echo "EXPECTED_S01_SHA, EXPECTED_S01_STATE_HASH, and exact S01_OUTPUT_ROOT are required" >&2
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
S01_STATE_HASH="$({
  find fl_v3/src/fl_v3/data/nuscenes -type f ! -path '*/__pycache__/*'
  printf '%s\n' \
    fl_v3/scripts/s01_nuscenes_zip_manifest.py \
    fl_v3/scripts/s01_nuscenes_zip_smoke.py \
    fl_v3/scripts/run_s01_nuscenes_zip_smoke.sh
} | sort -u | while IFS= read -r path; do sha256sum "${path}"; done | sha256sum | awk '{print $1}')"
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
export NUSCENES_ZIP_MANIFEST="${S01_OUTPUT_ROOT}/trainval01_manifest.sqlite"
export S01_ARCHIVE="${S01_ARCHIVE:-trainval01_blobs.zip}"

# Fail closed before the first mkdir if the requested output resolves under the
# immutable shared dataset.
python - "${S01_OUTPUT_ROOT}" "${NUSCENES_DATAROOT}" <<'PY'
import sys
from fl_v3.data.nuscenes import paths as P
P.resolve_writable(sys.argv[1], sys.argv[2])
PY
mkdir -p "${S01_OUTPUT_ROOT}"

echo "[S01 smoke] host=$(hostname) arch=$(uname -m) job=${SLURM_JOB_ID:-unset}"
echo "[S01 smoke] sha=${ACTUAL_SHA} state_hash=${S01_STATE_HASH} archive=${S01_ARCHIVE}"
echo "[S01 smoke] dataroot=${NUSCENES_DATAROOT} output=${S01_OUTPUT_ROOT}"

python fl_v3/scripts/s01_nuscenes_zip_manifest.py \
  --dataroot "${NUSCENES_DATAROOT}" \
  --manifest "${NUSCENES_ZIP_MANIFEST}" \
  --archives "${S01_ARCHIVE}"

python fl_v3/scripts/s01_nuscenes_zip_smoke.py \
  --dataroot "${NUSCENES_DATAROOT}" \
  --manifest "${NUSCENES_ZIP_MANIFEST}" \
  --archive "${S01_ARCHIVE}" \
  --version v1.0-trainval \
  --n-sweeps 10 \
  --num-samples 4 \
  --max-candidates 4000 \
  --num-workers 2 \
  --output "${S01_OUTPUT_ROOT}/smoke_report.json"

sha256sum \
  "${NUSCENES_ZIP_MANIFEST}" \
  "${S01_OUTPUT_ROOT}/smoke_report.json" \
  > "${S01_OUTPUT_ROOT}/sha256sums.txt"
cat "${S01_OUTPUT_ROOT}/sha256sums.txt"
