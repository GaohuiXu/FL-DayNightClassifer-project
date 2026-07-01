#!/bin/bash
# Arrhenius GH200 environment smoke launcher.
#
# Mini regression checks:
#   sbatch fl_v3/scripts/run_arrhenius_smoke.sh
#
# Non-data checks:
#   sbatch --export=ALL,REQUIRE_DATA=0,SMOKE_MODES='import spconv sparse-lidar dummy-train' fl_v3/scripts/run_arrhenius_smoke.sh
#SBATCH -A naiss2025-22-1113-gpu
#SBATCH -p gpu
#SBATCH --job-name=flv3_smoke
#SBATCH --gpus-per-node=nvidia_gh200_120gb:1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:40:00
#SBATCH --output=fl_v3/scripts/logs/arrhenius_smoke_%j.out
#SBATCH --error=fl_v3/scripts/logs/arrhenius_smoke_%j.err
set -euo pipefail

REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "${REPO}"
mkdir -p fl_v3/scripts/logs

# shellcheck disable=SC1091
source fl_v3/scripts/arrhenius_env.sh

# Source-built cumm/spconv can still trigger ccimport/ninja checks on import,
# so the smoke path defaults to the CUDA build environment as well.
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
if [ "${LOAD_BUILDENV:-1}" = "1" ]; then
  arrhenius_load_modules build
else
  arrhenius_load_modules run
fi
arrhenius_activate_env

export ARRHENIUS_NUSCENES_DATAROOT="${ARRHENIUS_NUSCENES_DATAROOT:-${REPO}/data/nuscenes_mini}"
export ARRHENIUS_NUSCENES_CACHE="${ARRHENIUS_NUSCENES_CACHE:-${ARRHENIUS_OUTPUT_ROOT}/nuscenes/info_cache_mini_from_main}"
mkdir -p "${ARRHENIUS_NUSCENES_CACHE}" "${ARRHENIUS_OUTPUT_ROOT}"

MODES="${SMOKE_MODES:-import spconv sparse-lidar data eval train}"
REQ=()
if [ "${REQUIRE_DATA:-1}" = "1" ]; then
  REQ+=(--require-data)
fi

echo "[run_arrhenius_smoke] host=$(hostname) arch=$(uname -m) modes=${MODES}"
echo "[run_arrhenius_smoke] env=${ARRHENIUS_VENV}"
echo "[run_arrhenius_smoke] precision=${PRECISION:-fp16} lidar_encoder=${LIDAR_ENCODER:-voxel}"
echo "[run_arrhenius_smoke] dataroot=${ARRHENIUS_NUSCENES_DATAROOT:-<unset>}"
echo "[run_arrhenius_smoke] cache=${ARRHENIUS_NUSCENES_CACHE}"

python fl_v3/scripts/arrhenius_smoke.py \
  --dataroot "${ARRHENIUS_NUSCENES_DATAROOT:-}" \
  --cache-dir "${ARRHENIUS_NUSCENES_CACHE}" \
  --output-dir "${ARRHENIUS_OUTPUT_ROOT}" \
  --precision "${PRECISION:-fp16}" \
  --lidar-encoder "${LIDAR_ENCODER:-voxel}" \
  --eval-limit "${EVAL_LIMIT:-2}" \
  --train-steps "${TRAIN_STEPS:-1}" \
  --batch-size "${BATCH_SIZE:-1}" \
  --num-workers "${NUM_WORKERS:-0}" \
  --min-keyframes-per-client "${MIN_KEYFRAMES_PER_CLIENT:-0}" \
  "${REQ[@]}" \
  ${MODES}
