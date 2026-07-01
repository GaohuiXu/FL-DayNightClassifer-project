#!/bin/bash
# Arrhenius Stop C mini tiny-overfit matrix launcher.
#
# Engineering-only mini validation:
#   sbatch fl_v3/scripts/run_arrhenius_mini_matrix.sh
#
# This does not run Best Config Smoke and must not be used for scientific claims.
# Outputs are written under ARRHENIUS_OUTPUT_ROOT, never under $HOME.
#SBATCH -A naiss2025-22-1113-gpu
#SBATCH -p gpu
#SBATCH --job-name=flv3_stopc_tiny
#SBATCH --gpus-per-node=nvidia_gh200_120gb:1
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH --output=fl_v3/scripts/logs/arrhenius_mini_matrix_%j.out
#SBATCH --error=fl_v3/scripts/logs/arrhenius_mini_matrix_%j.err
set -euo pipefail

REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "${REPO}"
mkdir -p fl_v3/scripts/logs

# shellcheck disable=SC1091
source fl_v3/scripts/arrhenius_env.sh

export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
if [ "${LOAD_BUILDENV:-1}" = "1" ]; then
  arrhenius_load_modules build
else
  arrhenius_load_modules run
fi
arrhenius_activate_env

export ARRHENIUS_NUSCENES_DATAROOT="${ARRHENIUS_NUSCENES_DATAROOT:-${REPO}/data/nuscenes_mini}"
export ARRHENIUS_NUSCENES_CACHE="${ARRHENIUS_NUSCENES_CACHE:-${ARRHENIUS_OUTPUT_ROOT}/nuscenes/info_cache_mini_from_main}"

RUN_STAMP="${SLURM_JOB_ID:-manual_$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="${OUT_DIR:-${ARRHENIUS_OUTPUT_ROOT}/stop_c_mini_tiny_overfit_${RUN_STAMP}}"
mkdir -p "${OUT_DIR}" "${ARRHENIUS_NUSCENES_CACHE}" "${ARRHENIUS_OUTPUT_ROOT}"

echo "[run_arrhenius_mini_matrix] host=$(hostname) arch=$(uname -m)"
echo "[run_arrhenius_mini_matrix] env=${ARRHENIUS_VENV}"
echo "[run_arrhenius_mini_matrix] repo=${REPO}"
echo "[run_arrhenius_mini_matrix] dataroot=${ARRHENIUS_NUSCENES_DATAROOT}"
echo "[run_arrhenius_mini_matrix] cache=${ARRHENIUS_NUSCENES_CACHE}"
echo "[run_arrhenius_mini_matrix] out=${OUT_DIR}"
echo "[run_arrhenius_mini_matrix] matrix=${MATRIX:-pillar_fp32,voxel_fp32,voxel_fp16} steps=${STEPS:-30} tokens=${NUM_TOKENS:-2}"
echo "[run_arrhenius_mini_matrix] Best Config Smoke is intentionally not run in Stop C."

EXTRA=()
if [ "${PRETRAINED_BACKBONE:-0}" = "1" ]; then
  EXTRA+=(--pretrained-backbone)
fi

python fl_v3/scripts/arrhenius_mini_matrix.py \
  --config "${CONFIG:-fl_v3/configs/t4_mini_smoke.json}" \
  --dataroot "${ARRHENIUS_NUSCENES_DATAROOT}" \
  --cache-dir "${ARRHENIUS_NUSCENES_CACHE}" \
  --output-dir "${OUT_DIR}" \
  --matrix "${MATRIX:-pillar_fp32,voxel_fp32,voxel_fp16}" \
  --steps "${STEPS:-30}" \
  --num-tokens "${NUM_TOKENS:-2}" \
  --batch-size "${BATCH_SIZE:-1}" \
  --num-workers "${NUM_WORKERS:-0}" \
  --seed "${SEED:-42}" \
  --learning-rate "${LEARNING_RATE:-1e-4}" \
  --weight-decay "${WEIGHT_DECAY:-0.0}" \
  --backbone "${BACKBONE:-resnet18}" \
  "${EXTRA[@]}"
