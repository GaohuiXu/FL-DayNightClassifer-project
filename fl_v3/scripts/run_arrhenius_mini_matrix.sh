#!/bin/bash
# Arrhenius camera-branch mini audit / tiny-overfit matrix launcher.
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

CANONICAL_MINI="/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/fl_weather_project/data/nuscenes_mini"
DEFAULT_MINI="${REPO}/data/nuscenes_mini"
if [ ! -d "${DEFAULT_MINI}" ] && [ -d "${CANONICAL_MINI}" ]; then
  DEFAULT_MINI="${CANONICAL_MINI}"
fi
export ARRHENIUS_NUSCENES_DATAROOT="${ARRHENIUS_NUSCENES_DATAROOT:-${DEFAULT_MINI}}"
export ARRHENIUS_NUSCENES_CACHE="${ARRHENIUS_NUSCENES_CACHE:-${ARRHENIUS_OUTPUT_ROOT}/nuscenes/info_cache_mini_from_main}"

RUN_STAMP="${SLURM_JOB_ID:-manual_$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="${OUT_DIR:-${ARRHENIUS_OUTPUT_ROOT}/camera_audit_mini_matrix_${RUN_STAMP}}"
mkdir -p "${OUT_DIR}" "${ARRHENIUS_NUSCENES_CACHE}" "${ARRHENIUS_OUTPUT_ROOT}"

echo "[run_arrhenius_mini_matrix] host=$(hostname) arch=$(uname -m)"
echo "[run_arrhenius_mini_matrix] env=${ARRHENIUS_VENV}"
echo "[run_arrhenius_mini_matrix] repo=${REPO}"
echo "[run_arrhenius_mini_matrix] dataroot=${ARRHENIUS_NUSCENES_DATAROOT}"
echo "[run_arrhenius_mini_matrix] cache=${ARRHENIUS_NUSCENES_CACHE}"
echo "[run_arrhenius_mini_matrix] out=${OUT_DIR}"
echo "[run_arrhenius_mini_matrix] matrix=${MATRIX:-camera_iso_020_fp16_swin} steps=${STEPS:-30} tokens=${NUM_TOKENS:-2}"
echo "[run_arrhenius_mini_matrix] branch_topology=${BRANCH_TOPOLOGY:-full_fusion} train_policy=${TRAIN_POLICY:-all_trainable} respect_config_shape=${RESPECT_CONFIG_SHAPE:-0} branch_delta=${BRANCH_DELTA_SANITY:-0}"
echo "[run_arrhenius_mini_matrix] grad_scale_init=${GRAD_SCALE_INIT:-512.0}"
echo "[run_arrhenius_mini_matrix] backbone=${BACKBONE:-swin_t} pretrained=${PRETRAINED_BACKBONE:-1}"
echo "[run_arrhenius_mini_matrix] Best Config Smoke is intentionally not run in the camera audit."

EXTRA=()
if [ "${PRETRAINED_BACKBONE:-1}" = "0" ]; then
  EXTRA+=(--no-pretrained-backbone)
else
  EXTRA+=(--pretrained-backbone)
fi
if [ "${RESPECT_CONFIG_SHAPE:-0}" = "1" ]; then
  EXTRA+=(--respect-config-shape)
fi
if [ "${BRANCH_DELTA_SANITY:-0}" = "1" ]; then
  EXTRA+=(--branch-delta-sanity)
fi

python fl_v3/scripts/arrhenius_mini_matrix.py \
  --config "${CONFIG:-fl_v3/configs/p1_bb02d_voxel.json}" \
  --dataroot "${ARRHENIUS_NUSCENES_DATAROOT}" \
  --cache-dir "${ARRHENIUS_NUSCENES_CACHE}" \
  --output-dir "${OUT_DIR}" \
  --matrix "${MATRIX:-camera_iso_020_fp16_swin}" \
  --branch-topology "${BRANCH_TOPOLOGY:-full_fusion}" \
  --train-policy "${TRAIN_POLICY:-all_trainable}" \
  --steps "${STEPS:-30}" \
  --num-tokens "${NUM_TOKENS:-2}" \
  --batch-size "${BATCH_SIZE:-1}" \
  --num-workers "${NUM_WORKERS:-0}" \
  --seed "${SEED:-42}" \
  --learning-rate "${LEARNING_RATE:-1e-4}" \
  --weight-decay "${WEIGHT_DECAY:-0.0}" \
  --grad-scale-init "${GRAD_SCALE_INIT:-512.0}" \
  --backbone "${BACKBONE:-swin_t}" \
  "${EXTRA[@]}"
