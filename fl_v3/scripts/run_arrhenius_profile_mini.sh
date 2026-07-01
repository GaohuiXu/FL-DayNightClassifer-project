#!/bin/bash
# Arrhenius Stop D mini profiling launcher.
#
# Engineering-only mini profiling:
#   sbatch fl_v3/scripts/run_arrhenius_profile_mini.sh
#
# This records module/stage time plus GPU telemetry. It does not run Best Config
# Smoke and must not be used for scientific performance or detection claims.
#SBATCH -A naiss2025-22-1113-gpu
#SBATCH -p gpu
#SBATCH --job-name=flv3_stopd_prof
#SBATCH --gpus-per-node=nvidia_gh200_120gb:1
#SBATCH --cpus-per-task=16
#SBATCH --time=01:00:00
#SBATCH --output=fl_v3/scripts/logs/arrhenius_profile_mini_%j.out
#SBATCH --error=fl_v3/scripts/logs/arrhenius_profile_mini_%j.err
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
OUT_DIR="${OUT_DIR:-${ARRHENIUS_OUTPUT_ROOT}/stop_d_profile_mini_${RUN_STAMP}}"
mkdir -p "${OUT_DIR}" "${ARRHENIUS_NUSCENES_CACHE}" "${ARRHENIUS_OUTPUT_ROOT}"

echo "[run_arrhenius_profile_mini] host=$(hostname) arch=$(uname -m)"
echo "[run_arrhenius_profile_mini] env=${ARRHENIUS_VENV}"
echo "[run_arrhenius_profile_mini] repo=${REPO}"
echo "[run_arrhenius_profile_mini] dataroot=${ARRHENIUS_NUSCENES_DATAROOT}"
echo "[run_arrhenius_profile_mini] cache=${ARRHENIUS_NUSCENES_CACHE}"
echo "[run_arrhenius_profile_mini] out=${OUT_DIR}"
echo "[run_arrhenius_profile_mini] matrix=${MATRIX:-pillar_fp32,voxel_fp32,voxel_fp16} warmup=${WARMUP_ITERS:-2} iters=${PROFILE_ITERS:-5}"
echo "[run_arrhenius_profile_mini] batch=${BATCH_SIZE:-1} workers=${NUM_WORKERS:-0} pin=${PIN_MEMORY:-1} prefetch=${PREFETCH_FACTOR:-4}"
echo "[run_arrhenius_profile_mini] Best Config Smoke is intentionally not run in Stop D."

EXTRA=()
if [ "${PRETRAINED_BACKBONE:-0}" = "1" ]; then
  EXTRA+=(--pretrained-backbone)
fi
if [ "${PIN_MEMORY:-1}" = "0" ]; then
  EXTRA+=(--no-pin-memory)
else
  EXTRA+=(--pin-memory)
fi
if [ "${PERSISTENT_WORKERS:-1}" = "0" ]; then
  EXTRA+=(--no-persistent-workers)
else
  EXTRA+=(--persistent-workers)
fi

python fl_v3/scripts/arrhenius_profile_mini.py \
  --config "${CONFIG:-fl_v3/configs/t4_mini_smoke.json}" \
  --dataroot "${ARRHENIUS_NUSCENES_DATAROOT}" \
  --cache-dir "${ARRHENIUS_NUSCENES_CACHE}" \
  --output-dir "${OUT_DIR}" \
  --matrix "${MATRIX:-pillar_fp32,voxel_fp32,voxel_fp16}" \
  --warmup-iters "${WARMUP_ITERS:-2}" \
  --profile-iters "${PROFILE_ITERS:-5}" \
  --num-tokens "${NUM_TOKENS:-2}" \
  --batch-size "${BATCH_SIZE:-1}" \
  --num-workers "${NUM_WORKERS:-0}" \
  --prefetch-factor "${PREFETCH_FACTOR:-4}" \
  --seed "${SEED:-42}" \
  --learning-rate "${LEARNING_RATE:-1e-4}" \
  --weight-decay "${WEIGHT_DECAY:-0.0}" \
  --backbone "${BACKBONE:-resnet18}" \
  --lidar-sweeps "${LIDAR_SWEEPS:-1}" \
  --max-pillars "${MAX_PILLARS:-30000}" \
  --max-points-per-pillar "${MAX_POINTS_PER_PILLAR:-32}" \
  --gpu-sample-ms "${GPU_SAMPLE_MS:-200}" \
  "${EXTRA[@]}"
