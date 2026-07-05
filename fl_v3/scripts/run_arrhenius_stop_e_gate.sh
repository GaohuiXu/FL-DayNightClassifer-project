#!/bin/bash
# Arrhenius Stop E capped gate launcher.
#
# Capped engineering gates only:
#   GATE=gate1 CELL=bb02d_020_control sbatch fl_v3/scripts/run_arrhenius_stop_e_gate.sh
#   GATE=gate2 CELL=sparse_020_matched sbatch fl_v3/scripts/run_arrhenius_stop_e_gate.sh
#
# This never launches an uncapped full scientific run. Full trainval science must
# be submitted separately after explicit approval.
#SBATCH -A naiss2025-22-1113-gpu
#SBATCH -p gpu
#SBATCH --job-name=flv3_stop_e_gate
#SBATCH --gpus-per-node=nvidia_gh200_120gb:1
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00
#SBATCH --output=fl_v3/scripts/logs/arrhenius_stop_e_gate_%j.out
#SBATCH --error=fl_v3/scripts/logs/arrhenius_stop_e_gate_%j.err
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

DEFAULT_TRAINVAL="/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/fl_weather_project/data/nuscenes_trainval"
export ARRHENIUS_NUSCENES_DATAROOT="${ARRHENIUS_NUSCENES_DATAROOT:-${DEFAULT_TRAINVAL}}"
export ARRHENIUS_NUSCENES_CACHE="${ARRHENIUS_NUSCENES_CACHE:-${ARRHENIUS_OUTPUT_ROOT}/nuscenes/info_cache_msweep10}"

if [ ! -d "${ARRHENIUS_NUSCENES_DATAROOT}/v1.0-trainval" ]; then
  echo "[stop_e_gate] ERROR: trainval dataroot missing: ${ARRHENIUS_NUSCENES_DATAROOT}/v1.0-trainval" >&2
  exit 3
fi
if ! compgen -G "${ARRHENIUS_NUSCENES_CACHE}/nuscenes_info_v1.0-trainval_train_*.pkl" >/dev/null; then
  echo "[stop_e_gate] ERROR: trainval train info-cache missing under ${ARRHENIUS_NUSCENES_CACHE}" >&2
  exit 3
fi

GATE="${GATE:-gate1}"
case "${GATE}" in
  gate1) DEFAULT_MAX_STEPS=100 ;;
  gate2) DEFAULT_MAX_STEPS=1000 ;;
  *)
    echo "[stop_e_gate] ERROR: GATE must be gate1 or gate2, got ${GATE}" >&2
    exit 2
    ;;
esac
MAX_STEPS="${MAX_STEPS:-${DEFAULT_MAX_STEPS}}"
if [ "${MAX_STEPS}" -le 0 ]; then
  echo "[stop_e_gate] ERROR: Stop E gates must be capped; MAX_STEPS=${MAX_STEPS}" >&2
  exit 2
fi

CELL="${CELL:-bb02d_020_control}"
CONFIG="${CONFIG:-fl_v3/configs/p1_bb02d.json}"
RUN_STAMP="${SLURM_JOB_ID:-manual_$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="${OUT_DIR:-${ARRHENIUS_OUTPUT_ROOT}/stop_e_${GATE}_${RUN_STAMP}}"
TAG="${TAG:-${CELL}}"
mkdir -p "${OUT_DIR}" "${ARRHENIUS_OUTPUT_ROOT}"

OVERRIDES=(
  "nuscenes-dataroot=${ARRHENIUS_NUSCENES_DATAROOT}"
  "nuscenes-cache-dir=${ARRHENIUS_NUSCENES_CACHE}"
  "stop-e-cell=${CELL}"
  "wandb-enabled=false"
  "wandb-mode=disabled"
  "train-telemetry-interval=${TRAIN_TELEMETRY_INTERVAL:-10}"
)

case "${CELL}" in
  bb02d_020_control)
    OVERRIDES+=(
      "det-lidar-encoder=pillar"
      "det-bev-voxel=0.2"
    )
    ;;
  sparse_020_matched)
    OVERRIDES+=(
      "det-lidar-encoder=voxel"
      "det-bev-voxel=0.2"
      "det-lidar-z-voxel=0.2"
      "det-sparse-grad-scale-init=${SPARSE_GRAD_SCALE_INIT:-1.0}"
    )
    ;;
  sparse_075_z020_ch256_parity)
    OVERRIDES+=(
      "batch-size=${BATCH_SIZE:-2}"
      "det-lidar-encoder=voxel"
      "det-bev-voxel=0.075"
      "det-lidar-z-voxel=0.2"
      "det-lidar-sparse-z-size=41"
      "det-pc-range=[-54.0,-54.0,-5.0,54.0,54.0,3.0]"
      "det-max-pillars=120000"
      "det-max-points-per-pillar=10"
      "det-lidar-backbone-out=256"
      "det-fusion-channels=256"
      "det-sparse-grad-scale-init=${SPARSE_GRAD_SCALE_INIT:-1.0}"
    )
    ;;
  *)
    echo "[stop_e_gate] ERROR: unknown CELL=${CELL}" >&2
    exit 2
    ;;
esac
if [ -n "${BATCH_SIZE:-}" ] && [ "${CELL}" != "sparse_075_z020_ch256_parity" ]; then
  OVERRIDES+=("batch-size=${BATCH_SIZE}")
fi

echo "[stop_e_gate] host=$(hostname) arch=$(uname -m)"
echo "[stop_e_gate] env=${ARRHENIUS_VENV}"
echo "[stop_e_gate] repo=${REPO}"
echo "[stop_e_gate] gate=${GATE} max_steps=${MAX_STEPS} cell=${CELL}"
echo "[stop_e_gate] dataroot=${ARRHENIUS_NUSCENES_DATAROOT}"
echo "[stop_e_gate] cache=${ARRHENIUS_NUSCENES_CACHE}"
echo "[stop_e_gate] out=${OUT_DIR} tag=${TAG}"
echo "[stop_e_gate] scientific_claim=false; capped gate only"

python fl_v3/scripts/centralized_train.py \
  --config "${CONFIG}" \
  --epochs 1 \
  --max-steps "${MAX_STEPS}" \
  --out-dir "${OUT_DIR}" \
  --tag "${TAG}" \
  "${OVERRIDES[@]}"
