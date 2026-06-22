#!/bin/bash
# Centralized training on 4×A100 via torchrun DistributedDataParallel (MCR Phase 2). batch-size is
# PER-GPU (default 16 → global 64 over 4 ranks; scale the LR for the larger global batch). Trains then
# runs the official readiness eval --diagnostic on a single GPU. The single-GPU path is run_centralized_a40.sh.
#   Smoke: MAX_STEPS=20 EPOCHS=1 sbatch fl_v3/scripts/run_centralized_a100_ddp.sh   (validate DDP plumbing)
#   Run:   EPOCHS=15 LR=0.006 sbatch fl_v3/scripts/run_centralized_a100_ddp.sh
#
#SBATCH -A NAISS2025-22-1113
#SBATCH -p alvis
#SBATCH --job-name=centr_ddp
#SBATCH --gpus-per-node=A100:4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=08:00:00
#SBATCH --output=fl_v3/scripts/logs/centr_ddp_%j.out
#SBATCH --error=fl_v3/scripts/logs/centr_ddp_%j.err
set -euo pipefail

PROJ_ROOT="/mimer/NOBACKUP/groups/naiss2024-22-991/gaohui/thesis_workspace/fl_weather_project"
if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then REPO="${SLURM_SUBMIT_DIR}"; else REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; fi
cd "$REPO"; mkdir -p fl_v3/scripts/logs

CONFIG="${CONFIG:-fl_v3/configs/p1_unfrozen.json}"
EPOCHS="${EPOCHS:-5}"
BATCH_PER_GPU="${BATCH_PER_GPU:-16}"          # global batch = BATCH_PER_GPU * 4
NPROC="${NPROC:-4}"
TAG="${TAG:-centr_ddp_r${EPOCHS}}"
TRAINVAL_CACHE="${TRAINVAL_CACHE:-${PROJ_ROOT}/.claude/worktrees/infallible-feistel-d42c34/fl_outputs/nuscenes/info_cache}"
OUT_DIR="${REPO}/fl_outputs/nuscenes/experiments/cycle_04/p2_ddp"
LR_OV=""; [ -n "${LR:-}" ] && LR_OV="learning-rate=${LR}"
MAXSTEPS_FLAG=""; [ -n "${MAX_STEPS:-}" ] && MAXSTEPS_FLAG="--max-steps ${MAX_STEPS}"
RESUME_FLAG=""; [ "${RESUME:-0}" = "1" ] && RESUME_FLAG="--resume"

export CUBLAS_WORKSPACE_CONFIG=":4096:8"      # pre-CUDA on every rank (torchrun inherits the env)
export TORCH_HOME="/cephyr/users/gaohui/Alvis/.cache/torch"
export WANDB_MODE=offline
export OMP_NUM_THREADS=8
if ! type module >/dev/null 2>&1; then [ -f /usr/share/lmod/lmod/init/bash ] && source /usr/share/lmod/lmod/init/bash; fi
module purge; module load PyTorch/2.7.1-foss-2024a-CUDA-12.6.0
# shellcheck disable=SC1091
source "${PROJ_ROOT}/.venv_v3/bin/activate"
export PYTHONPATH="${REPO}/fl_v3/src${PYTHONPATH:+:$PYTHONPATH}"

echo "===== centralized DDP (epochs=${EPOCHS} batch/gpu=${BATCH_PER_GPU} global=$((BATCH_PER_GPU*NPROC)) lr_ov=${LR_OV:-config}) =====  node=$(hostname) job=${SLURM_JOB_ID:-local}"
nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv,noheader || true
[ -f "${TRAINVAL_CACHE}/nuscenes_info_v1.0-trainval_train_t1.v1.pkl" ] || { echo "[ddp] FATAL: trainval cache missing"; exit 3; }
[ -f "${TORCH_HOME}/hub/checkpoints/swin_t-704ceda3.pth" ] || { echo "[ddp] FATAL: swin_t weights missing"; exit 3; }

# --- train (4 ranks via torchrun; --standalone = single-node rendezvous) ---
torchrun --standalone --nproc_per_node="${NPROC}" fl_v3/scripts/centralized_train.py \
    --config "$CONFIG" --epochs "$EPOCHS" --out-dir "$OUT_DIR" --tag "$TAG" --compile --compile-bev \
    $RESUME_FLAG $MAXSTEPS_FLAG \
    "precision=bf16" "batch-size=${BATCH_PER_GPU}" "num-workers=8" $LR_OV "nuscenes-cache-dir=${TRAINVAL_CACHE}"

CKPT="${OUT_DIR}/${TAG}/final_model.pt"
[ -f "$CKPT" ] || { echo "[ddp] FATAL: no checkpoint at ${CKPT}"; exit 1; }
if [ -n "${MAX_STEPS:-}" ]; then echo "[ddp] capped smoke — skipping readiness eval"; exit 0; fi

# --- official diagnostic eval on ONE GPU (the SDPA backbone + bf16, matching training) ---
EVAL_OUT="${OUT_DIR}/${TAG}/readiness_diag"; mkdir -p "$EVAL_OUT"
CUDA_VISIBLE_DEVICES=0 python fl_v3/scripts/t4_readiness_eval.py \
    --config "$CONFIG" --checkpoint "$CKPT" --output-dir "$EVAL_OUT" --diagnostic \
    "nuscenes-cache-dir=${TRAINVAL_CACHE}" "precision=bf16"
echo "===== DDP centralized verdict (DIAGNOSTIC) ====="
python3 -c "import json;d=json.load(open('${EVAL_OUT}/benchmark_readiness.json'));print('verdict',d['verdict'],'| mAP',round(d['mAP'],4),'NDS',round(d['NDS'],4),'car_recall',round(d['official_clean_car_recall'],4),'eligible_N',d['eligible_count'])" || true
echo "Elapsed: ${SECONDS}s"
