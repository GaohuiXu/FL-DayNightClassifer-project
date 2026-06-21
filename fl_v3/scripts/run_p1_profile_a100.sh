#!/bin/bash
# MCR Phase-1 per-component profiler for the UNFROZEN model on ONE A100 (single-GPU). SHORT job
# (<<3h) — the prerequisite diagnostic before any heavy backbone-training run (orchestrator rule):
# per-component fwd+bwd teardown + GPU-util + activation-ckpt overhead + batch-size sweep.
# Submit:  sbatch fl_v3/scripts/run_p1_profile_a100.sh
#
#SBATCH -A NAISS2025-22-1113
#SBATCH -p alvis
#SBATCH --job-name=p1_profile
#SBATCH --gpus-per-node=A100:1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:40:00
#SBATCH --output=fl_v3/scripts/logs/p1_profile_%j.out
#SBATCH --error=fl_v3/scripts/logs/p1_profile_%j.err
set -euo pipefail

PROJ_ROOT="/mimer/NOBACKUP/groups/naiss2024-22-991/gaohui/thesis_workspace/fl_weather_project"
if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then REPO="${SLURM_SUBMIT_DIR}"; else REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; fi
cd "$REPO"; mkdir -p fl_v3/scripts/logs fl_v3/collab/model_capability

CONFIG="${CONFIG:-fl_v3/configs/p1_unfrozen.json}"
STEPS="${STEPS:-20}"; WARMUP="${WARMUP:-6}"; BATCH_SIZES="${BATCH_SIZES:-16,24,32}"
OUT="${OUT:-fl_v3/collab/model_capability/p1_profile_a100.json}"
TRAINVAL_CACHE="${TRAINVAL_CACHE:-${PROJ_ROOT}/.claude/worktrees/infallible-feistel-d42c34/fl_outputs/nuscenes/info_cache}"

export CUBLAS_WORKSPACE_CONFIG=":4096:8"
export TORCH_HOME="/cephyr/users/gaohui/Alvis/.cache/torch"
export WANDB_MODE=offline
if ! type module >/dev/null 2>&1; then [ -f /usr/share/lmod/lmod/init/bash ] && source /usr/share/lmod/lmod/init/bash; fi
module purge; module load PyTorch/2.7.1-foss-2024a-CUDA-12.6.0
# shellcheck disable=SC1091
source "${PROJ_ROOT}/.venv_v3/bin/activate"
export PYTHONPATH="${REPO}/fl_v3/src${PYTHONPATH:+:$PYTHONPATH}"

echo "===== P1 per-component profile (UNFROZEN, single A100) =====  node=$(hostname) job=${SLURM_JOB_ID:-local}"
nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv,noheader || true
[ -f "${TRAINVAL_CACHE}/nuscenes_info_v1.0-trainval_train_t1.v1.pkl" ] || { echo "[p1-profile] FATAL: trainval cache missing at ${TRAINVAL_CACHE}"; exit 3; }
[ -f "${TORCH_HOME}/hub/checkpoints/swin_t-704ceda3.pth" ] || { echo "[p1-profile] FATAL: swin_t weights missing under TORCH_HOME"; exit 3; }

python fl_v3/scripts/p1_profile_a100.py \
    --config "$CONFIG" --steps "$STEPS" --warmup "$WARMUP" --batch-sizes "$BATCH_SIZES" --out "$OUT" \
    "nuscenes-cache-dir=${TRAINVAL_CACHE}"
echo "[p1-profile] done; elapsed ${SECONDS}s; wrote ${OUT}"
