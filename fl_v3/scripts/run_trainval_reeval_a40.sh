#!/bin/bash
# T3 Codex F1 fix — re-evaluate the trainval checkpoints on the FULL val split (eval-only).
# Submit: sbatch fl_v3/scripts/run_trainval_reeval_a40.sh
#
#SBATCH -A NAISS2025-22-1113
#SBATCH -p alvis
#SBATCH --job-name=t3_trainval_reeval
#SBATCH --gpus-per-node=A40:1
#SBATCH --time=01:30:00
#SBATCH --output=fl_v3/scripts/logs/t3_reeval_%j.out
#SBATCH --error=fl_v3/scripts/logs/t3_reeval_%j.err
set -euo pipefail

PROJ_ROOT="/mimer/NOBACKUP/groups/naiss2024-22-991/gaohui/thesis_workspace/fl_weather_project"
if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then REPO="${SLURM_SUBMIT_DIR}"; else REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; fi
cd "$REPO"; mkdir -p fl_v3/scripts/logs

export CUBLAS_WORKSPACE_CONFIG=":4096:8"
export TORCH_HOME="/cephyr/users/gaohui/Alvis/.cache/torch"
export WANDB_MODE=offline
if ! type module >/dev/null 2>&1; then [ -f /usr/share/lmod/lmod/init/bash ] && source /usr/share/lmod/lmod/init/bash; fi
module purge; module load PyTorch/2.7.1-foss-2024a-CUDA-12.6.0
# shellcheck disable=SC1091
source "${PROJ_ROOT}/.venv_v3/bin/activate"

echo "===== T3 trainval full-val re-eval =====  node=$(hostname) job=${SLURM_JOB_ID:-local}"
nvidia-smi --query-gpu=name --format=csv,noheader | head -1
python fl_v3/scripts/t3_trainval_reeval_fullval.py
echo "Elapsed: ${SECONDS}s"
