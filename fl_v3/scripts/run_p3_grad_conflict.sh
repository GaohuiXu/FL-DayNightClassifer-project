#!/bin/bash
# MCR Phase-3 STEP 2 — per-client gradient/update conflict analysis (single A100, DIAGNOSTIC, ~45 min).
# Runs each of the 25 log_group clients' 1 local epoch from a fixed global snapshot, computes per-module
# cross-client cosine + per-class heatmap sign-agreement. Reuses train_local; no platform code change.
#   CKPT=<round_N>/final_model.pt CACHE=<msweep10> sbatch fl_v3/scripts/run_p3_grad_conflict.sh
#SBATCH -A NAISS2025-22-1113
#SBATCH -p alvis
#SBATCH --job-name=p3_grad
#SBATCH --gpus-per-node=A100:1
#SBATCH --cpus-per-task=16
#SBATCH --time=01:30:00
#SBATCH --output=fl_v3/scripts/logs/p3_grad_%j.out
#SBATCH --error=fl_v3/scripts/logs/p3_grad_%j.err
set -euo pipefail

PROJ_ROOT="/mimer/NOBACKUP/groups/naiss2024-22-991/gaohui/thesis_workspace/fl_weather_project"
if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then REPO="${SLURM_SUBMIT_DIR}"; else REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; fi
cd "$REPO"; mkdir -p fl_v3/scripts/logs

CONFIG="${CONFIG:-fl_v3/configs/fl_bb02d_fedadam.json}"
CKPT="${CKPT:?set CKPT=<dir>/final_model.pt}"
CACHE="${CACHE:-${PROJ_ROOT}/.claude/worktrees/unruffled-chaplygin-0e43e5/fl_outputs/nuscenes/info_cache_msweep10}"
MAXSTEPS="${MAXSTEPS:-0}"
OUT="${OUT:-fl_v3/collab/fl_baseline/p3_grad_conflict.json}"

export CUBLAS_WORKSPACE_CONFIG=":4096:8"
export TORCH_HOME="/cephyr/users/gaohui/Alvis/.cache/torch"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
if ! type module >/dev/null 2>&1; then [ -f /usr/share/lmod/lmod/init/bash ] && source /usr/share/lmod/lmod/init/bash; fi
module purge; module load PyTorch/2.7.1-foss-2024a-CUDA-12.6.0
# shellcheck disable=SC1091
source "${PROJ_ROOT}/.venv_v3/bin/activate"
export PYTHONPATH="${REPO}/fl_v3/src${PYTHONPATH:+:$PYTHONPATH}"

echo "===== P3 grad-conflict =====  node=$(hostname) job=${SLURM_JOB_ID:-local}  ckpt=${CKPT}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
python fl_v3/scripts/p3_grad_conflict.py --config "$CONFIG" --checkpoint "$CKPT" --max-steps "$MAXSTEPS" \
    --out "$OUT" "nuscenes-cache-dir=${CACHE}" "precision=bf16"
echo "Elapsed: ${SECONDS}s"
