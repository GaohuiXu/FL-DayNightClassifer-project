#!/bin/bash
# Depth-sup CL smoke (single A100, ~3-5 min): real-data forward+backward through bb02d+depth-supervision,
# prints the loss breakdown (hm/reg/depth) + GT coverage + grad-norm + no-NaN gate. R1 validation before the
# full CL run.   CACHE=<msweep10> sbatch fl_v3/scripts/run_p1_depth_smoke.sh
#SBATCH -A NAISS2025-22-1113
#SBATCH -p alvis
#SBATCH --job-name=p1_depth_smoke
#SBATCH --gpus-per-node=A100:1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:20:00
#SBATCH --output=fl_v3/scripts/logs/p1_depth_smoke_%j.out
#SBATCH --error=fl_v3/scripts/logs/p1_depth_smoke_%j.err
set -euo pipefail

PROJ_ROOT="/mimer/NOBACKUP/groups/naiss2024-22-991/gaohui/thesis_workspace/fl_weather_project"
if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then REPO="${SLURM_SUBMIT_DIR}"; else REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; fi
cd "$REPO"; mkdir -p fl_v3/scripts/logs

CONFIG="${CONFIG:-fl_v3/configs/p1_bb02d_depth.json}"
CACHE="${CACHE:-${PROJ_ROOT}/.claude/worktrees/unruffled-chaplygin-0e43e5/fl_outputs/nuscenes/info_cache_msweep10}"
BATCH="${BATCH:-4}"          # small batch for the smoke (safe on a 40GB card)
STEPS="${STEPS:-3}"
FP16="${FP16:-0}"            # 1 = fp16-AMP+GradScaler precision de-risk (Arrhenius) instead of bf16
EXTRA_OVERRIDES="${EXTRA_OVERRIDES:-}"

export CUBLAS_WORKSPACE_CONFIG=":4096:8"
export TORCH_HOME="/cephyr/users/gaohui/Alvis/.cache/torch"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
if ! type module >/dev/null 2>&1; then [ -f /usr/share/lmod/lmod/init/bash ] && source /usr/share/lmod/lmod/init/bash; fi
module purge; module load PyTorch/2.7.1-foss-2024a-CUDA-12.6.0
# shellcheck disable=SC1091
source "${PROJ_ROOT}/.venv_v3/bin/activate"
export PYTHONPATH="${REPO}/fl_v3/src${PYTHONPATH:+:$PYTHONPATH}"

echo "===== depth-sup smoke ===== node=$(hostname) job=${SLURM_JOB_ID:-local} config=${CONFIG}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true
FP16_FLAG=""; [ "$FP16" = "1" ] && FP16_FLAG="--fp16"
# shellcheck disable=SC2086
python fl_v3/scripts/p1_depth_smoke.py --config "$CONFIG" --steps "$STEPS" $FP16_FLAG \
    "nuscenes-cache-dir=${CACHE}" "batch-size=${BATCH}" "precision=bf16" ${EXTRA_OVERRIDES}
echo "Elapsed: ${SECONDS}s"
