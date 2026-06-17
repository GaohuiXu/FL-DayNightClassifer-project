#!/bin/bash
# T3 milestone MEASUREMENT on the A40 — IID-mini FedAvg ≈ centralized (SPEC §2.4, falsifiable).
# In-process (local_runner + a central loop), no Ray needed. Submit: sbatch run_t3_milestone_a40.sh
#
#SBATCH -A NAISS2025-22-1113
#SBATCH -p alvis
#SBATCH --job-name=t3_milestone
#SBATCH --gpus-per-node=A40:1
#SBATCH --time=02:00:00
#SBATCH --output=fl_v3/scripts/logs/t3_milestone_%j.out
#SBATCH --error=fl_v3/scripts/logs/t3_milestone_%j.err
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

CONFIG="${CONFIG:-fl_v3/configs/t3_fl_gate.json}"
NUM_ROUNDS="${NUM_ROUNDS:-15}"
NUM_CLIENTS="${NUM_CLIENTS:-4}"
LOCAL_EPOCHS="${LOCAL_EPOCHS:-2}"
R_FLOOR="${R_FLOOR:-0.05}"
DELTA="${DELTA:-0.15}"
OUT="${OUT:-fl_v3/collab/T3/iid_vs_central_mini.json}"

echo "===== T3 milestone (IID-mini vs central) =====  node=$(hostname) job=${SLURM_JOB_ID:-local}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
python fl_v3/scripts/t3_iid_vs_central.py --config "$CONFIG" \
    --num-rounds "$NUM_ROUNDS" --num-clients "$NUM_CLIENTS" --local-epochs "$LOCAL_EPOCHS" \
    --r-floor "$R_FLOOR" --delta "$DELTA" --out "$OUT"
echo "Elapsed: ${SECONDS}s"
