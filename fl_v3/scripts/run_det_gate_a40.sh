#!/bin/bash
# A40-pinned bit-determinism gate + wall-clock (T2 GATE, SPEC §0).
# Submit:  sbatch fl_v3/scripts/run_det_gate_a40.sh
# The job ERRORS LOUD (exit 2) on no-CUDA / non-A40 — never a false green.
#
#SBATCH -A NAISS2025-22-1113
#SBATCH -p alvis
#SBATCH --job-name=t2_det_gate
#SBATCH --gpus-per-node=A40:1
#SBATCH --time=00:30:00
#SBATCH --output=fl_v3/scripts/logs/det_gate_%j.out
#SBATCH --error=fl_v3/scripts/logs/det_gate_%j.err
set -euo pipefail

PROJ_ROOT="/mimer/NOBACKUP/groups/naiss2024-22-991/gaohui/thesis_workspace/fl_weather_project"
# Under SLURM, BASH_SOURCE is the spool copy in /tmp/slurmd; the submission dir
# ($SLURM_SUBMIT_DIR, where `sbatch` was run = the repo/worktree root) is preserved
# and is the right anchor. Fall back to BASH_SOURCE for a local (non-SLURM) run.
if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    FL_V3_DIR="${SLURM_SUBMIT_DIR}/fl_v3"
else
    FL_V3_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
mkdir -p "${FL_V3_DIR}/scripts/logs"

# --- bit-determinism: cuBLAS workspace MUST be set BEFORE any CUDA context ---
export CUBLAS_WORKSPACE_CONFIG=":4096:8"
export TORCH_HOME="/cephyr/users/gaohui/Alvis/.cache/torch"

if ! type module >/dev/null 2>&1; then
    [ -f /usr/share/lmod/lmod/init/bash ] && source /usr/share/lmod/lmod/init/bash
fi
module purge
module load PyTorch/2.7.1-foss-2024a-CUDA-12.6.0
# shellcheck disable=SC1091
source "${PROJ_ROOT}/.venv_v3/bin/activate"

echo "[run_det_gate_a40] node=$(hostname) job=${SLURM_JOB_ID:-local}"
python "${FL_V3_DIR}/scripts/det_gate_a40.py"
