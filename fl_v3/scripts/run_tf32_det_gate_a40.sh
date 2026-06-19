#!/bin/bash
# A40-pinned TF32 determinism gate (Cycle-04 D14, Phase-1 C step 0 — the prerequisite for any
# TF32 scientific run). Exercises the REAL enforce_determinism(numeric_mode='tf32') path on real
# TF32 hardware; the gate script HARD-REFUSES cc<8 so it can never false-pass on the login T4.
# Submit:  sbatch fl_v3/scripts/run_tf32_det_gate_a40.sh
#
#SBATCH -A NAISS2025-22-1113
#SBATCH -p alvis
#SBATCH --job-name=tf32_det_gate
#SBATCH --gpus-per-node=A40:1
#SBATCH --time=00:15:00
#SBATCH --output=fl_v3/scripts/logs/tf32_det_gate_%j.out
#SBATCH --error=fl_v3/scripts/logs/tf32_det_gate_%j.err
set -euo pipefail

PROJ_ROOT="/mimer/NOBACKUP/groups/naiss2024-22-991/gaohui/thesis_workspace/fl_weather_project"
if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then REPO="${SLURM_SUBMIT_DIR}"; else REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; fi
cd "$REPO"; mkdir -p fl_v3/scripts/logs fl_v3/collab/speedup

export CUBLAS_WORKSPACE_CONFIG=":4096:8"
export TORCH_HOME="/cephyr/users/gaohui/Alvis/.cache/torch"
if ! type module >/dev/null 2>&1; then [ -f /usr/share/lmod/lmod/init/bash ] && source /usr/share/lmod/lmod/init/bash; fi
module purge; module load PyTorch/2.7.1-foss-2024a-CUDA-12.6.0
# shellcheck disable=SC1091
source "${PROJ_ROOT}/.venv_v3/bin/activate"
# Make THIS worktree's code authoritative over the shared editable-install (.pth points elsewhere).
export PYTHONPATH="${REPO}/fl_v3/src${PYTHONPATH:+:$PYTHONPATH}"

echo "[run_tf32_det_gate_a40] node=$(hostname) job=${SLURM_JOB_ID:-local}"
nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv,noheader || true
python "${REPO}/fl_v3/scripts/tf32_det_gate_a40.py" "fl_v3/collab/speedup/tf32_det_gate_a40.json"
echo "[run_tf32_det_gate_a40] done; elapsed ${SECONDS}s"
