#!/bin/bash
# fl_v3 SLURM launcher — SCAFFOLD (T0). Hardened in T3 (the real Ray run).
#
# T0 establishes the determinism-critical bits; the full same-node startup-race
# hardening + per-job Ray ports + monitoring carry over from fl_v2/run_alvis.sh
# in T3 when the real BEVFusion FedAvg run lands. Heavy Flower/Ray runs go
# through SLURM (NEVER the login node).
#
#SBATCH --job-name=fl_v3
#SBATCH --gpus-per-node=A40:1
#SBATCH --time=04:00:00
#SBATCH --output=fl_v3/scripts/logs/slurm_%j.out
#SBATCH --error=fl_v3/scripts/logs/slurm_%j.err
set -euo pipefail

PROJ_ROOT="/mimer/NOBACKUP/groups/naiss2024-22-991/gaohui/thesis_workspace/fl_weather_project"

# --- bit-determinism: cuBLAS workspace MUST be set before any CUDA context ---
export CUBLAS_WORKSPACE_CONFIG=":4096:8"
# Pinned torch.hub cache for the T2 camera-backbone ImageNet weights (pre-cached on the
# login node by build_venv.sh; compute nodes are offline). Must match build_venv.sh.
export TORCH_HOME="/cephyr/users/gaohui/Alvis/.cache/torch"

if ! type module >/dev/null 2>&1; then
    [ -f /usr/share/lmod/lmod/init/bash ] && source /usr/share/lmod/lmod/init/bash
fi
module purge
module load PyTorch/2.7.1-foss-2024a-CUDA-12.6.0
# shellcheck disable=SC1091
source "${PROJ_ROOT}/.venv_v3/bin/activate"

# Per-job FLWR_HOME isolation (carry-over from fl_v2 — do NOT remove the /tmp
# indirection; it isolates concurrent jobs). T3 adds the per-job Ray ports +
# SuperLink startup-race waits.
export FLWR_HOME="/tmp/flwr_${SLURM_JOB_ID:-local}"
mkdir -p "${FLWR_HOME}"
cp fl_v3/configs/flwr_config.toml "${FLWR_HOME}/config.toml"
trap 'rm -rf "${FLWR_HOME}"' EXIT

# Example: flwr run . local-simulation-gpu  (real AD run config — T3)
echo "[run_alvis] T0 scaffold — wire 'flwr run' here in T3."
