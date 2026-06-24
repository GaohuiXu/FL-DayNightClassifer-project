#!/bin/bash
# A100:1 VRAM/step-time probe for the LiDAR backbone × voxel-resolution matrix (MCR P1 OOM gate). ~10 min.
#SBATCH -A NAISS2025-22-1113
#SBATCH -p alvis
#SBATCH --job-name=vram_probe
#SBATCH --gpus-per-node=A100:1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:30:00
#SBATCH --output=fl_v3/scripts/logs/vram_probe_%j.out
#SBATCH --error=fl_v3/scripts/logs/vram_probe_%j.err
set -euo pipefail
PROJ_ROOT="/mimer/NOBACKUP/groups/naiss2024-22-991/gaohui/thesis_workspace/fl_weather_project"
if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then REPO="${SLURM_SUBMIT_DIR}"; else REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; fi
cd "$REPO"; mkdir -p fl_v3/scripts/logs
export CUBLAS_WORKSPACE_CONFIG=":4096:8" TORCH_HOME="/cephyr/users/gaohui/Alvis/.cache/torch"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
if ! type module >/dev/null 2>&1; then [ -f /usr/share/lmod/lmod/init/bash ] && source /usr/share/lmod/lmod/init/bash; fi
module purge; module load PyTorch/2.7.1-foss-2024a-CUDA-12.6.0
# shellcheck disable=SC1091
source "${PROJ_ROOT}/.venv_v3/bin/activate"
export PYTHONPATH="${REPO}/fl_v3/src${PYTHONPATH:+:$PYTHONPATH}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
python fl_v3/scripts/p1_vram_probe.py
echo "Elapsed: ${SECONDS}s"
