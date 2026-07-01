#!/bin/bash
# Submit this from the arrhenius/env-bringup-v3 worktree:
#   sbatch fl_v3/scripts/run_arrhenius_env_build.sh
# Optional:
#   sbatch --export=ALL,RECREATE=1 fl_v3/scripts/run_arrhenius_env_build.sh
#SBATCH -A naiss2025-22-1113-gpu
#SBATCH -p gpu
#SBATCH --job-name=flv3_env_build
#SBATCH --gpus-per-node=nvidia_gh200_120gb:1
#SBATCH --cpus-per-task=32
#SBATCH --time=02:00:00
#SBATCH --output=fl_v3/scripts/logs/arrhenius_env_build_%j.out
#SBATCH --error=fl_v3/scripts/logs/arrhenius_env_build_%j.err
set -euo pipefail

REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "${REPO}"
mkdir -p fl_v3/scripts/logs
exec bash fl_v3/scripts/build_arrhenius_env.sh
