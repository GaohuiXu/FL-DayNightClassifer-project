#!/bin/bash
# Offline GT-database build as a SLURM job (survives session restarts; CPU/IO-bound — the A100 is unused but is
# the simplest node request, and GPU hours are plentiful pre-sunset). ~30-40 min over 28k 10-sweep keyframes.
#   CACHE=<msweep10> OUT=<gt_database dir> sbatch --exclude=alvis4-01,alvis4-02 fl_v3/scripts/run_build_gt_database.sh
#SBATCH -A NAISS2025-22-1113
#SBATCH -p alvis
#SBATCH --job-name=gtdb
#SBATCH --gpus-per-node=A100:1
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH --output=fl_v3/scripts/logs/gtdb_%j.out
#SBATCH --error=fl_v3/scripts/logs/gtdb_%j.err
set -euo pipefail

PROJ_ROOT="/mimer/NOBACKUP/groups/naiss2024-22-991/gaohui/thesis_workspace/fl_weather_project"
if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then REPO="${SLURM_SUBMIT_DIR}"; else REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; fi
cd "$REPO"; mkdir -p fl_v3/scripts/logs

CACHE="${CACHE:-${PROJ_ROOT}/.claude/worktrees/unruffled-chaplygin-0e43e5/fl_outputs/nuscenes/info_cache_msweep10}"
OUT="${OUT:-${REPO}/fl_outputs/nuscenes/gt_database_msweep10}"
DATAROOT="${DATAROOT:-/mimer/NOBACKUP/Datasets/NuScenes_v1.0}"
CLASSES="${CLASSES:-trailer,construction_vehicle,bus,truck,bicycle,motorcycle}"
NSWEEPS="${NSWEEPS:-10}"

export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
if ! type module >/dev/null 2>&1; then [ -f /usr/share/lmod/lmod/init/bash ] && source /usr/share/lmod/lmod/init/bash; fi
module purge
module load PyTorch/2.7.1-foss-2024a-CUDA-12.6.0 || module --ignore_cache load PyTorch/2.7.1-foss-2024a-CUDA-12.6.0
# shellcheck disable=SC1091
source "${PROJ_ROOT}/.venv_v3/bin/activate"
export PYTHONPATH="${REPO}/fl_v3/src${PYTHONPATH:+:$PYTHONPATH}"

echo "===== GT-DB build (cache=${CACHE} out=${OUT} n_sweeps=${NSWEEPS}) =====  node=$(hostname) job=${SLURM_JOB_ID:-local}"
python fl_v3/scripts/build_gt_database.py \
    --cache-dir "$CACHE" --version v1.0-trainval --split train --n-sweeps "$NSWEEPS" \
    --dataroot "$DATAROOT" --out-dir "$OUT" \
    --classes "$CLASSES" --min-points 5 --max-per-class 8000
echo "Elapsed: ${SECONDS}s"
