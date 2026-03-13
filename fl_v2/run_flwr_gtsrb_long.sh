#!/usr/bin/env bash
#SBATCH -A NAISS2025-22-1113
#SBATCH -p alvis
#SBATCH --gpus-per-node=T4:1
#SBATCH -t 0-02:00:00
#SBATCH -J flwr_gtsrb_long
#SBATCH -o /mimer/NOBACKUP/groups/naiss2024-22-991/gaohui/fl_outputs/slurm/%x_%j.out
#SBATCH -e /mimer/NOBACKUP/groups/naiss2024-22-991/gaohui/fl_outputs/slurm/%x_%j.err

set -euo pipefail

echo "===== Job info ====="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Start: $(date)"

mkdir -p /mimer/NOBACKUP/groups/naiss2024-22-991/gaohui/fl_outputs/slurm
mkdir -p /mimer/NOBACKUP/groups/naiss2024-22-991/gaohui/fl_outputs/gtsrb_v2

cd /cephyr/users/gaohui/Alvis/thesis_workspace/fl_weather_project/fl_v2
source /cephyr/users/gaohui/Alvis/thesis_workspace/fl_weather_project/.venv/bin/activate

export FLWR_HOME=/cephyr/users/gaohui/Alvis/.flwr
export RAY_DEDUP_LOGS=0

echo "===== Environment ====="
which python
python --version
which flwr

echo "===== Running Flower ====="
flwr run . local-simulation-gpu --stream

echo "End: $(date)"