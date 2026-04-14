#!/usr/bin/env bash
#SBATCH -A NAISS2025-22-1113
#SBATCH -p alvis
#SBATCH -C NOGPU
#SBATCH -t 0-08:00:00
#SBATCH -J phaseD_extract
#SBATCH -o /mimer/NOBACKUP/groups/naiss2024-22-991/gaohui/fl_outputs/slurm/%x_%j.out
#SBATCH -e /mimer/NOBACKUP/groups/naiss2024-22-991/gaohui/fl_outputs/slurm/%x_%j.err

# ──────────────────────────────────────────────────────────────
# Phase D — STAGE 1: Feature extraction for model-replacement experiments.
#
# Runs forward pass on test set for all saved checkpoints.
# Idempotent: skips files that already exist.
#
# Usage:
#   sbatch analysis/run_phaseD_extract.sh
#   # or with dependency on training jobs:
#   sbatch --dependency=afterany:<jobA>:<jobB> analysis/run_phaseD_extract.sh
# ──────────────────────────────────────────────────────────────
set -euo pipefail

echo "===== Job info ====="
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Node: $(hostname)"
echo "Start: $(date)"

module purge
module load PyTorch/2.7.1-foss-2024a-CUDA-12.6.0

cd /cephyr/users/gaohui/Alvis/thesis_workspace/fl_weather_project/fl_v2
source /cephyr/users/gaohui/Alvis/thesis_workspace/fl_weather_project/.venv/bin/activate

export OPENBLAS_NUM_THREADS=16
export PYTHONUNBUFFERED=1

DATA_ROOT="/mimer/NOBACKUP/groups/naiss2024-22-991/gaohui/fl_datasets/gtsrb"
BASE_DIR="/mimer/NOBACKUP/groups/naiss2024-22-991/gaohui/fl_outputs/gtsrb_v2/phaseD"

ROUNDS=(0 5 10 25 50 75 100)

EXPS=(
    phaseD-modelrep-5mal-nodefense
    phaseD-modelrep-15mal-nodefense
)

EXP_DIRS=()
for exp in "${EXPS[@]}"; do
    EXP_DIRS+=("$BASE_DIR/${exp}_r100_seed42")
done

echo ""
echo "═══════════════════════════════════════════════════════"
echo "  Phase D Feature extraction (all rounds + final)"
echo "═══════════════════════════════════════════════════════"
for dir in "${EXP_DIRS[@]}"; do
    if [[ ! -d "$dir" ]]; then
        echo "  [skip] $(basename "$dir"): experiment not found"
        continue
    fi
    echo ""
    echo "── $(basename "$dir") ──"
    for r in "${ROUNDS[@]}"; do
        npz="$dir/checkpoints/round_$(printf '%04d' $r)_features.npz"
        ckpt="$dir/checkpoints/round_$(printf '%04d' $r).pt"
        if [[ -f "$npz" ]]; then
            echo "  [skip] round $r: already extracted"
        elif [[ -f "$ckpt" ]]; then
            python -m analysis.extract_features --exp-dir "$dir" --data-root "$DATA_ROOT" --round "$r" --device cpu
        else
            echo "  [skip] round $r: checkpoint not found"
        fi
    done
    # Also extract final model
    npz="$dir/checkpoints/features_test.npz"
    if [[ -f "$npz" ]]; then
        echo "  [skip] final: already extracted"
    elif [[ -f "$dir/checkpoints/final_model.pt" ]]; then
        python -m analysis.extract_features --exp-dir "$dir" --data-root "$DATA_ROOT" --device cpu
    fi
done

echo ""
echo "===== Done ====="
echo "End: $(date)"
