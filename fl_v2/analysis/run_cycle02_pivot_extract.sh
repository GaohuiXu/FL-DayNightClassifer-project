#!/usr/bin/env bash
#SBATCH -A NAISS2025-22-1113
#SBATCH -p alvis
#SBATCH --gpus-per-node=A40:1
#SBATCH -t 0-01:00:00
#SBATCH -J cycle02_pivot_extract
#SBATCH -o /mimer/NOBACKUP/groups/naiss2024-22-991/gaohui/fl_outputs/slurm/%x_%j.out
#SBATCH -e /mimer/NOBACKUP/groups/naiss2024-22-991/gaohui/fl_outputs/slurm/%x_%j.err

# ──────────────────────────────────────────────────────────────
# Cycle 02 pivot — STAGE 1: Feature extraction.
#
# Runs analysis.extract_features on every saved checkpoint of the 9-cell
# 3x3 design matrix (full_ft / last_block / head_only × clean / 5mal / 15mal).
# Idempotent — skips files that already exist.
#
# Usage:
#   sbatch analysis/run_cycle02_pivot_extract.sh
#   # or with dependency on training jobs:
#   sbatch --dependency=afterany:<jobids> analysis/run_cycle02_pivot_extract.sh
# ──────────────────────────────────────────────────────────────
set -euo pipefail

echo "===== Job info ====="
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Node: $(hostname)"
echo "Start: $(date)"

module purge
module load PyTorch/2.7.1-foss-2024a-CUDA-12.6.0

cd /mimer/NOBACKUP/groups/naiss2024-22-991/gaohui/thesis_workspace/fl_weather_project/fl_v2
source /mimer/NOBACKUP/groups/naiss2024-22-991/gaohui/thesis_workspace/fl_weather_project/.venv/bin/activate

export OPENBLAS_NUM_THREADS=16
export PYTHONUNBUFFERED=1

DATA_ROOT="/mimer/NOBACKUP/groups/naiss2024-22-991/gaohui/fl_datasets/gtsrb"
BASE_DIR="/mimer/NOBACKUP/groups/naiss2024-22-991/gaohui/fl_outputs/gtsrb/experiments/cycle_02/phaseD2"

ROUNDS=(0 5 10 25 50 75 100)

EXPS=(
    cycle02-pretrained-full-ft-clean
    cycle02-pretrained-full-ft-pixel5
    cycle02-pretrained-full-ft-pixel15
    cycle02-pretrained-lastblock-clean
    cycle02-pretrained-lastblock-pixel5
    cycle02-pretrained-lastblock-pixel15
    cycle02-pretrained-headonly-clean
    cycle02-pretrained-headonly-pixel5
    cycle02-pretrained-headonly-pixel15
    # canonical-conv1 fallback cells (image-size 64); skipped silently
    # if the SLURM training jobs have not yet finished.
    cycle02-pretrained-headonly-canonconv1-clean
    cycle02-pretrained-headonly-canonconv1-pixel5
    cycle02-pretrained-headonly-canonconv1-pixel15
)

EXP_DIRS=()
for exp in "${EXPS[@]}"; do
    EXP_DIRS+=("$BASE_DIR/${exp}_r100_seed42")
done

echo ""
echo "═══════════════════════════════════════════════════════"
echo "  Cycle 02 pivot — feature extraction (all rounds + final)"
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
            python -m analysis.extract_features --exp-dir "$dir" --data-root "$DATA_ROOT" --round "$r" --device cuda
        else
            echo "  [skip] round $r: checkpoint not found"
        fi
    done
    # Also extract final model
    npz="$dir/checkpoints/features_test.npz"
    if [[ -f "$npz" ]]; then
        echo "  [skip] final: already extracted"
    elif [[ -f "$dir/checkpoints/final_model.pt" ]]; then
        python -m analysis.extract_features --exp-dir "$dir" --data-root "$DATA_ROOT" --device cuda
    fi
done

echo ""
echo "===== Done ====="
echo "End: $(date)"
