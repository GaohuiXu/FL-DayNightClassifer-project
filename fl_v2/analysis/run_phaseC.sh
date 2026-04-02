#!/usr/bin/env bash
#SBATCH -A NAISS2025-22-1113
#SBATCH -p alvis
#SBATCH -C NOGPU
#SBATCH -t 0-02:00:00
#SBATCH -J phaseC
#SBATCH -o /mimer/NOBACKUP/groups/naiss2024-22-991/gaohui/fl_outputs/slurm/%x_%j.out
#SBATCH -e /mimer/NOBACKUP/groups/naiss2024-22-991/gaohui/fl_outputs/slurm/%x_%j.err

# ──────────────────────────────────────────────────────────────
# Phase C: Feature extraction + t-SNE representation analysis.
# Runs on a CPU-only compute node (NOGPU).
#
# Step 1: Extract penultimate-layer features from trained models
#         (CPU forward pass on 12k test images per model)
# Step 2: t-SNE visualization (CPU-heavy dimensionality reduction)
#
# Usage:
#   sbatch analysis/run_phaseC.sh
# ──────────────────────────────────────────────────────────────
set -euo pipefail

echo "===== Job info ====="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Start: $(date)"

module purge
module load PyTorch/2.7.1-foss-2024a-CUDA-12.6.0

cd /cephyr/users/gaohui/Alvis/thesis_workspace/fl_weather_project/fl_v2
source /cephyr/users/gaohui/Alvis/thesis_workspace/fl_weather_project/.venv/bin/activate

export OPENBLAS_NUM_THREADS=16
export PYTHONUNBUFFERED=1

DATA_ROOT="/mimer/NOBACKUP/groups/naiss2024-22-991/gaohui/fl_datasets/gtsrb"
BASE_DIR="/mimer/NOBACKUP/groups/naiss2024-22-991/gaohui/fl_outputs/gtsrb_v2/phaseC"
OUT_DIR="/mimer/NOBACKUP/groups/naiss2024-22-991/gaohui/fl_outputs/gtsrb_v2/phaseC/figures"

EXPS=(phaseC-clean phaseC-backdoor-nodefense phaseC-backdoor-fedmedian phaseC-backdoor-krum)
EXP_DIRS=()
for exp in "${EXPS[@]}"; do
    EXP_DIRS+=("$BASE_DIR/${exp}_r100_seed42")
done

# ── Step 1: Feature extraction (skip if already done) ──
echo ""
echo "═══════════════════════════════════════════════════════"
echo "  Step 1: Feature extraction"
echo "═══════════════════════════════════════════════════════"
for dir in "${EXP_DIRS[@]}"; do
    npz="$dir/checkpoints/features_test.npz"
    if [[ -f "$npz" ]]; then
        echo "  [skip] $(basename "$dir"): features already extracted"
    else
        python -m analysis.extract_features --exp-dir "$dir" --data-root "$DATA_ROOT" --device cpu
    fi
done

# ── Step 2: t-SNE visualization ──
echo ""
echo "═══════════════════════════════════════════════════════"
echo "  Step 2: t-SNE visualization"
echo "═══════════════════════════════════════════════════════"
python -m analysis.plot_features --exp-dirs "${EXP_DIRS[@]}" \
    --target-label 2 --output-dir "$OUT_DIR"

echo ""
echo "===== Done ====="
echo "End: $(date)"
echo "Figures saved to: $OUT_DIR"
