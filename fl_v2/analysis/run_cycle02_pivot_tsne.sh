#!/usr/bin/env bash
#SBATCH -A NAISS2025-22-1113
#SBATCH -p alvis
#SBATCH -C NOGPU
#SBATCH -c 32
#SBATCH -t 0-02:30:00
#SBATCH -J cycle02_pivot_tsne
#SBATCH -o /mimer/NOBACKUP/groups/naiss2024-22-991/gaohui/fl_outputs/slurm/%x_%j.out
#SBATCH -e /mimer/NOBACKUP/groups/naiss2024-22-991/gaohui/fl_outputs/slurm/%x_%j.err

# ──────────────────────────────────────────────────────────────
# Cycle 02 pivot — t-SNE visualizations.
#
# Generates two figures for the supervisor:
# 1. Cross-cell comparison panel: 4 most informative attack cells
#    + the full_ft clean baseline as reference.
# 2. Trajectory filmstrips for the same 4 attack cells (rounds
#    0, 5, 10, 25, 50, 75, 100).
#
# Reuses analysis/plot_features.py (existing). All inputs are
# already-extracted features_test.npz / round_NNNN_features.npz files.
#
# Usage:
#   sbatch analysis/run_cycle02_pivot_tsne.sh
# ──────────────────────────────────────────────────────────────
set -euo pipefail

echo "===== Job info ====="
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Node: $(hostname)"
echo "Start: $(date)"

if [[ "${SLURM_JOB_ID:-}" != "" ]]; then
    module purge
    module load PyTorch/2.7.1-foss-2024a-CUDA-12.6.0
    cd /mimer/NOBACKUP/groups/naiss2024-22-991/gaohui/thesis_workspace/fl_weather_project/fl_v2
    source /mimer/NOBACKUP/groups/naiss2024-22-991/gaohui/thesis_workspace/fl_weather_project/.venv/bin/activate
fi

export OPENBLAS_NUM_THREADS=16
export PYTHONUNBUFFERED=1

BASE_DIR="/mimer/NOBACKUP/groups/naiss2024-22-991/gaohui/fl_outputs/gtsrb/experiments/cycle_02/phaseD2"
FIG_DIR="$BASE_DIR/figures/tsne"
ROUNDS=(0 5 10 25 50 75 100)

# Comparison panel: 5 most informative cells (1 clean reference + 4 attacks
# spanning the regime space). The 4 attack cells are chosen to highlight
# distinct (head_attr, centroid_l2) regimes:
#   full_ft 5mal     — anchored encoder, almost-pure head attack
#   last_block 5mal  — close-feature attack (centroid_l2 = 1.14)
#   headonly modified 15mal — broken-conv1 utility cell (artifact)
#   headonly canonconv1 15mal — sterile encoder (centroid_l2 = 13)
COMPARISON_DIRS=(
    "$BASE_DIR/cycle02-pretrained-full-ft-clean_r100_seed42"
    "$BASE_DIR/cycle02-pretrained-full-ft-pixel5_r100_seed42"
    "$BASE_DIR/cycle02-pretrained-lastblock-pixel5_r100_seed42"
    "$BASE_DIR/cycle02-pretrained-headonly-pixel15_r100_seed42"
    "$BASE_DIR/cycle02-pretrained-headonly-canonconv1-pixel15_r100_seed42"
)

# Trajectory filmstrips: same 4 attack cells (rounds 0..100)
TRAJECTORY_DIRS=(
    "$BASE_DIR/cycle02-pretrained-full-ft-pixel5_r100_seed42"
    "$BASE_DIR/cycle02-pretrained-lastblock-pixel5_r100_seed42"
    "$BASE_DIR/cycle02-pretrained-headonly-pixel15_r100_seed42"
    "$BASE_DIR/cycle02-pretrained-headonly-canonconv1-pixel15_r100_seed42"
)

# ── Step 1: Cross-cell comparison panel ──
echo ""
echo "═══════════════════════════════════════════════════════"
echo "  Step 1: Cross-cell t-SNE comparison panel"
echo "          (full_ft clean ref + 4 attack regimes)"
echo "═══════════════════════════════════════════════════════"
AVAILABLE_COMPARISON_DIRS=()
for dir in "${COMPARISON_DIRS[@]}"; do
    if [[ -d "$dir" ]] && [[ -f "$dir/checkpoints/features_test.npz" ]]; then
        AVAILABLE_COMPARISON_DIRS+=("$dir")
    else
        echo "  [skip] $(basename "$dir"): features_test.npz not found"
    fi
done

if [[ ${#AVAILABLE_COMPARISON_DIRS[@]} -gt 0 ]]; then
    python -m analysis.plot_features \
        --exp-dirs "${AVAILABLE_COMPARISON_DIRS[@]}" \
        --target-label 2 \
        --output-dir "$FIG_DIR/comparison"
fi

# ── Step 2: Trajectory filmstrips ──
echo ""
echo "═══════════════════════════════════════════════════════"
echo "  Step 2: t-SNE trajectory filmstrips per regime"
echo "═══════════════════════════════════════════════════════"
for dir in "${TRAJECTORY_DIRS[@]}"; do
    if [[ ! -d "$dir" ]]; then
        echo "  [skip] $(basename "$dir"): not found"
        continue
    fi
    python -m analysis.plot_features \
        --exp-dirs "$dir" \
        --rounds "${ROUNDS[@]}" \
        --target-label 2 \
        --output-dir "$FIG_DIR/trajectory"
done

echo ""
echo "===== Done ====="
echo "End: $(date)"
echo "Figures saved to: $FIG_DIR"
