#!/usr/bin/env bash
#SBATCH -A NAISS2025-22-1113
#SBATCH -p alvis
#SBATCH -C NOGPU
#SBATCH -t 0-02:00:00
#SBATCH -J phaseD_viz
#SBATCH -o /mimer/NOBACKUP/groups/naiss2024-22-991/gaohui/fl_outputs/slurm/%x_%j.out
#SBATCH -e /mimer/NOBACKUP/groups/naiss2024-22-991/gaohui/fl_outputs/slurm/%x_%j.err

# ──────────────────────────────────────────────────────────────
# Phase D — Quantitative trajectory + t-SNE visualization.
#
# Assumes feature .npz files are already extracted for both the
# phaseD model-replacement experiments AND the phaseC_v2 experiments
# we use as reference rows.
#
# Step 1: Quantitative trajectory analysis (cosine sim, spectral, etc.)
#         over both ModelRep runs. Output: quantitative/*.{png,pdf,csv}
# Step 2a: t-SNE final-model comparison panel. Includes:
#          - phaseC_v2 Clean (reference geometry)
#          - phaseC_v2 Pixel-Trigger no-defense 5mal and 15mal (old baseline)
#          - phaseD ModelRep 5mal and 15mal (new attack)
# Step 2b: t-SNE trajectory filmstrips for the two ModelRep experiments
#          (matches the phaseC_v2 trajectory format).
#
# Usage:
#   sbatch analysis/run_phaseD_viz.sh
#   # or locally after sourcing activate_env.sh:
#   ./analysis/run_phaseD_viz.sh
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

PHASE_C_DIR="/mimer/NOBACKUP/groups/naiss2024-22-991/gaohui/fl_outputs/gtsrb_v2/phaseC_v2"
PHASE_D_DIR="/mimer/NOBACKUP/groups/naiss2024-22-991/gaohui/fl_outputs/gtsrb_v2/phaseD"
FIG_DIR="$PHASE_D_DIR/figures"
ROUNDS=(0 5 10 25 50 75 100)

# Phase D backdoor experiments (model replacement)
PHASE_D_EXPS=(
    phaseD-modelrep-5mal-nodefense
    phaseD-modelrep-15mal-nodefense
)

PHASE_D_DIRS=()
for exp in "${PHASE_D_EXPS[@]}"; do
    PHASE_D_DIRS+=("$PHASE_D_DIR/${exp}_r100_seed42")
done

# Cross-phase comparison panel: Clean + old pixel-trigger baselines + new ModelRep
COMPARISON_DIRS=(
    "$PHASE_C_DIR/phaseC2-clean_r100_seed42"
    "$PHASE_C_DIR/phaseC2-backdoor-5mal-nodefense_r100_seed42"
    "$PHASE_C_DIR/phaseC2-backdoor-15mal-nodefense_r100_seed42"
    "$PHASE_D_DIR/phaseD-modelrep-5mal-nodefense_r100_seed42"
    "$PHASE_D_DIR/phaseD-modelrep-15mal-nodefense_r100_seed42"
)

# ── Step 1: Quantitative trajectory analysis ──
echo ""
echo "═══════════════════════════════════════════════════════"
echo "  Step 1: Quantitative trajectory analysis (phaseD)"
echo "═══════════════════════════════════════════════════════"
if [[ ${#PHASE_D_DIRS[@]} -gt 0 ]]; then
    python -m analysis.analyze_features \
        --exp-dirs "${PHASE_D_DIRS[@]}" \
        --rounds "${ROUNDS[@]}" \
        --target-label 2 \
        --output-dir "$FIG_DIR/quantitative"
fi

# ── Step 2a: t-SNE final model comparison (cross-phase) ──
echo ""
echo "═══════════════════════════════════════════════════════"
echo "  Step 2a: t-SNE final model comparison"
echo "           (phaseC_v2 Clean + Pixel-Trigger vs phaseD ModelRep)"
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

# ── Step 2b: t-SNE trajectory filmstrips ──
echo ""
echo "═══════════════════════════════════════════════════════"
echo "  Step 2b: t-SNE trajectory filmstrips (phaseD ModelRep)"
echo "═══════════════════════════════════════════════════════"
for dir in "${PHASE_D_DIRS[@]}"; do
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
