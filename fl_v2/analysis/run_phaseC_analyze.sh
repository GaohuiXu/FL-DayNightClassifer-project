#!/usr/bin/env bash
#SBATCH -A NAISS2025-22-1113
#SBATCH -p alvis
#SBATCH -C NOGPU
#SBATCH -t 0-10:00:00
#SBATCH -J phaseC2_analyze
#SBATCH -o /mimer/NOBACKUP/groups/naiss2024-22-991/gaohui/fl_outputs/slurm/%x_%j.out
#SBATCH -e /mimer/NOBACKUP/groups/naiss2024-22-991/gaohui/fl_outputs/slurm/%x_%j.err

# ──────────────────────────────────────────────────────────────
# Phase C v2 — STAGE 2: Quantitative analysis + t-SNE plots.
#
# Assumes features_test.npz and round_XXXX_features.npz exist for
# all checkpoints (run run_phaseC_extract.sh first).
#
# Usage:
#   sbatch analysis/run_phaseC_analyze.sh
#   # or with dependency on extract job:
#   sbatch --dependency=afterok:<extract_jobid> analysis/run_phaseC_analyze.sh
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

BASE_DIR="/mimer/NOBACKUP/groups/naiss2024-22-991/gaohui/fl_outputs/gtsrb_v2/phaseC_v2"
FIG_DIR="$BASE_DIR/figures"

ROUNDS=(0 5 10 25 50 75 100)

EXPS=(
    phaseC2-clean
    phaseC2-backdoor-5mal-nodefense
    phaseC2-backdoor-15mal-nodefense
    phaseC2-backdoor-15mal-fedmedian
    phaseC2-backdoor-5mal-krum
    phaseC2-backdoor-15mal-krum
    phaseC2-backdoor-5mal-nodefense-partial
    phaseC2-backdoor-15mal-nodefense-partial
)

EXP_DIRS=()
for exp in "${EXPS[@]}"; do
    EXP_DIRS+=("$BASE_DIR/${exp}_r100_seed42")
done

# Backdoor experiments (skip clean for quantitative / trajectory analysis)
BACKDOOR_DIRS=()
AVAILABLE_DIRS=()
for dir in "${EXP_DIRS[@]}"; do
    if [[ -d "$dir" ]]; then
        AVAILABLE_DIRS+=("$dir")
        if [[ "$(basename "$dir")" != *"clean"* ]]; then
            BACKDOOR_DIRS+=("$dir")
        fi
    fi
done

# ── Step 1: Quantitative trajectory analysis ──
echo ""
echo "═══════════════════════════════════════════════════════"
echo "  Step 1: Quantitative trajectory analysis"
echo "═══════════════════════════════════════════════════════"
if [[ ${#BACKDOOR_DIRS[@]} -gt 0 ]]; then
    python -m analysis.analyze_features \
        --exp-dirs "${BACKDOOR_DIRS[@]}" \
        --rounds "${ROUNDS[@]}" \
        --target-label 2 \
        --output-dir "$FIG_DIR/quantitative"
fi

# ── Step 2a: t-SNE final model comparison ──
echo ""
echo "═══════════════════════════════════════════════════════"
echo "  Step 2a: t-SNE final model comparison"
echo "═══════════════════════════════════════════════════════"
if [[ ${#AVAILABLE_DIRS[@]} -gt 0 ]]; then
    python -m analysis.plot_features \
        --exp-dirs "${AVAILABLE_DIRS[@]}" \
        --target-label 2 \
        --output-dir "$FIG_DIR/comparison"
fi

# ── Step 2b: t-SNE trajectory filmstrips ──
echo ""
echo "═══════════════════════════════════════════════════════"
echo "  Step 2b: t-SNE trajectory filmstrips"
echo "═══════════════════════════════════════════════════════"
for dir in "${BACKDOOR_DIRS[@]}"; do
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
