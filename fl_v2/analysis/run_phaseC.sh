#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────
# Phase C: Extract features, generate t-SNE plots, and re-run
# training curve / defense comparison analysis for integrity.
#
# Usage:
#   ./run_phaseC.sh
#
# Assumes activate_env.sh has already been sourced.
# ──────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/.."

DATA_ROOT="/mimer/NOBACKUP/groups/naiss2024-22-991/gaohui/fl_datasets/gtsrb"
BASE_DIR="/mimer/NOBACKUP/groups/naiss2024-22-991/gaohui/fl_outputs/gtsrb_v2/phaseC"
OUT_DIR="analysis/figures/phaseC_representation"

EXPS=(phaseC-clean phaseC-backdoor-nodefense phaseC-backdoor-fedmedian phaseC-backdoor-krum)
EXP_DIRS=()
for exp in "${EXPS[@]}"; do
    EXP_DIRS+=("$BASE_DIR/${exp}_r100_seed42")
done

# Verify all experiments exist
for dir in "${EXP_DIRS[@]}"; do
    if [[ ! -d "$dir" ]]; then
        echo "ERROR: experiment dir not found: $dir"
        exit 1
    fi
done

# ── Step 1: Training curves + defense comparison (integrity) ──
echo ""
echo "═══════════════════════════════════════════════════════"
echo "  Step 1: Training curves + defense comparison"
echo "═══════════════════════════════════════════════════════"
python -m analysis.plot_experiment --base-dir "$BASE_DIR" \
    --output-dir "$OUT_DIR"

python -m analysis.plot_comparison --exp-dirs "${EXP_DIRS[@]}" \
    --output-dir "$OUT_DIR"

# ── Step 2: Extract features from each model ──
echo ""
echo "═══════════════════════════════════════════════════════"
echo "  Step 2: Feature extraction"
echo "═══════════════════════════════════════════════════════"
for dir in "${EXP_DIRS[@]}"; do
    echo ""
    python -m analysis.extract_features --exp-dir "$dir" --data-root "$DATA_ROOT"
done

# ── Step 3: t-SNE visualization ──
echo ""
echo "═══════════════════════════════════════════════════════"
echo "  Step 3: t-SNE visualization"
echo "═══════════════════════════════════════════════════════"
python -m analysis.plot_features --exp-dirs "${EXP_DIRS[@]}" \
    --target-label 2 --output-dir "$OUT_DIR"

echo ""
echo "═══════════════════════════════════════════════════════"
echo "  Phase C analysis complete!"
echo "  Figures saved to: $OUT_DIR"
echo "═══════════════════════════════════════════════════════"
