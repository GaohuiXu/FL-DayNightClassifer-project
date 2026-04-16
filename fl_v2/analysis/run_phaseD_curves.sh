#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────
# Phase D: Training curves + defense comparison (local analysis).
#
# Runs quickly on the login node once training is done — no feature
# extraction, just CSV plotting.
#
# Usage:
#   source activate_env.sh
#   ./analysis/run_phaseD_curves.sh
# ──────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/.."

BASE_DIR="/mimer/NOBACKUP/groups/naiss2024-22-991/gaohui/fl_outputs/gtsrb_v2/phaseD"
PHASE_C_DIR="/mimer/NOBACKUP/groups/naiss2024-22-991/gaohui/fl_outputs/gtsrb_v2/phaseC_v2"
OUT_DIR="$BASE_DIR/figures/training_curves"

# Per-experiment training curves for both phaseD runs.
# The cross-phase defense comparison additionally pulls in the phaseC_v2
# Clean baseline and the matching Pixel-Trigger no-defense runs so the user
# can read the delta between heuristic pixel trigger and Bagdasaryan-scaled
# model replacement directly on the same axes.
EXPS=(
    phaseD-modelrep-5mal-nodefense
    phaseD-modelrep-15mal-nodefense
)

EXP_DIRS=()
for exp in "${EXPS[@]}"; do
    EXP_DIRS+=("$BASE_DIR/${exp}_r100_seed42")
done

# Comparison row set includes reference rows from phaseC_v2
COMPARISON_DIRS=(
    "$PHASE_C_DIR/phaseC2-clean_r100_seed42"
    "$PHASE_C_DIR/phaseC2-backdoor-5mal-nodefense_r100_seed42"
    "$PHASE_C_DIR/phaseC2-backdoor-15mal-nodefense_r100_seed42"
    "$BASE_DIR/phaseD-modelrep-5mal-nodefense_r100_seed42"
    "$BASE_DIR/phaseD-modelrep-15mal-nodefense_r100_seed42"
)

# Filter to experiments that exist (some may still be running)
AVAILABLE_DIRS=()
for dir in "${EXP_DIRS[@]}"; do
    if [[ -d "$dir" ]]; then
        AVAILABLE_DIRS+=("$dir")
    else
        echo "  [skip] $(basename "$dir"): not found yet"
    fi
done
EXP_DIRS=("${AVAILABLE_DIRS[@]}")

AVAILABLE_COMP_DIRS=()
for dir in "${COMPARISON_DIRS[@]}"; do
    if [[ -d "$dir" ]]; then
        AVAILABLE_COMP_DIRS+=("$dir")
    else
        echo "  [skip] $(basename "$dir"): not found yet"
    fi
done
COMPARISON_DIRS=("${AVAILABLE_COMP_DIRS[@]}")

echo ""
echo "═══════════════════════════════════════════════════════"
echo "  Training curves (per-experiment)"
echo "═══════════════════════════════════════════════════════"
python -m analysis.plot_experiment --base-dir "$BASE_DIR" \
    --output-dir "$OUT_DIR" "$@"

echo ""
echo "═══════════════════════════════════════════════════════"
echo "  Cross-phase defense comparison"
echo "  (phaseC_v2 Clean + Pixel-Trigger vs phaseD ModelRep)"
echo "═══════════════════════════════════════════════════════"
python -m analysis.plot_comparison --exp-dirs "${COMPARISON_DIRS[@]}" \
    --output-dir "$OUT_DIR"

echo ""
echo "═══════════════════════════════════════════════════════"
echo "  Done! Figures saved to: $OUT_DIR"
echo "═══════════════════════════════════════════════════════"
