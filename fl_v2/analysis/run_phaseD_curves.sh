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
OUT_DIR="$BASE_DIR/figures/training_curves"

EXPS=(
    phaseD-modelrep-5mal-nodefense
    phaseD-modelrep-15mal-nodefense
)

EXP_DIRS=()
for exp in "${EXPS[@]}"; do
    EXP_DIRS+=("$BASE_DIR/${exp}_r100_seed42")
done

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

echo ""
echo "═══════════════════════════════════════════════════════"
echo "  Training curves (per-experiment)"
echo "═══════════════════════════════════════════════════════"
python -m analysis.plot_experiment --base-dir "$BASE_DIR" \
    --output-dir "$OUT_DIR" "$@"

echo ""
echo "═══════════════════════════════════════════════════════"
echo "  Defense comparison"
echo "═══════════════════════════════════════════════════════"
python -m analysis.plot_comparison --exp-dirs "${EXP_DIRS[@]}" \
    --output-dir "$OUT_DIR"

echo ""
echo "═══════════════════════════════════════════════════════"
echo "  Done! Figures saved to: $OUT_DIR"
echo "═══════════════════════════════════════════════════════"
