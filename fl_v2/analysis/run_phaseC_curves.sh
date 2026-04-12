#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────
# Phase C: Training curves + defense comparison analysis.
#
# Replaces the old analyze.sh + compare.sh scripts, scoped to
# the Phase C experiment set.
#
# Usage:
#   ./analysis/run_phaseC_curves.sh                   # all experiments
#   ./analysis/run_phaseC_curves.sh --compare-norms   # also compare norms
#   ./analysis/run_phaseC_curves.sh --smooth 5        # smooth curves
#
# Assumes activate_env.sh has already been sourced.
# ──────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/.."

BASE_DIR="/mimer/NOBACKUP/groups/naiss2024-22-991/gaohui/fl_outputs/gtsrb_v2/phaseC_v2"
OUT_DIR="$BASE_DIR/figures/training_curves"

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
