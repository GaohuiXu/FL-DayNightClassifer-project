#!/usr/bin/env bash
#SBATCH -A NAISS2025-22-1113
#SBATCH -p alvis
#SBATCH -C NOGPU
#SBATCH -t 0-02:00:00
#SBATCH -J cycle02_pivot_framework
#SBATCH -o /mimer/NOBACKUP/groups/naiss2024-22-991/gaohui/fl_outputs/slurm/%x_%j.out
#SBATCH -e /mimer/NOBACKUP/groups/naiss2024-22-991/gaohui/fl_outputs/slurm/%x_%j.err

# ──────────────────────────────────────────────────────────────
# Cycle 02 pivot — Framework metrics on the 6 attack cells.
#
# Computes the 4-axis profile for the six pixel-trigger cells of the
# 3x3 design matrix (clean cells skipped — no triggered features).
# Reads the pre-extracted .npz files; no forward passes. Cross-experiment
# comparison via analysis.compare_profiles produces a single side-by-side
# table.
#
# Usage:
#   sbatch analysis/run_cycle02_pivot_framework.sh
#   # or with dependency:
#   sbatch --dependency=afterok:<extract_jobid> analysis/run_cycle02_pivot_framework.sh
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

PROFILE_DIR="$BASE_DIR/figures/framework/profiles"
FIG_DIR="$BASE_DIR/figures/framework"
ROUNDS=(0 5 10 25 50 75 100)

# Only the six attack cells; clean cells have no triggered features.
ATTACK_EXPS=(
    cycle02-pretrained-full-ft-pixel5
    cycle02-pretrained-full-ft-pixel15
    cycle02-pretrained-lastblock-pixel5
    cycle02-pretrained-lastblock-pixel15
    cycle02-pretrained-headonly-pixel5
    cycle02-pretrained-headonly-pixel15
    # canonical-conv1 fallback cells; skipped silently if not yet available.
    cycle02-pretrained-headonly-canonconv1-pixel5
    cycle02-pretrained-headonly-canonconv1-pixel15
)

echo ""
echo "═══════════════════════════════════════════════════════"
echo "  Step 1: Per-experiment framework profiles"
echo "═══════════════════════════════════════════════════════"
# SEEDS env var overrides default 42-only sweep. Example for multi-seed:
#   SEEDS="43 44" sbatch analysis/run_cycle02_pivot_framework.sh
SEEDS=${SEEDS:-42}

for seed in $SEEDS; do
for exp in "${ATTACK_EXPS[@]}"; do
    exp_dir="$BASE_DIR/${exp}_r100_seed${seed}"
    if [[ ! -d "$exp_dir" ]]; then
        echo "  [skip] $exp: not found"
        continue
    fi
    python -m analysis.framework_metrics \
        --exp-dir "$exp_dir" \
        --rounds "${ROUNDS[@]}" \
        --target-label 2 \
        --output-dir "$PROFILE_DIR" \
        --seed 4242 \
        --load-head
done
done  # close SEEDS loop

echo ""
echo "═══════════════════════════════════════════════════════"
echo "  Step 2: Cross-experiment comparison table + figures"
echo "═══════════════════════════════════════════════════════"
python -m analysis.compare_profiles \
    --profile-dir "$PROFILE_DIR" \
    --output-dir "$FIG_DIR"

echo ""
echo "===== Done ====="
echo "End: $(date)"
echo "Profiles: $PROFILE_DIR"
echo "Figures:  $FIG_DIR"
