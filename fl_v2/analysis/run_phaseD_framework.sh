#!/usr/bin/env bash
#SBATCH -A NAISS2025-22-1113
#SBATCH -p alvis
#SBATCH -C NOGPU
#SBATCH -t 0-02:00:00
#SBATCH -J phaseD_framework
#SBATCH -o /mimer/NOBACKUP/groups/naiss2024-22-991/gaohui/fl_outputs/slurm/%x_%j.out
#SBATCH -e /mimer/NOBACKUP/groups/naiss2024-22-991/gaohui/fl_outputs/slurm/%x_%j.err

# ──────────────────────────────────────────────────────────────
# Phase D — Framework metrics on model-replacement experiments.
#
# Computes the 4-axis profile for both phaseD experiments using
# already-extracted features (reads .npz, no forward passes).
# Also copies the phaseC_v2 profiles into the phaseD framework dir
# so that compare_profiles.py produces a single side-by-side table.
#
# Usage:
#   sbatch analysis/run_phaseD_framework.sh
#   # or locally after sourcing activate_env.sh:
#   ./analysis/run_phaseD_framework.sh
# ──────────────────────────────────────────────────────────────
set -euo pipefail

echo "===== Job info ====="
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Node: $(hostname)"
echo "Start: $(date)"

if [[ "${SLURM_JOB_ID:-}" != "" ]]; then
    module purge
    module load PyTorch/2.7.1-foss-2024a-CUDA-12.6.0
    cd /cephyr/users/gaohui/Alvis/thesis_workspace/fl_weather_project/fl_v2
    source /cephyr/users/gaohui/Alvis/thesis_workspace/fl_weather_project/.venv/bin/activate
fi

export OPENBLAS_NUM_THREADS=16
export PYTHONUNBUFFERED=1

PHASE_C_DIR="/mimer/NOBACKUP/groups/naiss2024-22-991/gaohui/fl_outputs/gtsrb_v2/phaseC_v2"
PHASE_D_DIR="/mimer/NOBACKUP/groups/naiss2024-22-991/gaohui/fl_outputs/gtsrb_v2/phaseD"

PROFILE_DIR="$PHASE_D_DIR/figures/framework/profiles"
FIG_DIR="$PHASE_D_DIR/figures/framework"
ROUNDS=(0 5 10 25 50 75 100)

# Only the two phaseD backdoor experiments (no clean — clean reuses phaseC_v2)
PHASE_D_EXPS=(
    phaseD-modelrep-5mal-nodefense
    phaseD-modelrep-15mal-nodefense
)

# ── Step 1: Compute per-experiment profiles for phaseD ──
echo ""
echo "═══════════════════════════════════════════════════════"
echo "  Step 1: Phase D profiles (model replacement)"
echo "═══════════════════════════════════════════════════════"
for exp in "${PHASE_D_EXPS[@]}"; do
    exp_dir="$PHASE_D_DIR/${exp}_r100_seed42"
    if [[ ! -d "$exp_dir" ]]; then
        echo "  [skip] $exp: not found"
        continue
    fi
    python -m analysis.framework_metrics \
        --exp-dir "$exp_dir" \
        --rounds "${ROUNDS[@]}" \
        --target-label 2 \
        --output-dir "$PROFILE_DIR" \
        --load-head
done

# ── Step 2: Mirror phaseC_v2 profiles into phaseD profile dir for comparison ──
echo ""
echo "═══════════════════════════════════════════════════════"
echo "  Step 2: Copy phaseC_v2 profiles for side-by-side comparison"
echo "═══════════════════════════════════════════════════════"
mkdir -p "$PROFILE_DIR"
for f in "$PHASE_C_DIR/figures/framework/profiles/"*_profile.json; do
    if [[ -f "$f" ]]; then
        cp -n "$f" "$PROFILE_DIR/"
    fi
done
for f in "$PHASE_C_DIR/figures/framework/profiles/"*_trajectory.csv; do
    if [[ -f "$f" ]]; then
        cp -n "$f" "$PROFILE_DIR/"
    fi
done
echo "  [done] profiles copied (existing files preserved)"

# ── Step 3: Cross-phase comparison (pixel trigger vs model replacement) ──
echo ""
echo "═══════════════════════════════════════════════════════"
echo "  Step 3: Cross-phase comparison table + radar"
echo "═══════════════════════════════════════════════════════"
python -m analysis.compare_profiles \
    --profile-dir "$PROFILE_DIR" \
    --output-dir "$FIG_DIR"

echo ""
echo "===== Done ====="
echo "End: $(date)"
echo "Profiles: $PROFILE_DIR"
echo "Figures:  $FIG_DIR"
