#!/usr/bin/env bash
#SBATCH -A NAISS2025-22-1113
#SBATCH -p alvis
#SBATCH -C NOGPU
#SBATCH -t 0-04:00:00
#SBATCH -J phaseC2_framework
#SBATCH -o /mimer/NOBACKUP/groups/naiss2024-22-991/gaohui/fl_outputs/slurm/%x_%j.out
#SBATCH -e /mimer/NOBACKUP/groups/naiss2024-22-991/gaohui/fl_outputs/slurm/%x_%j.err

# ──────────────────────────────────────────────────────────────
# Phase C v2 — Representation-space framework metrics.
#
# Computes the 4-axis profile for each experiment using already-
# extracted features (reads .npz, no forward passes needed).
# Produces per-experiment profiles + cross-experiment comparison.
#
# Usage:
#   sbatch analysis/run_framework.sh
#   # or run locally after sourcing activate_env.sh:
#   ./analysis/run_framework.sh
# ──────────────────────────────────────────────────────────────
set -euo pipefail

echo "===== Job info ====="
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Node: $(hostname)"
echo "Start: $(date)"

if [[ "${SLURM_JOB_ID:-}" != "" ]]; then
    # Running under SLURM — need to load env
    module purge
    module load PyTorch/2.7.1-foss-2024a-CUDA-12.6.0
    cd /mimer/NOBACKUP/groups/naiss2024-22-991/gaohui/thesis_workspace/fl_weather_project/fl_v2
    source /mimer/NOBACKUP/groups/naiss2024-22-991/gaohui/thesis_workspace/fl_weather_project/.venv/bin/activate
fi

export OPENBLAS_NUM_THREADS=16
export PYTHONUNBUFFERED=1

BASE_DIR="/mimer/NOBACKUP/groups/naiss2024-22-991/gaohui/fl_outputs/gtsrb_v2/phaseC_v2"
PROFILE_DIR="$BASE_DIR/figures/framework/profiles"
FIG_DIR="$BASE_DIR/figures/framework"
ROUNDS=(0 5 10 25 50 75 100)

# Only backdoor experiments (clean has no triggered features)
BACKDOOR_EXPS=(
    phaseC2-backdoor-5mal-nodefense
    phaseC2-backdoor-15mal-nodefense
    phaseC2-backdoor-15mal-fedmedian
    phaseC2-backdoor-5mal-krum
    phaseC2-backdoor-15mal-krum
    phaseC2-backdoor-5mal-nodefense-partial
    phaseC2-backdoor-15mal-nodefense-partial
)

# ── Step 1: Compute per-experiment profiles ──
echo ""
echo "═══════════════════════════════════════════════════════"
echo "  Step 1: Per-experiment framework profiles"
echo "═══════════════════════════════════════════════════════"
for exp in "${BACKDOOR_EXPS[@]}"; do
    exp_dir="$BASE_DIR/${exp}_r100_seed42"
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

# ── Step 2: Cross-experiment comparison ──
echo ""
echo "═══════════════════════════════════════════════════════"
echo "  Step 2: Cross-experiment comparison"
echo "═══════════════════════════════════════════════════════"
python -m analysis.compare_profiles \
    --profile-dir "$PROFILE_DIR" \
    --output-dir "$FIG_DIR"

echo ""
echo "===== Done ====="
echo "End: $(date)"
echo "Profiles: $PROFILE_DIR"
echo "Figures:  $FIG_DIR"
