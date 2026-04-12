#!/usr/bin/env bash
#SBATCH -A NAISS2025-22-1113
#SBATCH -p alvis
#SBATCH -C NOGPU
#SBATCH -t 0-08:00:00
#SBATCH -J phaseC2
#SBATCH -o /mimer/NOBACKUP/groups/naiss2024-22-991/gaohui/fl_outputs/slurm/%x_%j.out
#SBATCH -e /mimer/NOBACKUP/groups/naiss2024-22-991/gaohui/fl_outputs/slurm/%x_%j.err

# ──────────────────────────────────────────────────────────────
# Phase C v2: Feature extraction + t-SNE + quantitative analysis.
# Runs on a CPU-only compute node (NOGPU).
#
# Step 1: Extract features at all checkpoint rounds (CPU forward pass)
# Step 2: Quantitative trajectory analysis (cosine sim, spectral, etc.)
# Step 3: t-SNE visualization (final model comparison + trajectory filmstrips)
#
# Usage:
#   sbatch analysis/run_phaseC.sh
# ──────────────────────────────────────────────────────────────
set -euo pipefail

echo "===== Job info ====="
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Node: $(hostname)"
echo "Start: $(date)"

module purge
module load PyTorch/2.7.1-foss-2024a-CUDA-12.6.0

cd /cephyr/users/gaohui/Alvis/thesis_workspace/fl_weather_project/fl_v2
source /cephyr/users/gaohui/Alvis/thesis_workspace/fl_weather_project/.venv/bin/activate

export OPENBLAS_NUM_THREADS=16
export PYTHONUNBUFFERED=1

DATA_ROOT="/mimer/NOBACKUP/groups/naiss2024-22-991/gaohui/fl_datasets/gtsrb"
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

# ── Step 1: Extract features at all checkpoint rounds ──
echo ""
echo "═══════════════════════════════════════════════════════"
echo "  Step 1: Feature extraction (all rounds)"
echo "═══════════════════════════════════════════════════════"
for dir in "${EXP_DIRS[@]}"; do
    if [[ ! -d "$dir" ]]; then
        echo "  [skip] $(basename "$dir"): experiment not found"
        continue
    fi
    for r in "${ROUNDS[@]}"; do
        npz="$dir/checkpoints/round_$(printf '%04d' $r)_features.npz"
        ckpt="$dir/checkpoints/round_$(printf '%04d' $r).pt"
        if [[ -f "$npz" ]]; then
            echo "  [skip] $(basename "$dir") round $r: already extracted"
        elif [[ -f "$ckpt" ]]; then
            python -m analysis.extract_features --exp-dir "$dir" --data-root "$DATA_ROOT" --round "$r" --device cpu
        fi
    done
    # Also extract final model
    npz="$dir/checkpoints/features_test.npz"
    if [[ ! -f "$npz" ]] && [[ -f "$dir/checkpoints/final_model.pt" ]]; then
        python -m analysis.extract_features --exp-dir "$dir" --data-root "$DATA_ROOT" --device cpu
    fi
done

# ── Step 2: Quantitative trajectory analysis ──
echo ""
echo "═══════════════════════════════════════════════════════"
echo "  Step 2: Quantitative trajectory analysis"
echo "═══════════════════════════════════════════════════════"
# Only analyze backdoor experiments (skip clean baseline)
BACKDOOR_DIRS=()
for dir in "${EXP_DIRS[@]}"; do
    if [[ -d "$dir" ]] && [[ "$(basename "$dir")" != *"clean"* ]]; then
        BACKDOOR_DIRS+=("$dir")
    fi
done

if [[ ${#BACKDOOR_DIRS[@]} -gt 0 ]]; then
    python -m analysis.analyze_features \
        --exp-dirs "${BACKDOOR_DIRS[@]}" \
        --rounds "${ROUNDS[@]}" \
        --target-label 2 \
        --output-dir "$FIG_DIR/quantitative"
fi

# ── Step 3: t-SNE visualization ──
echo ""
echo "═══════════════════════════════════════════════════════"
echo "  Step 3a: t-SNE final model comparison"
echo "═══════════════════════════════════════════════════════"
# Final model comparison across all experiments
AVAILABLE_DIRS=()
for dir in "${EXP_DIRS[@]}"; do
    if [[ -d "$dir" ]]; then
        AVAILABLE_DIRS+=("$dir")
    fi
done

if [[ ${#AVAILABLE_DIRS[@]} -gt 0 ]]; then
    python -m analysis.plot_features \
        --exp-dirs "${AVAILABLE_DIRS[@]}" \
        --target-label 2 \
        --output-dir "$FIG_DIR/comparison"
fi

echo ""
echo "═══════════════════════════════════════════════════════"
echo "  Step 3b: t-SNE trajectory filmstrips"
echo "═══════════════════════════════════════════════════════"
# Trajectory filmstrip for each backdoor experiment
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
