#!/usr/bin/env bash
#SBATCH -A NAISS2025-22-1113
#SBATCH -p alvis
#SBATCH --gpus-per-node=A40:1
#SBATCH -t 0-02:00:00
#SBATCH -J cycle02_pivot_head_attr
#SBATCH -o /mimer/NOBACKUP/groups/naiss2024-22-991/gaohui/fl_outputs/slurm/%x_%j.out
#SBATCH -e /mimer/NOBACKUP/groups/naiss2024-22-991/gaohui/fl_outputs/slurm/%x_%j.err

# ──────────────────────────────────────────────────────────────
# Cycle 02 pivot — Head-feature decomposition diagnostic.
#
# For each of the 6 attack cells, runs analysis.head_feature_decomposition:
#   1. load final_model.pt
#   2. measure original clean_acc + ASR
#   3. freeze features+avgpool, reinit fc, retrain head on clean GTSRB train
#   4. measure clean-head clean_acc + ASR
#   5. compute head_attribution_pct = (orig_asr - new_asr) / orig_asr * 100
#
# Outputs head_feature_decomposition.json next to each final_model.pt.
# Runs on GPU because the head-training pass is fast there (~1-2 min/cell)
# and we have one big SLURM job processing all 6 sequentially (~10 min total).
#
# Usage:
#   sbatch analysis/run_cycle02_pivot_head_attribution.sh
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

DATA_ROOT="/mimer/NOBACKUP/groups/naiss2024-22-991/gaohui/fl_datasets/gtsrb"
BASE_DIR="/mimer/NOBACKUP/groups/naiss2024-22-991/gaohui/fl_outputs/gtsrb/experiments/cycle_02/phaseD2"

# Hyperparameters confirmed via pilot on a Cycle 01 checkpoint
# (see docs/cycle_02_pivot_results.md when filled in).
EPOCHS=10
LR=1e-3
SEED=4242

ATTACK_EXPS=(
    cycle02-pretrained-full-ft-pixel5
    cycle02-pretrained-full-ft-pixel15
    cycle02-pretrained-lastblock-pixel5
    cycle02-pretrained-lastblock-pixel15
    cycle02-pretrained-headonly-pixel5
    cycle02-pretrained-headonly-pixel15
)

echo ""
echo "═══════════════════════════════════════════════════════"
echo "  Cycle 02 pivot — head-feature decomposition (6 cells)"
echo "═══════════════════════════════════════════════════════"
for exp in "${ATTACK_EXPS[@]}"; do
    exp_dir="$BASE_DIR/${exp}_r100_seed42"
    if [[ ! -d "$exp_dir" ]]; then
        echo "  [skip] $exp: not found"
        continue
    fi
    json_out="$exp_dir/head_feature_decomposition.json"
    if [[ -f "$json_out" ]]; then
        echo "  [skip] $exp: head_feature_decomposition.json already exists"
        continue
    fi
    echo ""
    echo "── $exp ──"
    python -m analysis.head_feature_decomposition \
        --exp-dir "$exp_dir" \
        --data-root "$DATA_ROOT" \
        --epochs $EPOCHS \
        --lr $LR \
        --seed $SEED \
        --device auto
done

echo ""
echo "===== Done ====="
echo "End: $(date)"
echo "Outputs in: $BASE_DIR/<exp>_r100_seed42/head_feature_decomposition.json"
