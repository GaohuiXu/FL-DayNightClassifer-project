#!/usr/bin/env bash
#SBATCH -A NAISS2025-22-1113
#SBATCH -p alvis
#SBATCH --gpus-per-node=A40:1
#SBATCH -t 0-01:30:00
#SBATCH -J sentinel_head_attr
#SBATCH -o /mimer/NOBACKUP/groups/naiss2024-22-991/gaohui/fl_outputs/slurm/%x_%j.out
#SBATCH -e /mimer/NOBACKUP/groups/naiss2024-22-991/gaohui/fl_outputs/slurm/%x_%j.err
#
# Phase 3.0 sentinel head-feature decomposition.
#
# Runs the convergent v2 diagnostic on the sentinel checkpoint
# (phaseC2-backdoor-5mal-nodefense, deterministic-pipeline rerun of the
# Cycle 01 cell). Designed to be submitted with --dependency=afterok:<sentinel_jobid>
# so it auto-fires once sentinel training completes.

set -euo pipefail

echo "===== Job info ====="
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Node: $(hostname)"
echo "Start: $(date)"

module purge
module load PyTorch/2.7.1-foss-2024a-CUDA-12.6.0
cd /mimer/NOBACKUP/groups/naiss2024-22-991/gaohui/thesis_workspace/fl_weather_project/fl_v2
source /mimer/NOBACKUP/groups/naiss2024-22-991/gaohui/thesis_workspace/fl_weather_project/.venv/bin/activate
export PYTHONUNBUFFERED=1

DATA_ROOT=/mimer/NOBACKUP/groups/naiss2024-22-991/gaohui/fl_datasets/gtsrb
SENTINEL_DIR=/mimer/NOBACKUP/groups/naiss2024-22-991/gaohui/fl_outputs/gtsrb_v2/phaseC_v2_sentinel/phaseC2-backdoor-5mal-nodefense-sentinel_r100_seed42

if [[ ! -f "$SENTINEL_DIR/checkpoints/final_model.pt" ]]; then
    echo "FATAL: sentinel final_model.pt not found at $SENTINEL_DIR/checkpoints/" >&2
    exit 1
fi

echo "── sentinel head-attr v2 (convergent) ──"
python -m analysis.head_feature_decomposition \
    --exp-dir "$SENTINEL_DIR" \
    --data-root "$DATA_ROOT" \
    --epochs 100 --patience 8 --min-improvement 1e-4 \
    --lr 1e-3 \
    --seed 4242 --device auto \
    --output "$SENTINEL_DIR/head_feature_decomposition_v2.json"

echo ""
echo "===== Sentinel result ====="
python3 -c "
import json
hd = json.load(open('$SENTINEL_DIR/head_feature_decomposition_v2.json'))
sm = json.load(open('$SENTINEL_DIR/summary.json'))
fin = sm['final']
print(f'phaseC2-backdoor-5mal-nodefense-sentinel  seed=42  acc={fin[\"test_accuracy\"]:.4f}  asr={fin[\"asr\"]:.4f}  ch_acc={hd[\"clean_head_clean_acc\"]:.4f}  ch_asr={hd[\"clean_head_asr\"]:.4f}  HEAD_ATTR={hd[\"head_attribution_pct\"]:6.2f}%  best_ep={hd[\"head_train_best_epoch\"]}/{hd[\"head_train_total_epochs_run\"]}  early_stop={hd[\"head_train_early_stopped\"]}')
"

echo ""
echo "End: $(date)"
