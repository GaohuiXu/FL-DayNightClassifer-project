#!/usr/bin/env bash
#SBATCH -A NAISS2025-22-1113
#SBATCH -p alvis
#SBATCH --gpus-per-node=A40:1
#SBATCH -t 0-02:00:00
#SBATCH -J centralised_clean
#SBATCH -o /mimer/NOBACKUP/groups/naiss2024-22-991/gaohui/fl_outputs/slurm/%x_%j.out
#SBATCH -e /mimer/NOBACKUP/groups/naiss2024-22-991/gaohui/fl_outputs/slurm/%x_%j.err
#
# Centralised clean upper-bound: ResNet18 trained centrally on GTSRB,
# matching the FL setup's architecture (pretrained, modified-conv1,
# image-size 32) so the comparison is apples-to-apples with our FL
# clean baseline.
#
# Wallclock 2 h is generous; 50 epochs × ~30 s/epoch ≈ 25 min actual.

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
export PYTHONHASHSEED=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8

DATA_ROOT=/mimer/NOBACKUP/groups/naiss2024-22-991/gaohui/fl_datasets/gtsrb
OUT_DIR=/mimer/NOBACKUP/groups/naiss2024-22-991/gaohui/fl_outputs/gtsrb/baselines

# Pretrained + modified-conv1 (matches the Cycle 02 full_ft cell)
python -m analysis.eval_centralized_clean \
    --data-root "$DATA_ROOT" \
    --output    "$OUT_DIR/centralised_clean_pretrained_modified_conv1.json" \
    --epochs 50 --batch-size 128 --lr 0.05 --lr-min 1e-4 \
    --seed 42 --pretrained --device auto

echo ""
echo "===== Final summary ====="
python3 -c "
import json
d = json.load(open('$OUT_DIR/centralised_clean_pretrained_modified_conv1.json'))
f = d['final']
print('Centralised clean (pretrained + modified-conv1, image-size 32, 50 epochs):')
print(f'  test_clean_acc            = {f[\"test_clean_acc\"]:.4f}')
print(f'  test_target_class_clean_acc = {f[\"test_target_class_clean_acc\"]:.4f}')
print(f'  test_asr (no defense, no attack train) = {f[\"test_asr\"]:.4f}')
"

echo "End: $(date)"
