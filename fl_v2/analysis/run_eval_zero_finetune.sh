#!/usr/bin/env bash
#SBATCH -A NAISS2025-22-1113
#SBATCH -p alvis
#SBATCH --gpus-per-node=A40:1
#SBATCH -t 0-00:15:00
#SBATCH -J zero_ft_baseline
#SBATCH -o /mimer/NOBACKUP/groups/naiss2024-22-991/gaohui/fl_outputs/slurm/%x_%j.out
#SBATCH -e /mimer/NOBACKUP/groups/naiss2024-22-991/gaohui/fl_outputs/slurm/%x_%j.err
#
# Strict zero-fine-tuning baseline: ImageNet-pretrained ResNet18 on GTSRB
# without any GTSRB-specific training. Tells us whether FL is necessary
# (if zero-FT clean acc is already ~95 %, FL adds nothing; if zero-FT is
# at chance ~2.3 %, FL is doing all the work).
#
# Tiny job: ~3 fc seeds × 2 architectures × ~10 s = ~1 min compute on
# A40. SBATCH wallclock 15 min is generous to account for cuDNN warmup
# and dataset-loading latency on first read.

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
OUTPUT=/mimer/NOBACKUP/groups/naiss2024-22-991/gaohui/fl_outputs/gtsrb/baselines/zero_finetune_baseline.json

python -m analysis.eval_zero_finetune \
    --data-root "$DATA_ROOT" \
    --output "$OUTPUT" \
    --device auto

echo ""
echo "===== Final results ====="
python3 -c "
import json
d = json.load(open('$OUTPUT'))
print('Pretrained ResNet18 evaluated on GTSRB WITHOUT any FL fine-tuning')
print('=' * 70)
for arch_label, arch in d['by_architecture'].items():
    print(f'  {arch_label}  (image-size={arch[\"image_size\"]}, fc_seeds={d[\"fc_seeds\"]})')
    print(f'    clean_acc        = {arch[\"clean_acc_mean\"]:.4f} +/- {arch[\"clean_acc_std\"]:.4f}')
    print(f'    target_class_acc = {arch[\"target_class_clean_acc_mean\"]:.4f} +/- {arch[\"target_class_clean_acc_std\"]:.4f}')
    print(f'    asr (no defense) = {arch[\"asr_mean\"]:.4f} +/- {arch[\"asr_std\"]:.4f}')
    print()
"

echo "End: $(date)"
