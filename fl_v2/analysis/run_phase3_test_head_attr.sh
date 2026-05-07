#!/usr/bin/env bash
#SBATCH -A NAISS2025-22-1113
#SBATCH -p alvis
#SBATCH --gpus-per-node=A40:1
#SBATCH -t 0-01:00:00
#SBATCH -J phase3_test_head_attr
#SBATCH -o /mimer/NOBACKUP/groups/naiss2024-22-991/gaohui/fl_outputs/slurm/%x_%j.out
#SBATCH -e /mimer/NOBACKUP/groups/naiss2024-22-991/gaohui/fl_outputs/slurm/%x_%j.err

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
BASE=/mimer/NOBACKUP/groups/naiss2024-22-991/gaohui/fl_outputs/gtsrb/experiments/cycle_02/phaseD2

for seed in 42 43 44; do
    exp_dir=$BASE/cycle02-fixed-full-ft-pixel5_r100_seed${seed}
    if [[ ! -f "$exp_dir/checkpoints/final_model.pt" ]]; then
        echo "  [skip] seed=$seed: final_model.pt not found"
        continue
    fi
    if [[ -f "$exp_dir/head_feature_decomposition.json" ]]; then
        echo "  [skip] seed=$seed: diagnostic already ran"
        continue
    fi
    echo ""
    echo "── seed=$seed ──"
    python -m analysis.head_feature_decomposition \
        --exp-dir "$exp_dir" \
        --data-root "$DATA_ROOT" \
        --epochs 10 --lr 1e-3 \
        --seed 4242 --device auto
done

echo ""
echo "===== Phase 3 fixed-pipeline test summary ====="
for seed in 42 43 44; do
    f=$BASE/cycle02-fixed-full-ft-pixel5_r100_seed${seed}/head_feature_decomposition.json
    if [[ -f "$f" ]]; then
        python3 -c "
import json
d = json.load(open('$f'))
print(f'seed=${seed}  orig_acc=Will be loaded  orig_asr={d[\"original_asr\"]:.4f}  ch_asr={d[\"clean_head_asr\"]:.4f}  ch_acc={d[\"clean_head_clean_acc\"]:.4f}  HEAD_ATTR={d[\"head_attribution_pct\"]:.2f}%')
"
    fi
done

echo "End: $(date)"
