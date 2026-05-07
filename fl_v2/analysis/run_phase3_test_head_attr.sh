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

CELLS=(
    cycle02-fixed-full-ft-pixel5
    cycle02-fixed-lastblock-pixel5
    cycle02-fixed-headonly-canonconv1-pixel15
)

for cell in "${CELLS[@]}"; do
    for seed in 42 43 44; do
        exp_dir=$BASE/${cell}_r100_seed${seed}
        if [[ ! -f "$exp_dir/checkpoints/final_model.pt" ]]; then
            echo "  [skip] $cell seed=$seed: final_model.pt not found"
            continue
        fi
        if [[ -f "$exp_dir/head_feature_decomposition.json" ]]; then
            echo "  [skip] $cell seed=$seed: diagnostic already ran"
            continue
        fi
        echo ""
        echo "── $cell seed=$seed ──"
        python -m analysis.head_feature_decomposition \
            --exp-dir "$exp_dir" \
            --data-root "$DATA_ROOT" \
            --epochs 10 --lr 1e-3 \
            --seed 4242 --device auto
    done
done

echo ""
echo "===== Phase 3.1 fixed-pipeline summary (9 cells: 3 cells x 3 seeds) ====="
for cell in "${CELLS[@]}"; do
    for seed in 42 43 44; do
        f=$BASE/${cell}_r100_seed${seed}/head_feature_decomposition.json
        s=$BASE/${cell}_r100_seed${seed}/summary.json
        if [[ -f "$f" && -f "$s" ]]; then
            python3 -c "
import json
hd = json.load(open('$f'))
sm = json.load(open('$s'))
fin = sm['final']
print(f'$cell  seed=${seed}  acc={fin[\"test_accuracy\"]:.4f}  asr={fin[\"asr\"]:.4f}  ch_acc={hd[\"clean_head_clean_acc\"]:.4f}  ch_asr={hd[\"clean_head_asr\"]:.4f}  HEAD_ATTR={hd[\"head_attribution_pct\"]:6.2f}%')
"
        fi
    done
done

echo "End: $(date)"
