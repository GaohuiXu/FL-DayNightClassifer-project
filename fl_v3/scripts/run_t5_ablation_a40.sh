#!/bin/bash
# T5 5-condition ablation — the per-target fan-out worker (D9 across-cell parallelism). A SLURM ARRAY
# job: each task processes target-shard ${SLURM_ARRAY_TASK_ID} of NUM_SHARDS over the frozen subset
# (2ad8f8da…) at batch_size=1, decoding cond-2/3/4/5a on the poisoned model + cond-4 on the clean
# model (occlusion). Aggregate afterwards with TASK=aggregate run_t5_eval_a40.sh.
# Submit:  sbatch --array=0-39 fl_v3/scripts/run_t5_ablation_a40.sh        (NUM_SHARDS must match: 40)
#
#SBATCH -A NAISS2025-22-1113
#SBATCH -p alvis
#SBATCH --job-name=t5_ablation
#SBATCH --gpus-per-node=A40:1
#SBATCH --time=06:00:00
#SBATCH --array=0-39
#SBATCH --output=fl_v3/scripts/logs/t5_ablation_%A_%a.out
#SBATCH --error=fl_v3/scripts/logs/t5_ablation_%A_%a.err
set -euo pipefail

PROJ_ROOT="/mimer/NOBACKUP/groups/naiss2024-22-991/gaohui/thesis_workspace/fl_weather_project"
if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then REPO="${SLURM_SUBMIT_DIR}"; else REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; fi
cd "$REPO"; mkdir -p fl_v3/scripts/logs

NUM_SHARDS="${NUM_SHARDS:-40}"
SHARD="${SLURM_ARRAY_TASK_ID:-0}"
CONFIG="${CONFIG:-fl_v3/configs/t5_attack.json}"
# inherited T4 artifacts (read-only) + the poisoned checkpoint (this worktree)
T4WT="${T4WT:-${PROJ_ROOT}/.claude/worktrees/infallible-feistel-d42c34}"
CLEAN_CKPT="${CLEAN_CKPT:-${T4WT}/fl_outputs/nuscenes/experiments/cycle_04/t4_reference/t4_reference/final_model.pt}"
FROZEN_SUB="${FROZEN_SUB:-${T4WT}/fl_outputs/nuscenes/experiments/cycle_04/t4_reference/readiness_bs1/frozen_asr_subset.json}"
CKPT="${CKPT:-${REPO}/fl_outputs/nuscenes/experiments/cycle_04/t5_attack/t5_relocation/final_model.pt}"
OUT="${OUT:-${REPO}/fl_outputs/nuscenes/experiments/cycle_04/t5_attack/t5_relocation/eval}"
ABS_CACHE="${REPO}/$(python3 -c "import json;print(json.load(open('${CONFIG}'))['nuscenes-cache-dir'])" | sed 's#^\./##')"

export CUBLAS_WORKSPACE_CONFIG=":4096:8"
export TORCH_HOME="/cephyr/users/gaohui/Alvis/.cache/torch"; export WANDB_MODE=offline
export PYTHONPATH="${REPO}/fl_v3/src${PYTHONPATH:+:${PYTHONPATH}}"
if ! type module >/dev/null 2>&1; then [ -f /usr/share/lmod/lmod/init/bash ] && source /usr/share/lmod/lmod/init/bash; fi
module purge; module load PyTorch/2.7.1-foss-2024a-CUDA-12.6.0
# shellcheck disable=SC1091
source "${PROJ_ROOT}/.venv_v3/bin/activate"
python -c "import fl_v3,sys; sys.exit(0 if fl_v3.__file__=='${REPO}/fl_v3/src/fl_v3/__init__.py' else 1)" \
  || { echo "[t5-ablation] FATAL: fl_v3 not this worktree"; exit 4; }

COND4_FLAG=""; [ "${COND4_ONLY:-0}" = "1" ] && COND4_FLAG="--cond4-only"   # lean CONTROL fan-out
echo "===== T5 ablation shard ${SHARD}/${NUM_SHARDS} ${COND4_FLAG} =====  node=$(hostname) job=${SLURM_JOB_ID:-local}"
nvidia-smi --query-gpu=name --format=csv,noheader || true
mkdir -p "$OUT"
python fl_v3/scripts/t5_attack_eval.py --task shard ${COND4_FLAG} \
    --config "$CONFIG" --checkpoint "$CKPT" --clean-checkpoint "$CLEAN_CKPT" \
    --subset "$FROZEN_SUB" --output-dir "$OUT" --shard "$SHARD" --num-shards "$NUM_SHARDS" \
    "nuscenes-cache-dir=${ABS_CACHE}"
echo "Elapsed: ${SECONDS}s"
