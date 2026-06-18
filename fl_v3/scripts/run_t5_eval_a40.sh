#!/bin/bash
# T5 single-task eval launcher (1 A40): TASK ∈ {aggregate, stealth, guards, viz, null-verify}.
#   aggregate   — combine the ablation shards → 5-condition table + fusion-aware verdict + occlusion
#   stealth     — poisoned clean DetectionEval (mAP/NDS + official car recall ≥ floor)  [~1-2 h, bs=1]
#   guards      — cond-5a LiDAR-invariance + camera-only clean-recall precondition
#   viz         — V5 + V3(trigger) for a few subset samples
#   null-verify — diff the null checkpoint's full-state sha256 vs the pinned clean 0fe444e3
# Submit:  TASK=aggregate sbatch fl_v3/scripts/run_t5_eval_a40.sh
#          TASK=null-verify CKPT=/path/t5_null/final_model.pt sbatch fl_v3/scripts/run_t5_eval_a40.sh
#
#SBATCH -A NAISS2025-22-1113
#SBATCH -p alvis
#SBATCH --job-name=t5_eval
#SBATCH --gpus-per-node=A40:1
#SBATCH --time=04:00:00
#SBATCH --output=fl_v3/scripts/logs/t5_eval_%j.out
#SBATCH --error=fl_v3/scripts/logs/t5_eval_%j.err
set -euo pipefail

PROJ_ROOT="/mimer/NOBACKUP/groups/naiss2024-22-991/gaohui/thesis_workspace/fl_weather_project"
if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then REPO="${SLURM_SUBMIT_DIR}"; else REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; fi
cd "$REPO"; mkdir -p fl_v3/scripts/logs

TASK="${TASK:-aggregate}"
CONFIG="${CONFIG:-fl_v3/configs/t5_attack.json}"
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
  || { echo "[t5-eval] FATAL: fl_v3 not this worktree"; exit 4; }

echo "===== T5 eval TASK=${TASK} =====  node=$(hostname) job=${SLURM_JOB_ID:-local}"
nvidia-smi --query-gpu=name --format=csv,noheader || true
mkdir -p "$OUT"
python fl_v3/scripts/t5_attack_eval.py --task "$TASK" \
    --config "$CONFIG" --checkpoint "$CKPT" --clean-checkpoint "$CLEAN_CKPT" \
    --subset "$FROZEN_SUB" --output-dir "$OUT" \
    "nuscenes-cache-dir=${ABS_CACHE}"
echo "Elapsed: ${SECONDS}s"
