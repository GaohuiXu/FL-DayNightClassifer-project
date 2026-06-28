#!/bin/bash
# Standalone readiness eval of one or more checkpoints on a single A100 (post-hoc convergence-curve / snapshot
# eval). Pass CKPTS as a space-sep env list of <dir> each containing final_model.pt + trainable_checksum.txt +
# provenance.json. Decode at the official utility protocol (score 0.01 / maxobj 500). Eval-only, no training.
#   CKPTS="a/ema_ep20 a/ema_ep30" CONFIG=fl_v3/configs/p1_msweep.json TRAINVAL_CACHE=... sbatch run_eval_ckpt_a100.sh
#SBATCH -A NAISS2025-22-1113
#SBATCH -p alvis
#SBATCH --job-name=eval_ckpt
#SBATCH --gpus-per-node=A100:1
#SBATCH --cpus-per-task=16
#SBATCH --time=01:30:00
#SBATCH --output=fl_v3/scripts/logs/eval_ckpt_%j.out
#SBATCH --error=fl_v3/scripts/logs/eval_ckpt_%j.err
set -euo pipefail

PROJ_ROOT="/mimer/NOBACKUP/groups/naiss2024-22-991/gaohui/thesis_workspace/fl_weather_project"
if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then REPO="${SLURM_SUBMIT_DIR}"; else REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; fi
cd "$REPO"; mkdir -p fl_v3/scripts/logs

CONFIG="${CONFIG:-fl_v3/configs/p1_msweep.json}"
TRAINVAL_CACHE="${TRAINVAL_CACHE:-${PROJ_ROOT}/.claude/worktrees/unruffled-chaplygin-0e43e5/fl_outputs/nuscenes/info_cache_msweep10}"

export CUBLAS_WORKSPACE_CONFIG=":4096:8"
export TORCH_HOME="/cephyr/users/gaohui/Alvis/.cache/torch"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
if ! type module >/dev/null 2>&1; then [ -f /usr/share/lmod/lmod/init/bash ] && source /usr/share/lmod/lmod/init/bash; fi
module purge; module load PyTorch/2.7.1-foss-2024a-CUDA-12.6.0
# shellcheck disable=SC1091
source "${PROJ_ROOT}/.venv_v3/bin/activate"
export PYTHONPATH="${REPO}/fl_v3/src${PYTHONPATH:+:$PYTHONPATH}"

for d in ${CKPTS}; do
  echo "===== eval ${d} ====="
  # EXTRA = optional space-sep key=value overrides (e.g. "det-eval-limit=0" to force the FULL val set; the
  # FL config may carry a small det-eval-limit for the in-run proxy). Placed LAST so it wins.
  CUDA_VISIBLE_DEVICES=0 python fl_v3/scripts/t4_readiness_eval.py \
    --config "$CONFIG" --checkpoint "${d}/final_model.pt" --output-dir "${d}/readiness_diag" \
    --diagnostic --no-gt-sanity \
    "nuscenes-cache-dir=${TRAINVAL_CACHE}" "precision=bf16" "det-score-threshold=0.01" "det-max-objects=500" \
    ${EXTRA:-}
done
echo "Elapsed: ${SECONDS}s"
