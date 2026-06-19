#!/bin/bash
# Official mAP/NDS + ASR-eligibility + false-disappearance + readiness verdict on a checkpoint, in the
# D14 TF32 regime (numeric-mode=tf32, batch_size=1 canonical decode). Serves BOTH:
#   * E (clean FL TF32 reference): default (reference gate, D10 provenance verified).
#   * D (centralized baseline): DIAGNOSTIC=1 -> --diagnostic (same evaluator, no D10 RAISE).
# Submit:  CKPT=/path/final_model.pt OUT=/path [DIAGNOSTIC=1] sbatch fl_v3/scripts/run_readiness_tf32_a40.sh
#
#SBATCH -A NAISS2025-22-1113
#SBATCH -p alvis
#SBATCH --job-name=readiness_tf32
#SBATCH --gpus-per-node=A40:1
#SBATCH --time=04:00:00
#SBATCH --output=fl_v3/scripts/logs/readiness_tf32_%j.out
#SBATCH --error=fl_v3/scripts/logs/readiness_tf32_%j.err
set -euo pipefail

PROJ_ROOT="/mimer/NOBACKUP/groups/naiss2024-22-991/gaohui/thesis_workspace/fl_weather_project"
if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then REPO="${SLURM_SUBMIT_DIR}"; else REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; fi
cd "$REPO"; mkdir -p fl_v3/scripts/logs

CONFIG="${CONFIG:-fl_v3/configs/t4_reference.json}"
NUMERIC_MODE="${NUMERIC_MODE:-tf32}"
CKPT="${CKPT:?set CKPT=/path/final_model.pt}"
OUT="${OUT:-$(dirname "$CKPT")/readiness_tf32}"
TRAINVAL_CACHE="${TRAINVAL_CACHE:-${PROJ_ROOT}/.claude/worktrees/infallible-feistel-d42c34/fl_outputs/nuscenes/info_cache}"
DIAG_FLAG=""; [ "${DIAGNOSTIC:-0}" = "1" ] && DIAG_FLAG="--diagnostic"

export CUBLAS_WORKSPACE_CONFIG=":4096:8"
export TORCH_HOME="/cephyr/users/gaohui/Alvis/.cache/torch"
export WANDB_MODE=offline
if ! type module >/dev/null 2>&1; then [ -f /usr/share/lmod/lmod/init/bash ] && source /usr/share/lmod/lmod/init/bash; fi
module purge; module load PyTorch/2.7.1-foss-2024a-CUDA-12.6.0
# shellcheck disable=SC1091
source "${PROJ_ROOT}/.venv_v3/bin/activate"
export PYTHONPATH="${REPO}/fl_v3/src${PYTHONPATH:+:$PYTHONPATH}"

echo "===== readiness eval (numeric=${NUMERIC_MODE} diagnostic=${DIAGNOSTIC:-0}) =====  node=$(hostname) job=${SLURM_JOB_ID:-local}"
nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader || true
echo "checkpoint=${CKPT}"; echo "output=${OUT}"
[ -f "$CKPT" ] || { echo "[readiness-tf32] ERROR: checkpoint not found: ${CKPT}" >&2; exit 2; }
mkdir -p "$OUT"

python fl_v3/scripts/t4_readiness_eval.py \
    --config "$CONFIG" --checkpoint "$CKPT" --output-dir "$OUT" $DIAG_FLAG \
    "nuscenes-cache-dir=${TRAINVAL_CACHE}" "numeric-mode=${NUMERIC_MODE}"

echo "===== readiness verdict ====="
python3 -c "import json;d=json.load(open('${OUT}/benchmark_readiness.json'));print(d['verdict'],'scope',d.get('verdict_scope'),'numeric',d.get('numeric_mode'),'| mAP',round(d['mAP'],4),'NDS',round(d['NDS'],4),'car_recall',round(d['official_clean_car_recall'],4),'eligible_N',d['eligible_count'],'false_disappear',d['false_disappearance'].get('false_disappearance_rate'),'| hash',d['frozen_subset_hash'][:16])" || true
echo "Elapsed: ${SECONDS}s"
