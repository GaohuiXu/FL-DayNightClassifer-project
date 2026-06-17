#!/bin/bash
# DT4-A benchmark-readiness eval (T4_SPEC §0.2/§0.3) — the official DetectionEval mAP/NDS + the
# ASR-eligibility harness + the frozen subset + the false-disappearance baseline + the READY/
# NOT-READY verdict, on the FULL trainval val split, bound to the full-participation log-group
# checkpoint. NOT a Flower run (plain python eval); 1 A40. Run AFTER run_t4_reference_a40.sh.
# Submit:  sbatch fl_v3/scripts/run_t4_readiness_eval_a40.sh
#   (override CKPT=/path/final_model.pt and OUT=/path to point elsewhere)
#
#SBATCH -A NAISS2025-22-1113
#SBATCH -p alvis
#SBATCH --job-name=t4_readiness
#SBATCH --gpus-per-node=A40:1
#SBATCH --time=04:00:00
#SBATCH --output=fl_v3/scripts/logs/t4_readiness_%j.out
#SBATCH --error=fl_v3/scripts/logs/t4_readiness_%j.err
set -euo pipefail

PROJ_ROOT="/mimer/NOBACKUP/groups/naiss2024-22-991/gaohui/thesis_workspace/fl_weather_project"
if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then REPO="${SLURM_SUBMIT_DIR}"; else REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; fi
cd "$REPO"; mkdir -p fl_v3/scripts/logs

CONFIG="${CONFIG:-fl_v3/configs/t4_reference.json}"
REF_OUT="$(python3 -c "import json;print(json.load(open('${CONFIG}'))['output-dir'])" 2>/dev/null || echo ./fl_outputs/nuscenes/experiments/cycle_04/t4_reference)"
CKPT="${CKPT:-${REPO}/${REF_OUT#./}/t4_reference/final_model.pt}"
OUT="${OUT:-${REPO}/${REF_OUT#./}/readiness}"
ABS_CACHE="${REPO}/$(python3 -c "import json;print(json.load(open('${CONFIG}'))['nuscenes-cache-dir'])" 2>/dev/null | sed 's#^\./##')"

export CUBLAS_WORKSPACE_CONFIG=":4096:8"
export TORCH_HOME="/cephyr/users/gaohui/Alvis/.cache/torch"
export WANDB_MODE=offline
if ! type module >/dev/null 2>&1; then [ -f /usr/share/lmod/lmod/init/bash ] && source /usr/share/lmod/lmod/init/bash; fi
module purge; module load PyTorch/2.7.1-foss-2024a-CUDA-12.6.0
# shellcheck disable=SC1091
source "${PROJ_ROOT}/.venv_v3/bin/activate"

echo "===== T4 readiness eval =====  node=$(hostname) job=${SLURM_JOB_ID:-local}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
echo "config=${CONFIG}"; echo "checkpoint=${CKPT}"; echo "output=${OUT}"
if [ ! -f "$CKPT" ]; then
    echo "[t4-readiness] ERROR: checkpoint not found: ${CKPT}" >&2
    echo "  run the reference first:  sbatch fl_v3/scripts/run_t4_reference_a40.sh" >&2
    exit 2
fi
mkdir -p "$OUT"

python fl_v3/scripts/t4_readiness_eval.py \
    --config "$CONFIG" \
    --checkpoint "$CKPT" \
    --output-dir "$OUT" \
    "nuscenes-cache-dir=${ABS_CACHE}"

echo "===== readiness verdict ====="
python3 -c "import json;d=json.load(open('${OUT}/benchmark_readiness.json'));print(d['verdict'],'| mAP',round(d['mAP'],4),'NDS',round(d['NDS'],4),'car_recall',round(d['official_clean_car_recall'],4),'eligible_N',d['eligible_count'],'| hash',d['frozen_subset_hash'][:16])" || true
echo "Elapsed: ${SECONDS}s"
