#!/bin/bash
# MCR Phase-1 Step 0 — protocol-faithful RE-DECODE of an EXISTING checkpoint (NO retrain). The gap
# analysis found our decode deflates our own mAP: it drops score<0.1 and caps at 200 boxes/sample
# BEFORE the devkit, zeroing real recall bins for the low-AP rare classes. The official nuScenes
# protocol submits the full tail at max_boxes_per_sample=500. So we re-decode the saved ckpt at
# SCORE=0.01 / MAXOBJ=500 (the devkit's own min_recall/min_precision=0.1 clip handles the floor) to
# get the honest official utility number — and the per-class AP table (now surfaced in the readiness
# JSON). The 0.1 floor is kept ONLY for the ASR path (its tau_clean gate is separate). A100:1, batch-1
# inference over the full val split (~20-30 min). Reusable for the Exp3 eval (override CKPT/OUT/CONFIG).
#   Submit: sbatch fl_v3/scripts/run_p1_redecode_a100.sh
#   Reuse:  CKPT=/path/final_model.pt OUT=/path/readiness_xyz sbatch fl_v3/scripts/run_p1_redecode_a100.sh
#
#SBATCH -A NAISS2025-22-1113
#SBATCH -p alvis
#SBATCH --job-name=p1_redecode
#SBATCH --gpus-per-node=A100:1
#SBATCH --time=01:30:00
#SBATCH --output=fl_v3/scripts/logs/p1_redecode_%j.out
#SBATCH --error=fl_v3/scripts/logs/p1_redecode_%j.err
set -euo pipefail

PROJ_ROOT="/mimer/NOBACKUP/groups/naiss2024-22-991/gaohui/thesis_workspace/fl_weather_project"
if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then REPO="${SLURM_SUBMIT_DIR}"; else REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; fi
cd "$REPO"; mkdir -p fl_v3/scripts/logs

CONFIG="${CONFIG:-fl_v3/configs/p1_unfrozen.json}"
DDP_OUT="${REPO}/fl_outputs/nuscenes/experiments/cycle_04/p2_ddp"
CKPT="${CKPT:-${DDP_OUT}/ddp_g16_r15/final_model.pt}"
OUT="${OUT:-${DDP_OUT}/ddp_g16_r15/readiness_redecoded}"
SCORE="${SCORE:-0.01}"
MAXOBJ="${MAXOBJ:-500}"
TRAINVAL_CACHE="${TRAINVAL_CACHE:-${PROJ_ROOT}/.claude/worktrees/infallible-feistel-d42c34/fl_outputs/nuscenes/info_cache}"

export CUBLAS_WORKSPACE_CONFIG=":4096:8"
export TORCH_HOME="/cephyr/users/gaohui/Alvis/.cache/torch"
export WANDB_MODE=offline
if ! type module >/dev/null 2>&1; then [ -f /usr/share/lmod/lmod/init/bash ] && source /usr/share/lmod/lmod/init/bash; fi
module purge; module load PyTorch/2.7.1-foss-2024a-CUDA-12.6.0
# shellcheck disable=SC1091
source "${PROJ_ROOT}/.venv_v3/bin/activate"
export PYTHONPATH="${REPO}/fl_v3/src${PYTHONPATH:+:$PYTHONPATH}"

echo "===== P1 Step-0 re-decode (SCORE=${SCORE} MAXOBJ=${MAXOBJ}) =====  node=$(hostname) job=${SLURM_JOB_ID:-local}"
nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv,noheader || true
echo "config=${CONFIG}"; echo "checkpoint=${CKPT}"; echo "output=${OUT}"
[ -f "$CKPT" ] || { echo "[redecode] FATAL: checkpoint not found: ${CKPT}" >&2; exit 2; }
mkdir -p "$OUT"

# --diagnostic (centralized ckpt, not D10); --no-gt-sanity (conversion already confirmed AP@2m=1.0 in 6770205).
python fl_v3/scripts/t4_readiness_eval.py \
    --config "$CONFIG" --checkpoint "$CKPT" --output-dir "$OUT" --diagnostic --no-gt-sanity \
    "precision=bf16" "nuscenes-cache-dir=${TRAINVAL_CACHE}" \
    "det-score-threshold=${SCORE}" "det-max-objects=${MAXOBJ}"

echo "===== re-decoded verdict (DIAGNOSTIC, score=${SCORE} maxobj=${MAXOBJ}) ====="
python3 -c "import json;d=json.load(open('${OUT}/benchmark_readiness.json'));print('verdict',d['verdict'],'| mAP',round(d['mAP'],4),'NDS',round(d['NDS'],4),'car_recall',round(d['official_clean_car_recall'],4),'eligible_N',d['eligible_count']);p=d.get('per_class_mean_ap') or {};[print(f'  {k:26s} {v:.4f}') for k,v in sorted(p.items(), key=lambda x:-x[1])]" || true
echo "Elapsed: ${SECONDS}s"
