#!/bin/bash
# D — centralized (single-node, full-pooled-data, NO FedAvg) baseline (D14 Phase-2 D), matched budget
# EPOCHS == FL rounds, numeric-mode=tf32. Trains then runs the official readiness eval --diagnostic
# (same evaluator as E; no D10 RAISE since centralized is not a D10 FL checkpoint). 1 A40 (centralized
# does not FedAvg-parallelize); generous walltime (~50 min/epoch TF32 at batch-16).
#   D1 = clean (this script). D2 = the gated centralized attack — run ONLY if D1 clears the readiness
#   bar (eligible_count >= N_min AND car_recall > floor); wired in a follow-up once D1 lands.
# Submit:  EPOCHS=15 sbatch fl_v3/scripts/run_centralized_a40.sh
#
#SBATCH -A NAISS2025-22-1113
#SBATCH -p alvis
#SBATCH --job-name=centralized_d
#SBATCH --gpus-per-node=A40:1
#SBATCH --time=24:00:00
#SBATCH --output=fl_v3/scripts/logs/centralized_d_%j.out
#SBATCH --error=fl_v3/scripts/logs/centralized_d_%j.err
set -euo pipefail

PROJ_ROOT="/mimer/NOBACKUP/groups/naiss2024-22-991/gaohui/thesis_workspace/fl_weather_project"
if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then REPO="${SLURM_SUBMIT_DIR}"; else REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; fi
cd "$REPO"; mkdir -p fl_v3/scripts/logs fl_v3/collab/speedup

CONFIG="${CONFIG:-fl_v3/configs/t4_reference.json}"
EPOCHS="${EPOCHS:-15}"
NUMERIC_MODE="${NUMERIC_MODE:-tf32}"
LEVEL="${LEVEL:-strict}"                       # D15: strict (byte-identical) | relaxed (atomics+bf16+compile)
MAXSTEPS_FLAG=""; [ -n "${MAX_STEPS:-}" ] && MAXSTEPS_FLAG="--max-steps ${MAX_STEPS}"
COMPILE_FLAG=""; [ "${COMPILE:-0}" = "1" ] && COMPILE_FLAG="--compile"
TAG="${TAG:-centralized_clean_r${EPOCHS}_${LEVEL}}"
TRAINVAL_CACHE="${TRAINVAL_CACHE:-${PROJ_ROOT}/.claude/worktrees/infallible-feistel-d42c34/fl_outputs/nuscenes/info_cache}"
OUT_DIR="${REPO}/fl_outputs/nuscenes/experiments/cycle_04/speedup_D"
RESUME_FLAG=""; [ "${RESUME:-0}" = "1" ] && RESUME_FLAG="--resume"

export CUBLAS_WORKSPACE_CONFIG=":4096:8"
export TORCH_HOME="/cephyr/users/gaohui/Alvis/.cache/torch"
export WANDB_MODE=offline
if ! type module >/dev/null 2>&1; then [ -f /usr/share/lmod/lmod/init/bash ] && source /usr/share/lmod/lmod/init/bash; fi
module purge; module load PyTorch/2.7.1-foss-2024a-CUDA-12.6.0
# shellcheck disable=SC1091
source "${PROJ_ROOT}/.venv_v3/bin/activate"
export PYTHONPATH="${REPO}/fl_v3/src${PYTHONPATH:+:$PYTHONPATH}"

echo "===== D centralized baseline (epochs=${EPOCHS} numeric=${NUMERIC_MODE}) =====  node=$(hostname) job=${SLURM_JOB_ID:-local}"
nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv,noheader || true
[ -f "${TRAINVAL_CACHE}/nuscenes_info_v1.0-trainval_train_t1.v1.pkl" ] || { echo "[D] FATAL: trainval cache missing at ${TRAINVAL_CACHE}"; exit 3; }
[ -f "${TORCH_HOME}/hub/checkpoints/swin_t-704ceda3.pth" ] || { echo "[D] FATAL: swin_t ImageNet weights missing under TORCH_HOME"; exit 3; }

# --- train ---
python fl_v3/scripts/centralized_train.py \
    --config "$CONFIG" --epochs "$EPOCHS" --out-dir "$OUT_DIR" --tag "$TAG" $RESUME_FLAG $MAXSTEPS_FLAG $COMPILE_FLAG \
    "numeric-mode=${NUMERIC_MODE}" "determinism-level=${LEVEL}" "nuscenes-cache-dir=${TRAINVAL_CACHE}"

CKPT="${OUT_DIR}/${TAG}/final_model.pt"
[ -f "$CKPT" ] || { echo "[D] FATAL: training produced no checkpoint at ${CKPT}"; exit 1; }

# A capped run (MAX_STEPS set) is a SMOKE (NaN/reasonableness check) — skip the expensive readiness eval.
if [ -n "${MAX_STEPS:-}" ]; then
    echo "[D] capped smoke (MAX_STEPS=${MAX_STEPS}) — skipping readiness eval; train loss is the signal."
    exit 0
fi

# --- official diagnostic eval (same evaluator as E; --diagnostic skips the D10 FL gate) ---
EVAL_OUT="${OUT_DIR}/${TAG}/readiness_diag"
mkdir -p "$EVAL_OUT"
python fl_v3/scripts/t4_readiness_eval.py \
    --config "$CONFIG" --checkpoint "$CKPT" --output-dir "$EVAL_OUT" --diagnostic \
    "nuscenes-cache-dir=${TRAINVAL_CACHE}" "numeric-mode=${NUMERIC_MODE}"

echo "===== D1 centralized verdict (DIAGNOSTIC) ====="
python3 -c "import json;d=json.load(open('${EVAL_OUT}/benchmark_readiness.json'));print('verdict',d['verdict'],'scope',d.get('verdict_scope'),'| mAP',round(d['mAP'],4),'NDS',round(d['NDS'],4),'car_recall',round(d['official_clean_car_recall'],4),'eligible_N',d['eligible_count'],'false_disappear',d['false_disappearance'].get('false_disappearance_rate'))" || true
echo "[D] D2 (centralized attack) gate: run ONLY if the above clears eligible_N>=N_min AND car_recall>floor."
echo "Elapsed: ${SECONDS}s"
