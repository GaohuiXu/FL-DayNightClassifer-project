#!/bin/bash
# MCR Phase-3 STEP 3 — cRT decoupling probe (single A100). Freeze the FL-converged feature stack of the
# FedAvg-0.247 global, balanced-retrain ONLY the head (CBGS), then (EVAL=1) score raw+EMA on FULL val with
# the SAME protocol as the 0.247 baseline → decisive head-vs-representation answer + per-class AP table.
#   INIT=<fedavg round_15>/final_model.pt CACHE=<msweep10> [EPOCHS=2 EVAL=1] sbatch fl_v3/scripts/run_p3_crt_probe.sh
#SBATCH -A NAISS2025-22-1113
#SBATCH -p alvis
#SBATCH --job-name=p3_crt
#SBATCH --gpus-per-node=A100:1
#SBATCH --cpus-per-task=16
#SBATCH --time=04:00:00
#SBATCH --output=fl_v3/scripts/logs/p3_crt_%j.out
#SBATCH --error=fl_v3/scripts/logs/p3_crt_%j.err
set -euo pipefail

PROJ_ROOT="/mimer/NOBACKUP/groups/naiss2024-22-991/gaohui/thesis_workspace/fl_weather_project"
if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then REPO="${SLURM_SUBMIT_DIR}"; else REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; fi
cd "$REPO"; mkdir -p fl_v3/scripts/logs

CONFIG="${CONFIG:-fl_v3/configs/fl_bb02d_fedadam.json}"
INIT="${INIT:?set INIT=<fedavg round_15 dir>/final_model.pt (the 0.247 FedAvg global to decouple)}"
CACHE="${CACHE:-${PROJ_ROOT}/.claude/worktrees/unruffled-chaplygin-0e43e5/fl_outputs/nuscenes/info_cache_msweep10}"
OUT_DIR="${OUT_DIR:-fl_outputs/nuscenes/experiments/cycle_04/p3_crt}"
TAG="${TAG:-crt_head_fedavg15}"
EPOCHS="${EPOCHS:-2}"
MAX_STEPS="${MAX_STEPS:-0}"
TRAINABLE_PREFIX="${TRAINABLE_PREFIX:-head}"
SNAP_EPOCHS="${SNAP_EPOCHS:-1}"
EVAL="${EVAL:-1}"                       # 1 = chain a FULL-val eval (raw+ema) after the retrain, same A100
DRY="${DRY:-0}"                        # 1 = login wiring smoke (build+freeze+CBGS+provenance, NO training)
EXTRA_OVERRIDES="${EXTRA_OVERRIDES:-}"

export CUBLAS_WORKSPACE_CONFIG=":4096:8"
export TORCH_HOME="/cephyr/users/gaohui/Alvis/.cache/torch"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-/cephyr/users/gaohui/Alvis/.cache/inductor_crt}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
if ! type module >/dev/null 2>&1; then [ -f /usr/share/lmod/lmod/init/bash ] && source /usr/share/lmod/lmod/init/bash; fi
module purge; module load PyTorch/2.7.1-foss-2024a-CUDA-12.6.0
# shellcheck disable=SC1091
source "${PROJ_ROOT}/.venv_v3/bin/activate"
# CRITICAL: put the worktree's fl_v3/src FIRST so this code (not a sibling worktree) runs (the FL stale-code bug).
export PYTHONPATH="${REPO}/fl_v3/src${PYTHONPATH:+:$PYTHONPATH}"

echo "===== P3 cRT probe =====  node=$(hostname) job=${SLURM_JOB_ID:-local}"
echo "  config=${CONFIG}  init=${INIT}  cache=${CACHE}"
echo "  out=${OUT_DIR}/${TAG}  epochs=${EPOCHS} prefix=${TRAINABLE_PREFIX} eval=${EVAL} dry=${DRY}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true

DRY_FLAG=""; [ "${DRY}" = "1" ] && DRY_FLAG="--dry-run"
# shellcheck disable=SC2086
python fl_v3/scripts/p3_crt_probe.py \
    --config "$CONFIG" --init-from "$INIT" --out-dir "$OUT_DIR" --tag "$TAG" \
    --epochs "$EPOCHS" --max-steps "$MAX_STEPS" --trainable-prefix "$TRAINABLE_PREFIX" \
    --snapshot-epochs "$SNAP_EPOCHS" $DRY_FLAG \
    "nuscenes-cache-dir=${CACHE}" "precision=bf16" ${EXTRA_OVERRIDES}

if [ "${DRY}" = "1" ]; then echo "Elapsed: ${SECONDS}s (dry-run)"; exit 0; fi

# --- chained FULL-val eval (raw + ema), SAME protocol as the 0.247 FedAvg baseline (score 0.01 / maxobj 500,
#     det-eval-limit=0 = full val, --diagnostic = no D10 gate). Per-class AP lands in <dir>/readiness_diag/. ---
if [ "${EVAL}" = "1" ]; then
  EVAL_DIRS="${OUT_DIR}/${TAG}"
  [ -f "${OUT_DIR}/${TAG}/ema/final_model.pt" ] && EVAL_DIRS="${EVAL_DIRS} ${OUT_DIR}/${TAG}/ema"
  for d in ${EVAL_DIRS}; do
    echo "===== eval ${d} (full val) ====="
    CUDA_VISIBLE_DEVICES=0 python fl_v3/scripts/t4_readiness_eval.py \
      --config "$CONFIG" --checkpoint "${d}/final_model.pt" --output-dir "${d}/readiness_diag" \
      --diagnostic --no-gt-sanity \
      "nuscenes-cache-dir=${CACHE}" "precision=bf16" "det-score-threshold=0.01" "det-max-objects=500" \
      "det-eval-limit=0"
  done
fi
echo "Elapsed: ${SECONDS}s"
