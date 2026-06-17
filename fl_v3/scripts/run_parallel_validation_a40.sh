#!/bin/bash
# T3 FOLLOW-UP — concurrent-actor determinism validation (the T5-T7 speedup question).
# Runs the gate config (resnet18, N=8, 3 rounds, fraction 0.5) TWICE same-seed under a
# CONCURRENT-actor federation and compares the final FL_TRAINABLE_CHECKSUM to itself
# (run-to-run reproducibility) AND to the single-actor reference d82ef500... .
#
#   Path B (shared GPU):  sbatch --gpus-per-node=A40:1 \
#       --export=ALL,FEDERATION=local-simulation-gpu-shared run_parallel_validation_a40.sh
#   Path A (1 client/GPU): sbatch --gpus-per-node=A40:4 \
#       --export=ALL,FEDERATION=local-simulation-gpu-4x     run_parallel_validation_a40.sh
#
#SBATCH -A NAISS2025-22-1113
#SBATCH -p alvis
#SBATCH --job-name=t3_parallel_val
#SBATCH --gpus-per-node=A40:1
#SBATCH --time=01:00:00
#SBATCH --output=fl_v3/scripts/logs/parallel_val_%j.out
#SBATCH --error=fl_v3/scripts/logs/parallel_val_%j.err
set -euo pipefail

PROJ_ROOT="/mimer/NOBACKUP/groups/naiss2024-22-991/gaohui/thesis_workspace/fl_weather_project"
if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then REPO="${SLURM_SUBMIT_DIR}"; else REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; fi
cd "$REPO"; mkdir -p fl_v3/scripts/logs

FEDERATION="${FEDERATION:-local-simulation-gpu-shared}"
GATE_JSON="${GATE_JSON:-fl_v3/configs/t3_fl_gate.json}"
SINGLE_ACTOR_REF="d82ef5001b88bd157161fe7b3eb658a9493fd169b4be28d3b5d3d9c34c08b236"

export CUBLAS_WORKSPACE_CONFIG=":4096:8"
export TORCH_HOME="/cephyr/users/gaohui/Alvis/.cache/torch"
if ! type module >/dev/null 2>&1; then [ -f /usr/share/lmod/lmod/init/bash ] && source /usr/share/lmod/lmod/init/bash; fi
module purge; module load PyTorch/2.7.1-foss-2024a-CUDA-12.6.0
# shellcheck disable=SC1091
source "${PROJ_ROOT}/.venv_v3/bin/activate"
# shellcheck disable=SC1091
source fl_v3/scripts/_fl_env.sh
fl_setup_env
trap 'rm -rf "${FLWR_HOME:-/tmp/none}" "${RAY_TMPDIR:-/tmp/none}"' EXIT

APP_DIR="${REPO}/fl_v3"
CACHE_DIR="$(python -c "import json;print(json.load(open('${GATE_JSON}')).get('nuscenes-cache-dir','./fl_outputs/nuscenes/info_cache'))")"
OUT_DIR="$(python -c "import json;print(json.load(open('${GATE_JSON}'))['output-dir'])")"
ABS_CACHE="${REPO}/${CACHE_DIR#./}"; ABS_OUT="${REPO}/${OUT_DIR#./}"
PATH_OVERRIDES=("nuscenes-cache-dir=${ABS_CACHE}" "output-dir=${ABS_OUT}")
fl_preflight_offline "$ABS_CACHE"
# These validation federations carry num-supernodes=8 inline (no stamping needed).
cp fl_v3/configs/flwr_config.toml "${FLWR_HOME}/config.toml"

echo "===== T3 concurrent-actor determinism validation ====="
echo "node=$(hostname) job=${SLURM_JOB_ID:-local} FEDERATION=${FEDERATION} GPUs=${SLURM_GPUS_PER_NODE:-?}"
nvidia-smi --query-gpu=name --format=csv,noheader | head -1
nvidia-smi -L | head

# ONE flwr run per job (chaining two simulations in one process throws flwr Exit-700).
# The reference d82ef500... is the gate's serial baseline, already proven reproducible
# (runA==runB in job 6764008) — so concurrent==reference establishes BOTH determinism and
# byte-identity to serial in a single run. Run TWO separate jobs only if it differs.
RUN_LOG="${FLWR_HOME}/val.log"
RC="$(python fl_v3/scripts/runconfig.py "$GATE_JSON" "experiment-name=t3_paraval_${FEDERATION}" "${PATH_OVERRIDES[@]}")"
echo "----- ${FEDERATION} single run -----"
set +e; flwr run "$APP_DIR" "$FEDERATION" --stream --run-config "$RC" | tee "$RUN_LOG"; FLWR_EXIT=${PIPESTATUS[0]}; set -e
fl_silent_exit_guard "$RUN_LOG" || exit 1

CHK="$(cat "${ABS_OUT}/t3_paraval_${FEDERATION}/trainable_checksum.txt" 2>/dev/null || echo NA)"
echo "===== PARALLEL VALIDATION RESULT (${FEDERATION}) ====="
echo "device          = $(python -c 'import torch;print(torch.cuda.get_device_name())')"
echo "concurrent cksum= ${CHK}"
echo "single-actor ref= ${SINGLE_ACTOR_REF}"
if [ "$CHK" = "$SINGLE_ACTOR_REF" ]; then
    echo "[val] PASS-STRONG: concurrent (${FEDERATION}) is BYTE-IDENTICAL to the single-actor serial baseline"
    echo "[val]   => this concurrency is a drop-in speedup with ZERO determinism cost (null-configs safe)."
elif [ "$CHK" != "NA" ]; then
    echo "[val] DIFFERS from serial: concurrent=${CHK}. Run a 2nd same-seed job to test reproducible-but-different vs non-deterministic."
else
    echo "[val] FAIL: no checksum (silent zero-round). Resubmit."
fi
echo "Elapsed: ${SECONDS}s"
exit "$FLWR_EXIT"
