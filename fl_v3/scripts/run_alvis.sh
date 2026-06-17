#!/bin/bash
# fl_v3 SLURM launcher — flwr-1.27 clean FedAvg on the A40 (T3, PLATFORM MILESTONE).
#
# Runs ONE clean `nuscenes_detection` (or any task) FedAvg via `flwr run`. flwr 1.27
# AUTO-MANAGES the local SuperLink (T3_SPEC §0.1) — NO manual `flower-superlink`, NO
# SuperExec wait. Hardening that still applies (per-job SuperLink/Ray ports + tmp,
# offline preflight, silent-exit guard) lives in scripts/_fl_env.sh.
#
# Usage (env-parametrised):
#   sbatch --export=ALL,NUM_SUPERNODES=8,RUN_CONFIG_FILE=fl_v3/configs/t3_fl_gate.json \
#          [,RUN_CONFIG_OVERRIDES="experiment-name='iid_mini' fraction-train=1.0"] \
#          fl_v3/scripts/run_alvis.sh
#   - NUM_SUPERNODES         : REQUIRED — the DERIVED client count N (stamped into the fed).
#   - RUN_CONFIG_FILE        : JSON run_config (default fl_v3/configs/t3_fl_gate.json).
#   - RUN_CONFIG_OVERRIDES   : extra `key=value` pairs appended to the run-config.
#
#SBATCH -A NAISS2025-22-1113
#SBATCH -p alvis
#SBATCH --job-name=fl_v3_fedavg
#SBATCH --gpus-per-node=A40:1
#SBATCH --time=04:00:00
#SBATCH --output=fl_v3/scripts/logs/fedavg_%j.out
#SBATCH --error=fl_v3/scripts/logs/fedavg_%j.err
set -euo pipefail

PROJ_ROOT="/mimer/NOBACKUP/groups/naiss2024-22-991/gaohui/thesis_workspace/fl_weather_project"
# Anchor at the submission dir (the repo/worktree root); fall back to BASH_SOURCE locally.
if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then REPO="${SLURM_SUBMIT_DIR}"; else REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; fi
cd "$REPO"
mkdir -p fl_v3/scripts/logs

NUM_SUPERNODES="${NUM_SUPERNODES:?set NUM_SUPERNODES=<derived client count N> (stamped into num-supernodes)}"
RUN_CONFIG_FILE="${RUN_CONFIG_FILE:-fl_v3/configs/t3_fl_gate.json}"
RUN_CONFIG_OVERRIDES="${RUN_CONFIG_OVERRIDES:-}"

# --- determinism env BEFORE any CUDA context + module/venv ---
export CUBLAS_WORKSPACE_CONFIG=":4096:8"
export TORCH_HOME="/cephyr/users/gaohui/Alvis/.cache/torch"
if ! type module >/dev/null 2>&1; then
    [ -f /usr/share/lmod/lmod/init/bash ] && source /usr/share/lmod/lmod/init/bash
fi
module purge
module load PyTorch/2.7.1-foss-2024a-CUDA-12.6.0
# shellcheck disable=SC1091
source "${PROJ_ROOT}/.venv_v3/bin/activate"

# shellcheck disable=SC1091
source fl_v3/scripts/_fl_env.sh
fl_setup_env
trap 'rm -rf "${FLWR_HOME:-/tmp/none}" "${RAY_TMPDIR:-/tmp/none}"' EXIT

# Absolute app dir + data/output paths (the Ray actors' cwd is not the repo root; `flwr run`
# loads the App pyproject from the APP dir, which is fl_v3 not the repo root).
APP_DIR="${REPO}/fl_v3"
CACHE_DIR="$(python -c "import json;print(json.load(open('${RUN_CONFIG_FILE}')).get('nuscenes-cache-dir','./fl_outputs/nuscenes/info_cache'))")"
OUT_DIR="$(python -c "import json;print(json.load(open('${RUN_CONFIG_FILE}')).get('output-dir','./outputs'))")"
ABS_CACHE="${REPO}/${CACHE_DIR#./}"
ABS_OUT="${REPO}/${OUT_DIR#./}"
fl_preflight_offline "$ABS_CACHE"
fl_stamp_supernodes "$NUM_SUPERNODES" fl_v3/configs/flwr_config.toml "${FLWR_HOME}/config.toml"

echo "===== fl_v3 FedAvg ====="
echo "node=$(hostname) job=${SLURM_JOB_ID:-local} N=${NUM_SUPERNODES}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

RUN_CONFIG="$(python fl_v3/scripts/runconfig.py "$RUN_CONFIG_FILE" "nuscenes-cache-dir=${ABS_CACHE}" "output-dir=${ABS_OUT}" $RUN_CONFIG_OVERRIDES)"
echo "run-config: ${RUN_CONFIG}"

RUN_LOG="${FLWR_HOME}/flwr_run.log"
# flwr run auto-starts + manages the local SuperLink; --stream relays ServerApp stdout.
set +e
flwr run "$APP_DIR" local-simulation-gpu --stream --run-config "$RUN_CONFIG" | tee "$RUN_LOG"
FLWR_EXIT=${PIPESTATUS[0]}
set -e

fl_silent_exit_guard "$RUN_LOG" || exit 1
echo "Elapsed: ${SECONDS}s"
exit "$FLWR_EXIT"
