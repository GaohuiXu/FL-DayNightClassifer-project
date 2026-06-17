#!/bin/bash
# DT4-A reference clean checkpoint (T4_SPEC §0.2/§0.3; D10) — the FULL-participation log-group
# trainval clean FedAvg the T5 attack binds to. fraction-train=1.0 (ALL N clients/round) via D9
# Path-A multi-GPU (one client per A40 on a 4-GPU node) for wall-clock. Single run by default;
# set T4_PAIRED=1 to run a 2nd same-seed run and assert byte-identical FL_TRAINABLE_CHECKSUM
# (full-participation byte-determinism re-assertion — a NEW checksum vs the T3 5-of-25 gate).
# Submit:  sbatch fl_v3/scripts/run_t4_reference_a40.sh
#
#SBATCH -A NAISS2025-22-1113
#SBATCH -p alvis
#SBATCH --job-name=t4_reference
#SBATCH --gpus-per-node=A40:4
#SBATCH --time=12:00:00
#SBATCH --output=fl_v3/scripts/logs/t4_reference_%j.out
#SBATCH --error=fl_v3/scripts/logs/t4_reference_%j.err
set -euo pipefail

PROJ_ROOT="/mimer/NOBACKUP/groups/naiss2024-22-991/gaohui/thesis_workspace/fl_weather_project"
if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then REPO="${SLURM_SUBMIT_DIR}"; else REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; fi
cd "$REPO"; mkdir -p fl_v3/scripts/logs

CONFIG="${CONFIG:-fl_v3/configs/t4_reference.json}"
FEDERATION="${FEDERATION:-local-simulation-gpu-4x}"   # D9 Path-A (4 A40s, 1 client/GPU)

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

N="$(python -c "import json;print(json.load(open('${CONFIG}'))['nuscenes-num-clients'])")"
CACHE_DIR="$(python -c "import json;print(json.load(open('${CONFIG}'))['nuscenes-cache-dir'])")"
OUT_DIR="$(python -c "import json;print(json.load(open('${CONFIG}'))['output-dir'])")"
ROUNDS="$(python -c "import json;print(json.load(open('${CONFIG}'))['num-server-rounds'])")"
FRACTION="$(python -c "import json;print(json.load(open('${CONFIG}'))['fraction-train'])")"
APP_DIR="${REPO}/fl_v3"
ABS_CACHE="${REPO}/${CACHE_DIR#./}"
OUT_DIR="${REPO}/${OUT_DIR#./}"
PATH_OVERRIDES=("nuscenes-cache-dir=${ABS_CACHE}" "output-dir=${OUT_DIR}")
fl_preflight_offline "$ABS_CACHE"
# Stamp the DERIVED N into the Path-A (-4x) federation block (default block is the single-GPU one).
fl_stamp_supernodes "$N" fl_v3/configs/flwr_config.toml "${FLWR_HOME}/config.toml" "[superlink.${FEDERATION}]"

echo "===== T4 reference (FULL participation, D10) =====  node=$(hostname) job=${SLURM_JOB_ID:-local}"
echo "N=${N} fraction-train=${FRACTION} rounds=${ROUNDS} federation=${FEDERATION}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
[ "$(printf '%.0f' "$FRACTION")" = "1" ] || echo "[t4-ref] WARNING: fraction-train != 1.0 (D10 wants FULL participation)"

run_one() {  # $1 = experiment tag
    local tag="$1" log="${FLWR_HOME}/flwr_${1}.log"
    local rc; rc="$(python fl_v3/scripts/runconfig.py "$CONFIG" "experiment-name=${tag}" "${PATH_OVERRIDES[@]}")"
    echo "----- reference run: ${tag} -----"; echo "run-config: ${rc}"
    set +e; flwr run "$APP_DIR" "$FEDERATION" --stream --run-config "$rc" | tee "$log"; local ex=${PIPESTATUS[0]}; set -e
    fl_silent_exit_guard "$log" || return 1
    [ "$ex" -eq 0 ] || { echo "[t4-ref] ${tag} flwr exit ${ex}"; return "$ex"; }
}
# Read the checksum from the artifact FILE, never via run_one's stdout (run_one streams the full
# flwr log to the SLURM .out; capturing its stdout would (a) swallow the live log and (b) make the
# paired comparison diff whole log-blobs instead of the 64-char checksum — a false DIVERGED).
read_chk() { cat "${OUT_DIR}/$1/trainable_checksum.txt" 2>/dev/null; }

run_one t4_reference || exit 1
CHK_A="$(read_chk t4_reference)"
echo "===== T4 REFERENCE RESULT ====="
echo "device          = $(python -c 'import torch;print(torch.cuda.get_device_name())')"
echo "checkpoint      = ${OUT_DIR}/t4_reference/final_model.pt"
echo "FL_TRAINABLE_CHECKSUM = ${CHK_A}"

if [ "${T4_PAIRED:-0}" = "1" ]; then
    run_one t4_reference_paired || exit 1
    CHK_B="$(read_chk t4_reference_paired)"
    echo "paired checksum = ${CHK_B}"
    if [ -n "$CHK_A" ] && [ "$CHK_A" = "$CHK_B" ]; then
        echo "[t4-ref] PASS: full-participation same-seed byte-identity (A==B)"
    else
        echo "[t4-ref] FAIL: full-participation runs DIVERGED (A!=B)"; exit 1
    fi
fi
echo "Elapsed: ${SECONDS}s"
echo "[t4-ref] Next: sbatch fl_v3/scripts/run_t4_readiness_eval_a40.sh  (CKPT defaults to the above)"
