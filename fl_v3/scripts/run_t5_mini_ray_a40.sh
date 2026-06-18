#!/bin/bash
# T5 mini Ray smoke (1 A40, ~5 min): a TINY real flwr/Ray run on v1.0-mini (resnet18, 1 round, N=4,
# attack-enabled, m=1) — validates (a) the Ray WORKERS import THIS worktree's fl_v3 via PYTHONPATH
# (the "[ATTACK] client … ∈ roster" line MUST appear in the log; its absence = a silent clean run =
# Ray used the wrong fl_v3 → STOP, re-point the editable install), and (b) the end-to-end flwr-poison
# integration (server saves a checkpoint, the routing fires). NOT a scientific result (scale=mini).
# Submit:  sbatch fl_v3/scripts/run_t5_mini_ray_a40.sh
#
#SBATCH -A NAISS2025-22-1113
#SBATCH -p alvis
#SBATCH --job-name=t5_mini_ray
#SBATCH --gpus-per-node=A40:1
#SBATCH --time=00:30:00
#SBATCH --output=fl_v3/scripts/logs/t5_mini_ray_%j.out
#SBATCH --error=fl_v3/scripts/logs/t5_mini_ray_%j.err
set -euo pipefail

PROJ_ROOT="/mimer/NOBACKUP/groups/naiss2024-22-991/gaohui/thesis_workspace/fl_weather_project"
if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then REPO="${SLURM_SUBMIT_DIR}"; else REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; fi
cd "$REPO"; mkdir -p fl_v3/scripts/logs

CONFIG="${CONFIG:-fl_v3/configs/t5_mini_smoke.json}"
FEDERATION="${FEDERATION:-local-simulation-gpu}"   # single GPU

export CUBLAS_WORKSPACE_CONFIG=":4096:8"
export TORCH_HOME="/cephyr/users/gaohui/Alvis/.cache/torch"
export PYTHONPATH="${REPO}/fl_v3/src${PYTHONPATH:+:${PYTHONPATH}}"
if ! type module >/dev/null 2>&1; then [ -f /usr/share/lmod/lmod/init/bash ] && source /usr/share/lmod/lmod/init/bash; fi
module purge; module load PyTorch/2.7.1-foss-2024a-CUDA-12.6.0
# shellcheck disable=SC1091
source "${PROJ_ROOT}/.venv_v3/bin/activate"
# shellcheck disable=SC1091
source fl_v3/scripts/_fl_env.sh
fl_setup_env
trap 'rm -rf "${FLWR_HOME:-/tmp/none}" "${RAY_TMPDIR:-/tmp/none}"' EXIT
python -c "import fl_v3,sys; sys.exit(0 if fl_v3.__file__=='${REPO}/fl_v3/src/fl_v3/__init__.py' else 1)" \
  || { echo "[t5-mini-ray] FATAL: driver fl_v3 not this worktree"; exit 4; }

N=4
CACHE_DIR="$(python -c "import json;print(json.load(open('${CONFIG}'))['nuscenes-cache-dir'])")"
OUT_DIR="$(python -c "import json;print(json.load(open('${CONFIG}'))['output-dir'])")"
APP_DIR="${REPO}/fl_v3"; ABS_CACHE="${REPO}/${CACHE_DIR#./}"; OUT_DIR="${REPO}/${OUT_DIR#./}"
fl_preflight_offline "$ABS_CACHE"
fl_stamp_supernodes "$N" fl_v3/configs/flwr_config.toml "${FLWR_HOME}/config.toml" "[superlink.${FEDERATION}]"

echo "===== T5 mini Ray smoke =====  node=$(hostname) job=${SLURM_JOB_ID:-local}"
RC="$(python fl_v3/scripts/runconfig.py "$CONFIG" "experiment-name=t5_mini_ray" "device=cuda" \
      "nuscenes-cache-dir=${ABS_CACHE}" "output-dir=${OUT_DIR}")"
LOG="${FLWR_HOME}/flwr_mini.log"
set +e; flwr run "$APP_DIR" "$FEDERATION" --stream --run-config "$RC" | tee "$LOG"; EX=${PIPESTATUS[0]}; set -e
fl_silent_exit_guard "$LOG" || exit 1

echo "===== PROPAGATION CHECK ====="
if grep -q "\[ATTACK\] client" "$LOG"; then
    echo "[t5-mini-ray] PASS: the attack code path FIRED in the Ray workers (PYTHONPATH propagates) ✓"
    grep "\[ATTACK\] client" "$LOG" | sort -u
else
    echo "[t5-mini-ray] FAIL: NO '[ATTACK]' line — Ray workers did NOT use this worktree's fl_v3."
    echo "[t5-mini-ray]       The heavy runs would silently train CLEAN. Re-point the editable install."
    exit 5
fi
echo "Elapsed: ${SECONDS}s"
