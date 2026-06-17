#!/bin/bash
# fl_v3 FL bit-determinism GATE on the A40 (T3 GATE, SPEC §2.5 — the crown jewel).
#
#   (0) assert_a40 (LOUD exit-2 on non-A40/CPU — reused from det_gate_a40).
#   (1) local_runner: two same-seed run_clean_rounds (>=3 rounds, fraction<1) byte-identical.
#   (2) Ray: two same-seed `flwr run` drivers -> byte-identical final global (FL_TRAINABLE_
#       CHECKSUM). Different node_ids per driver; the DT3-B sampler picks the same pids.
#   (3) cross-check: Ray runA FL_TRAINABLE_CHECKSUM == local_runner LOCAL_RUNNER_CHECKSUM
#       (byte-identical on the SAME A40).
#   (4) substrate stability: runA vs runB norm_log byte-identical (participant set + cosines).
#
# Submit:  sbatch fl_v3/scripts/run_fedavg_a40.sh
#
#SBATCH -A NAISS2025-22-1113
#SBATCH -p alvis
#SBATCH --job-name=t3_fl_gate
#SBATCH --gpus-per-node=A40:1
#SBATCH --time=04:00:00
#SBATCH --output=fl_v3/scripts/logs/fl_gate_%j.out
#SBATCH --error=fl_v3/scripts/logs/fl_gate_%j.err
set -euo pipefail

PROJ_ROOT="/mimer/NOBACKUP/groups/naiss2024-22-991/gaohui/thesis_workspace/fl_weather_project"
if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then REPO="${SLURM_SUBMIT_DIR}"; else REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; fi
cd "$REPO"
mkdir -p fl_v3/scripts/logs

GATE_JSON="${GATE_JSON:-fl_v3/configs/t3_fl_gate.json}"

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

# All python-dependent derivations AFTER the venv is on PATH (else exit 127).
NUM_SUPERNODES="$(python -c "import json;print(json.load(open('${GATE_JSON}'))['nuscenes-num-clients'])")"
CACHE_DIR="$(python -c "import json;print(json.load(open('${GATE_JSON}')).get('nuscenes-cache-dir','./fl_outputs/nuscenes/info_cache'))")"
OUT_DIR="$(python -c "import json;print(json.load(open('${GATE_JSON}'))['output-dir'])")"
GATE_SEED="$(python -c "import json;print(json.load(open('${GATE_JSON}'))['seed'])")"
# `flwr run` loads the Flower-App pyproject from the APP dir (fl_v3, not the repo root) and
# the Ray actors' cwd is not guaranteed to be the repo root — so pass the app path absolutely
# and make data/output paths ABSOLUTE (cwd-independent).
APP_DIR="${REPO}/fl_v3"
ABS_CACHE="${REPO}/${CACHE_DIR#./}"
OUT_DIR="${REPO}/${OUT_DIR#./}"
PATH_OVERRIDES=("nuscenes-cache-dir=${ABS_CACHE}" "output-dir=${OUT_DIR}")
fl_preflight_offline "$ABS_CACHE"
fl_stamp_supernodes "$NUM_SUPERNODES" fl_v3/configs/flwr_config.toml "${FLWR_HOME}/config.toml"
echo "===== T3 FL bit-determinism gate =====   node=$(hostname) job=${SLURM_JOB_ID:-local} N=${NUM_SUPERNODES}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# (0)+(1) assert_a40 + local_runner two-run determinism (in-process, fast).
LR_LOG="${FLWR_HOME}/fl_gate_localrunner.log"
python fl_v3/scripts/fl_gate_a40.py --config "$GATE_JSON" --mode localrunner | tee "$LR_LOG"
LR_CHECKSUM="$(grep 'LOCAL_RUNNER_CHECKSUM = ' "$LR_LOG" | awk '{print $NF}')"
echo "[gate] LOCAL_RUNNER_CHECKSUM = ${LR_CHECKSUM}"

# (2) two same-seed Ray runs -> byte-identical final global.
run_one() {  # $1 = run tag (A/B)
    local tag="$1" run_log="${FLWR_HOME}/flwr_run_${1}.log"
    local rc; rc="$(python fl_v3/scripts/runconfig.py "$GATE_JSON" "experiment-name=t3_gate_run${tag}" "${PATH_OVERRIDES[@]}")"
    echo "----- Ray run ${tag} -----"
    echo "run-config: ${rc}"
    set +e
    flwr run "$APP_DIR" local-simulation-gpu --stream --run-config "$rc" | tee "$run_log"
    local ex=${PIPESTATUS[0]}
    set -e
    fl_silent_exit_guard "$run_log" || return 1
    [ "$ex" -eq 0 ] || { echo "[gate] flwr run ${tag} exit ${ex}"; return "$ex"; }
}
run_one A
run_one B

CHK_A="$(cat "${OUT_DIR}/t3_gate_runA/trainable_checksum.txt")"
CHK_B="$(cat "${OUT_DIR}/t3_gate_runB/trainable_checksum.txt")"
echo "===== T3 FL GATE RESULTS ====="
echo "device              = $(python -c "import torch;print(torch.cuda.get_device_name())")"
echo "RAY runA checksum   = ${CHK_A}"
echo "RAY runB checksum   = ${CHK_B}"
echo "LOCAL_RUNNER cksum  = ${LR_CHECKSUM}"

FAIL=0
if [ "$CHK_A" != "$CHK_B" ]; then echo "[gate] FAIL: two same-seed Ray runs DIVERGED (A!=B)"; FAIL=1
else echo "[gate] PASS: two same-seed Ray runs byte-identical (A==B) — crown jewel"; fi
# Cross-check: same-A40 local_runner vs Ray MUST be byte-identical (HARD fail). The SPEC's
# allclose fallback is only for a documented cross-DEVICE case (CPU<->A40); here both run on
# the same A40, so byte-identity is required and a mismatch is a real determinism failure.
if [ "$CHK_A" != "$LR_CHECKSUM" ]; then
    echo "[gate] FAIL: cross-check Ray != local_runner on the same A40 (CHK_A=$CHK_A LR=$LR_CHECKSUM)"; FAIL=1
else echo "[gate] PASS: cross-check Ray==local_runner byte-identical on the same A40"; fi
# (4) substrate stability: norm_log content identical across the two runs (deterministic).
# A MISSING norm_log is itself a failure (no gradient-space substrate produced).
NL_A="${OUT_DIR}/t3_gate_runA/t3_gate_runA_seed${GATE_SEED}_norm_log.json"
NL_B="${OUT_DIR}/t3_gate_runB/t3_gate_runB_seed${GATE_SEED}_norm_log.json"
if [ ! -f "$NL_A" ] || [ ! -f "$NL_B" ]; then
    echo "[gate] FAIL: norm_log missing (NL_A exists=$([ -f "$NL_A" ] && echo y || echo n), NL_B=$([ -f "$NL_B" ] && echo y || echo n)) — no substrate artifact"; FAIL=1
elif diff -q <(python -c "import json,sys;print(json.dumps(json.load(open('$NL_A')),sort_keys=True))") \
              <(python -c "import json,sys;print(json.dumps(json.load(open('$NL_B')),sort_keys=True))") >/dev/null; then
    echo "[gate] PASS: norm_log (participant set + gradient-space substrate) byte-identical A==B"
else echo "[gate] FAIL: norm_log differs across runs (substrate not reproducible)"; FAIL=1; fi

echo "Elapsed: ${SECONDS}s"
[ "$FAIL" -eq 0 ] && echo "[gate] OVERALL: PASS" || { echo "[gate] OVERALL: FAIL"; exit 1; }
