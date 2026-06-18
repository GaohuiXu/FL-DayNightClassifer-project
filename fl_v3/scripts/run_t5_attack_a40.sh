#!/bin/bash
# T5 fusion attack × FedAvg (D10 full-participation log-group trainval, Path-A 4×A40) — the poisoned
# checkpoints the T5 GATE binds to. ONE launcher, MODE-parameterized:
#   MODE=relocation     trigger + CENTER-RELOCATION (D4 primary, §0.A)        — the headline attack
#   MODE=trigger_only   trigger, no label edit (control)
#   MODE=label_only     relocate, no trigger (control)
#   MODE=delete         trigger + box-DELETION (the §0.A label_only_delete control; deletion-fails)
#   MODE=null           poison_rate=0 — must be BYTE-IDENTICAL to the clean a80466c3 (§0.C5)
# Submit:  MODE=relocation sbatch fl_v3/scripts/run_t5_attack_a40.sh
# Determinism re-check (cheap): T5_PAIRED=1 NUM_ROUNDS=3 MODE=relocation sbatch …  (A==B byte-identity)
#
#SBATCH -A NAISS2025-22-1113
#SBATCH -p alvis
#SBATCH --job-name=t5_attack
#SBATCH --gpus-per-node=A40:4
#SBATCH --time=12:00:00
#SBATCH --output=fl_v3/scripts/logs/t5_attack_%j.out
#SBATCH --error=fl_v3/scripts/logs/t5_attack_%j.err
set -euo pipefail

PROJ_ROOT="/mimer/NOBACKUP/groups/naiss2024-22-991/gaohui/thesis_workspace/fl_weather_project"
if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then REPO="${SLURM_SUBMIT_DIR}"; else REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; fi
cd "$REPO"; mkdir -p fl_v3/scripts/logs

MODE="${MODE:-relocation}"
CONFIG="${CONFIG:-fl_v3/configs/t5_attack.json}"
FEDERATION="${FEDERATION:-local-simulation-gpu-4x}"   # D9 Path-A (4 A40s, 1 client/GPU)

export CUBLAS_WORKSPACE_CONFIG=":4096:8"
export TORCH_HOME="/cephyr/users/gaohui/Alvis/.cache/torch"
# Make THIS worktree's fl_v3 the one imported by flwr/Ray workers (the venv editable-install may
# point elsewhere). PYTHONPATH is prepended ahead of the site .pth; the preflight below HARD-FAILS
# if a worker would import a different fl_v3 (a silent clean run masquerading as poisoned is the trap).
export PYTHONPATH="${REPO}/fl_v3/src${PYTHONPATH:+:${PYTHONPATH}}"
if ! type module >/dev/null 2>&1; then [ -f /usr/share/lmod/lmod/init/bash ] && source /usr/share/lmod/lmod/init/bash; fi
module purge; module load PyTorch/2.7.1-foss-2024a-CUDA-12.6.0
# shellcheck disable=SC1091
source "${PROJ_ROOT}/.venv_v3/bin/activate"
# shellcheck disable=SC1091
source fl_v3/scripts/_fl_env.sh
fl_setup_env
trap 'rm -rf "${FLWR_HOME:-/tmp/none}" "${RAY_TMPDIR:-/tmp/none}"' EXIT

# HARD preflight: the imported fl_v3 MUST be this worktree's source (so the attack code actually runs).
python - <<PY || { echo "[t5] FATAL: fl_v3 does NOT resolve to this worktree — refusing (would run CLEAN code)"; exit 4; }
import fl_v3, sys
f = fl_v3.__file__
want = "${REPO}/fl_v3/src/fl_v3/__init__.py"
print("[t5] fl_v3.__file__ =", f)
import fl_v3.attacks.poisoned_client  # must exist (T5 code present)
sys.exit(0 if f == want else 1)
PY

case "$MODE" in
  relocation)   ATTACK_MODE=relocation;   POISON_RATE=0.5 ;;
  trigger_only) ATTACK_MODE=trigger_only; POISON_RATE=0.5 ;;
  label_only)   ATTACK_MODE=label_only;   POISON_RATE=0.5 ;;
  delete)       ATTACK_MODE=delete;       POISON_RATE=0.5 ;;
  null)         ATTACK_MODE=relocation;   POISON_RATE=0.0 ;;
  *) echo "[t5] unknown MODE=$MODE"; exit 2 ;;
esac
NUM_ROUNDS="${NUM_ROUNDS:-$(python -c "import json;print(json.load(open('${CONFIG}'))['num-server-rounds'])")}"
TAG="${EXP_TAG:-t5_${MODE}}"   # override (e.g. a short paired determinism run that must NOT clobber the headline)

N="$(python -c "import json;print(json.load(open('${CONFIG}'))['nuscenes-num-clients'])")"
CACHE_DIR="$(python -c "import json;print(json.load(open('${CONFIG}'))['nuscenes-cache-dir'])")"
OUT_DIR="$(python -c "import json;print(json.load(open('${CONFIG}'))['output-dir'])")"
APP_DIR="${REPO}/fl_v3"
ABS_CACHE="${REPO}/${CACHE_DIR#./}"
OUT_DIR="${REPO}/${OUT_DIR#./}"
PATH_OVERRIDES=("nuscenes-cache-dir=${ABS_CACHE}" "output-dir=${OUT_DIR}"
                "attack-mode=${ATTACK_MODE}" "attack-poison-rate=${POISON_RATE}"
                "num-server-rounds=${NUM_ROUNDS}")
fl_preflight_offline "$ABS_CACHE"
fl_stamp_supernodes "$N" fl_v3/configs/flwr_config.toml "${FLWR_HOME}/config.toml" "[superlink.${FEDERATION}]"

echo "===== T5 attack (MODE=${MODE}, poison_rate=${POISON_RATE}, rounds=${NUM_ROUNDS}) =====  node=$(hostname) job=${SLURM_JOB_ID:-local}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
# D10-base preflight (the attack run is still full-participation log-group trainval clean-aggregator).
python - "$CONFIG" <<'PY' || { echo "[t5] FATAL: config not D10-base-compliant — refusing"; exit 3; }
import json, sys
c = json.load(open(sys.argv[1]))
req = {"task-type":"nuscenes_detection","nuscenes-version":"v1.0-trainval","nuscenes-train-split":"train",
       "nuscenes-val-split":"val","nuscenes-partition-mode":"log_group","defense-type":"none"}
bad = [f"{k}={c.get(k)!r}!={v!r}" for k,v in req.items() if str(c.get(k))!=v]
if float(c.get("fraction-train",-1))!=1.0: bad.append("fraction-train!=1.0")
if bad: print("[t5] D10-base violations:", "; ".join(bad), file=sys.stderr); sys.exit(1)
PY

run_one() {
    local tag="$1" log="${FLWR_HOME}/flwr_${1}.log"
    local rc; rc="$(python fl_v3/scripts/runconfig.py "$CONFIG" "experiment-name=${tag}" "${PATH_OVERRIDES[@]}")"
    echo "----- run: ${tag} -----"; echo "run-config: ${rc}"
    set +e; flwr run "$APP_DIR" "$FEDERATION" --stream --run-config "$rc" | tee "$log"; local ex=${PIPESTATUS[0]}; set -e
    fl_silent_exit_guard "$log" || return 1
    [ "$ex" -eq 0 ] || { echo "[t5] ${tag} flwr exit ${ex}"; return "$ex"; }
}
read_chk() { cat "${OUT_DIR}/$1/trainable_checksum.txt" 2>/dev/null; }

run_one "$TAG" || exit 1
CHK_A="$(read_chk "$TAG")"

# Provenance + roster manifest beside the checkpoint (the eval driver hard-verifies these).
CHK_A="$CHK_A" MODE="$MODE" POISON_RATE="$POISON_RATE" python - "$CONFIG" "${OUT_DIR}/${TAG}/provenance.json" "${OUT_DIR}/${TAG}/roster.json" <<'PY'
import json, os, sys
cfg = json.load(open(sys.argv[1]))
cfg["attack-mode"] = os.environ["MODE"] if os.environ["MODE"]!="null" else "relocation"
cfg["attack-poison-rate"] = float(os.environ["POISON_RATE"])
chk = os.environ.get("CHK_A","")
from fl_v3.attacks.poisoned_client import roster_record, malicious_roster
roster = malicious_roster(int(cfg["seed"]), int(cfg["nuscenes-num-clients"]), float(cfg["attack-rho"]))
rec = roster_record(int(cfg["seed"]), int(cfg["nuscenes-num-clients"]), float(cfg["attack-rho"]), cfg)
json.dump(rec, open(sys.argv[3],"w"), indent=2, sort_keys=True)
if float(cfg["attack-poison-rate"]) > 0.0:
    from fl_v3.eval.provenance import build_attack_provenance, check_attack
    prov = build_attack_provenance(cfg, chk, roster, len(roster))
    bad = check_attack(prov)
else:  # null: poison_rate=0 ⇒ clean D10 provenance (byte-identical to a80466c3)
    from fl_v3.eval.provenance import build_provenance, check_d10
    prov = build_provenance(cfg, chk); bad = check_d10(prov)
if bad: print("[t5] WARNING provenance non-compliant:", bad, file=sys.stderr)
json.dump(prov, open(sys.argv[2],"w"), indent=2, sort_keys=True)
print("[t5] wrote provenance.json:", prov["regime"], "| roster", roster, "| m_r", len(roster))
PY

echo "===== T5 ATTACK RESULT (${MODE}) ====="
echo "device          = $(python -c 'import torch;print(torch.cuda.get_device_name())')"
echo "checkpoint      = ${OUT_DIR}/${TAG}/final_model.pt"
echo "FL_TRAINABLE_CHECKSUM = ${CHK_A}"

if [ "${T5_PAIRED:-0}" = "1" ]; then
    run_one "${TAG}_paired" || exit 1
    CHK_B="$(read_chk "${TAG}_paired")"
    echo "paired checksum = ${CHK_B}"
    if [ -n "$CHK_A" ] && [ "$CHK_A" = "$CHK_B" ]; then
        echo "[t5] PASS: same-seed poisoned byte-identity (A==B)"
    else
        echo "[t5] FAIL: poisoned runs DIVERGED (A!=B)"; exit 1
    fi
fi
echo "Elapsed: ${SECONDS}s"
