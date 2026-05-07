#!/usr/bin/env bash
#SBATCH -A NAISS2025-22-1113
#SBATCH -p alvis
#SBATCH --gpus-per-node=A40:1
#SBATCH -t 0-02:00:00
#SBATCH -J audit_v8_singleactor
#SBATCH --nodelist=alvis8-09
#SBATCH -o /mimer/NOBACKUP/groups/naiss2024-22-991/gaohui/fl_outputs/slurm/%x_%j.out
#SBATCH -e /mimer/NOBACKUP/groups/naiss2024-22-991/gaohui/fl_outputs/slurm/%x_%j.err

# v8: same node, but force Ray to use a SINGLE actor (num-gpus=1.0 per
# supernode). Tests whether Ray multi-actor scheduling is the 7th
# non-determinism source.
#
# Workflow per run: temporarily override num-gpus in a per-job config,
# point FLWR_HOME at it, run the simulation, save outputs to a clearly
# tagged exp dir. Repeat with different exp name for the second run.
# ──────────────────────────────────────────────────────────────
set -euo pipefail

echo "===== Job info ====="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Start: $(date)"

module purge
module load PyTorch/2.7.1-foss-2024a-CUDA-12.6.0
cd /mimer/NOBACKUP/groups/naiss2024-22-991/gaohui/thesis_workspace/fl_weather_project/fl_v2
source /mimer/NOBACKUP/groups/naiss2024-22-991/gaohui/thesis_workspace/fl_weather_project/.venv/bin/activate

export PYTHONHASHSEED=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export RAY_DEDUP_LOGS=0

run_one() {
    local label="$1"
    local exp_yaml="$2"
    local jid_offset="$3"
    local jid_suffix=$((SLURM_JOB_ID * 10 + jid_offset))

    local flwr_home="/tmp/flwr_${jid_suffix}_${label}"
    mkdir -p "$flwr_home/local-superlink/ffs"

    # Generate single-actor flwr config
    cat > "$flwr_home/config.toml" <<EOF
[superlink]
default = "local-simulation-gpu"

[superlink.local-simulation-gpu]
options.num-supernodes = 50
options.backend.client-resources.num-cpus = 1
options.backend.client-resources.num-gpus = 1.0
EOF

    export FLWR_HOME="$flwr_home"
    export FLWR_LOCAL_CONTROL_API_PORT=$((39000 + (jid_suffix % 1000) * 2))
    export FLWR_LOCAL_SIMULATIONIO_API_PORT=$((FLWR_LOCAL_CONTROL_API_PORT + 1))

    echo ""
    echo "═══════════════════════════════════════════════════════"
    echo "  Run $label (single-actor Ray)"
    echo "  ports control=$FLWR_LOCAL_CONTROL_API_PORT sio=$FLWR_LOCAL_SIMULATIONIO_API_PORT"
    echo "═══════════════════════════════════════════════════════"

    # Start SuperLink
    flower-superlink \
      --insecure \
      --simulation \
      --isolation subprocess \
      --control-api-address "127.0.0.1:$FLWR_LOCAL_CONTROL_API_PORT" \
      --simulationio-api-address "127.0.0.1:$FLWR_LOCAL_SIMULATIONIO_API_PORT" \
      --database "$flwr_home/local-superlink/state.db" \
      --storage-dir "$flwr_home/local-superlink/ffs" \
      --log-file "$flwr_home/local-superlink/superlink.log" &
    local sl_pid=$!

    # Wait for ready
    for i in $(seq 1 30); do
        if kill -0 "$sl_pid" 2>/dev/null && \
           python -c "import socket, sys; s=socket.socket(); s.settimeout(1); s.connect(('127.0.0.1',int(sys.argv[1]))); s.close()" "$FLWR_LOCAL_CONTROL_API_PORT" 2>/dev/null; then
            echo "SuperLink ready"
            break
        fi
        sleep 1
    done

    # Build run-config
    local run_cfg=$(awk '
      /^[[:space:]]*#/ { next }
      /^[[:space:]]*$/ { next }
      /^[^:]+:/ {
        key = $0; sub(/:.*/, "", key);   gsub(/^[[:space:]]+|[[:space:]]+$/, "", key)
        val = $0; sub(/^[^:]+:[[:space:]]*/, "", val); gsub(/^[[:space:]]+|[[:space:]]+$/, "", val)
        if (val ~ /^".*"$/ || val ~ /^'\''.*'\''$/) { val = substr(val, 2, length(val)-2) }
        if (val ~ /^-?[0-9]*\.?[0-9]+$/) { printf "%s=%s ", key, val }
        else { printf "%s='\''%s'\'' ", key, val }
      }
    ' "$exp_yaml")

    flwr run . local-simulation-gpu --stream --run-config "$run_cfg" || true
    kill "$sl_pid" 2>/dev/null || true
    wait "$sl_pid" 2>/dev/null || true
    rm -rf "$flwr_home"
}

cd /mimer/NOBACKUP/groups/naiss2024-22-991/gaohui/thesis_workspace/fl_weather_project/fl_v2
run_one k configs/experiments/cycle_02/phaseD2/_audit_v8_singleactor_k.yaml 1
run_one l configs/experiments/cycle_02/phaseD2/_audit_v8_singleactor_l.yaml 2

echo ""
echo "===== Comparison ====="
EXP_BASE=/mimer/NOBACKUP/groups/naiss2024-22-991/gaohui/fl_outputs/gtsrb/experiments/cycle_02/phaseD2
for d in cycle02-audit-v8-singleactor-k cycle02-audit-v8-singleactor-l; do
    f=$EXP_BASE/${d}_r30_seed42/summary.json
    if [[ -f "$f" ]]; then
        python3 -c "import json; d=json.load(open('$f')); fin=d['final']; print(f'$d  acc={fin[\"test_accuracy\"]:.10f} asr={fin.get(\"asr\",0):.10f}')"
    fi
done
for d in cycle02-audit-v8-singleactor-k cycle02-audit-v8-singleactor-l; do
    f=$EXP_BASE/${d}_r30_seed42/checkpoints/round_0030.pt
    if [[ -f "$f" ]]; then
        sha256sum "$f" | awk -v d=$d '{print d" sha256="$1}'
    fi
done

echo "End: $(date)"
