#!/usr/bin/env bash
#SBATCH -A NAISS2025-22-1113
#SBATCH -p alvis
#SBATCH --gpus-per-node=A40:1
#SBATCH -t 0-02:30:00
#SBATCH -J audit_v5_samenode
#SBATCH -o /mimer/NOBACKUP/groups/naiss2024-22-991/gaohui/fl_outputs/slurm/%x_%j.out
#SBATCH -e /mimer/NOBACKUP/groups/naiss2024-22-991/gaohui/fl_outputs/slurm/%x_%j.err

# v5 verification: run the same YAML TWICE in sequence on the same node,
# same GPU. Isolates cross-node GPU non-determinism (different A40 cards
# producing slightly different floating-point results) from any remaining
# software-level non-determinism. Result interpretation:
#   - same-node bit-identical → cross-node-only issue (real but separable)
#   - same-node still diverges → 7th software-level source still unfixed

set -euo pipefail

echo "===== Job info ====="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Start: $(date)"
nvidia-smi --query-gpu=uuid --format=csv,noheader

# All env vars + module load happen via run_alvis.sh; just call it twice.
SCRIPT_DIR=/mimer/NOBACKUP/groups/naiss2024-22-991/gaohui/thesis_workspace/fl_weather_project/fl_v2
YAML_E=$SCRIPT_DIR/configs/experiments/cycle_02/phaseD2/_audit_v5_samenode_e.yaml
YAML_F=$SCRIPT_DIR/configs/experiments/cycle_02/phaseD2/_audit_v5_samenode_f.yaml

JID_E=$((SLURM_JOB_ID * 10 + 1))
JID_F=$((SLURM_JOB_ID * 10 + 2))

echo ""
echo "═══════════════════════════════════════════════════════"
echo "  Run E (first)  derived JID=$JID_E"
echo "═══════════════════════════════════════════════════════"
EXPERIMENT_YAML="$YAML_E" SLURM_JOB_ID=$JID_E bash $SCRIPT_DIR/run_alvis.sh

echo ""
echo "═══════════════════════════════════════════════════════"
echo "  Run F (second, same node, same GPU)  derived JID=$JID_F"
echo "═══════════════════════════════════════════════════════"
EXPERIMENT_YAML="$YAML_F" SLURM_JOB_ID=$JID_F bash $SCRIPT_DIR/run_alvis.sh

echo ""
echo "===== Comparison ====="
EXP_BASE=/mimer/NOBACKUP/groups/naiss2024-22-991/gaohui/fl_outputs/gtsrb/experiments/cycle_02/phaseD2
for d in cycle02-audit-v5-samenode-e cycle02-audit-v5-samenode-f; do
    f=$EXP_BASE/${d}_r30_seed42/summary.json
    if [[ -f "$f" ]]; then
        python3 -c "import json; d=json.load(open('$f')); fin=d['final']; print(f'$d  acc={fin[\"test_accuracy\"]:.10f} asr={fin.get(\"asr\",0):.10f}')"
    fi
done
echo "--- checkpoint hashes ---"
for d in cycle02-audit-v5-samenode-e cycle02-audit-v5-samenode-f; do
    f=$EXP_BASE/${d}_r30_seed42/checkpoints/round_0030.pt
    if [[ -f "$f" ]]; then
        sha256sum "$f" | awk -v d=$d '{print d"  "$1}'
    fi
done

echo "End: $(date)"
