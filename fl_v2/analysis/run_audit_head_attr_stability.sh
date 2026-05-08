#!/usr/bin/env bash
#SBATCH -A NAISS2025-22-1113
#SBATCH -p alvis
#SBATCH --gpus-per-node=A40:1
#SBATCH -t 0-01:00:00
#SBATCH -J audit_head_attr_stability
#SBATCH -o /mimer/NOBACKUP/groups/naiss2024-22-991/gaohui/fl_outputs/slurm/%x_%j.out
#SBATCH -e /mimer/NOBACKUP/groups/naiss2024-22-991/gaohui/fl_outputs/slurm/%x_%j.err

# ──────────────────────────────────────────────────────────────
# Audit: head-feature decomposition stability at fixed checkpoint.
#
# Tests two questions for the Cycle 02 reliability audit:
# 1. Does running the diagnostic TWICE with the same diagnostic seed on
#    the same checkpoint give bit-identical clean_head_asr?
# 2. Does using DIFFERENT diagnostic seeds (4242, 4243, 4244, 4245)
#    swing the head_attribution number meaningfully?
#
# Test checkpoint: full_ft + 5mal seed=42 (the original 97.2% headline cell).
# If same-seed gives identical numbers and different-seeds give close numbers
# (e.g. ±2pp), the diagnostic is stable.
# If same-seed reproduces but different-seeds swing wildly, the diagnostic
# itself is sensitive to its own initialisation.
# If same-seed does NOT reproduce, the diagnostic is non-deterministic and
# we have a bug to fix.
# ──────────────────────────────────────────────────────────────
set -euo pipefail

echo "===== Job info ====="
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Node: $(hostname)"
echo "Start: $(date)"

module purge
module load PyTorch/2.7.1-foss-2024a-CUDA-12.6.0
cd /mimer/NOBACKUP/groups/naiss2024-22-991/gaohui/thesis_workspace/fl_weather_project/fl_v2
source /mimer/NOBACKUP/groups/naiss2024-22-991/gaohui/thesis_workspace/fl_weather_project/.venv/bin/activate

export PYTHONUNBUFFERED=1

DATA_ROOT="/mimer/NOBACKUP/groups/naiss2024-22-991/gaohui/fl_datasets/gtsrb"
EXP="/mimer/NOBACKUP/groups/naiss2024-22-991/gaohui/fl_outputs/gtsrb/experiments/cycle_02/phaseD2/cycle02-pretrained-full-ft-pixel5_r100_seed42"
OUT_DIR="$EXP/audit_head_attr_stability"
mkdir -p "$OUT_DIR"

# Stash the original head_feature_decomposition.json so the diagnostic
# does not skip-if-exists; we want fresh runs.
ORIG_JSON="$EXP/head_feature_decomposition.json"
ORIG_BACKUP="$OUT_DIR/original_head_feature_decomposition.json"
if [[ -f "$ORIG_JSON" && ! -f "$ORIG_BACKUP" ]]; then
    cp "$ORIG_JSON" "$ORIG_BACKUP"
fi

run_one() {
    local diag_seed="$1"
    local label="$2"
    rm -f "$ORIG_JSON"  # force fresh run
    echo ""
    echo "── run: $label (--seed $diag_seed) ──"
    python -m analysis.head_feature_decomposition \
        --exp-dir "$EXP" \
        --data-root "$DATA_ROOT" \
        --epochs 10 \
        --lr 1e-3 \
        --seed "$diag_seed" \
        --device auto
    cp "$ORIG_JSON" "$OUT_DIR/${label}.json"
}

# Test A: same diagnostic seed, twice. Should be bit-identical clean_head_asr.
run_one 4242 "diag_seed4242_run1"
run_one 4242 "diag_seed4242_run2"

# Test B: different diagnostic seeds. How much does the result swing?
run_one 4243 "diag_seed4243"
run_one 4244 "diag_seed4244"
run_one 4245 "diag_seed4245"

# Restore the original so downstream consumers are unaffected.
cp "$ORIG_BACKUP" "$ORIG_JSON"

echo ""
echo "===== Audit summary ====="
for f in "$OUT_DIR"/diag_seed*.json; do
    python3 -c "
import json
d = json.load(open('$f'))
print(f'  {\"$f\".split(\"/\")[-1].replace(\".json\",\"\"):30s}  orig_asr={d[\"original_asr\"]:.6f}  ch_asr={d[\"clean_head_asr\"]:.6f}  ch_acc={d[\"clean_head_clean_acc\"]:.6f}  head_attr={d[\"head_attribution_pct\"]:.4f}%')"
done

echo ""
echo "End: $(date)"
