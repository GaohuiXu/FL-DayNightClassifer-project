#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────
# Submit an experiment to Alvis via SLURM.
#
# Usage:
#   ./submit_experiment.sh configs/experiments/clean_cnn.yaml
#   ./submit_experiment.sh configs/experiments/backdoor_cnn_krum.yaml
#
# The YAML filename (without .yaml) is used as the SLURM job name,
# so `squeue` output is easy to read.
# ──────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <experiment-yaml>"
    echo ""
    echo "Available experiments:"
    ls -1 "$SCRIPT_DIR/configs/experiments/"*.yaml 2>/dev/null | while read -r f; do
        name=$(basename "$f" .yaml)
        desc=$(head -1 "$f" | sed 's/^# *//')
        printf "  %-35s  %s\n" "$name" "$desc"
    done
    exit 1
fi

YAML_FILE="$(realpath "$1")"

if [[ ! -f "$YAML_FILE" ]]; then
    echo "Error: file not found: $YAML_FILE" >&2
    exit 1
fi

JOB_NAME="$(basename "$YAML_FILE" .yaml)"

echo "Submitting: $JOB_NAME"
echo "Config:     $YAML_FILE"

sbatch --job-name="$JOB_NAME" --export=ALL,EXPERIMENT_YAML="$YAML_FILE" "$SCRIPT_DIR/run_alvis.sh"
