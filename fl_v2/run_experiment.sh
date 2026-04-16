#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────
# Run a Flower experiment from a YAML config file.
#
# Usage:
#   ./run_experiment.sh configs/experiments/phaseC_v2/1_clean.yaml
#   ./run_experiment.sh configs/experiments/phaseD/1_modelrep_5mal_nodefense.yaml
#
# The YAML file should contain only the overrides from pyproject.toml defaults.
# Each key-value pair is converted to a --run-config argument for `flwr run`.
# ──────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/flwr_local_env.sh"

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <experiment-yaml> [extra --run-config overrides...]"
    echo ""
    echo "Available experiments:"
    for dir in "$SCRIPT_DIR/configs/experiments"/*/; do
        group=$(basename "$dir")
        echo "  [$group]"
        find "$dir" -name "*.yaml" -maxdepth 1 2>/dev/null | sort | while read -r f; do
            name=$(basename "$f" .yaml)
            desc=$(head -1 "$f" | sed 's/^# *//')
            printf "    %-30s  %s\n" "$name" "$desc"
        done
    done
    exit 1
fi

YAML_FILE="$1"
shift  # remaining args are extra overrides

if [[ ! -f "$YAML_FILE" ]]; then
    echo "Error: file not found: $YAML_FILE" >&2
    exit 1
fi

# Parse flat YAML key-value pairs (no Python dependency — pure awk)
# Skips comments and blank lines.  Produces: key1=value1 key2='str' ...
RUN_CONFIG=$(awk '
  /^[[:space:]]*#/ { next }          # skip comments
  /^[[:space:]]*$/ { next }          # skip blank lines
  /^[^:]+:/ {
    key = $0; sub(/:.*/, "", key);   gsub(/^[[:space:]]+|[[:space:]]+$/, "", key)
    val = $0; sub(/^[^:]+:[[:space:]]*/, "", val); gsub(/^[[:space:]]+|[[:space:]]+$/, "", val)
    # Remove surrounding quotes if present
    if (val ~ /^".*"$/ || val ~ /^'\''.*'\''$/) {
      val = substr(val, 2, length(val)-2)
    }
    # Quote strings (non-numeric values)
    if (val ~ /^-?[0-9]*\.?[0-9]+$/) {
      printf "%s=%s ", key, val
    } else {
      printf "%s='\''%s'\'' ", key, val
    }
  }
' "$YAML_FILE")

echo "──────────────────────────────────────────────"
echo "Experiment config: $YAML_FILE"
echo "Run config:        $RUN_CONFIG"
if [[ $# -gt 0 ]]; then
    echo "Extra overrides:   $*"
fi
echo "──────────────────────────────────────────────"

cd "$SCRIPT_DIR"

if [[ -n "$RUN_CONFIG" ]]; then
    flwr run . local-simulation-gpu --stream --run-config "$RUN_CONFIG $*"
else
    flwr run . local-simulation-gpu --stream "$@"
fi
