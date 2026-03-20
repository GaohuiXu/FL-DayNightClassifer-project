#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/flwr_local_env.sh"

echo "Using FLWR_HOME: $FLWR_HOME"

cd "$SCRIPT_DIR"
flwr run . local-simulation-gpu --stream "$@"