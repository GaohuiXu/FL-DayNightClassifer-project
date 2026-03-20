#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/flwr_local_env.sh"

echo "Using FLWR_HOME: $FLWR_HOME"

"$SCRIPT_DIR/stop_superlink.sh" || true

mkdir -p "$FLWR_HOME/local-superlink/ffs"

flower-superlink \
  --insecure \
  --simulation \
  --isolation subprocess \
  --control-api-address 127.0.0.1:39093 \
  --simulationio-api-address 127.0.0.1:39094 \
  --database "$FLWR_HOME/local-superlink/state.db" \
  --storage-dir "$FLWR_HOME/local-superlink/ffs" \
  --log-file "$FLWR_HOME/local-superlink/superlink.log"