#!/usr/bin/env bash
set -euo pipefail

export TEST_FLWR_HOME=${TEST_FLWR_HOME:-/tmp/flwr_manual_${USER}}
mkdir -p "$TEST_FLWR_HOME/local-superlink/ffs"

echo "Using FLWR_HOME: $TEST_FLWR_HOME"

# Kill stale local Flower SuperLink processes
pkill -f flower-superlink 2>/dev/null || true
pkill -f flwr-superlink 2>/dev/null || true
sleep 2

flower-superlink \
  --insecure \
  --simulation \
  --isolation subprocess \
  --control-api-address 127.0.0.1:39093 \
  --simulationio-api-address 127.0.0.1:39094 \
  --database "$TEST_FLWR_HOME/local-superlink/state.db" \
  --storage-dir "$TEST_FLWR_HOME/local-superlink/ffs" \
  --log-file "$TEST_FLWR_HOME/local-superlink/superlink.log"