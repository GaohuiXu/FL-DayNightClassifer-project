#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/flwr_local_env.sh"

echo "Stopping local Flower processes ..."
echo "Using FLWR_HOME: $FLWR_HOME"

pids=$(lsof -tiTCP:39093 -sTCP:LISTEN -n -P || true)
pids2=$(lsof -tiTCP:39094 -sTCP:LISTEN -n -P || true)

all_pids="$(printf "%s\n%s\n" "$pids" "$pids2" | sort -u | sed '/^$/d')"

if [[ -n "$all_pids" ]]; then
  echo "Found listener PID(s):"
  echo "$all_pids"
  while read -r pid; do
    [[ -z "$pid" ]] && continue
    kill "$pid" 2>/dev/null || true
  done <<< "$all_pids"
fi

sleep 2

pkill -f flwr-simulation 2>/dev/null || true
pkill -f flower-superlink 2>/dev/null || true
pkill -f flwr-superlink 2>/dev/null || true
pkill -f "control-api-address 127.0.0.1:39093" 2>/dev/null || true
pkill -f "simulationio-api-address 127.0.0.1:39094" 2>/dev/null || true

sleep 2

remaining="$(printf "%s\n%s\n" \
  "$(lsof -tiTCP:39093 -sTCP:LISTEN -n -P || true)" \
  "$(lsof -tiTCP:39094 -sTCP:LISTEN -n -P || true)" \
  | sort -u | sed '/^$/d')"

if [[ -n "$remaining" ]]; then
  echo "Forcing kill on remaining PID(s):"
  echo "$remaining"
  while read -r pid; do
    [[ -z "$pid" ]] && continue
    kill -9 "$pid" 2>/dev/null || true
  done <<< "$remaining"
fi

rm -rf "$FLWR_HOME/local-superlink"

echo "Done."