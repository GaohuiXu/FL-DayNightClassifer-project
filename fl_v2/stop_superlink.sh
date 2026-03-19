#!/usr/bin/env bash
set -euo pipefail

echo "Stopping local Flower SuperLink on 127.0.0.1:39093/39094 ..."

pids=$(lsof -tiTCP:39093 -sTCP:LISTEN -n -P || true)
pids2=$(lsof -tiTCP:39094 -sTCP:LISTEN -n -P || true)

all_pids="$(printf "%s\n%s\n" "$pids" "$pids2" | sort -u | sed '/^$/d')"

if [[ -z "$all_pids" ]]; then
  echo "No listening SuperLink process found."
  exit 0
fi

echo "Found PID(s):"
echo "$all_pids"

while read -r pid; do
  [[ -z "$pid" ]] && continue
  kill "$pid" 2>/dev/null || true
done <<< "$all_pids"

sleep 2

remaining=$(printf "%s\n%s\n" \
  "$(lsof -tiTCP:39093 -sTCP:LISTEN -n -P || true)" \
  "$(lsof -tiTCP:39094 -sTCP:LISTEN -n -P || true)" \
  | sort -u | sed '/^$/d')

if [[ -n "$remaining" ]]; then
  echo "Some process(es) still alive, forcing kill:"
  echo "$remaining"
  while read -r pid; do
    [[ -z "$pid" ]] && continue
    kill -9 "$pid" 2>/dev/null || true
  done <<< "$remaining"
fi

echo "Done."