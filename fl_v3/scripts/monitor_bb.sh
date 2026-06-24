#!/bin/bash
# Health monitor for the two LiDAR-backbone runs (Step A 6775191, Step B 6775192). Polls every 10 min:
# waits through PENDING, then checks each job is training healthily. Exits early (re-invoking the agent) on
# the FIRST unhealthy signal (FAILED/CANCELLED/TIMEOUT state, stderr Traceback/OOM/CUDA-error, or non-finite
# loss), or when BOTH complete, or after a 6.6h cap. The benign DDP "grad strides" warning is NOT fatal.
JA=${JA:-6775191}; JB=${JB:-6775192}
D=fl_v3/scripts/logs
state() { sacct -j "$1" --format=State -n 2>/dev/null | head -1 | tr -d ' '; }
epline() { grep -oE "epoch [0-9]+/[0-9]+ loss=[0-9.]+ n=[0-9]+ time=[0-9]+s" "$D/centr_ddp_$1.out" 2>/dev/null | tail -1; }
unhealthy() {  # echo a reason if job $1 is unhealthy, else nothing
  local j=$1 st; st=$(state "$j")
  case "$st" in FAILED*|CANCELLED*|TIMEOUT*|OUT_OF*|NODE_FAIL*|DEADLINE*) echo "state=$st"; return;; esac
  grep -qE "Traceback \(most recent call last\)|out of memory|CUDA error|Segmentation fault|terminate called after" \
       "$D/centr_ddp_$j.err" 2>/dev/null && { echo "stderr error"; return; }
  grep -qiE "loss=(nan|inf|-inf)" "$D/centr_ddp_$j.out" 2>/dev/null && { echo "non-finite loss"; return; }
  echo ""
}
for i in $(seq 1 40); do
  ra=$(unhealthy "$JA"); rb=$(unhealthy "$JB")
  if [ -n "$ra" ] || [ -n "$rb" ]; then
    echo "=== BACKBONE MONITOR: UNHEALTHY @ $(date +%H:%M) ==="
    [ -n "$ra" ] && echo "  StepA $JA UNHEALTHY: $ra"
    [ -n "$rb" ] && echo "  StepB $JB UNHEALTHY: $rb"
    echo "  StepA: $(state $JA) | $(epline $JA)"; echo "  StepB: $(state $JB) | $(epline $JB)"
    exit 0
  fi
  sa=$(state "$JA"); sb=$(state "$JB")
  if echo "$sa" | grep -q COMPLETED && echo "$sb" | grep -q COMPLETED; then
    echo "=== BACKBONE MONITOR: BOTH COMPLETED @ $(date +%H:%M) ==="
    echo "  StepA: $(grep -E 'verdict READY|OFFICIAL mAP' $D/centr_ddp_$JA.out 2>/dev/null | tail -1)"
    echo "  StepB: $(grep -E 'verdict READY|OFFICIAL mAP' $D/centr_ddp_$JB.out 2>/dev/null | tail -1)"
    exit 0
  fi
  echo "[mon $(date +%H:%M) iter $i] StepA=$sa $(epline $JA) | StepB=$sb $(epline $JB)"
  sleep 600
done
echo "=== BACKBONE MONITOR: 6.6h cap reached (still pending/running?) ==="
echo "  StepA: $(state $JA) | $(epline $JA)"; echo "  StepB: $(state $JB) | $(epline $JB)"
