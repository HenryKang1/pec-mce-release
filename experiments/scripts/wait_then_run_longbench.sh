#!/usr/bin/env bash
# Wait for GPU to be less saturated, then run LongBench evals sequentially.
# Starts when GPU utilization stays <= threshold for N consecutive checks.

PY="python"
SCRIPT="./experiments/scripts/longbench_pipeline.py"
LOGDIR="./experiments/logs"

THRESHOLD="${THRESHOLD:-30}"   # percent
QUIET_CHECKS="${QUIET_CHECKS:-3}"  # need this many consecutive low readings
CHECK_INTERVAL="${CHECK_INTERVAL:-30}"  # seconds

echo "[Wait] Waiting for GPU util <= ${THRESHOLD}% for ${QUIET_CHECKS} consecutive checks..."
ok=0
while [ "$ok" -lt "$QUIET_CHECKS" ]; do
  util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | head -1 | tr -d ' ')
  echo "[Wait] util=${util}%  ok=${ok}/${QUIET_CHECKS}"
  if [ "$util" -le "$THRESHOLD" ]; then
    ok=$((ok + 1))
  else
    ok=0
  fi
  sleep "$CHECK_INTERVAL"
done

echo "[Start] GPU idle. Launching LongBench LFM pilot."

MODEL="lfm2.5-1.2b-instruct"
TASK="hotpotqa"
for variant in raw_trunc raw_topk summary anchors anchored; do
  echo "=== $MODEL / $TASK / $variant ==="
  "$PY" -u "$SCRIPT" \
    --task "$TASK" \
    --model "$MODEL" \
    --variant "$variant" \
    --top-k 5 \
    2>&1 | tee "$LOGDIR/lb_${MODEL//\//_}_${TASK}_${variant}.log" | tail -5
done

echo "[Done] LFM / $TASK pilot complete. Review results before expanding."
