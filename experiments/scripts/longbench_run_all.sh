#!/usr/bin/env bash
# Queue all LongBench variants sequentially (avoid GPU contention).
# Usage: bash longbench_run_all.sh [model] [tasks] [variants]
set -e

MODEL="${1:-lfm2.5-1.2b-instruct}"
TASKS="${2:-hotpotqa 2wikimqa musique}"
VARIANTS="${3:-raw_trunc raw_topk summary anchors anchored}"

PY="python"
SCRIPT="./experiments/scripts/longbench_pipeline.py"
LOGDIR="./experiments/logs"
mkdir -p "$LOGDIR"

for task in $TASKS; do
  for variant in $VARIANTS; do
    echo "=== $MODEL / $task / $variant ==="
    "$PY" -u "$SCRIPT" \
      --task "$task" \
      --model "$MODEL" \
      --variant "$variant" \
      --top-k 5 \
      2>&1 | tee "$LOGDIR/lb_${MODEL//\//_}_${task}_${variant}.log" | tail -10
  done
done
