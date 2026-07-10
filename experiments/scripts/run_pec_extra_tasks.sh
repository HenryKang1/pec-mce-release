#!/usr/bin/env bash
# Add 2 more LongBench tasks for non-inferiority + Pareto aggregate story.
# multifieldqa_en (n=150 max) and qasper (n=200) for both LFM and Qwen3-1.7B.
set -e
PY="python"
cd "$(dirname "$0")/.."

VARIANTS=(raw_topk pec_bridge pec_bridge_k3 pec_hop)
MODELS=(lfm2.5-1.2b-instruct qwen3-1.7b)

declare -A TASK_N
TASK_N[multifieldqa_en]=150
TASK_N[qasper]=200

for m in "${MODELS[@]}"; do
  for v in "${VARIANTS[@]}"; do
    for t in multifieldqa_en qasper; do
      n=${TASK_N[$t]}
      echo "=== [$(date +%H:%M:%S)] $m / $t / $v / n=$n ==="
      "$PY" scripts/longbench_pipeline.py \
        --task "$t" --model "$m" --variant "$v" --n-samples $n 2>&1 \
        | grep -E "EM=|Saved|Skip|Error|error" || true
    done
  done
done
echo "=== ALL DONE ==="
