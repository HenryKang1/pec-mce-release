#!/usr/bin/env bash
# PEC-Hop novelty-defense ablations on LFM2.5-1.2B (main reader).
# 3 ablations x 5 tasks @ task-specific n.
set -e
PY="python"
cd "$(dirname "$0")/.."

VARIANTS=(pec_hop_no_anchor pec_hop_fact_only pec_hop_no_hydration)
MODELS=(lfm2.5-1.2b-instruct)

declare -A TASK_N
TASK_N[hotpotqa]=200
TASK_N[2wikimqa]=200
TASK_N[musique]=200
TASK_N[multifieldqa_en]=150
TASK_N[qasper]=200

for m in "${MODELS[@]}"; do
  for v in "${VARIANTS[@]}"; do
    for t in hotpotqa 2wikimqa musique multifieldqa_en qasper; do
      n=${TASK_N[$t]}
      echo "=== [$(date +%H:%M:%S)] $m / $t / $v / n=$n ==="
      "$PY" scripts/longbench_pipeline.py \
        --task "$t" --model "$m" --variant "$v" --n-samples $n 2>&1 \
        | grep -E "EM=|Saved|Skip|Error|error" || true
    done
  done
done
echo "=== ALL DONE ==="
