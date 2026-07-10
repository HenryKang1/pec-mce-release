#!/usr/bin/env bash
# PEC variant pilot: 3 variants x 3 tasks x 2 models @ n=50
# Used to pick winners before promoting to n=200.
set -e
PY="python"
cd "$(dirname "$0")/.."

VARIANTS=(pec_hop pec_bridge_k3 pec_query_expand)
TASKS=(hotpotqa 2wikimqa musique)
MODELS=(lfm2.5-1.2b-instruct qwen3-1.7b)
N=50

for m in "${MODELS[@]}"; do
  for v in "${VARIANTS[@]}"; do
    for t in "${TASKS[@]}"; do
      echo "=== [$(date +%H:%M:%S)] $m / $t / $v / n=$N ==="
      "$PY" scripts/longbench_pipeline.py \
        --task "$t" --model "$m" --variant "$v" --n-samples $N 2>&1 \
        | grep -E "EM=|Saved|Skip|Error|error" || true
    done
  done
done
echo "=== ALL DONE ==="
