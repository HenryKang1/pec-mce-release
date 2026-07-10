#!/usr/bin/env bash
# Phase 8: hydration window sweep + dynamic fallback.
# - hydration sweep (pec_hop_w0, pec_hop_w2) on LFM × 5 tasks
# - dynamic fallback on LFM × 5 tasks AND Qwen3-1.7B × 3 multi-hop tasks
set -e
PY="python"
cd "$(dirname "$0")/.."

declare -A TASK_N
TASK_N[hotpotqa]=200
TASK_N[2wikimqa]=200
TASK_N[musique]=200
TASK_N[multifieldqa_en]=150
TASK_N[qasper]=200

# LFM hydration sweep
for v in pec_hop_w0 pec_hop_w2; do
  for t in hotpotqa 2wikimqa musique multifieldqa_en qasper; do
    n=${TASK_N[$t]}
    echo "=== [$(date +%H:%M:%S)] LFM / $t / $v / n=$n ==="
    "$PY" scripts/longbench_pipeline.py \
      --task "$t" --model lfm2.5-1.2b-instruct --variant "$v" --n-samples $n 2>&1 \
      | grep -E "EM=|Saved|Skip|Error|error" || true
  done
done

# LFM dynamic fallback
for t in hotpotqa 2wikimqa musique multifieldqa_en qasper; do
  n=${TASK_N[$t]}
  echo "=== [$(date +%H:%M:%S)] LFM / $t / pec_hop_dynamic / n=$n ==="
  "$PY" scripts/longbench_pipeline.py \
    --task "$t" --model lfm2.5-1.2b-instruct --variant pec_hop_dynamic --n-samples $n 2>&1 \
    | grep -E "EM=|Saved|Skip|Error|error" || true
done

# Qwen3 dynamic fallback (multi-hop only — that's where capacity threshold appears)
for t in hotpotqa 2wikimqa musique; do
  n=${TASK_N[$t]}
  echo "=== [$(date +%H:%M:%S)] Qwen3 / $t / pec_hop_dynamic / n=$n ==="
  "$PY" scripts/longbench_pipeline.py \
    --task "$t" --model qwen3-1.7b --variant pec_hop_dynamic --n-samples $n 2>&1 \
    | grep -E "EM=|Saved|Skip|Error|error" || true
done

echo "=== ALL DONE ==="
