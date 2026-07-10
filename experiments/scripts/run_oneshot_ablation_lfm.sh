#!/bin/bash
# 1-shot ablation on LFM: compare oneextractive vs fewextractive latency/EM
set -e
PY="python"
TASKS="hotpotqa 2wikimqa musique"
READER="lfm2.5-1.2b-instruct"
VARIANTS="raw_topk_oneextractive sentence_only_oneextractive llmlingua2_oneextractive provence_oneextractive pec_hop_rerank_oneextractive"

cd "$(dirname "$0")"
for T in $TASKS; do
  for V in $VARIANTS; do
    echo "=== $READER / $T / $V ==="
    "$PY" longbench_pipeline.py --task "$T" --model "$READER" --variant "$V" --n-samples 200 2>&1 | tail -3
  done
done
echo "=== DONE ==="
