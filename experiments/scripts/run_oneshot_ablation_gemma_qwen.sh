#!/bin/bash
# 1-shot ablation on Gemma + Qwen3 (sequential to share GPU with LFM job)
set -e
PY="python"
TASKS="hotpotqa 2wikimqa musique"
READERS="gemma-4-e4b qwen3-4b"
VARIANTS="raw_topk_oneextractive sentence_only_oneextractive llmlingua2_oneextractive provence_oneextractive pec_hop_rerank_oneextractive"

cd "$(dirname "$0")"
for M in $READERS; do
  for T in $TASKS; do
    for V in $VARIANTS; do
      echo "=== $M / $T / $V ==="
      "$PY" longbench_pipeline.py --task "$T" --model "$M" --variant "$V" --n-samples 200 2>&1 | tail -3
    done
  done
done
echo "=== DONE ==="
