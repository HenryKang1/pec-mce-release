#!/bin/bash
# Comprehensive head-to-head: 3 tasks x 3 readers x 8 variants = 72 cells
# (already-cached cells skip automatically inside the pipeline)
set -e
PY="python"
TASKS="hotpotqa 2wikimqa musique"
READERS="lfm2.5-1.2b-instruct gemma-4-e4b qwen3-4b"
VARIANTS="llmlingua2 llmlingua2_extractive provence provence_extractive pec_hop_rerank pec_hop_rerank_extractive pec_hop_fewextractive pec_hop_rerank_fewextractive"

cd "$(dirname "$0")"
for M in $READERS; do
  for T in $TASKS; do
    for V in $VARIANTS; do
      echo "=== $M / $T / $V ==="
      "$PY" longbench_pipeline.py --task "$T" --model "$M" --variant "$V" --n-samples 200 2>&1 | tail -3
    done
  done
done
echo "=== ALL DONE ==="
