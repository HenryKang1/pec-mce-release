#!/bin/bash
# Reviewer-defense ablation: apply 2-shot extractive prompt to all baselines
# on LFM2.5 so PEC-Hop's fewshot lift is comparable to baseline lift.
set -e
PY="python"
TASKS="hotpotqa 2wikimqa musique"
READER="lfm2.5-1.2b-instruct"
VARIANTS="raw_topk_fewextractive sentence_only_fewextractive llmlingua2_fewextractive provence_fewextractive"

cd "$(dirname "$0")"
for T in $TASKS; do
  for V in $VARIANTS; do
    echo "=== $READER / $T / $V ==="
    "$PY" longbench_pipeline.py --task "$T" --model "$READER" --variant "$V" --n-samples 200 2>&1 | tail -3
  done
done
echo "=== DONE ==="
