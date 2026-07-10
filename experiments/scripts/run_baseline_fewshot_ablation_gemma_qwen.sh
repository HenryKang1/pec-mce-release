#!/bin/bash
# Reviewer-defense ablation, secondary readers: Gemma and Qwen3.
# Run sequentially (not parallel) to limit concurrent GPU pressure with the
# LFM job that runs in another shell.
set -e
PY="python"
TASKS="hotpotqa 2wikimqa musique"
READERS="gemma-4-e4b qwen3-4b"
VARIANTS="raw_topk_fewextractive sentence_only_fewextractive llmlingua2_fewextractive provence_fewextractive"

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
