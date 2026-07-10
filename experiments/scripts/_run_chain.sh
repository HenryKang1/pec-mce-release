#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
PY="python"

# Qwen3-0.6B HotpotQA entity + compress full
"$PY" run_baseline.py --model qwen3-0.6b --dataset hotpotqa --mode entity   --max-samples 7405
"$PY" run_baseline.py --model qwen3-0.6b --dataset hotpotqa --mode compress --max-samples 7405

# Qwen3-0.6B 2WikiMQA full
"$PY" run_baseline.py --model qwen3-0.6b --dataset 2wikimqa --mode rag    --max-samples 12576
"$PY" run_baseline.py --model qwen3-0.6b --dataset 2wikimqa --mode entity --max-samples 12576

# Qwen3-1.7B 2WikiMQA full
"$PY" run_baseline.py --model qwen3-1.7b --dataset 2wikimqa --mode rag    --max-samples 12576
"$PY" run_baseline.py --model qwen3-1.7b --dataset 2wikimqa --mode entity --max-samples 12576

echo "CHAIN_DONE"
