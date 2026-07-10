#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
PY="python"
LOG="../results/_resume_chain.log"

echo "[resume] waiting for in-flight compress (PID 23804) to finish..." | tee -a "$LOG"
while tasklist //FI "PID eq 23804" 2>/dev/null | grep -q "python.exe"; do
  sleep 30
done
echo "[resume] compress finished at $(date)" | tee -a "$LOG"

run() {
  echo "[resume] >>> $* at $(date)" | tee -a "$LOG"
  "$PY" run_baseline.py "$@" 2>&1 | tee -a "$LOG"
  echo "[resume] <<< $* done at $(date)" | tee -a "$LOG"
}

run --model qwen3-0.6b --dataset 2wikimqa --mode rag    --max-samples 12576
run --model qwen3-0.6b --dataset 2wikimqa --mode entity --max-samples 12576
run --model qwen3-1.7b --dataset 2wikimqa --mode rag    --max-samples 12576
run --model qwen3-1.7b --dataset 2wikimqa --mode entity --max-samples 12576

echo "CHAIN_DONE at $(date)" | tee -a "$LOG"
