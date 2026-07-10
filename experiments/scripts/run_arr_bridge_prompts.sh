#!/usr/bin/env bash
# Cross-prompt for pec_bridge / pec_bridge_k3.
# Hypothesis: on Qwen3, PEC-Bridge variants (raw passages + cards) with
# concise prompt may beat Raw RAG + concise. On LFM, may match or beat
# pec_hop on some tasks.
set +e
PY="python"
ROOT="."
SCRIPT="$ROOT/experiments/scripts/longbench_pipeline.py"
LOG="$ROOT/experiments/results/longbench/_arr_bridge_prompts.log"
: > "$LOG"

run_it () {
    local model="$1"; local task="$2"; local variant="$3"; local n="$4"
    echo "[$(date +%H:%M:%S)] $model / $task / $variant / n=$n" | tee -a "$LOG"
    "$PY" "$SCRIPT" --task "$task" --model "$model" --variant "$variant" --n-samples "$n" >> "$LOG" 2>&1
}

declare -A NS
NS[hotpotqa]=200
NS[2wikimqa]=200
NS[musique]=200
NS[qasper]=200
NS[multifieldqa_en]=150

# Priority order: Qwen first (paper-saver), then LFM
for model in qwen3-1.7b lfm2.5-1.2b-instruct ; do
    for task in hotpotqa 2wikimqa musique qasper multifieldqa_en ; do
        for variant in \
            pec_bridge_concise pec_bridge_extractive pec_bridge_short15 \
            pec_bridge_k3_concise pec_bridge_k3_extractive pec_bridge_k3_short15 ; do
            run_it "$model" "$task" "$variant" "${NS[$task]}"
        done
    done
done

echo "[$(date +%H:%M:%S)] DONE" | tee -a "$LOG"
