#!/usr/bin/env bash
# Qwen3-4B grid v2: skip raw_topk_b840 variants (pathological slowdown observed
# at ~20min/question, root cause likely tokenizer/cache interaction). LFM data
# already provides the budget-matched comparison.
set +e
PY="python"
ROOT="."
SCRIPT="$ROOT/experiments/scripts/longbench_pipeline.py"
LOG="$ROOT/experiments/results/longbench/_arr_qwen4b_v2.log"
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

# Multi-hop tasks first (primary claim), then single-doc (boundary analysis).
# 12 variants per task: pec_hop x 4 prompts, raw_topk x 4 prompts, sentence_only x 4 prompts.
for task in hotpotqa 2wikimqa musique qasper multifieldqa_en ; do
    for variant in \
        pec_hop pec_hop_extractive pec_hop_short15 pec_hop_concise \
        raw_topk raw_topk_extractive raw_topk_short15 raw_topk_concise \
        sentence_only sentence_only_extractive sentence_only_short15 sentence_only_concise ; do
        run_it "qwen3-4b" "$task" "$variant" "${NS[$task]}"
    done
done

echo "[$(date +%H:%M:%S)] DONE" | tee -a "$LOG"
