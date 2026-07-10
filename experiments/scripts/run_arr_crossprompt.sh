#!/usr/bin/env bash
# Cross-prompt baselines: apply the same decoding prompts to non-PEC variants
# so we can disentangle prompt-effect from representation-effect.
set +e
PY="python"
ROOT="."
SCRIPT="$ROOT/experiments/scripts/longbench_pipeline.py"
LOG="$ROOT/experiments/results/longbench/_arr_crossprompt.log"
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

for model in lfm2.5-1.2b-instruct qwen3-1.7b ; do
    for task in hotpotqa 2wikimqa musique qasper multifieldqa_en ; do
        for variant in \
            raw_topk_extractive raw_topk_short15 raw_topk_concise \
            raw_topk_b840_extractive raw_topk_b840_short15 raw_topk_b840_concise \
            sentence_only_extractive sentence_only_short15 sentence_only_concise ; do
            run_it "$model" "$task" "$variant" "${NS[$task]}"
        done
    done
done

echo "[$(date +%H:%M:%S)] DONE" | tee -a "$LOG"
