#!/usr/bin/env bash
# Run pec_hop_concise and pec_hop_short15 across all 5 tasks x 2 readers.
set +e
PY="python"
ROOT="."
SCRIPT="$ROOT/experiments/scripts/longbench_pipeline.py"
LOG="$ROOT/experiments/results/longbench/_arr_prompts.log"
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
        for variant in pec_hop_concise pec_hop_short15 ; do
            run_it "$model" "$task" "$variant" "${NS[$task]}"
        done
    done
done

echo "[$(date +%H:%M:%S)] DONE" | tee -a "$LOG"
