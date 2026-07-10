#!/usr/bin/env bash
# Run all ARR-defense variants on LFM and Qwen.
# Corruption stress (shuffle/random) is LFM-only on multi-hop tasks.
set +e
PY="python"
ROOT="."
SCRIPT="$ROOT/experiments/scripts/longbench_pipeline.py"
LOG="$ROOT/experiments/results/longbench/_arr_defense.log"
: > "$LOG"

run_it () {
    local model="$1"
    local task="$2"
    local variant="$3"
    local n="$4"
    echo "[$(date +%H:%M:%S)] $model / $task / $variant / n=$n" | tee -a "$LOG"
    "$PY" "$SCRIPT" --task "$task" --model "$model" --variant "$variant" --n-samples "$n" >> "$LOG" 2>&1
}

# Task -> n_samples (matches existing files)
declare -A NS
NS[hotpotqa]=200
NS[2wikimqa]=200
NS[musique]=200
NS[qasper]=200
NS[multifieldqa_en]=150

# Main variants on both models, all 5 tasks
for model in lfm2.5-1.2b-instruct qwen3-1.7b ; do
    for task in hotpotqa 2wikimqa musique qasper multifieldqa_en ; do
        for variant in sentence_only pec_router pec_hop_extractive pec_hop_relations ; do
            run_it "$model" "$task" "$variant" "${NS[$task]}"
        done
    done
done

# Corruption stress: LFM only, on 3 multi-hop tasks
for task in hotpotqa 2wikimqa musique ; do
    for variant in pec_hop_shuffle_ptr pec_hop_random_anchor ; do
        run_it "lfm2.5-1.2b-instruct" "$task" "$variant" "${NS[$task]}"
    done
done

echo "[$(date +%H:%M:%S)] DONE" | tee -a "$LOG"
