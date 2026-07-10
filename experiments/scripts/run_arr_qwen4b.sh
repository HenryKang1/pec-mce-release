#!/usr/bin/env bash
# Add Qwen3-4B as a third reader: full 4 base x 4 prompt x 5 task grid.
# Hypothesis: Qwen3-1.7B's PEC underperformance is largely due to its weak
# instruction-following at 1.7B; a stronger 4B reader should restore the
# representation advantage we observe on LFM2.5-1.2B.
set +e
PY="python"
ROOT="."
SCRIPT="$ROOT/experiments/scripts/longbench_pipeline.py"
LOG="$ROOT/experiments/results/longbench/_arr_qwen4b.log"
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

# Priority order: multi-hop tasks first (the primary claim), then single-doc
# Within each task, run PEC-Hop first (so we see headline number early),
# then baselines for fair comparison.
for task in hotpotqa 2wikimqa musique qasper multifieldqa_en ; do
    for variant in \
        pec_hop pec_hop_extractive pec_hop_short15 pec_hop_concise \
        raw_topk raw_topk_extractive raw_topk_short15 raw_topk_concise \
        raw_topk_b840 raw_topk_b840_extractive raw_topk_b840_short15 raw_topk_b840_concise \
        sentence_only sentence_only_extractive sentence_only_short15 sentence_only_concise ; do
        run_it "qwen3-4b" "$task" "$variant" "${NS[$task]}"
    done
done

echo "[$(date +%H:%M:%S)] DONE" | tee -a "$LOG"
