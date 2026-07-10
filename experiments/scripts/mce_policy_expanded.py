"""MCE-COMPASS with EXPANDED candidate pool.

Adds fewextractive prompt (2-shot demo) and the strong baselines
(llmlingua2, provence) plus pec_hop_rerank to the candidate pool. This
answers: under a fair, prompt-inclusive pool, does the rule policy still
pick PEC-Hop, or does Raw RAG + fewextractive dominate on verbose readers?

Run from anywhere -- writes results to stdout and to
experiments/results/MCE_RULE_EXPANDED.md.
"""
import json
import sys
from pathlib import Path
from collections import defaultdict, Counter

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "experiments" / "results" / "longbench"

MODELS = ["lfm2.5-1.2b-instruct", "qwen3-4b", "gemma-4-e4b"]
TASKS = ["hotpotqa", "2wikimqa", "musique", "qasper", "multifieldqa_en"]
NS = {"hotpotqa": 200, "2wikimqa": 200, "musique": 200, "qasper": 200, "multifieldqa_en": 150}
DEV_SIZE = {"hotpotqa": 50, "2wikimqa": 50, "musique": 50, "qasper": 50, "multifieldqa_en": 40}

# Expanded pool
BASES = [
    "raw_topk", "raw_topk_b840",
    "sentence_only",
    "llmlingua2", "provence",
    "pec_hop", "pec_hop_rerank",
]
PROMPTS = ["default", "extractive", "short15", "concise", "fewextractive"]


def variant_name(base, prompt):
    return base if prompt == "default" else f"{base}_{prompt}"


def load_per_q(model, task, base, prompt):
    v = variant_name(base, prompt)
    p = RES / f"{model}_{task}_{v}_n{NS[task]}.json"
    if not p.exists():
        return None
    d = json.load(open(p, encoding="utf-8"))
    return [r["em"] for r in d["results"]]


def mean_at(values, indices):
    return sum(values[i] for i in indices) / len(indices) if indices else 0.0


def main():
    out_lines = []
    def emit(s=""):
        out_lines.append(s)
        print(s)

    emit("# MCE-COMPASS rule policy with EXPANDED candidate pool\n")
    emit(f"Bases ({len(BASES)}): {', '.join(BASES)}")
    emit(f"Prompts ({len(PROMPTS)}): {', '.join(PROMPTS)}\n")

    # Load
    all_em = {}
    for model in MODELS:
        for task in TASKS:
            for base in BASES:
                for prompt in PROMPTS:
                    em = load_per_q(model, task, base, prompt)
                    if em is not None:
                        all_em[(model, task, base, prompt)] = em

    # Per-(model, task) candidate count and rule choice
    emit("## Per-(reader, task) rule choice on dev (first 50/40 questions)\n")
    emit("| reader | task | n_candidates | dev winner | dev EM | test EM |")
    emit("|---|---|---:|---|---:|---:|")

    rule_test_macro = defaultdict(lambda: [0.0, 0])
    rule_choices = defaultdict(list)  # model -> [(base, prompt), ...]
    base_only_choices = defaultdict(Counter)
    prompt_only_choices = defaultdict(Counter)
    rule_test_em_per_q = {}

    for model in MODELS:
        for task in TASKS:
            cands = {(b, p): all_em[(model, task, b, p)]
                     for b in BASES for p in PROMPTS
                     if (model, task, b, p) in all_em}
            if not cands:
                continue
            n_total = min(len(v) for v in cands.values())
            n_dev = DEV_SIZE[task]
            dev_idx = list(range(n_dev))
            test_idx = list(range(n_dev, n_total))
            dev_means = {bp: mean_at(em, dev_idx) for bp, em in cands.items()}
            best_bp = max(dev_means, key=dev_means.get)
            dev_em = dev_means[best_bp] * 100
            test_em = mean_at(cands[best_bp], test_idx) * 100
            rule_test_macro[model][0] += test_em
            rule_test_macro[model][1] += 1
            rule_choices[model].append(best_bp)
            base_only_choices[model][best_bp[0]] += 1
            prompt_only_choices[model][best_bp[1]] += 1
            rule_test_em_per_q[(model, task)] = [cands[best_bp][i] for i in test_idx]
            emit(f"| {model} | {task} | {len(cands)} | "
                 f"{best_bp[0]} + {best_bp[1]} | {dev_em:.2f} | {test_em:.2f} |")

    emit("")
    emit("## Selection frequency per reader (over 5 tasks)\n")
    emit("| reader | bases picked | prompts picked |")
    emit("|---|---|---|")
    for model in MODELS:
        bs = ", ".join(f"{b}({c})" for b, c in base_only_choices[model].most_common())
        ps = ", ".join(f"{p}({c})" for p, c in prompt_only_choices[model].most_common())
        emit(f"| {model} | {bs} | {ps} |")
    emit("")

    # === Best-fixed-single baseline within the EXPANDED pool ===
    emit("## Best-fixed-single config (same (base, prompt) across all reader×task)\n")
    bp_dev_macro = defaultdict(list)
    seen_cells = []
    for model in MODELS:
        for task in TASKS:
            cands = {(b, p): all_em[(model, task, b, p)]
                     for b in BASES for p in PROMPTS
                     if (model, task, b, p) in all_em}
            if not cands:
                continue
            seen_cells.append((model, task))
            n_dev = DEV_SIZE[task]
            dev_idx = list(range(n_dev))
            for bp, em in cands.items():
                bp_dev_macro[bp].append(mean_at(em, dev_idx))
    n_cells = len(seen_cells)
    bp_complete = {bp: v for bp, v in bp_dev_macro.items() if len(v) == n_cells}
    if bp_complete:
        bp_ranked = sorted(bp_complete.items(),
                           key=lambda kv: -sum(kv[1]) / len(kv[1]))
        emit("Top 10 single configs by dev macro EM (over all complete cells):\n")
        emit("| rank | (base, prompt) | dev macro EM |")
        emit("|---:|---|---:|")
        for i, (bp, vs) in enumerate(bp_ranked[:10], 1):
            emit(f"| {i} | {bp[0]} + {bp[1]} | {100*sum(vs)/len(vs):.2f} |")
        emit("")
        best_single = bp_ranked[0][0]
        emit(f"**Best single = {best_single[0]} + {best_single[1]}**\n")
        emit("| reader | task | test EM (best single) |")
        emit("|---|---|---:|")
        single_test_macro = defaultdict(lambda: [0.0, 0])
        for (model, task) in seen_cells:
            if (model, task, *best_single) not in all_em:
                continue
            em = all_em[(model, task, *best_single)]
            test_idx = list(range(DEV_SIZE[task], len(em)))
            v = mean_at(em, test_idx) * 100
            single_test_macro[model][0] += v
            single_test_macro[model][1] += 1
            emit(f"| {model} | {task} | {v:.2f} |")
        emit("")
        emit("| reader | macro test EM (best single) | macro test EM (rule) | Δ |")
        emit("|---|---:|---:|---:|")
        for model in MODELS:
            s, ns = single_test_macro[model]
            r, nr = rule_test_macro[model]
            ss = s/ns if ns else 0
            rr = r/nr if nr else 0
            emit(f"| {model} | {ss:.2f} | {rr:.2f} | +{rr-ss:.2f} |")
    emit("")

    # === Bases picked by readers under matched prompt fewextractive only ===
    emit("## What wins under fewextractive (matched-prompt sub-pool)\n")
    emit("| reader | task | best base (fewextractive only) | dev EM |")
    emit("|---|---|---|---:|")
    for model in MODELS:
        for task in TASKS:
            cands = {b: all_em[(model, task, b, "fewextractive")]
                     for b in BASES
                     if (model, task, b, "fewextractive") in all_em}
            if not cands:
                continue
            n_dev = DEV_SIZE[task]
            dev_idx = list(range(n_dev))
            means = {b: mean_at(em, dev_idx) for b, em in cands.items()}
            best_b = max(means, key=means.get)
            emit(f"| {model} | {task} | {best_b} | {100*means[best_b]:.2f} |")
    emit("")

    out_path = ROOT / "experiments" / "results" / "MCE_RULE_EXPANDED.md"
    out_path.write_text("\n".join(out_lines), encoding="utf-8")
    emit(f"\n[Saved] {out_path}")


if __name__ == "__main__":
    main()
