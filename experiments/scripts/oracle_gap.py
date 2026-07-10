"""Compute Oracle EIC gap from existing JSON files.

For each (model, task), align per-query EM across all (base, prompt)
combinations and compute:
  - best_fixed_em       = max_{(b,p)} mean_q EM
  - oracle_em           = mean_q max_{(b,p)} EM_q
  - oracle_gap          = oracle_em - best_fixed_em
Aggregate macro across tasks (per model and overall).

This is the go/no-go decision for the EIC-RAG pivot:
  gap >= 5.0 EM macro -> EIC-RAG plan is viable
  gap in 3-5         -> need a strong new primitive (bridge_sentence)
  gap < 3            -> need a different framing entirely
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "experiments" / "results" / "longbench"

MODELS = ["lfm2.5-1.2b-instruct", "qwen3-4b", "gemma-4-e4b"]
TASKS = ["hotpotqa", "2wikimqa", "musique", "qasper", "multifieldqa_en"]
NS_BY_TASK = {
    "hotpotqa": 200, "2wikimqa": 200, "musique": 200,
    "qasper": 200, "multifieldqa_en": 150,
}
BASES = ["raw_topk", "raw_topk_b840", "sentence_only", "pec_hop"]
PROMPTS = ["default", "extractive", "short15", "concise"]


def variant_name(base: str, prompt: str) -> str:
    if prompt == "default":
        return base
    return f"{base}_{prompt}"


def load_per_query_em(model: str, task: str, base: str, prompt: str):
    n = NS_BY_TASK[task]
    v = variant_name(base, prompt)
    p = RES / f"{model}_{task}_{v}_n{n}.json"
    if not p.exists():
        return None
    with open(p, encoding="utf-8") as f:
        d = json.load(f)
    em_list = [r["em"] for r in d["results"]]
    f1_list = [r["f1"] for r in d["results"]]
    return em_list, f1_list


def compute_for_model(model: str):
    print(f"\n=== {model} ===")
    print(f"{'task':<18} {'best_fixed':>12} {'best_var':>32} {'oracle_em':>10} {'gap':>8} {'n':>5}")
    print("-" * 95)

    macro_best = 0.0
    macro_oracle = 0.0
    macro_gap = 0.0
    n_tasks = 0
    available_per_task = {}

    for task in TASKS:
        per_query = {}  # (base, prompt) -> list of EM scores
        for base in BASES:
            for prompt in PROMPTS:
                got = load_per_query_em(model, task, base, prompt)
                if got is None:
                    continue
                em_list, _ = got
                per_query[(base, prompt)] = em_list

        if not per_query:
            print(f"{task:<18} (no data)")
            continue

        # Pin n to the smallest available list (some runs may differ)
        n = min(len(v) for v in per_query.values())
        if n < 5:
            continue

        # Best fixed: max over (b,p) of mean EM
        means = {(b, p): sum(em[:n]) / n for (b, p), em in per_query.items()}
        best_var, best_mean = max(means.items(), key=lambda kv: kv[1])

        # Oracle: per-query max
        oracle_em_per_q = []
        for q in range(n):
            em_q = max(em[q] for em in per_query.values())
            oracle_em_per_q.append(em_q)
        oracle_em = sum(oracle_em_per_q) / n

        gap = oracle_em - best_mean
        macro_best += best_mean
        macro_oracle += oracle_em
        macro_gap += gap
        n_tasks += 1

        best_var_str = f"{best_var[0]}+{best_var[1]}"
        print(f"{task:<18} {best_mean*100:>11.2f}% {best_var_str:>32} "
              f"{oracle_em*100:>9.2f}% {gap*100:>7.2f}% {n:>5}")
        available_per_task[task] = len(per_query)

    if n_tasks:
        macro_best /= n_tasks
        macro_oracle /= n_tasks
        macro_gap /= n_tasks
        print("-" * 95)
        print(f"{'MACRO':<18} {macro_best*100:>11.2f}% {' ':>32} "
              f"{macro_oracle*100:>9.2f}% {macro_gap*100:>7.2f}%")
        print(f"variants per task (avg): {sum(available_per_task.values())/len(available_per_task):.1f}")

    return macro_best, macro_oracle, macro_gap, n_tasks


def main():
    print("Oracle EIC gap analysis")
    print(f"  {len(BASES)} bases x {len(PROMPTS)} prompts = {len(BASES)*len(PROMPTS)} candidates per (model, task)")

    overall = {}
    for model in MODELS:
        b, o, g, nt = compute_for_model(model)
        overall[model] = (b, o, g, nt)

    print("\n\n=== SUMMARY ===")
    print(f"{'reader':<25} {'best_fixed_em':>15} {'oracle_em':>12} {'gap':>10}")
    print("-" * 65)
    for model, (b, o, g, _) in overall.items():
        print(f"{model:<25} {b*100:>14.2f}% {o*100:>11.2f}% {g*100:>9.2f}%")

    print("\nDecision:")
    avg_gap = sum(g for (_, _, g, _) in overall.values()) / len(overall)
    print(f"  Average oracle gap across readers: {avg_gap*100:.2f}% EM")
    if avg_gap >= 0.05:
        print("  >= 5% EM: EIC-RAG plan is viable. Proceed with bridge_sentence + rule router.")
    elif avg_gap >= 0.03:
        print("  3-5% EM: Marginal. Need a strong new primitive (bridge_sentence) to push it.")
    else:
        print("  < 3% EM: Pivot. EIC-RAG by selection alone won't carry the paper.")


if __name__ == "__main__":
    main()
