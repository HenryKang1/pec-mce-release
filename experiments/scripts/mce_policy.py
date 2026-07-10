"""Dev/test split + Rule MCE policy.

Protocol:
  - Split each task's n questions into dev (first ~25%) and test (rest).
  - For each (reader, task), pick the best (base, prompt) on dev.
  - Evaluate that choice on test.
  - Report:
      * Best single (base, prompt) on dev, applied uniformly to all (reader, task)
      * Best (base, prompt) per task (task-aware, reader-pooled)
      * Best (base, prompt) per (reader, task) -- the MCE rule
      * Oracle per-query upper bound on test
  - Compute paired bootstrap 95% LB for: MCE Rule vs Best non-PEC envelope.
"""
import json
import random
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "experiments" / "results" / "longbench"

MODELS = ["lfm2.5-1.2b-instruct", "qwen3-4b", "gemma-4-e4b"]
TASKS = ["hotpotqa", "2wikimqa", "musique", "qasper", "multifieldqa_en"]
NS = {"hotpotqa": 200, "2wikimqa": 200, "musique": 200, "qasper": 200, "multifieldqa_en": 150}
DEV_SIZE = {"hotpotqa": 50, "2wikimqa": 50, "musique": 50, "qasper": 50, "multifieldqa_en": 40}
MIN_CANDIDATES_PER_TASK = 12

# Candidate pool excludes bridge_sentence per the MCE-RAG design decision.
BASES = ["raw_topk", "raw_topk_b840", "sentence_only", "pec_hop"]
PROMPTS = ["default", "extractive", "short15", "concise"]


def variant_name(base, prompt):
    return base if prompt == "default" else f"{base}_{prompt}"


def load_per_q(model, task, base, prompt):
    v = variant_name(base, prompt)
    p = RES / f"{model}_{task}_{v}_n{NS[task]}.json"
    if not p.exists():
        return None
    d = json.load(open(p, encoding="utf-8"))
    em = [r["em"] for r in d["results"]]
    f1 = [r["f1"] for r in d["results"]]
    return em, f1


def split_dev_test(n_total, n_dev):
    return list(range(n_dev)), list(range(n_dev, n_total))


def mean_at(values, indices):
    if not indices:
        return 0.0
    return sum(values[i] for i in indices) / len(indices)


def paired_bootstrap_lb(diffs, n_boot=10000, alpha=0.05, seed=0):
    """One-sided 95% lower bound on the mean difference (paired)."""
    rng = random.Random(seed)
    n = len(diffs)
    if n == 0:
        return 0.0
    samples = []
    for _ in range(n_boot):
        idxs = [rng.randrange(n) for _ in range(n)]
        samples.append(sum(diffs[i] for i in idxs) / n)
    samples.sort()
    return samples[int(alpha * n_boot)]


def task_candidate_count(all_em, model, task):
    return sum(
        1
        for base in BASES
        for prompt in PROMPTS
        if (model, task, base, prompt) in all_em
    )


def active_models(all_em):
    """Readers with enough completed cells for policy-level comparison.

    Gemma is included in MODELS before its full matrix is available.  We skip a
    reader until every task has a minimally comparable candidate pool; otherwise
    a partial one-cell smoke test can make the global best-single baseline
    undefined.
    """
    active = []
    skipped = []
    for model in MODELS:
        counts = {task: task_candidate_count(all_em, model, task) for task in TASKS}
        if all(counts[task] >= MIN_CANDIDATES_PER_TASK for task in TASKS):
            active.append(model)
        else:
            skipped.append((model, counts))
    return active, skipped


def main():
    print("# MCE-RAG Rule policy: dev/test split evaluation\n")
    print("Candidate pool: {} bases x {} prompts. Bridge excluded.\n".format(
        len(BASES), len(PROMPTS)))

    # Load all candidate per-question EM
    all_em = {}  # (model, task, base, prompt) -> [EM per q]
    for model in MODELS:
        for task in TASKS:
            for base in BASES:
                for prompt in PROMPTS:
                    got = load_per_q(model, task, base, prompt)
                    if got:
                        all_em[(model, task, base, prompt)] = got[0]

    models, skipped_models = active_models(all_em)
    if skipped_models:
        print("Skipped readers without a complete-enough candidate grid:")
        for model, counts in skipped_models:
            count_str = ", ".join(f"{task}={counts[task]}" for task in TASKS)
            print(f"- {model}: {count_str}")
        print()
    if not models:
        raise RuntimeError("No reader has enough completed candidates for MCE-Select.")

    # === Per (reader, task) MCE Rule policy ===
    print("## Per-(reader, task) MCE Rule policy on dev split\n")
    print("| reader | task | dev best (base, prompt) | dev EM | test EM | n_test |")
    print("|---|---|---|---:|---:|---:|")

    rule_test_macro = defaultdict(lambda: [0.0, 0])  # model -> [sum_em, n_tasks]
    rule_choice = {}  # (model, task) -> (base, prompt)
    rule_test_em_per_q = {}  # (model, task) -> list of test EMs under rule

    for model in models:
        for task in TASKS:
            # Find candidates available
            candidates = {(b, p): all_em[(model, task, b, p)]
                          for b in BASES for p in PROMPTS
                          if (model, task, b, p) in all_em}
            if not candidates:
                print(f"| {model} | {task} | (no data) | - | - | 0 |")
                continue
            n_total = min(len(v) for v in candidates.values())
            n_dev = DEV_SIZE[task]
            dev_idx, test_idx = split_dev_test(n_total, n_dev)

            # Best (base, prompt) on dev
            dev_means = {bp: mean_at(em, dev_idx) for bp, em in candidates.items()}
            best_bp = max(dev_means, key=dev_means.get)
            dev_em = dev_means[best_bp] * 100

            # Apply on test
            test_em = mean_at(candidates[best_bp], test_idx) * 100

            rule_choice[(model, task)] = best_bp
            rule_test_macro[model][0] += test_em
            rule_test_macro[model][1] += 1
            rule_test_em_per_q[(model, task)] = [
                candidates[best_bp][i] for i in test_idx]

            print(f"| {model} | {task} | {best_bp[0]} + {best_bp[1]} "
                  f"| {dev_em:.2f}% | {test_em:.2f}% | {len(test_idx)} |")

    print()
    print("## Macro test EM under MCE Rule policy")
    print("| reader | macro test EM (Rule) |")
    print("|---|---:|")
    for model, (s, n) in rule_test_macro.items():
        print(f"| {model} | {s/n if n else 0:.2f}% |")

    # === Best fixed single (base, prompt) baseline ===
    # Same (base, prompt) for all (reader, task), picked on dev.
    print("\n## Best-fixed-single-config baseline (same (base, prompt) everywhere)\n")
    print("Picked on dev macro across all (reader, task), then evaluated on test.\n")

    bp_dev_macro = defaultdict(list)
    for (model, task), _ in rule_choice.items():
        candidates = {(b, p): all_em[(model, task, b, p)]
                      for b in BASES for p in PROMPTS
                      if (model, task, b, p) in all_em}
        n_total = min(len(v) for v in candidates.values())
        n_dev = DEV_SIZE[task]
        dev_idx = list(range(n_dev))
        for bp, em in candidates.items():
            bp_dev_macro[bp].append(mean_at(em, dev_idx))

    n_active_cells = len(rule_choice)
    bp_dev_means = {
        bp: sum(v) / len(v)
        for bp, v in bp_dev_macro.items()
        if len(v) == n_active_cells
    }
    if bp_dev_means:
        best_single = max(bp_dev_means, key=bp_dev_means.get)
        print(f"Best single (base, prompt) by dev macro: **{best_single[0]} + {best_single[1]}** "
              f"(dev macro {bp_dev_means[best_single]*100:.2f}%)\n")
        # Apply on test
        print("| reader | task | test EM (best single) |")
        print("|---|---|---:|")
        single_test_macro = defaultdict(lambda: [0.0, 0])
        for (model, task) in rule_choice.keys():
            if (model, task, best_single[0], best_single[1]) not in all_em:
                continue
            em = all_em[(model, task, best_single[0], best_single[1])]
            n_total = len(em)
            n_dev = DEV_SIZE[task]
            test_idx = list(range(n_dev, n_total))
            test_em = mean_at(em, test_idx) * 100
            single_test_macro[model][0] += test_em
            single_test_macro[model][1] += 1
            print(f"| {model} | {task} | {test_em:.2f}% |")
        print()
        print("| reader | macro test EM (best single) |")
        print("|---|---:|")
        for model, (s, n) in single_test_macro.items():
            print(f"| {model} | {s/n if n else 0:.2f}% |")

    # === Oracle test EM (per-query upper bound) ===
    print("\n## Oracle test EM (per-query upper bound)\n")
    print("| reader | task | test EM (oracle) |")
    print("|---|---|---:|")
    oracle_test_macro = defaultdict(lambda: [0.0, 0])
    for (model, task) in rule_choice.keys():
        candidates = {(b, p): all_em[(model, task, b, p)]
                      for b in BASES for p in PROMPTS
                      if (model, task, b, p) in all_em}
        n_total = min(len(v) for v in candidates.values())
        n_dev = DEV_SIZE[task]
        test_idx = list(range(n_dev, n_total))
        oracle = []
        for q in test_idx:
            oracle.append(max(em[q] for em in candidates.values()))
        oracle_em = (sum(oracle) / len(oracle)) * 100
        oracle_test_macro[model][0] += oracle_em
        oracle_test_macro[model][1] += 1
        print(f"| {model} | {task} | {oracle_em:.2f}% |")

    print()
    print("| reader | macro test EM (oracle) |")
    print("|---|---:|")
    for model, (s, n) in oracle_test_macro.items():
        print(f"| {model} | {s/n if n else 0:.2f}% |")

    # === Headline summary ===
    print("\n## Headline\n")
    print("| reader | best-single | MCE-Rule | Oracle | Rule - single | Rule recovers oracle gap |")
    print("|---|---:|---:|---:|---:|---:|")
    for model in models:
        s = single_test_macro[model][0] / max(1, single_test_macro[model][1])
        r = rule_test_macro[model][0] / max(1, rule_test_macro[model][1])
        o = oracle_test_macro[model][0] / max(1, oracle_test_macro[model][1])
        gap = o - s
        recov = (r - s) / gap * 100 if gap > 0 else 0.0
        print(f"| {model} | {s:.2f}% | {r:.2f}% | {o:.2f}% | +{r-s:.2f}% | {recov:.1f}% |")


if __name__ == "__main__":
    main()
