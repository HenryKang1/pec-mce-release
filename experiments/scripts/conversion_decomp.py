"""Conversion decomposition: why does MCE-RAG work?

For each (reader, task, base, prompt), compute:
  - EM rate
  - loose rate (answer-in-prediction; gold string contained anywhere)
  - empty rate (prediction is empty or just whitespace)
  - verbose-failure rate = (loose - EM) / loose -- model knew the answer but
    surrounded it with a sentence
  - conversion rate = EM / loose -- given that the model produced an answer
    containing the gold span, did it produce a clean exact match

The hypothesis under MCE-RAG:
  Raw RAG -> high loose, low conversion (model knows but writes verbose).
  PEC-Hop + extractive -> medium loose, high conversion.
  sentence-only + extractive -> medium loose, high conversion (cheap variant).

If true, the gain of MCE-RAG over Raw RAG is *predominantly conversion*,
not retrieval recall.
"""
import json
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "experiments" / "results" / "longbench"

MODELS = ["lfm2.5-1.2b-instruct", "qwen3-4b"]  # focus on positive readers
TASKS = ["hotpotqa", "2wikimqa", "musique", "qasper", "multifieldqa_en"]
NS = {"hotpotqa": 200, "2wikimqa": 200, "musique": 200, "qasper": 200, "multifieldqa_en": 150}

# Three families to compare:
INTERFACES = [
    ("raw_topk", "default"),       # Raw RAG default
    ("raw_topk", "concise"),       # Raw + concise (Qwen-4B optimal raw)
    ("sentence_only", "extractive"),
    ("sentence_only", "short15"),
    ("pec_hop", "extractive"),
    ("pec_hop", "concise"),
]


def variant_name(base, prompt):
    return base if prompt == "default" else f"{base}_{prompt}"


def load_results(model, task, base, prompt):
    v = variant_name(base, prompt)
    p = RES / f"{model}_{task}_{v}_n{NS[task]}.json"
    if not p.exists():
        return None
    return json.load(open(p, encoding="utf-8"))["results"]


def compute_metrics(results):
    n = len(results)
    if n == 0:
        return None
    em = sum(r["em"] for r in results) / n
    loose = sum(r.get("loose", 0) for r in results) / n
    empty = sum(1 for r in results if not r.get("prediction", "").strip()) / n
    n_loose = sum(1 for r in results if r.get("loose", 0) == 1)
    n_em = sum(1 for r in results if r["em"] == 1)
    verbose_fail = (n_loose - n_em) / n_loose if n_loose > 0 else 0.0
    conversion = n_em / n_loose if n_loose > 0 else 0.0
    return {
        "em": em * 100,
        "loose": loose * 100,
        "empty": empty * 100,
        "verbose_fail_pct": verbose_fail * 100,
        "conversion": conversion * 100,
        "n": n,
    }


def main():
    print("# Conversion decomposition\n")
    print("Per (reader, task, interface, prompt):")
    print("  EM = exact match")
    print("  loose = answer-in-prediction (gold contained anywhere)")
    print("  empty = prediction was empty/whitespace")
    print("  verbose_fail = fraction of loose-correct predictions that missed EM (model knew but verbose)")
    print("  conversion = EM / loose (given answer was produced, did it match exactly)\n")

    for model in MODELS:
        print(f"## {model}\n")
        print("| task | base | prompt | EM | loose | empty | verbose_fail | conversion |")
        print("|---|---|---|---:|---:|---:|---:|---:|")
        # Aggregate across tasks for summary
        agg = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0, 0.0, 0])
        for task in TASKS:
            for base, prompt in INTERFACES:
                results = load_results(model, task, base, prompt)
                if not results:
                    continue
                m = compute_metrics(results)
                if not m:
                    continue
                print(f"| {task} | {base} | {prompt} | "
                      f"{m['em']:.2f}% | {m['loose']:.2f}% | {m['empty']:.2f}% | "
                      f"{m['verbose_fail_pct']:.2f}% | {m['conversion']:.2f}% |")
                a = agg[(base, prompt)]
                a[0] += m["em"]; a[1] += m["loose"]; a[2] += m["empty"]
                a[3] += m["verbose_fail_pct"]; a[4] += m["conversion"]; a[5] += 1
        print()
        print(f"### {model} -- macro across tasks\n")
        print("| base | prompt | EM | loose | empty | verbose_fail | conversion |")
        print("|---|---|---:|---:|---:|---:|---:|")
        for (base, prompt), (em, loose, empty, vf, conv, n) in agg.items():
            if n == 0:
                continue
            print(f"| {base} | {prompt} | "
                  f"{em/n:.2f}% | {loose/n:.2f}% | {empty/n:.2f}% | "
                  f"{vf/n:.2f}% | {conv/n:.2f}% |")
        print()


if __name__ == "__main__":
    main()
