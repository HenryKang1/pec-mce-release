"""Aggregate comprehensive head-to-head: 3 readers x 3 tasks x N variants.

Produces:
  - LaTeX-ready table (long form)
  - Macro EM by method across all reader-task cells
  - Paired bootstrap deltas vs PEC-Hop+rerank+fewshot (the proposed final)
"""
import json
import random
from pathlib import Path

random.seed(0)

RESULTS = Path(__file__).resolve().parents[1] / "results" / "longbench"
READERS = ["lfm2.5-1.2b-instruct", "gemma-4-e4b", "qwen3-4b"]
TASKS = ["hotpotqa", "2wikimqa", "musique"]
VARIANTS = [
    ("Raw RAG",                         "raw_topk_extractive"),
    ("Sentence-only",                   "sentence_only_extractive"),
    ("LLMLingua-2",                     "llmlingua2_extractive"),
    ("Provence",                        "provence_extractive"),
    ("PEC-Hop (ext)",                   "pec_hop_extractive"),
    ("PEC-Hop+rerank",                  "pec_hop_rerank_extractive"),
    ("PEC-Hop+fewshot",                 "pec_hop_fewextractive"),
    ("PEC-Hop+rerank+fewshot",          "pec_hop_rerank_fewextractive"),
]
REFERENCE = "pec_hop_rerank_fewextractive"


def load(reader: str, task: str, variant: str) -> dict:
    f = RESULTS / f"{reader}_{task}_{variant}_n200.json"
    if not f.exists():
        return None
    return json.load(open(f, encoding="utf-8"))


def paired_bootstrap(records_a, records_b, metric: str = "em", B: int = 10_000):
    n = min(len(records_a), len(records_b))
    diffs = [records_a[i][metric] - records_b[i][metric] for i in range(n)]
    mean = sum(diffs) / n * 100
    samples = []
    for _ in range(B):
        idx = [random.randrange(n) for _ in range(n)]
        s = sum(diffs[i] for i in idx) / n * 100
        samples.append(s)
    samples.sort()
    return mean, samples[int(0.025 * B)], samples[int(0.975 * B)]


def main():
    # Per-cell table
    print(f"\n{'='*120}")
    print(f"{'Reader':<26s} {'Task':<12s} {'Variant':<28s} {'EM':>6s} {'F1':>6s} {'Lat(ms)':>9s} {'Ctx':>6s}")
    print("="*120)
    cells = []
    for r in READERS:
        for t in TASKS:
            for label, v in VARIANTS:
                d = load(r, t, v)
                if d is None:
                    continue
                m = d["metrics"]
                cells.append({
                    "reader": r, "task": t, "variant": v, "label": label,
                    "em": m["em"], "f1": m["f1"],
                    "lat": m["avg_latency_ms"], "ctx": m["avg_context_tokens"],
                })
                print(f"{r:<26s} {t:<12s} {label:<28s} {m['em']:>6.2f} {m['f1']:>6.2f} {m['avg_latency_ms']:>9.1f} {m['avg_context_tokens']:>6.0f}")

    # Macro EM by method, separately per reader
    print(f"\n{'='*100}")
    print("MACRO EM by method (averaged over 3 tasks per reader)")
    print("="*100)
    print(f"{'Variant':<28s}" + "".join(f"{r:>20s}" for r in READERS) + f"{'overall':>10s}")
    print("-"*100)
    method_overall = {}
    for label, v in VARIANTS:
        row = {}
        for r in READERS:
            ems = []
            for t in TASKS:
                d = load(r, t, v)
                if d is not None:
                    ems.append(d["metrics"]["em"])
            row[r] = sum(ems) / len(ems) if ems else None
        all_ems = [v for v in row.values() if v is not None]
        overall = sum(all_ems) / len(all_ems) if all_ems else None
        method_overall[v] = overall
        cells_str = "".join(
            f"{row[r]:>20.2f}" if row[r] is not None else f"{'-':>20s}" for r in READERS
        )
        overall_str = f"{overall:>10.2f}" if overall is not None else f"{'-':>10s}"
        print(f"{label:<28s}{cells_str}{overall_str}")

    # Paired bootstrap vs REFERENCE on each cell
    print(f"\n{'='*100}")
    print(f"Paired bootstrap (EM, %, paired by question index) vs {REFERENCE}")
    print("="*100)
    print(f"{'Reader':<26s} {'Task':<12s} {'Variant':<28s} {'ΔEM':>8s} {'95% CI':>20s}")
    print("-"*100)
    for r in READERS:
        for t in TASKS:
            ref = load(r, t, REFERENCE)
            if ref is None:
                continue
            for label, v in VARIANTS:
                if v == REFERENCE:
                    continue
                d = load(r, t, v)
                if d is None:
                    continue
                mean, lo, hi = paired_bootstrap(d["results"], ref["results"])
                print(f"{r:<26s} {t:<12s} {label:<28s} {mean:>+8.2f}    [{lo:>+6.2f},{hi:>+6.2f}]")

    out = Path(__file__).resolve().parents[1] / "results" / "h2h_comprehensive.json"
    json.dump(cells, open(out, "w", encoding="utf-8"), indent=2)
    print(f"\n[Saved] {out}")


if __name__ == "__main__":
    main()
