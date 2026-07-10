"""Head-to-head comparison: PEC-Hop vs external compression baselines.

Aggregates HotpotQA x LFM2.5-1.2B n=200 results for the ARR head-to-head
table. Computes paired bootstrap for EM differences vs PEC-Hop+extractive
(the MCE-Compass selection for that cell) and pretty-prints a LaTeX-ready
table plus a JSON dump.

Run after the four llmlingua2/provence variants have been executed:
    python compare_external_compressors.py
"""
import json
import random
from pathlib import Path

random.seed(0)

RESULTS = Path(__file__).resolve().parents[1] / "results" / "longbench"
MODEL = "lfm2.5-1.2b-instruct"
TASKS = ["hotpotqa", "2wikimqa"]

# Order matters: it's how the table prints.
VARIANTS = [
    ("Raw RAG (default)",              "raw_topk"),
    ("Raw RAG (extractive)",           "raw_topk_extractive"),
    ("Raw RAG$_{840}$ (default)",      "raw_topk_b840"),
    ("Raw RAG$_{840}$ (extractive)",   "raw_topk_b840_extractive"),
    ("Sentence-only (default)",        "sentence_only"),
    ("Sentence-only (extractive)",     "sentence_only_extractive"),
    ("LLMLingua-2 (default)",          "llmlingua2"),
    ("LLMLingua-2 (extractive)",       "llmlingua2_extractive"),
    ("Provence (default)",             "provence"),
    ("Provence (extractive)",          "provence_extractive"),
    ("\\textbf{PEC-Hop} (default)",    "pec_hop"),
    ("\\textbf{PEC-Hop} (extractive)", "pec_hop_extractive"),
    ("PEC-Hop+rerank (default)",       "pec_hop_rerank"),
    ("PEC-Hop+rerank (extractive)",    "pec_hop_rerank_extractive"),
    ("PEC-Hop (few-shot extractive)",  "pec_hop_fewextractive"),
    ("\\textbf{PEC-Hop+rerank} (few-shot)", "pec_hop_rerank_fewextractive"),
]

REFERENCE = "pec_hop_extractive"  # MCE-Compass pick for HotpotQA x LFM


def load(task: str, variant: str) -> dict:
    f = RESULTS / f"{MODEL}_{task}_{variant}_n200.json"
    if not f.exists():
        return None
    return json.load(open(f, encoding="utf-8"))


def paired_bootstrap_em(records_a: list[dict], records_b: list[dict],
                         B: int = 10_000) -> tuple[float, float, float]:
    """Return (mean delta, 2.5 percentile, 97.5 percentile) of EM(a) - EM(b)."""
    n = len(records_a)
    assert n == len(records_b)
    diffs = [records_a[i]["em"] - records_b[i]["em"] for i in range(n)]
    mean = sum(diffs) / n * 100
    samples = []
    for _ in range(B):
        idx = [random.randrange(n) for _ in range(n)]
        s = sum(diffs[i] for i in idx) / n * 100
        samples.append(s)
    samples.sort()
    return mean, samples[int(0.025 * B)], samples[int(0.975 * B)]


def run_task(task: str):
    rows = []
    ref = load(task, REFERENCE)
    assert ref is not None, f"Missing reference: {REFERENCE}/{task}"
    ref_records = ref["results"]

    for label, vname in VARIANTS:
        d = load(task, vname)
        if d is None:
            rows.append({"task": task, "label": label, "variant": vname, "missing": True})
            continue
        m = d["metrics"]
        if vname == REFERENCE:
            mean_delta, lo, hi = 0.0, 0.0, 0.0
        else:
            mean_delta, lo, hi = paired_bootstrap_em(d["results"], ref_records)
        rows.append({
            "task": task,
            "label": label,
            "variant": vname,
            "em": m["em"],
            "f1": m["f1"],
            "loose": m["loose"],
            "latency_ms": m["avg_latency_ms"],
            "ctx_tokens": m["avg_context_tokens"],
            "delta_em_vs_ref": mean_delta,
            "ci_lo": lo,
            "ci_hi": hi,
        })
    return rows


def main():
    all_rows = []
    for task in TASKS:
        print(f"\n=== {task.upper()} (n=200) ===")
        rows = run_task(task)
        all_rows.extend(rows)
        header = f"{'Variant':40s} {'EM':>6s} {'F1':>6s} {'Loose':>6s} {'Lat(ms)':>8s} {'Tokens':>7s}  {'ΔEM vs ref':>10s} {'95% CI':>16s}"
        print(header)
        print("-" * len(header))
        for r in rows:
            if r.get("missing"):
                print(f"{r['label']:40s}   MISSING")
                continue
            ci = f"[{r['ci_lo']:+5.2f},{r['ci_hi']:+5.2f}]" if r["variant"] != REFERENCE else "  (reference)   "
            print(f"{r['label']:40s} {r['em']:6.2f} {r['f1']:6.2f} {r['loose']:6.2f} {r['latency_ms']:8.1f} {r['ctx_tokens']:7.1f}  {r['delta_em_vs_ref']:+10.2f} {ci:>16s}")

    out_path = Path(__file__).resolve().parents[1] / "results" / "external_compressors_head_to_head.json"
    json.dump(all_rows, open(out_path, "w", encoding="utf-8"), indent=2)
    print(f"\n[Saved] {out_path}")


if __name__ == "__main__":
    main()
