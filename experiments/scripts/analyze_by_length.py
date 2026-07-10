"""
Stratify EM / F1 by article length to test the hypothesis that CTKS helps
for LONG articles (where context doesn't fit raw) and hurts for SHORT ones
(where raw already fits and summarization only adds hallucination).

For each question:
  - look up the supporting articles in the context field
  - compute the MAX article length among supporting articles (proxy for difficulty)
  - bucket the question by that length

Compare within each bucket:
  - fair RAG (hotpotqa_raw_index) EM
  - CTKS (entity_notes)          EM

If the hypothesis holds, we expect:
  - short-article bucket:  fair RAG >> CTKS   (the 'hurting' regime)
  - long-article bucket:   CTKS    >= fair RAG (the 'helping' regime)

Usage:
  python analyze_by_length.py --model lfm2.5-1.2b-instruct --dataset hotpotqa
"""
import argparse
import json
import sys
from pathlib import Path
from statistics import mean

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "shared"))
from utils.config import TOPIC6_DIR, DATASETS_DIR


def load_dataset(dataset: str):
    path = DATASETS_DIR / f"{dataset}_full_validation.json"
    if not path.exists():
        path = DATASETS_DIR / f"{dataset}_validation.json"
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def supporting_article_max_length(item: dict) -> int:
    sf = item.get("supporting_facts", {})
    gold_titles = set(sf.get("title", []))
    ctx = item.get("context", {})
    title_to_sents = dict(zip(ctx.get("title", []), ctx.get("sentences", [])))
    lens = []
    for t in gold_titles:
        s = title_to_sents.get(t, [])
        lens.append(sum(len(x.split()) for x in s))
    return max(lens) if lens else 0


def run(model: str, dataset: str, buckets: list[int]):
    data = load_dataset(dataset)
    q_to_maxlen = {item["question"]: supporting_article_max_length(item) for item in data}

    rd = TOPIC6_DIR / "experiments" / "results"
    variants = {
        "RAG_fair":   rd / f"{model}_{dataset}_rag_fairRag.json",
        "RAG_wiki":   rd / f"{model}_{dataset}_rag.json",
        "CTKS":       rd / f"{model}_{dataset}_entity.json",
        "Smart_CTKS": rd / f"{model}_{dataset}_entity_smartCtks.json",
        "Extract":    rd / f"{model}_{dataset}_entity_extractCtks.json",
    }

    cumu = {name: {} for name in variants}
    for name, p in variants.items():
        if not p.exists():
            continue
        with open(p, encoding="utf-8") as f:
            d = json.load(f)
        for r in d["results"]:
            ml = q_to_maxlen.get(r["question"])
            if ml is None:
                continue
            cumu[name].setdefault("all", []).append(r["em"])
            # Bucket
            bucket = "overflow"
            for b in buckets:
                if ml <= b:
                    bucket = f"<= {b}"
                    break
            cumu[name].setdefault(bucket, []).append(r["em"])

    print(f"\nEM (%) by supporting-article MAX length")
    print(f"Model={model} Dataset={dataset}")
    print("-" * 70)
    headers = [f"<= {b}" for b in buckets] + ["overflow", "all"]
    print(f"{'Variant':<14} " + " ".join(f"{h:>9}" for h in headers))
    for name in variants:
        if not cumu[name]:
            continue
        row = []
        for h in headers:
            vals = cumu[name].get(h, [])
            if vals:
                row.append(f"{sum(vals)/len(vals)*100:>8.2f}% (n={len(vals)})")
            else:
                row.append(f"{'--':>15}")
        # Trim to fit
        print(f"{name:<14}", end=" ")
        for h in headers:
            vals = cumu[name].get(h, [])
            if vals:
                print(f"{sum(vals)/len(vals)*100:>8.2f}%", end=" ")
            else:
                print(f"{'--':>9}", end=" ")
        print()

    # n per bucket (from first variant)
    first = next((k for k, v in cumu.items() if v), None)
    if first:
        print(f"\n{'n':<14}", end=" ")
        for h in headers:
            vals = cumu[first].get(h, [])
            print(f"{len(vals):>9}", end=" ")
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="lfm2.5-1.2b-instruct")
    parser.add_argument("--dataset", default="hotpotqa")
    parser.add_argument("--buckets", type=int, nargs="+",
                        default=[60, 100, 150, 250, 500])
    args = parser.parse_args()
    run(args.model, args.dataset, args.buckets)
