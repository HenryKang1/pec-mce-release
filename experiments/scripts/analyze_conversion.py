"""
Conversion-rate analysis: joins coverage records with actual QA predictions.

For each question in the existing QA result files:
  - answer_present_in_retrieval (R): gold answer substring appears in retrieved text
  - em (Y): model got the answer right

Buckets:
  (R=1, Y=1)  Useful: retrieval had it AND model extracted it
  (R=1, Y=0)  Wasted: retrieval had it but model missed it
  (R=0, Y=1)  Lucky:  model got it despite retrieval missing it (rare; priors)
  (R=0, Y=0)  Missed: neither had it

Conversion rate = P(Y=1 | R=1) — what fraction of "retrieval showed the answer"
                                   actually converts to a correct prediction.

Intuition: higher conversion rate = the model is better at *using* whatever
evidence it was given. This is the extraction-side story for CTKS.

Usage:
  python analyze_conversion.py --model lfm2.5-1.2b-instruct --dataset hotpotqa
"""
import argparse
import json
import re
import string
import sys
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "shared"))
from utils.config import TOPIC6_DIR
from rag_pipeline import ChunkIndex


def norm(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def answer_in_text(answer: str, text: str) -> bool:
    a = norm(answer)
    t = norm(text)
    return bool(a) and a in t


def build_lookup(index: ChunkIndex, top_k: int, questions: list[str]) -> dict:
    """Precompute retrieved text per question."""
    out = {}
    for q in tqdm(questions, desc="Retrieving", leave=False):
        texts, _, _ = index.search(q, top_k=top_k)
        out[q] = " ".join(texts)
    return out


def conversion_for(result_file: Path, retrieve_lookup: dict) -> dict:
    with open(result_file, encoding="utf-8") as f:
        data = json.load(f)
    records = data.get("results", [])

    buckets = {"useful": 0, "wasted": 0, "lucky": 0, "missed": 0}
    for rec in records:
        q = rec["question"]
        retrieved = retrieve_lookup.get(q, "")
        if not retrieved:
            continue
        gt = rec.get("ground_truth", "")
        # Handle list-valued ground truth (aliases)
        gts = gt if isinstance(gt, list) else [gt]
        R = any(answer_in_text(g, retrieved) for g in gts if g)
        Y = bool(rec.get("em"))

        if R and Y:      buckets["useful"] += 1
        elif R and not Y: buckets["wasted"] += 1
        elif not R and Y: buckets["lucky"]  += 1
        else:             buckets["missed"] += 1

    total = sum(buckets.values()) or 1
    r_rate = (buckets["useful"] + buckets["wasted"]) / total
    y_rate = (buckets["useful"] + buckets["lucky"])  / total
    conv_given_r = buckets["useful"] / max(buckets["useful"] + buckets["wasted"], 1)
    return {
        "n": total,
        "retrieval_hit_rate": r_rate,       # P(R=1) — how often retrieval has answer
        "em_rate": y_rate,                  # P(Y=1) — overall EM
        "conversion_rate": conv_given_r,    # P(Y=1 | R=1)
        "buckets": buckets,
    }


def run(model: str, dataset: str, top_k: int):
    results_dir = TOPIC6_DIR / "experiments" / "results"
    cache = TOPIC6_DIR / "experiments" / "cache"

    rag_file = results_dir / f"{model}_{dataset}_rag.json"
    ctks_file = results_dir / f"{model}_{dataset}_entity.json"
    if not rag_file.exists() or not ctks_file.exists():
        print(f"[Error] Need both: {rag_file}, {ctks_file}")
        return

    # Load the question list from either (same questions)
    with open(rag_file, encoding="utf-8") as f:
        rag_data = json.load(f)
    questions = [r["question"] for r in rag_data["results"]]

    raw_dir = cache / f"{dataset}_raw_index"
    ent_dir = cache / f"{dataset}_entity_index"
    print(f"[Load] Raw index: {raw_dir}")
    raw_idx = ChunkIndex()
    raw_idx.load(raw_dir)
    print(f"[Load] Entity index: {ent_dir}")
    ent_idx = ChunkIndex()
    ent_idx.load(ent_dir)

    print(f"[Retrieve] Raw top-k for {len(questions)} questions")
    raw_lookup = build_lookup(raw_idx, top_k, questions)
    print(f"[Retrieve] Entity top-k for {len(questions)} questions")
    ent_lookup = build_lookup(ent_idx, top_k, questions)

    rag_stats = conversion_for(rag_file, raw_lookup)
    ctks_stats = conversion_for(ctks_file, ent_lookup)

    out = {
        "model": model, "dataset": dataset, "top_k": top_k,
        "rag": rag_stats, "ctks": ctks_stats,
        "delta": {
            "conversion_rate": ctks_stats["conversion_rate"] - rag_stats["conversion_rate"],
            "retrieval_hit_rate": ctks_stats["retrieval_hit_rate"] - rag_stats["retrieval_hit_rate"],
            "em_rate": ctks_stats["em_rate"] - rag_stats["em_rate"],
        },
    }

    out_file = results_dir / f"conversion_{model}_{dataset}.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*70}")
    print(f"Conversion Analysis — {model} / {dataset} (n={rag_stats['n']})")
    print(f"{'='*70}")
    print(f"{'Metric':<24} {'RAG':>12} {'CTKS':>12} {'Δ':>10}")
    for key, label in [("retrieval_hit_rate", "Retrieval has answer"),
                        ("em_rate",            "Overall EM"),
                        ("conversion_rate",    "Conv. EM | Retrieved")]:
        r, c = rag_stats[key], ctks_stats[key]
        print(f"{label:<24} {r*100:>11.2f}% {c*100:>11.2f}% {(c-r)*100:>+9.2f}%")
    print(f"\nBuckets (RAG) : {rag_stats['buckets']}")
    print(f"Buckets (CTKS): {ctks_stats['buckets']}")
    print(f"\n[Saved] {out_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="lfm2.5-1.2b-instruct")
    parser.add_argument("--dataset", default="hotpotqa")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()
    run(args.model, args.dataset, args.top_k)
