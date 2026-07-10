"""
Supporting-Fact Coverage Analysis (no LLM needed).

For each HotpotQA question we have gold supporting facts (title, sent_id).
We measure how well RAG vs CTKS retrieval preserves these answer-bearing
sentences, *per retrieved token*.

Metrics per question:
  - retrieved_tokens      : #tokens in top-k retrieved content
  - title_recall          : fraction of gold titles represented in top-k
  - answer_recall         : 1 if the gold answer substring is in retrieved text
  - entity_recall         : fraction of capitalized-entity tokens from gold sentences
                            that also appear in retrieved text (more robust than
                            raw token overlap — focuses on facts, not filler words)
  - token_recall          : fraction of gold (content) tokens present in retrieved
                            (HIGH for RAG by construction: raw passages ARE gold sents)
  - answer_density        = answer_recall / retrieved_tokens  (x1000 for readability)
                            ==> how many tokens to have the answer present
  - entity_density        = entity_recall / retrieved_tokens  (x1000)
                            ==> answer-bearing entities per token — the key metric

Aggregated across N sampled questions.

Usage:
  python analyze_coverage.py --n 1000 --top-k 5
  python analyze_coverage.py --n 500 --top-k 5 --dataset 2wikimqa
"""
import argparse
import json
import re
import string
import sys
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "shared"))
from utils.config import DATASETS_DIR, TOPIC6_DIR
from rag_pipeline import ChunkIndex


STOPWORDS = set("""a an the is was are were be been being am of on in at by for to from with
                   as and or but if then else when while do does did have has had this that
                   these those it its he she they them their there here which who whom what""".split())


def tokenize(text: str) -> list[str]:
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = text.split()
    return [t for t in tokens if t and t not in STOPWORDS]


def extract_entities(text: str) -> set[str]:
    """Heuristic: capitalized word sequences, plus standalone numbers.

    Avoids the need for an NER model. Returns lowercased entity tokens.
    """
    # Multi-word capitalized phrases
    phrases = re.findall(r"\b(?:[A-Z][a-zA-Z0-9]*(?:\s+[A-Z][a-zA-Z0-9]*)*)\b", text)
    # Numbers (dates, counts)
    numbers = re.findall(r"\b\d{2,4}\b", text)
    tokens = set()
    for p in phrases:
        for w in p.split():
            w = w.lower()
            if len(w) > 1 and w not in STOPWORDS:
                tokens.add(w)
    tokens.update(numbers)
    return tokens


def get_gold_sentences(item: dict) -> tuple[list[str], list[str]]:
    """Return (gold_titles, gold_sentences)."""
    sf = item.get("supporting_facts", {})
    titles = sf.get("title", [])
    sent_ids = sf.get("sent_id", [])
    ctx = item.get("context", {})
    ctx_titles = ctx.get("title", [])
    ctx_sents = ctx.get("sentences", [])
    title_to_sents = {t: s for t, s in zip(ctx_titles, ctx_sents)}
    gold_sents = []
    for t, sid in zip(titles, sent_ids):
        sents = title_to_sents.get(t, [])
        if 0 <= sid < len(sents):
            gold_sents.append(sents[sid])
    return list(set(titles)), gold_sents


def measure(retrieved_text: str, gold_titles: list[str], gold_sents: list[str],
            gold_answer: str) -> dict:
    gold_all = " ".join(gold_sents)
    gold_tok = tokenize(gold_all)
    gold_set = set(gold_tok)
    gold_entities = extract_entities(gold_all) | {t.lower() for title in gold_titles
                                                   for t in re.findall(r"\w+", title)
                                                   if t.lower() not in STOPWORDS}

    ret_tok = tokenize(retrieved_text)
    ret_set = set(ret_tok)
    ret_lower = retrieved_text.lower()
    ret_ent_tokens = set(ret_lower.split())

    if not gold_tok:
        return None

    # Title recall: fraction of gold titles that appear as substring
    titles_hit = sum(1 for t in gold_titles if t.lower() in ret_lower)
    title_recall = titles_hit / len(gold_titles) if gold_titles else 0.0

    # Answer recall: gold answer as substring (case-insensitive, punctuation-tolerant)
    ans_clean = re.sub(r"[^\w\s]", " ", (gold_answer or "")).lower().strip()
    ret_clean = re.sub(r"[^\w\s]", " ", ret_lower)
    answer_recall = 1.0 if (ans_clean and ans_clean in ret_clean) else 0.0

    # Entity recall: gold entities present in retrieved text
    ent_hit = sum(1 for e in gold_entities if e in ret_ent_tokens)
    entity_recall = ent_hit / len(gold_entities) if gold_entities else 0.0

    # Token recall / precision / f1 on *types* (lexical; favors RAG by construction)
    inter = gold_set & ret_set
    token_recall = len(inter) / len(gold_set) if gold_set else 0.0
    token_precision = len(inter) / len(ret_set) if ret_set else 0.0
    token_f1 = (
        2 * token_recall * token_precision / (token_recall + token_precision)
        if (token_recall + token_precision) > 0 else 0.0
    )

    nt = max(len(ret_tok), 1)
    return {
        "gold_tokens": len(gold_tok),
        "retrieved_tokens": len(ret_tok),
        "title_recall": title_recall,
        "answer_recall": answer_recall,
        "entity_recall": entity_recall,
        "token_recall": token_recall,
        "token_precision": token_precision,
        "token_f1": token_f1,
        "answer_density_per_1k": 1000 * answer_recall / nt,
        "entity_density_per_1k": 1000 * entity_recall / nt,
    }


def aggregate(records: list[dict]) -> dict:
    keys = ["gold_tokens", "retrieved_tokens", "title_recall",
            "answer_recall", "entity_recall",
            "token_recall", "token_precision", "token_f1",
            "answer_density_per_1k", "entity_density_per_1k"]
    n = len(records)
    return {k: sum(r[k] for r in records) / n for k in keys} | {"n": n}


def run(dataset: str, n: int, top_k: int, seed: int):
    import random
    random.seed(seed)

    ds_path = DATASETS_DIR / f"{dataset}_full_validation.json"
    if not ds_path.exists():
        ds_path = DATASETS_DIR / f"{dataset}_validation.json"
    with open(ds_path, encoding="utf-8") as f:
        data = json.load(f)

    # Sample
    if len(data) > n:
        data = random.sample(data, n)

    # Load both indices
    raw_dir = TOPIC6_DIR / "experiments" / "cache" / f"{dataset}_raw_index"
    ent_dir = TOPIC6_DIR / "experiments" / "cache" / f"{dataset}_entity_index"

    print(f"[Load] Raw index: {raw_dir}")
    raw_idx = ChunkIndex()
    raw_idx.load(raw_dir)

    print(f"[Load] Entity index: {ent_dir}")
    ent_idx = ChunkIndex()
    ent_idx.load(ent_dir)

    raw_recs, ent_recs = [], []
    for item in tqdm(data, desc="Coverage"):
        gold_titles, gold_sents = get_gold_sentences(item)
        if not gold_sents:
            continue
        q = item["question"]
        raw_texts, _, _ = raw_idx.search(q, top_k=top_k)
        ent_texts, _, _ = ent_idx.search(q, top_k=top_k)

        gold_answer = item.get("answer", "")
        r_raw = measure(" ".join(raw_texts), gold_titles, gold_sents, gold_answer)
        r_ent = measure(" ".join(ent_texts), gold_titles, gold_sents, gold_answer)
        if r_raw and r_ent:
            raw_recs.append(r_raw)
            ent_recs.append(r_ent)

    summary = {
        "dataset": dataset,
        "n_evaluated": len(raw_recs),
        "top_k": top_k,
        "raw": aggregate(raw_recs),
        "entity": aggregate(ent_recs),
    }
    # Derived
    r, e = summary["raw"], summary["entity"]
    summary["delta"] = {
        "title_recall": e["title_recall"] - r["title_recall"],
        "answer_recall": e["answer_recall"] - r["answer_recall"],
        "entity_recall": e["entity_recall"] - r["entity_recall"],
        "token_recall": e["token_recall"] - r["token_recall"],
        "retrieved_tokens_ratio": e["retrieved_tokens"] / r["retrieved_tokens"] if r["retrieved_tokens"] > 0 else 0,
        "answer_density_ratio": e["answer_density_per_1k"] / r["answer_density_per_1k"] if r["answer_density_per_1k"] > 0 else 0,
        "entity_density_ratio": e["entity_density_per_1k"] / r["entity_density_per_1k"] if r["entity_density_per_1k"] > 0 else 0,
    }

    out = TOPIC6_DIR / "experiments" / "results" / f"coverage_{dataset}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"Coverage Analysis — {dataset} (n={summary['n_evaluated']}, top-k={top_k})")
    print(f"{'='*60}")
    print(f"{'Metric':<24} {'RAG (raw)':>12} {'CTKS (entity)':>14} {'Δ':>10}")
    for k in ["retrieved_tokens", "title_recall",
              "answer_recall", "entity_recall",
              "token_recall", "token_f1",
              "answer_density_per_1k", "entity_density_per_1k"]:
        rv, ev = summary["raw"][k], summary["entity"][k]
        print(f"{k:<24} {rv:>12.4f} {ev:>14.4f} {ev-rv:>+10.4f}")
    print(f"\nRetrieved tokens ratio (CTKS/RAG): {summary['delta']['retrieved_tokens_ratio']:.2f}x")
    print(f"Answer-density multiple:  {summary['delta']['answer_density_ratio']:.2f}x")
    print(f"Entity-density multiple:  {summary['delta']['entity_density_ratio']:.2f}x")
    print(f"\n[Saved] {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="hotpotqa", choices=["hotpotqa", "2wikimqa"])
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run(args.dataset, args.n, args.top_k, args.seed)
