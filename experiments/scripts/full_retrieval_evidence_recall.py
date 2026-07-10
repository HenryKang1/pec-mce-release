"""Full-retrieval evidence recall diagnostic for EMNLP-readiness.

This is a retrieval-only experiment: no reader LLM is called.  It tests whether
minimal extractive evidence remains useful outside LongBench's pre-bundled
context setting by querying full validation indices.

Metrics:
  - answer_recall: gold answer string appears in retrieved evidence.
  - support_title_recall: fraction of gold supporting titles surfaced.
  - support_sentence_recall: fraction of gold supporting sentences surfaced.
  - evidence_words: average retrieved context words.
  - answer_density_per_1k: answer_recall per 1k retrieved words.

Usage:
  conda run -n research6 python experiments/scripts/full_retrieval_evidence_recall.py --dataset hotpotqa --n 7405
  conda run -n research6 python experiments/scripts/full_retrieval_evidence_recall.py --dataset 2wikimqa --n 12576
"""
import argparse
import json
import re
import string
import sys
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "shared"))
from utils.config import DATASETS_DIR, EMBEDDING_MODEL, MODELS_DIR, TOPIC6_DIR


STOPWORDS = set(
    """a an the is was are were be been being am of on in at by for to from with
    as and or but if then else when while do does did have has had this that
    these those it its he she they them their there here which who whom what"""
    .split()
)


def normalize(s: str) -> str:
    s = (s or "").lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = s.translate(str.maketrans("", "", string.punctuation))
    return " ".join(s.split())


def tokenize(text: str) -> list[str]:
    text = normalize(text)
    return [t for t in text.split() if t and t not in STOPWORDS]


def load_dataset(dataset: str, n: int) -> list[dict]:
    if dataset == "hotpotqa":
        path = DATASETS_DIR / "hotpotqa_full_validation.json"
    elif dataset == "2wikimqa":
        path = DATASETS_DIR / "2wikimqa_validation.json"
    else:
        raise ValueError(f"unsupported dataset: {dataset}")
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data[: min(n, len(data))]


def get_gold(item: dict) -> tuple[list[str], list[str]]:
    sf = item.get("supporting_facts", {})
    titles = list(sf.get("title", []))
    sent_ids = list(sf.get("sent_id", []))
    ctx = item.get("context", {})
    title_to_sents = {
        t: sents for t, sents in zip(ctx.get("title", []), ctx.get("sentences", []))
    }
    gold_sents = []
    for title, sent_id in zip(titles, sent_ids):
        sents = title_to_sents.get(title, [])
        if 0 <= sent_id < len(sents):
            gold_sents.append(sents[sent_id])
    return titles, gold_sents


def load_index(cache_name: str) -> tuple[faiss.Index, list[str]]:
    path = TOPIC6_DIR / "experiments" / "cache" / cache_name
    index = faiss.read_index(str(path / "index.faiss"))
    with open(path / "chunks.json", encoding="utf-8") as f:
        chunks = json.load(f)
    return index, chunks


def measure(texts: list[str], answer: str, gold_titles: list[str], gold_sents: list[str]) -> dict:
    evidence = "\n".join(texts)
    evidence_norm = normalize(evidence)
    evidence_words = max(len(tokenize(evidence)), 1)

    answer_norm = normalize(answer)
    answer_recall = 1.0 if answer_norm and answer_norm in evidence_norm else 0.0

    if gold_titles:
        support_title_recall = sum(
            1 for title in gold_titles if normalize(title) in evidence_norm
        ) / len(gold_titles)
        all_titles_hit = 1.0 if support_title_recall == 1.0 else 0.0
    else:
        support_title_recall = 0.0
        all_titles_hit = 0.0

    if gold_sents:
        support_sentence_recall = sum(
            1 for sent in gold_sents if normalize(sent) and normalize(sent) in evidence_norm
        ) / len(gold_sents)
        all_sents_hit = 1.0 if support_sentence_recall == 1.0 else 0.0
    else:
        support_sentence_recall = 0.0
        all_sents_hit = 0.0

    return {
        "answer_recall": answer_recall,
        "support_title_recall": support_title_recall,
        "all_support_titles": all_titles_hit,
        "support_sentence_recall": support_sentence_recall,
        "all_support_sentences": all_sents_hit,
        "evidence_words": evidence_words,
        "answer_density_per_1k": 1000.0 * answer_recall / evidence_words,
        "title_density_per_1k": 1000.0 * support_title_recall / evidence_words,
    }


def aggregate(rows: list[dict]) -> dict:
    keys = [
        "answer_recall",
        "support_title_recall",
        "all_support_titles",
        "support_sentence_recall",
        "all_support_sentences",
        "evidence_words",
        "answer_density_per_1k",
        "title_density_per_1k",
    ]
    return {k: sum(r[k] for r in rows) / len(rows) for k in keys} | {"n": len(rows)}


def markdown_report(summary: dict, out_json: Path) -> str:
    rows = summary["variants"]
    dataset_label = {
        "hotpotqa_full_validation": "HotpotQA full validation",
        "2wikimqa_validation": "2WikiMQA validation",
    }.get(summary["dataset"], summary["dataset"])
    lines = [
        "# Full-retrieval evidence recall",
        "",
        f"Retrieval-only diagnostic on {dataset_label}. No reader LLM is called.",
        f"Dataset size: **{summary['n']}**. Top-k: **{summary['top_k']}**.",
        "",
        "| Variant | Ans recall | Support-title recall | All titles | Support-sentence recall | Words | Ans density/1k |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, row in rows.items():
        lines.append(
            f"| {row['label']} | {row['answer_recall']*100:.2f}% | "
            f"{row['support_title_recall']*100:.2f}% | "
            f"{row['all_support_titles']*100:.2f}% | "
            f"{row['support_sentence_recall']*100:.2f}% | "
            f"{row['evidence_words']:.0f} | "
            f"{row['answer_density_per_1k']:.3f} |"
        )

    raw = rows.get("raw_article")
    ext = rows.get("extractive_note")
    if raw and ext:
        lines += [
            "",
            "## Key comparison",
            "",
            f"Extractive minimal evidence uses **{ext['evidence_words']/raw['evidence_words']:.2f}x** "
            "the words of raw article retrieval while retaining "
            f"**{ext['answer_recall']/raw['answer_recall']*100:.1f}%** of its answer recall "
            "and increasing answer density by "
            f"**{ext['answer_density_per_1k']/raw['answer_density_per_1k']:.2f}x**.",
            "",
            f"JSON: `{out_json}`",
        ]
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["hotpotqa", "2wikimqa"], default="hotpotqa")
    ap.add_argument("--n", type=int, default=7405)
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=128)
    args = ap.parse_args()

    items = load_dataset(args.dataset, args.n)
    questions = [item["question"] for item in items]

    cache_dir = MODELS_DIR / "embeddings"
    encoder = SentenceTransformer(
        EMBEDDING_MODEL,
        cache_folder=str(cache_dir) if cache_dir.exists() else None,
    )
    query_emb = encoder.encode(
        questions,
        batch_size=args.batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
    ).astype(np.float32)

    if args.dataset == "hotpotqa":
        variants = {
            "raw_article": ("Raw article", "hotpotqa_raw_index"),
            "chunk100": ("Raw chunk-100", "hotpotqa_chunk_index"),
            "chunk40": ("Raw chunk-40", "hotpotqa_chunk_index_c40"),
            "extractive_note": ("Extractive minimal note", "hotpotqa_extractive_index"),
            "smart_note": ("Smart minimal note", "hotpotqa_smart_index"),
            "entity_note": ("LLM entity note", "hotpotqa_entity_index"),
        }
        dataset_name = "hotpotqa_full_validation"
        out_stem = "full_retrieval_evidence_recall"
    else:
        variants = {
            "raw_article": ("Raw article", "2wikimqa_raw_index"),
            "extractive_note": ("Extractive minimal note", "2wikimqa_extractive_index"),
            "entity_note": ("LLM entity note", "2wikimqa_entity_index"),
        }
        dataset_name = "2wikimqa_validation"
        out_stem = "full_retrieval_evidence_recall_2wikimqa"

    all_rows = {}
    for key, (label, cache_name) in variants.items():
        index, chunks = load_index(cache_name)
        scores, idxs = index.search(query_emb, args.top_k)
        rows = []
        for i, item in enumerate(tqdm(items, desc=label)):
            texts = [chunks[j] for j in idxs[i] if 0 <= j < len(chunks)]
            titles, sents = get_gold(item)
            rows.append(measure(texts, item.get("answer", ""), titles, sents))
        agg = aggregate(rows)
        agg["label"] = label
        agg["cache"] = cache_name
        all_rows[key] = agg

    summary = {
        "dataset": dataset_name,
        "n": len(items),
        "top_k": args.top_k,
        "variants": all_rows,
    }

    out_json = TOPIC6_DIR / "experiments" / "results" / f"{out_stem}.json"
    out_md = TOPIC6_DIR / "experiments" / "results" / f"{out_stem.upper()}.md"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    with open(out_md, "w", encoding="utf-8") as f:
        f.write(markdown_report(summary, out_json))

    print(markdown_report(summary, out_json))
    print(f"[Saved] {out_json}")
    print(f"[Saved] {out_md}")


if __name__ == "__main__":
    main()
