"""Retrieval-only coverage diagnostic for PEC variants.

Re-runs the retrieval/prompt-construction stage of longbench_pipeline.py
without calling the LLM, and reports per-variant:
  - gold_in_ctx: fraction of questions where any normalized gold answer
                 string appears in the compiled prompt context
  - n_titles: average number of distinct titles surfaced in selected_titles
  - ctx_tokens: average compiled context length (whitespace tokens)
  - gold_in_ctx | answer_short: same as gold_in_ctx but conditioned on the
                                 first gold answer being <= 5 tokens (the
                                 cleanly-extractable subset)

Used to defend novelty: shows where each PEC variant retains vs loses
answer-bearing evidence relative to Raw RAG.

Usage:
  python coverage_diagnostic.py --variants raw_topk pec_bridge pec_hop \
      --task hotpotqa --model lfm2.5-1.2b-instruct --n 200
"""
import argparse
import json
import re
import string
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "shared"))
from utils.config import DATASETS_DIR, TOPIC6_DIR

# Reuse helper logic from the pipeline file, but only the parts that don't
# touch the LLM.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from longbench_pipeline import (
    split_passages, split_sentences, extract_anchors,
    build_pec_cards, build_fact_only_cards, hydrate_cards,
    evidence_is_weak, retrieve_texts,
)
from rag_pipeline import ChunkIndex


def normalize(s: str) -> str:
    s = s.lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = s.translate(str.maketrans("", "", string.punctuation))
    return " ".join(s.split())


def gold_in_text(answers: list[str], text: str) -> bool:
    txt_n = normalize(text)
    for a in answers:
        if not a:
            continue
        a_n = normalize(a)
        if a_n and a_n in txt_n:
            return True
    return False


def build_ctx(variant: str, retriever: ChunkIndex, question: str,
              context: str, top_k: int = 5) -> tuple[str, list[str]]:
    """Return (ctx_str, selected_titles). Mirrors longbench_pipeline.run_task
    for the retrieval-side variants we care about, without LLM calls."""
    if variant == "raw_trunc":
        return context, []

    passages = split_passages(context)
    if not passages:
        return context, []

    if variant == "raw_topk":
        chunks = [f"{t}: {' '.join(b.split()[:200])}" for t, b in passages]
        retriever.build_from_chunks(chunks, batch_size=128)
        texts, _, _ = retriever.search(question, top_k=top_k)
        ctx = "\n\n".join(f"[Document {i+1}]: {c}" for i, c in enumerate(texts))
        # extract titles from each chunk
        titles = []
        for t in texts:
            head = t.split(":", 1)[0]
            titles.append(head)
        return ctx, titles

    # PEC-family variants share the card-building stage.
    if variant == "pec_hop_fact_only":
        cards, card_meta = build_fact_only_cards(passages)
    else:
        cards, card_meta = build_pec_cards(passages)
    if not cards:
        return context, []
    retrieved_cards, card_indices, card_scores = retrieve_texts(
        retriever, question, cards, top_k=top_k
    )
    selected_meta = [card_meta[i] for i in card_indices]

    if variant in ("pec_bridge", "pec_bridge_k3"):
        n_raw = 2 if variant == "pec_bridge" else 3
        chunks = [f"{t}: {' '.join(b.split()[:200])}" for t, b in passages]
        raw_texts, raw_indices, _ = retrieve_texts(
            retriever, question, chunks, top_k=n_raw
        )
        raw_title_set = {passages[i][0] for i in raw_indices}
        bridge = [m for m in selected_meta if m["title"] not in raw_title_set]
        hyd = hydrate_cards(passages, bridge[:max(1, top_k - n_raw)], window=1)
        parts = [f"[Document {i+1} (RAW)]: {r}" for i, r in enumerate(raw_texts)]
        offset = len(parts)
        for i, h in enumerate(hyd):
            parts.append(f"[Document {offset+i+1} (HYD)]: {h}")
        titles = [passages[i][0] for i in raw_indices] + [m["title"] for m in bridge[:max(1, top_k - n_raw)]]
        return "\n\n".join(parts), titles

    if variant in ("pec_hop", "pec_hop_no_anchor",
                    "pec_hop_fact_only", "pec_hop_no_hydration"):
        use_anchor = variant in ("pec_hop", "pec_hop_no_hydration")
        use_hyd = variant in ("pec_hop", "pec_hop_no_anchor")
        if use_anchor:
            seed_anchors = []
            for m in selected_meta[:3]:
                for a in m.get("anchors", [])[:4]:
                    seed_anchors.append(a)
                seed_anchors.append(m["title"])
            seen, dedup = set(), []
            for a in seed_anchors:
                if not a: continue
                k = a.lower()
                if k in seen: continue
                seen.add(k); dedup.append(a)
            step2 = (question + " " + " ".join(dedup[:8])).strip()
        else:
            step2 = question
        expanded, _, _ = retriever.search(step2, top_k=top_k * 2)
        exp_meta = []
        seen_idx = set(card_indices)
        for r in expanded:
            for i, c in enumerate(cards):
                if i in seen_idx: continue
                if c == r:
                    exp_meta.append(card_meta[i])
                    seen_idx.add(i); break
            if len(exp_meta) >= top_k: break
        half = max(2, top_k - len(exp_meta))
        final = (selected_meta[:half] + exp_meta)[:top_k + 2]
        if use_hyd:
            hyd = hydrate_cards(passages, final, window=1)
            ctx = "\n\n".join(f"[Document {i+1}]: {c}" for i, c in enumerate(hyd))
        else:
            if variant == "pec_hop_fact_only":
                chunks = [m["sentence"] for m in final]
            else:
                # find cards back from meta
                final_idx = []
                for m in final:
                    for i, mm in enumerate(card_meta):
                        if mm is m:
                            final_idx.append(i); break
                chunks = [cards[i] for i in final_idx]
            ctx = "\n\n".join(f"[Document {i+1}]: {c}" for i, c in enumerate(chunks))
        titles = [m["title"] for m in final]
        return ctx, titles

    raise ValueError(f"Unsupported variant for diagnostic: {variant}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="+",
                    default=["raw_topk", "pec_bridge", "pec_hop"])
    ap.add_argument("--tasks", nargs="+",
                    default=["hotpotqa", "2wikimqa", "musique",
                             "multifieldqa_en", "qasper"])
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--top-k", type=int, default=5)
    args = ap.parse_args()

    print(f"\n=== Coverage diagnostic ===")
    print(f"{'task':<18} {'variant':<22} {'n':>4} {'gold_in':>8} {'ctx_w':>8} {'titles':>8}")
    print("-" * 80)

    retriever = ChunkIndex()

    rows = []
    for task in args.tasks:
        data_path = DATASETS_DIR / "longbench" / "data" / f"{task}.jsonl"
        items = []
        with open(data_path, encoding="utf-8") as f:
            for line in f:
                items.append(json.loads(line))
        n = min(args.n, len(items))
        items = items[:n]

        for v in args.variants:
            counts = {"gold": 0, "ctx_w": 0, "titles": 0, "n": 0}
            for it in items:
                q = it["input"]; ctx_full = it["context"]
                ans = it["answers"] if isinstance(it["answers"], list) else [it["answers"]]
                ctx, titles = build_ctx(v, retriever, q, ctx_full, top_k=args.top_k)
                counts["n"] += 1
                if gold_in_text(ans, ctx):
                    counts["gold"] += 1
                counts["ctx_w"] += len(ctx.split())
                counts["titles"] += len(set(t for t in titles if t))
            n_ = counts["n"]
            row = {
                "task": task, "variant": v, "n": n_,
                "gold_in_ctx": counts["gold"] / n_ * 100,
                "ctx_words": counts["ctx_w"] / n_,
                "n_titles": counts["titles"] / n_,
            }
            rows.append(row)
            print(f"{task:<18} {v:<22} {n_:>4} "
                  f"{row['gold_in_ctx']:>7.2f}% "
                  f"{row['ctx_words']:>8.0f} "
                  f"{row['n_titles']:>8.2f}")

    # Save JSON
    out = TOPIC6_DIR / "experiments" / "results" / "longbench" / "_coverage_diagnostic.json"
    json.dump(rows, open(out, "w"), indent=2)
    print(f"\n[Saved] {out}")


if __name__ == "__main__":
    main()
