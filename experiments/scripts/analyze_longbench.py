"""
LongBench per-question diagnostics.

For each (model, task, variant) result, re-run retrieval on the question's
context under the same rules used at eval time, and compute:

  - answer_in_ctx: fraction of questions whose retrieved context contains
                   the gold answer as a substring
  - gold_psg_retrieved: fraction of questions whose gold supporting passage
                        (proxy = passage containing the answer substring)
                        was among the top-k retrieved / used passages
  - avg_ctx_tokens: average tokens fed to the reader (coarse proxy for latency)

Also stratifies EM and answer_in_ctx by:
  - total context length (short / medium / long)
  - number of passages per question

Outputs:
  - results/longbench/DIAG_{model}_{task}_{variant}.json
  - results/longbench/FRONTIER.json  (EM / F1 / latency triples for plotting)

Usage:
  python analyze_longbench.py --model lfm2.5-1.2b-instruct
"""
import argparse
import json
import re
import string
import sys
from pathlib import Path
from statistics import mean
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "shared"))
from utils.config import DATASETS_DIR, TOPIC6_DIR
sys.path.insert(0, str(Path(__file__).resolve().parent))
from longbench_pipeline import (
    split_passages, extract_anchors, VARIANTS,
    NUM_RX, YEAR_RX, build_pec_cards, hydrate_cards, evidence_is_weak,
)
from rag_pipeline import ChunkIndex


MAX_CTX_TOKENS = 1500  # must match pipeline
TOP_K = 5


def norm(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def answer_in_text(answers, text):
    t = norm(text)
    return any(a and norm(a) in t for a in answers)


def load_task(task: str):
    path = DATASETS_DIR / "longbench" / "data" / f"{task}.jsonl"
    items = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            items.append(json.loads(line))
    return items


def compile_passage_no_llm(title: str, body: str, variant: str) -> str:
    """For analysis we only need the retrieval-index form; we approximate
    summary/anchored by using (title + first 80 words) + anchors, since the
    actual LLM summaries are not serialized at eval time and that's fine
    for coverage/answer-in-ctx diagnosis — we just need retrieval behavior."""
    if variant in ("raw_trunc", "raw_topk", "hybrid2", "hybrid3"):
        return f"{title}: {' '.join(body.split()[:200])}"
    if variant == "anchors":
        return f"{title}: ANCHORS: {', '.join(extract_anchors(title, body))}"
    if variant == "summary":
        return f"{title}: " + " ".join(body.split()[:40])
    if variant == "anchored":
        return f"{title}: " + " ".join(body.split()[:40]) + \
            f" | ANCHORS: {', '.join(extract_anchors(title, body))}"
    if variant in ("pec_card", "pec_hydrate", "pec_adaptive", "pec_bridge"):
        raise ValueError("PEC variants are reconstructed at the passage-set level")
    raise ValueError(variant)


def retrieve_texts_for_diag(retriever: ChunkIndex, query: str, texts: list[str],
                            top_k: int) -> tuple[list[str], list[int], list[float]]:
    retriever.build_from_chunks(texts, batch_size=128)
    retrieved, scores, _ = retriever.search(query, top_k=top_k)
    used_indices = []
    taken = set()
    for r in retrieved:
        for i, text in enumerate(texts):
            if i in taken:
                continue
            if text == r:
                taken.add(i)
                used_indices.append(i)
                break
    return retrieved, used_indices, scores


def reconstruct_context(question: str, passages: list[tuple[str, str]],
                        variant: str, retriever: ChunkIndex):
    """Approximate the reader context used by longbench_pipeline."""
    if variant == "raw_trunc":
        return "\n\n".join(f"{t}\n{b}" for t, b in passages), {t for t, _ in passages}

    if variant in ("pec_card", "pec_hydrate", "pec_adaptive", "pec_bridge"):
        cards, card_meta = build_pec_cards(passages)
        if not cards:
            return "\n\n".join(f"{t}\n{b}" for t, b in passages), {t for t, _ in passages}
        retrieved_cards, card_indices, scores = retrieve_texts_for_diag(
            retriever, question, cards, TOP_K
        )
        selected_meta = [card_meta[i] for i in card_indices]

        if variant == "pec_bridge":
            raw_chunks = [f"{t}: {' '.join(b.split()[:200])}" for t, b in passages]
            raw_texts, raw_indices, _ = retrieve_texts_for_diag(
                retriever, question, raw_chunks, 2
            )
            raw_title_set = {passages[i][0] for i in raw_indices}
            bridge_meta = [m for m in selected_meta if m["title"] not in raw_title_set]
            hydrated = hydrate_cards(passages, bridge_meta[:max(1, TOP_K - 2)], window=1)
            parts = []
            for i, r in enumerate(raw_texts):
                parts.append(f"[Document {i+1} (RAW)]: {r}")
            offset = len(parts)
            for i, h in enumerate(hydrated):
                parts.append(f"[Document {offset+i+1} (HYDRATED)]: {h}")
            used_titles = raw_title_set | {m["title"] for m in bridge_meta}
            return "\n\n".join(parts), used_titles

        if variant == "pec_adaptive" and evidence_is_weak(question, selected_meta, scores):
            raw_chunks = [f"{t}: {' '.join(b.split()[:200])}" for t, b in passages]
            retrieved, _, _ = retrieve_texts_for_diag(retriever, question, raw_chunks, TOP_K)
            used_titles = set()
            for r in retrieved:
                for t, _ in passages:
                    if r.startswith(f"{t}:") or r.startswith(f"{t} "):
                        used_titles.add(t)
                        break
            ctx = "\n\n".join(
                f"[Document {i+1} (RAW-FALLBACK)]: {c}"
                for i, c in enumerate(retrieved)
            )
            return ctx, used_titles

        used_titles = {m["title"] for m in selected_meta}
        if variant == "pec_card":
            ctx = "\n\n".join(
                f"[Document {i+1} (CARD)]: {c}"
                for i, c in enumerate(retrieved_cards)
            )
            return ctx, used_titles

        hydrated = hydrate_cards(passages, selected_meta, window=1)
        ctx = "\n\n".join(
            f"[Document {i+1} (HYDRATED)]: {c}"
            for i, c in enumerate(hydrated)
        )
        return ctx, used_titles

    compiled = [compile_passage_no_llm(t, b, variant) for t, b in passages]
    total_words = sum(len(c.split()) for c in compiled)
    est_tokens = int(total_words * 1.4)
    use_all = (variant in ("summary", "anchors", "anchored")
               and est_tokens <= MAX_CTX_TOKENS - 100)

    if use_all:
        ctx = "\n\n".join(f"[Document {i+1}]: {c}" for i, c in enumerate(compiled))
        return ctx, {t for t, _ in passages}

    retriever.build_from_chunks(compiled, batch_size=128)
    retrieved, _, _ = retriever.search(question, top_k=TOP_K)
    used_titles = set()
    for r in retrieved:
        for t, _ in passages:
            if r.startswith(f"{t}:") or r.startswith(f"{t} "):
                used_titles.add(t)
                break
    ctx = "\n\n".join(f"[Document {i+1}]: {c}" for i, c in enumerate(retrieved))
    return ctx, used_titles


def diagnose(model: str, task: str, variant: str, items: list):
    rd = TOPIC6_DIR / "experiments" / "results" / "longbench"
    result_file = rd / f"{model}_{task}_{variant}.json"
    if not result_file.exists():
        return None
    try:
        with open(result_file, encoding="utf-8") as f:
            result = json.load(f)
    except Exception:
        return None

    retriever = ChunkIndex()
    per_q = []
    for item, rec in tqdm(list(zip(items, result["results"]))[:len(result["results"])],
                          desc=f"{task}/{variant}", leave=False):
        question = item["input"]
        context = item["context"]
        answers = item["answers"] if isinstance(item["answers"], list) else [item["answers"]]
        passages = split_passages(context)

        # Gold passage = any passage whose body contains the gold answer
        gold_titles = set()
        for t, b in passages:
            if answer_in_text(answers, b):
                gold_titles.add(t)

        # Reconstruct the context the reader actually saw (approx)
        ctx_str, used_titles = reconstruct_context(question, passages, variant, retriever)

        # Token-level truncation (mirror pipeline logic coarsely)
        ctx_words = ctx_str.split()
        if len(ctx_words) > int(MAX_CTX_TOKENS / 1.4):
            ctx_str = " ".join(ctx_words[: int(MAX_CTX_TOKENS / 1.4)])

        ans_in_ctx = answer_in_text(answers, ctx_str)
        gold_covered = len(gold_titles & used_titles) > 0 if gold_titles else None

        per_q.append({
            "n_passages": len(passages),
            "ctx_len_chars": len(context),
            "gold_titles": sorted(gold_titles),
            "used_titles": sorted(used_titles),
            "gold_covered": gold_covered,
            "answer_in_ctx": ans_in_ctx,
            "em": rec["em"],
            "f1": rec["f1"],
        })

    # Aggregate
    gold_covs = [p["gold_covered"] for p in per_q if p["gold_covered"] is not None]
    aic = [p["answer_in_ctx"] for p in per_q]
    em = [p["em"] for p in per_q]

    summary = {
        "model": model, "task": task, "variant": variant,
        "n": len(per_q),
        "n_with_gold_passages": len(gold_covs),
        "gold_coverage": mean(gold_covs) if gold_covs else None,
        "answer_in_ctx": mean(aic),
        "em": mean(em),
        "latency_ms": result["metrics"].get("avg_latency_ms"),
        "per_question": per_q,
    }

    out_path = rd / f"DIAG_{model}_{task}_{variant}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False)

    return {
        "task": task, "variant": variant,
        "n": summary["n"],
        "gold_coverage": summary["gold_coverage"],
        "answer_in_ctx": summary["answer_in_ctx"],
        "em": summary["em"],
        "latency_ms": summary["latency_ms"],
    }


def length_buckets(per_q_records: list[dict], bucket_edges):
    """per_q_records: list with ctx_len_chars, em, answer_in_ctx"""
    buckets = {f"<={e}": {"em": [], "aic": [], "n": 0}
               for e in bucket_edges}
    buckets["> max"] = {"em": [], "aic": [], "n": 0}
    for p in per_q_records:
        placed = False
        for e in bucket_edges:
            if p["ctx_len_chars"] <= e:
                buckets[f"<={e}"]["em"].append(p["em"])
                buckets[f"<={e}"]["aic"].append(p["answer_in_ctx"])
                buckets[f"<={e}"]["n"] += 1
                placed = True
                break
        if not placed:
            buckets["> max"]["em"].append(p["em"])
            buckets["> max"]["aic"].append(p["answer_in_ctx"])
            buckets["> max"]["n"] += 1
    return {k: {
        "n": v["n"],
        "em": mean(v["em"]) if v["em"] else None,
        "aic": mean(v["aic"]) if v["aic"] else None,
    } for k, v in buckets.items()}


def main(models):
    rd = TOPIC6_DIR / "experiments" / "results" / "longbench"
    frontier = []
    per_task_diag = {t: [] for t in ["hotpotqa", "2wikimqa", "musique"]}

    for m in models:
        for t in ["hotpotqa", "2wikimqa", "musique"]:
            try:
                items = load_task(t)
            except FileNotFoundError:
                continue
            for v in VARIANTS:
                rec = diagnose(m, t, v, items)
                if rec:
                    rec["model"] = m
                    frontier.append(rec)
                    per_task_diag[t].append(rec)

    with open(rd / "FRONTIER.json", "w") as f:
        json.dump(frontier, f, indent=2)

    # Pretty print
    print(f"\n{'Model':<24} {'Task':<10} {'Variant':<12} "
          f"{'EM':>7} {'Ans∈Ctx':>9} {'Gold%':>7} {'Lat':>7}")
    print("-" * 82)
    for r in frontier:
        gc = f"{r['gold_coverage']*100:>6.2f}%" if r["gold_coverage"] is not None else "   --"
        aic = f"{r['answer_in_ctx']*100:>8.2f}%" if r["answer_in_ctx"] is not None else "   --"
        em = f"{r['em']*100:>6.2f}%" if r["em"] is not None else "   --"
        print(f"{r['model']:<24} {r['task']:<10} {r['variant']:<12} "
              f"{em} {aic} {gc} {r['latency_ms']:>7}")

    # Length buckets for each task (merging across variants for now)
    print("\n=== Length buckets (context chars) — per variant ===")
    bucket_edges = [10000, 30000, 60000, 100000]
    for t in ["hotpotqa", "2wikimqa", "musique"]:
        for m in models:
            for v in VARIANTS:
                diag_path = rd / f"DIAG_{m}_{t}_{v}.json"
                if not diag_path.exists():
                    continue
                with open(diag_path, encoding="utf-8") as f:
                    data = json.load(f)
                buckets = length_buckets(data["per_question"], bucket_edges)
                print(f"\n{m} / {t} / {v}")
                for k, stats in buckets.items():
                    if stats["n"] == 0:
                        continue
                    em = f"{stats['em']*100:>6.2f}%" if stats["em"] is not None else "  --"
                    aic = f"{stats['aic']*100:>6.2f}%" if stats["aic"] is not None else "  --"
                    print(f"  {k:<10}  n={stats['n']:>4}  EM={em}  AnsCtx={aic}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=["lfm2.5-1.2b-instruct"])
    args = parser.parse_args()
    main(args.models)
