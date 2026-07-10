"""
Extractive CTKS index — zero-hallucination alternative to LLM-compiled notes.

For each article, produce an entity note by concatenating a minimal,
fact-dense subset of its own sentences. No text is generated; every
token comes from the source article. Therefore:
  - No hallucinated dates, album names, or relationships.
  - No paraphrase-away of exact entity mentions.
  - Zero GPU time for compilation.

Selection heuristic per article:
  1. Keep the first sentence (usually the definitional lead in Wikipedia).
  2. Keep any sentence that contains >=1 capitalized multi-word entity
     AND at least one number/date.
  3. Cap note at N words; if under, also include any sentence with a
     named entity that is not already covered.

If the article is shorter than a threshold (default 100 words), keep the
entire raw text unchanged (same heuristic as Smart CTKS).

Usage:
  python build_extractive_index.py --dataset hotpotqa --max-words 80 --threshold 100
"""
import argparse
import json
import re
import sys
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "shared"))
from utils.config import TOPIC6_DIR, DATASETS_DIR
from rag_pipeline import ChunkIndex


SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")
ENTITY_RX = re.compile(r"\b[A-Z][a-zA-Z0-9]+(?:\s+[A-Z][a-zA-Z0-9]+)+\b")
NUM_RX = re.compile(r"\b\d{2,4}\b")


def extract_articles(dataset: str) -> dict[str, str]:
    ds_path = DATASETS_DIR / f"{dataset}_full_validation.json"
    if not ds_path.exists():
        ds_path = DATASETS_DIR / f"{dataset}_validation.json"
    with open(ds_path, encoding="utf-8") as f:
        data = json.load(f)
    articles = {}
    for item in data:
        ctx = item.get("context", {})
        for title, sents in zip(ctx.get("title", []), ctx.get("sentences", [])):
            text = " ".join(sents)
            if title not in articles or len(text) > len(articles[title]):
                articles[title] = text
    return articles


def extractive_note(text: str, max_words: int) -> str:
    """Pick fact-dense sentences from `text` while staying under `max_words`."""
    sents = [s.strip() for s in SENT_SPLIT.split(text) if s.strip()]
    if not sents:
        return text

    # Score each sentence: #entities + 2*#numbers. Always keep sent 0.
    scored = []
    for i, s in enumerate(sents):
        ent = len(ENTITY_RX.findall(s))
        num = len(NUM_RX.findall(s))
        base = ent + 2 * num
        scored.append((i, base, s))

    keep = [scored[0]] if scored else []
    rest = sorted(scored[1:], key=lambda t: -t[1])
    word_count = len(keep[0][2].split()) if keep else 0
    for i, score, s in rest:
        if score <= 0:
            break
        sw = len(s.split())
        if word_count + sw > max_words:
            continue
        keep.append((i, score, s))
        word_count += sw

    # Restore original order
    keep.sort(key=lambda t: t[0])
    return " ".join(s for _, _, s in keep)


def build(dataset: str, max_words: int, threshold: int):
    articles = extract_articles(dataset)

    out_dir = TOPIC6_DIR / "experiments" / "cache" / f"{dataset}_extractive_index"
    out_dir.mkdir(parents=True, exist_ok=True)

    chunks = []
    notes = {}
    used = {"raw_short": 0, "extracted": 0, "raw_fallback": 0}
    kept_words = 0

    for title, raw_text in tqdm(articles.items(), desc="Extractive"):
        raw_words = raw_text.split()
        if len(raw_words) <= threshold:
            note = raw_text
            used["raw_short"] += 1
        else:
            note = extractive_note(raw_text, max_words)
            if len(note.split()) < 10:
                note = " ".join(raw_words[:max_words])
                used["raw_fallback"] += 1
            else:
                used["extracted"] += 1
        notes[title] = note
        chunks.append(f"{title}: {note}")
        kept_words += len(note.split())

    print(f"\n[Stats]")
    print(f"  raw_short  (<= {threshold}w)       : {used['raw_short']}")
    print(f"  extracted                        : {used['extracted']}")
    print(f"  raw_fallback (no good sent)      : {used['raw_fallback']}")
    print(f"  avg words/note                   : {kept_words/len(chunks):.1f}")

    # Save the notes as entity_notes.json for convenience
    with open(out_dir / "entity_notes.json", "w", encoding="utf-8") as f:
        json.dump(notes, f, ensure_ascii=False)

    idx = ChunkIndex()
    idx.build_from_chunks(chunks, batch_size=512)
    idx.save(out_dir)

    meta = {
        "type": f"{dataset}_extractive",
        "max_words_per_note": max_words,
        "short_threshold": threshold,
        "used": used,
        "n_entries": len(chunks),
    }
    with open(out_dir / "compile_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[Saved] {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="hotpotqa", choices=["hotpotqa", "2wikimqa"])
    parser.add_argument("--max-words", type=int, default=80)
    parser.add_argument("--threshold", type=int, default=100)
    args = parser.parse_args()
    build(args.dataset, args.max_words, args.threshold)
