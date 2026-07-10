"""
Smart CTKS index — hybrid of raw and compiled notes.

Rationale: on HotpotQA, many articles are already short (~80 words) and
pre-filtered. Summarizing them is lossy (hallucinated dates/names) and barely
compresses. Long articles are the ones that actually need compilation.

Strategy:
  - If raw article <= threshold words  -> keep raw text verbatim (index it)
  - If raw article >  threshold words  -> use the LFM-compiled note (fallback)
  - If the compiled note is longer than the raw article for a given entity,
    ALSO fall back to raw (notes expanding means the compiler hallucinated).

Output index directory:
  cache/{dataset}_smart_index/   --> FAISS index + chunks.json

Usage:
  python build_smart_index.py --dataset hotpotqa --threshold 100
"""
import argparse
import json
import sys
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "shared"))
from utils.config import TOPIC6_DIR, DATASETS_DIR
from rag_pipeline import ChunkIndex


def extract_articles(dataset: str) -> dict[str, str]:
    """Reconstruct {title: raw_text} from the dataset's context field."""
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


def build(dataset: str, threshold: int, use_stricter: bool):
    articles = extract_articles(dataset)

    notes_path = TOPIC6_DIR / "experiments" / "cache" / f"{dataset}_entity_index" / "entity_notes.json"
    if not notes_path.exists():
        print(f"[Error] compiled notes not found: {notes_path}")
        sys.exit(1)
    with open(notes_path, encoding="utf-8") as f:
        notes = json.load(f)

    out_dir = TOPIC6_DIR / "experiments" / "cache" / f"{dataset}_smart_index"
    out_dir.mkdir(parents=True, exist_ok=True)

    chunks: list[str] = []
    used = {"raw_short": 0, "raw_note_longer": 0, "note": 0, "missing_note": 0}
    kept_word_total = 0
    raw_word_total = 0

    for title, raw_text in tqdm(articles.items(), desc="Smart index"):
        raw_words = raw_text.split()
        raw_len = len(raw_words)
        raw_word_total += raw_len
        note = notes.get(title, "")
        note_len = len(note.split())

        if raw_len <= threshold:
            # Short article: keep raw verbatim
            kept = f"{title}: {raw_text}"
            used["raw_short"] += 1
        elif not note:
            kept = f"{title}: {raw_text}"
            used["missing_note"] += 1
        elif use_stricter and note_len >= raw_len:
            # Compiler expanded instead of compressed -> likely hallucination
            kept = f"{title}: {raw_text}"
            used["raw_note_longer"] += 1
        else:
            kept = f"{title}: {note}"
            used["note"] += 1
        chunks.append(kept)
        kept_word_total += len(kept.split())

    print(f"\n[Stats]")
    print(f"  raw (short <= {threshold}w)    : {used['raw_short']}")
    print(f"  raw (note >= raw, dropped)   : {used['raw_note_longer']}")
    print(f"  raw (note missing)           : {used['missing_note']}")
    print(f"  note (kept)                  : {used['note']}")
    print(f"  total entries                : {len(chunks)}")
    print(f"  avg words/entry              : {kept_word_total/len(chunks):.1f}")
    print(f"  total word count ratio       : {kept_word_total/raw_word_total:.2f}x of raw")

    idx = ChunkIndex()
    idx.build_from_chunks(chunks, batch_size=512)
    idx.save(out_dir)

    meta = {
        "type": f"{dataset}_smart_hybrid",
        "threshold_words": threshold,
        "drop_note_if_longer": use_stricter,
        "used": used,
        "n_entries": len(chunks),
    }
    with open(out_dir / "compile_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[Saved] {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="hotpotqa", choices=["hotpotqa", "2wikimqa"])
    parser.add_argument("--threshold", type=int, default=100,
                        help="Articles at or below this word count keep raw text.")
    parser.add_argument("--use-stricter", action="store_true",
                        help="Also drop the compiled note when it is longer than the raw article (likely hallucination).")
    args = parser.parse_args()
    build(args.dataset, args.threshold, args.use_stricter)
