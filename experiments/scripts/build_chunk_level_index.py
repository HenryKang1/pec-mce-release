"""
Build a CHUNK-level raw index for the granularity ablation.

Current 'raw' index is actually article-level (each article = 1 FAISS entry
containing the whole article). To isolate the aggregation effect vs the
summarization effect, we also need a chunk-level raw index where each
article is split into multiple ~100-token chunks, each a separate entry.

Granularity ablation comparisons at eval time:
  chunk_raw   vs article_raw : isolates the AGGREGATION effect
  article_raw vs article_note: isolates the SUMMARIZATION effect
  chunk_raw   vs article_note: full CTKS gap (aggregation + summarization)

Usage:
  python build_chunk_level_index.py --dataset hotpotqa --chunk-size 100 --overlap 0
"""
import argparse
import json
import sys
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "shared"))
from utils.config import DATASETS_DIR, TOPIC6_DIR
from rag_pipeline import ChunkIndex


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


def split_into_chunks(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Word-based sliding window chunking."""
    words = text.split()
    if len(words) <= chunk_size:
        return [" ".join(words)]
    chunks = []
    step = max(chunk_size - overlap, 1)
    for i in range(0, len(words), step):
        piece = words[i:i + chunk_size]
        if len(piece) < 20:  # drop tiny trailing fragment
            break
        chunks.append(" ".join(piece))
    return chunks


def run(dataset: str, chunk_size: int, overlap: int, suffix: str = ""):
    dir_name = f"{dataset}_chunk_index"
    if suffix:
        dir_name = f"{dataset}_chunk_index_{suffix}"
    out_dir = TOPIC6_DIR / "experiments" / "cache" / dir_name
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[Load] Articles from {dataset}")
    articles = extract_articles(dataset)
    print(f"[Load] {len(articles)} unique articles")

    all_chunks = []
    for title, text in tqdm(articles.items(), desc="Chunking"):
        for i, piece in enumerate(split_into_chunks(text, chunk_size, overlap)):
            prefix = f"{title}" if i == 0 else f"{title} (cont.)"
            all_chunks.append(f"{prefix}: {piece}")

    print(f"[Chunk] Produced {len(all_chunks)} chunks "
          f"(~{len(all_chunks)/len(articles):.1f} per article)")

    idx = ChunkIndex()
    idx.build_from_chunks(all_chunks, batch_size=512)
    idx.save(out_dir)

    # Metadata
    from statistics import mean
    lens = [len(c.split()) for c in all_chunks]
    meta = {
        "type": f"{dataset}_chunk_level_raw",
        "chunk_size_words": chunk_size,
        "overlap": overlap,
        "n_articles": len(articles),
        "n_chunks": len(all_chunks),
        "chunks_per_article": round(len(all_chunks) / len(articles), 2),
        "avg_words_per_chunk": round(mean(lens), 1),
    }
    with open(out_dir / "compile_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[Meta] {meta}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="hotpotqa", choices=["hotpotqa", "2wikimqa"])
    parser.add_argument("--chunk-size", type=int, default=100, help="Words per chunk")
    parser.add_argument("--overlap", type=int, default=0, help="Word overlap between chunks")
    parser.add_argument("--suffix", default="", help="Output dir suffix (e.g. 'c40')")
    args = parser.parse_args()
    run(args.dataset, args.chunk_size, args.overlap, args.suffix)
