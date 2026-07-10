"""
Compile entity index from 2WikiMultiHopQA context.

Same approach as compile_hotpotqa_entities.py:
Extract all articles from 2WikiMQA context → SLM compiles entity notes → FAISS index.

Usage:
  python compile_2wikimqa_entities.py --model lfm2.5-1.2b-instruct
  python compile_2wikimqa_entities.py --resume
"""
import argparse
import json
import sys
import time
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "shared"))
from utils.config import GGUF_MODELS, MODELS_DIR, DATASETS_DIR, TOPIC6_DIR
from rag_pipeline import ChunkIndex, LLMGenerator


ENTITY_PROMPT = (
    "Summarize this article in one concise paragraph. "
    "Include: what it is, key facts, important names, dates, numbers.\n\n"
    "Title: {title}\n"
    "Text: {text}\n\n"
    "Summary:"
)


def extract_articles_from_2wikimqa() -> dict[str, str]:
    """Extract all unique articles from 2WikiMQA validation context."""
    ds_path = DATASETS_DIR / "2wikimqa_validation.json"
    if not ds_path.exists():
        print("[Error] Run: python setup_evaluation.py --download")
        return {}

    with open(ds_path, encoding="utf-8") as f:
        data = json.load(f)

    articles = {}
    for item in data:
        ctx = item.get("context", {})
        titles = ctx.get("title", [])
        sentences = ctx.get("sentences", [])
        for title, sents in zip(titles, sentences):
            if title not in articles:
                articles[title] = " ".join(sents)
            else:
                existing = articles[title]
                new_text = " ".join(sents)
                if len(new_text) > len(existing):
                    articles[title] = new_text

    return articles


def compile_entity(llm, title: str, text: str) -> str:
    """Compile a single article into an entity note."""
    words = text.split()
    if len(words) > 300:
        text = " ".join(words[:300])

    prompt = ENTITY_PROMPT.format(title=title, text=text)

    output = llm(
        prompt,
        max_tokens=100,
        temperature=0.0,
        echo=False,
        stop=["\n\n", "Title:", "Text:", "<|im_end|>"],
        repeat_penalty=1.1,
    )

    note = output["choices"][0]["text"].strip()
    note = note.split("\n\n")[0].strip()

    if not note or len(note) < 5:
        note = f"{title}: {text.split('.')[0]}."

    return note


def compile_2wikimqa_entities(
    model_name: str = "lfm2.5-1.2b-instruct",
    resume: bool = False,
    batch_save_every: int = 1000,
):
    """Compile all 2WikiMQA articles into entity notes."""

    output_dir = TOPIC6_DIR / "experiments" / "cache" / "2wikimqa_entity_index"
    output_dir.mkdir(parents=True, exist_ok=True)
    notes_file = output_dir / "entity_notes.json"

    # Extract articles
    print("[Extract] Loading articles from 2WikiMQA...")
    articles = extract_articles_from_2wikimqa()
    print(f"[Extract] Found {len(articles)} unique articles")

    # Resume
    existing_notes = {}
    if resume and notes_file.exists():
        with open(notes_file, encoding="utf-8") as f:
            existing_notes = json.load(f)
        print(f"[Resume] Found {len(existing_notes)} existing notes")

    # Load model
    model_info = GGUF_MODELS[model_name]
    model_path = str(MODELS_DIR / model_info["file"])
    print(f"[Model] Loading {model_name}...")
    generator = LLMGenerator(model_path, n_ctx=2048, n_threads=4, n_gpu_layers=-1)

    # Compile
    entity_notes = dict(existing_notes)
    t_start = time.time()
    errors = 0
    to_compile = [(t, txt) for t, txt in articles.items() if t not in entity_notes]

    print(f"[Compile] {len(to_compile)} articles to compile, {len(entity_notes)} already done")

    for title, text in tqdm(to_compile, desc="Compiling"):
        try:
            note = compile_entity(generator.llm, title, text)
            entity_notes[title] = note
        except Exception as e:
            entity_notes[title] = f"{title}: {text[:100]}."
            errors += 1

        # Periodic save
        if len(entity_notes) % batch_save_every == 0:
            with open(notes_file, "w", encoding="utf-8") as f:
                json.dump(entity_notes, f, ensure_ascii=False)
            elapsed = time.time() - t_start
            done = len(entity_notes) - len(existing_notes)
            rate = done / elapsed if elapsed > 0 else 0
            remaining = (len(to_compile) - done) / rate if rate > 0 else 0
            print(f"\n[Save] {len(entity_notes)}/{len(articles)} "
                  f"({rate:.1f}/sec, ~{remaining/3600:.1f}h remaining)")

    # Final save
    with open(notes_file, "w", encoding="utf-8") as f:
        json.dump(entity_notes, f, ensure_ascii=False)

    elapsed = time.time() - t_start
    compiled_count = len(entity_notes) - len(existing_notes)
    print(f"\n[Done] {len(entity_notes)} total notes, "
          f"{compiled_count} newly compiled in {elapsed:.0f}s, {errors} errors")

    # Build FAISS index
    print(f"\n[Index] Building FAISS index from {len(entity_notes)} entity notes...")
    note_list = [f"{title}: {note}" for title, note in entity_notes.items()]
    idx = ChunkIndex()
    idx.build_from_chunks(note_list, batch_size=512)
    idx.save(output_dir)

    # Also build raw article index for fair comparison
    raw_dir = TOPIC6_DIR / "experiments" / "cache" / "2wikimqa_raw_index"
    raw_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n[Index] Building raw article index ({len(articles)} articles)...")
    raw_list = [f"{title}: {text}" for title, text in articles.items()]
    raw_idx = ChunkIndex()
    raw_idx.build_from_chunks(raw_list, batch_size=512)
    raw_idx.save(raw_dir)

    # Save metadata
    avg_raw_words = sum(len(t.split()) for t in articles.values()) / len(articles)
    avg_note_words = sum(len(n.split()) for n in entity_notes.values()) / len(entity_notes)
    meta = {
        "type": "2wikimqa_entity_compiled",
        "model_used": model_name,
        "n_articles": len(articles),
        "n_entity_notes": len(entity_notes),
        "avg_raw_words": round(avg_raw_words, 1),
        "avg_note_words": round(avg_note_words, 1),
        "compression_ratio": round(avg_raw_words / avg_note_words, 1) if avg_note_words > 0 else 0,
        "compile_time_sec": round(elapsed, 1),
        "errors": errors,
    }
    with open(output_dir / "compile_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n[Meta] {avg_raw_words:.0f} words/article → {avg_note_words:.0f} words/note "
          f"({meta['compression_ratio']}x compression)")
    print(f"[Meta] Entity index: {output_dir}")
    print(f"[Meta] Raw index: {raw_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="lfm2.5-1.2b-instruct")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--batch-save-every", type=int, default=1000)
    args = parser.parse_args()

    compile_2wikimqa_entities(
        model_name=args.model,
        resume=args.resume,
        batch_save_every=args.batch_save_every,
    )
