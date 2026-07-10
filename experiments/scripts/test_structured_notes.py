"""
Test: SLM이 Markdown structured entity card를 잘 생성하는지 확인.
Plain text vs Markdown vs JSON 3가지 형식 비교.

Usage:
  python test_structured_notes.py
  python test_structured_notes.py --model qwen3-0.6b
  python test_structured_notes.py --eval  # 10개 article로 QA 성능까지 비교
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "shared"))
from utils.config import GGUF_MODELS, MODELS_DIR, DATASETS_DIR, TOPIC6_DIR
from rag_pipeline import LLMGenerator

# ============================================================
# 3가지 컴파일 프롬프트
# ============================================================

PROMPT_PLAIN = (
    "Summarize this article in one concise paragraph. "
    "Include: what it is, key facts, important names, dates, numbers.\n\n"
    "Title: {title}\n"
    "Text: {text}\n\n"
    "Summary:"
)

PROMPT_MARKDOWN = (
    "Create a structured note for this article using the format below.\n\n"
    "Title: {title}\n"
    "Text: {text}\n\n"
    "# {title}\n"
    "- Type:"
)

PROMPT_JSON = (
    'Extract structured information from this article as JSON.\n\n'
    'Title: {title}\n'
    'Text: {text}\n\n'
    '{{"title": "{title}", "type":'
)

# ============================================================
# Article extraction
# ============================================================

def get_sample_articles(n=10):
    """Get n sample articles from HotpotQA with their questions."""
    ds_path = DATASETS_DIR / "hotpotqa_full_validation.json"
    with open(ds_path, encoding="utf-8") as f:
        data = json.load(f)

    # Pick articles that are actually asked about
    articles = {}
    article_questions = {}
    for item in data[:200]:
        ctx = item.get("context", {})
        titles = ctx.get("title", [])
        sentences = ctx.get("sentences", [])
        sup_titles = item.get("supporting_facts", {}).get("title", [])

        for title, sents in zip(titles, sentences):
            if title in sup_titles and title not in articles:
                articles[title] = " ".join(sents)
                article_questions[title] = {
                    "question": item["question"],
                    "answer": item["answer"],
                }
            if len(articles) >= n:
                break
        if len(articles) >= n:
            break

    return articles, article_questions


def compile_note(llm, title, text, format_type="markdown"):
    """Compile one article into a note using specified format."""
    words = text.split()
    if len(words) > 300:
        text = " ".join(words[:300])

    if format_type == "plain":
        prompt = PROMPT_PLAIN.format(title=title, text=text)
        stop = ["\n\n", "Title:", "Text:"]
        max_tokens = 100
    elif format_type == "markdown":
        prompt = PROMPT_MARKDOWN.format(title=title, text=text)
        stop = ["\n\n\n", "Title:", "Text:", "<|im_end|>"]
        max_tokens = 150
    elif format_type == "json":
        prompt = PROMPT_JSON.format(title=title, text=text)
        stop = ["\n\n", "Title:", "Text:", "<|im_end|>"]
        max_tokens = 200
    else:
        raise ValueError(f"Unknown format: {format_type}")

    t0 = time.perf_counter()
    output = llm(
        prompt,
        max_tokens=max_tokens,
        temperature=0.0,
        echo=False,
        stop=stop,
        repeat_penalty=1.1,
    )
    ms = (time.perf_counter() - t0) * 1000
    raw = output["choices"][0]["text"].strip()

    # Reconstruct full note
    if format_type == "markdown":
        note = f"# {title}\n- Type:{raw}"
    elif format_type == "json":
        note = f'{{"title": "{title}", "type":{raw}'
    else:
        note = raw

    tokens = output.get("usage", {}).get("completion_tokens", len(raw.split()))
    return note, ms, tokens


def test_generation(model_name="lfm2.5-1.2b-instruct", n=10):
    """Test all 3 formats on n articles."""
    articles, questions = get_sample_articles(n)
    print(f"\nLoaded {len(articles)} sample articles\n")

    model_info = GGUF_MODELS[model_name]
    model_path = str(MODELS_DIR / model_info["file"])
    generator = LLMGenerator(model_path, n_ctx=2048, n_threads=4, n_gpu_layers=-1)

    formats = ["plain", "markdown", "json"]
    all_notes = {fmt: {} for fmt in formats}
    stats = {fmt: {"total_ms": 0, "total_tokens": 0, "total_chars": 0} for fmt in formats}

    for title, text in articles.items():
        print(f"\n{'='*60}")
        print(f"  {title}")
        print(f"{'='*60}")

        for fmt in formats:
            note, ms, tokens = compile_note(generator.llm, title, text, fmt)
            all_notes[fmt][title] = note
            stats[fmt]["total_ms"] += ms
            stats[fmt]["total_tokens"] += tokens
            stats[fmt]["total_chars"] += len(note)

            print(f"\n  [{fmt.upper()}] ({ms:.0f}ms, {tokens} tokens, {len(note)} chars)")
            # Show first 200 chars
            preview = note.replace('\n', '\n    ')
            if len(preview) > 300:
                preview = preview[:300] + "..."
            print(f"    {preview}")

    # Summary
    print(f"\n\n{'='*60}")
    print(f"  SUMMARY ({len(articles)} articles, {model_name})")
    print(f"{'='*60}")
    print(f"  {'Format':<12} {'Time':>8} {'Tokens':>8} {'Chars':>8} {'Chars/article':>14}")
    print(f"  {'-'*12} {'-'*8} {'-'*8} {'-'*8} {'-'*14}")
    for fmt in formats:
        s = stats[fmt]
        avg_chars = s['total_chars'] / len(articles)
        print(f"  {fmt:<12} {s['total_ms']:>7.0f}ms {s['total_tokens']:>8} "
              f"{s['total_chars']:>8} {avg_chars:>13.0f}")

    return all_notes, questions, generator


def test_qa(all_notes, questions, generator, articles):
    """Quick QA test: which format helps SLM answer better?"""
    from run_baseline import normalize_answer, exact_match, f1_score

    print(f"\n\n{'='*60}")
    print(f"  QA TEST: 각 형식의 note를 context로 답변 품질 비교")
    print(f"{'='*60}")

    formats = list(all_notes.keys())
    results = {fmt: {"em": 0, "f1": 0.0, "n": 0} for fmt in formats}
    # Add raw baseline
    results["raw"] = {"em": 0, "f1": 0.0, "n": 0}

    for title, qa in questions.items():
        q = qa["question"]
        a = qa["answer"]

        # Raw article as context
        raw_text = articles[title]
        raw_prompt = (
            f"Answer the question based on the document. Give a short, direct answer.\n\n"
            f"[Document]: {raw_text[:500]}\n\n"
            f"Question: {q}\nAnswer:"
        )
        raw_out = generator.generate(raw_prompt, max_tokens=30)
        raw_ans = raw_out[0]
        results["raw"]["em"] += int(exact_match(raw_ans, a))
        results["raw"]["f1"] += f1_score(raw_ans, a)
        results["raw"]["n"] += 1

        # Each compiled format
        for fmt in formats:
            note = all_notes[fmt].get(title, "")
            if not note:
                continue

            prompt = (
                f"Answer the question based on the document. Give a short, direct answer.\n\n"
                f"[Document]: {note[:500]}\n\n"
                f"Question: {q}\nAnswer:"
            )
            out = generator.generate(prompt, max_tokens=30)
            ans = out[0]

            results[fmt]["em"] += int(exact_match(ans, a))
            results[fmt]["f1"] += f1_score(ans, a)
            results[fmt]["n"] += 1

        # Show per-question
        print(f"\n  Q: {q}")
        print(f"  A: {a}")
        print(f"  Raw:      {raw_ans}")
        for fmt in formats:
            note = all_notes[fmt].get(title, "")
            if note:
                prompt = (
                    f"Answer the question based on the document. Give a short, direct answer.\n\n"
                    f"[Document]: {note[:500]}\n\n"
                    f"Question: {q}\nAnswer:"
                )
                # Re-use already generated answer (avoid double generation)
                pass
        for fmt in formats:
            n = results[fmt]["n"]
            if n > 0:
                last_em = results[fmt]["em"]
                # Just print current running average
        print(f"  Results so far: ", end="")
        for fmt in ["raw"] + formats:
            n = results[fmt]["n"]
            if n > 0:
                em_pct = results[fmt]["em"] / n * 100
                f1_pct = results[fmt]["f1"] / n * 100
                print(f" {fmt}={em_pct:.0f}/{f1_pct:.0f}", end="")
        print()

    # Final
    print(f"\n\n  {'Format':<12} {'EM%':>6} {'F1%':>6} {'n':>4}")
    print(f"  {'-'*12} {'-'*6} {'-'*6} {'-'*4}")
    for fmt in ["raw"] + formats:
        n = results[fmt]["n"]
        if n > 0:
            print(f"  {fmt:<12} {results[fmt]['em']/n*100:>5.1f}% "
                  f"{results[fmt]['f1']/n*100:>5.1f}% {n:>4}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="lfm2.5-1.2b-instruct")
    parser.add_argument("--n", type=int, default=10)
    parser.add_argument("--eval", action="store_true", help="QA 성능도 비교")
    args = parser.parse_args()

    all_notes, questions, generator = test_generation(args.model, args.n)

    if args.eval:
        articles, _ = get_sample_articles(args.n)
        test_qa(all_notes, questions, generator, articles)
