"""
Download and prepare QA datasets for Topic 6 + Topic 1.
Saves processed datasets as JSON files for fast loading.
"""
import json
import sys
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm

# Add parent to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.config import DATASETS_DIR

DATASETS_DIR.mkdir(parents=True, exist_ok=True)


def download_natural_questions():
    """Download NQ and extract simple short-answer questions."""
    out_path = DATASETS_DIR / "nq_validation.json"
    if out_path.exists():
        print(f"[NQ] Already exists: {out_path}")
        return

    print("[NQ] Downloading Natural Questions (validation)...")
    # NQ is very large; use streaming to get validation split
    ds = load_dataset(
        "google-research-datasets/natural_questions",
        split="validation",
    )

    processed = []
    for item in tqdm(ds, desc="Processing NQ"):
        # Extract short answers
        annotations = item.get("annotations", {})
        short_answers = annotations.get("short_answers", [])
        if not short_answers:
            continue

        # Get the first valid short answer
        sa = short_answers[0]
        if isinstance(sa, dict):
            start = sa.get("start_token", -1)
            end = sa.get("end_token", -1)
        else:
            continue

        question = item.get("question", {}).get("text", "")
        if not question:
            continue

        # Extract answer text from document tokens
        doc_tokens = item.get("document", {}).get("tokens", {})
        tokens = doc_tokens.get("token", [])
        if start >= 0 and end >= 0 and end <= len(tokens):
            answer = " ".join(tokens[start:end])
        else:
            continue

        if answer.strip():
            processed.append({
                "question": question,
                "answer": answer.strip(),
                "dataset": "nq",
            })

    print(f"[NQ] Processed {len(processed)} questions")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(processed, f, ensure_ascii=False, indent=2)
    print(f"[NQ] Saved to {out_path}")


def download_triviaqa():
    """Download TriviaQA (no context version for RAG eval)."""
    out_path = DATASETS_DIR / "triviaqa_validation.json"
    if out_path.exists():
        print(f"[TriviaQA] Already exists: {out_path}")
        return

    print("[TriviaQA] Downloading...")
    ds = load_dataset("trivia_qa", "rc.nocontext", split="validation",
                      trust_remote_code=True)

    processed = []
    for item in tqdm(ds, desc="Processing TriviaQA"):
        question = item.get("question", "")
        answer_obj = item.get("answer", {})
        aliases = answer_obj.get("aliases", [])
        value = answer_obj.get("value", "")

        if not question or not (value or aliases):
            continue

        processed.append({
            "question": question,
            "answer": value,
            "aliases": aliases,
            "dataset": "triviaqa",
        })

    print(f"[TriviaQA] Processed {len(processed)} questions")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(processed, f, ensure_ascii=False, indent=2)
    print(f"[TriviaQA] Saved to {out_path}")


def download_hotpotqa():
    """Download HotpotQA (distractor setting)."""
    out_path = DATASETS_DIR / "hotpotqa_validation.json"
    if out_path.exists():
        print(f"[HotpotQA] Already exists: {out_path}")
        return

    print("[HotpotQA] Downloading...")
    ds = load_dataset("hotpot_qa", "distractor", split="validation",
                      trust_remote_code=True)

    processed = []
    for item in tqdm(ds, desc="Processing HotpotQA"):
        question = item.get("question", "")
        answer = item.get("answer", "")

        if not question or not answer:
            continue

        # Also store supporting facts for analysis
        processed.append({
            "question": question,
            "answer": answer,
            "type": item.get("type", ""),
            "level": item.get("level", ""),
            "dataset": "hotpotqa",
        })

    print(f"[HotpotQA] Processed {len(processed)} questions")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(processed, f, ensure_ascii=False, indent=2)
    print(f"[HotpotQA] Saved to {out_path}")


def download_popqa():
    """Download PopQA (long-tail entity questions)."""
    out_path = DATASETS_DIR / "popqa_test.json"
    if out_path.exists():
        print(f"[PopQA] Already exists: {out_path}")
        return

    print("[PopQA] Downloading...")
    ds = load_dataset("akariasai/PopQA", split="test",
                      trust_remote_code=True)

    processed = []
    for item in tqdm(ds, desc="Processing PopQA"):
        question = item.get("question", "")
        answers = item.get("possible_answers", [])

        if not question:
            continue

        # PopQA has possible_answers as a list
        if isinstance(answers, str):
            answers = [answers]

        processed.append({
            "question": question,
            "answer": answers[0] if answers else "",
            "aliases": answers,
            "s_pop": item.get("s_pop", 0),  # popularity score
            "dataset": "popqa",
        })

    print(f"[PopQA] Processed {len(processed)} questions")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(processed, f, ensure_ascii=False, indent=2)
    print(f"[PopQA] Saved to {out_path}")


def download_gsm8k():
    """Download GSM8K for reasoning evaluation (Topic 1)."""
    out_path = DATASETS_DIR / "gsm8k_test.json"
    if out_path.exists():
        print(f"[GSM8K] Already exists: {out_path}")
        return

    print("[GSM8K] Downloading...")
    ds = load_dataset("openai/gsm8k", "main", split="test",
                      trust_remote_code=True)

    processed = []
    for item in tqdm(ds, desc="Processing GSM8K"):
        question = item.get("question", "")
        answer = item.get("answer", "")

        # Extract final numerical answer
        final_answer = ""
        if "####" in answer:
            final_answer = answer.split("####")[-1].strip()

        processed.append({
            "question": question,
            "answer": final_answer,
            "full_solution": answer,
            "dataset": "gsm8k",
        })

    print(f"[GSM8K] Processed {len(processed)} questions")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(processed, f, ensure_ascii=False, indent=2)
    print(f"[GSM8K] Saved to {out_path}")


if __name__ == "__main__":
    print("=" * 60)
    print("Downloading datasets for Topic 6 + Topic 1")
    print(f"Output directory: {DATASETS_DIR}")
    print("=" * 60)

    # Topic 6: QA datasets
    download_triviaqa()
    download_hotpotqa()
    download_popqa()

    # Topic 1: Reasoning
    download_gsm8k()

    # NQ last (largest and most complex)
    download_natural_questions()

    print("\n" + "=" * 60)
    print("All datasets downloaded!")
    print("=" * 60)
