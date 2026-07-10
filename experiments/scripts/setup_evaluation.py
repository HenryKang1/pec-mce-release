"""
Setup full evaluation matrix for CTKS paper.

1. Download datasets with oracle context (HotpotQA full, 2WikiMultiHopQA, TriviaQA)
2. Run oracle evaluation (ground truth context → SLM)
3. Run full matrix: 3 models × 3 datasets × 4 conditions

Usage:
  python setup_evaluation.py --download      # Download datasets
  python setup_evaluation.py --run-all       # Run full matrix
  python setup_evaluation.py --oracle-only   # Run oracle evaluation only
"""
import argparse
import json
import sys
import time
import re
import string
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "shared"))
from utils.config import GGUF_MODELS, MODELS_DIR, DATASETS_DIR, RAG_CONFIG, TOPIC6_DIR
from rag_pipeline import ChunkIndex, LLMGenerator, RAGPipeline, LatencyProfile


# ============================================================
# Evaluation Metrics
# ============================================================
def normalize_answer(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r'\b(a|an|the)\b', ' ', s)
    s = s.translate(str.maketrans('', '', string.punctuation))
    s = ' '.join(s.split())
    return s


def exact_match(prediction: str, ground_truth: str) -> bool:
    return normalize_answer(prediction) == normalize_answer(ground_truth)


def f1_score(prediction: str, ground_truth: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    gt_tokens = normalize_answer(ground_truth).split()
    if not pred_tokens or not gt_tokens:
        return float(pred_tokens == gt_tokens)
    common = set(pred_tokens) & set(gt_tokens)
    if not common:
        return 0.0
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(gt_tokens)
    return 2 * precision * recall / (precision + recall)


def evaluate_answer(prediction: str, item: dict) -> dict:
    answer = item.get("answer", "")
    aliases = item.get("aliases", [answer])
    best_em = False
    best_f1 = 0.0
    for ans in aliases:
        if exact_match(prediction, ans):
            best_em = True
        best_f1 = max(best_f1, f1_score(prediction, ans))
    return {"em": best_em, "f1": best_f1}


# ============================================================
# Dataset Download
# ============================================================
def download_datasets():
    """Download datasets with oracle context."""
    from datasets import load_dataset

    # 1. HotpotQA with full context
    print("[Download] HotpotQA (distractor, validation)...")
    hotpot_path = DATASETS_DIR / "hotpotqa_full_validation.json"
    if not hotpot_path.exists():
        ds = load_dataset("hotpot_qa", "distractor", split="validation",
                          trust_remote_code=True)
        data = []
        for item in tqdm(ds, desc="HotpotQA"):
            data.append({
                "question": item["question"],
                "answer": item["answer"],
                "type": item["type"],
                "level": item["level"],
                "supporting_facts": {
                    "title": item["supporting_facts"]["title"],
                    "sent_id": item["supporting_facts"]["sent_id"],
                },
                "context": {
                    "title": item["context"]["title"],
                    "sentences": item["context"]["sentences"],
                },
            })
        with open(hotpot_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
        print(f"  Saved {len(data)} items to {hotpot_path}")
    else:
        print(f"  Already exists: {hotpot_path}")

    # 2. 2WikiMultiHopQA (Parquet format from framolfese)
    print("[Download] 2WikiMultiHopQA...")
    wiki2_path = DATASETS_DIR / "2wikimqa_validation.json"
    if not wiki2_path.exists():
        try:
            ds = load_dataset("framolfese/2WikiMultihopQA", split="validation")
            data = []
            for item in tqdm(ds, desc="2WikiMQA"):
                data.append({
                    "question": item.get("question", ""),
                    "answer": item.get("answer", ""),
                    "type": item.get("type", ""),
                    "context": item.get("context", ""),
                    "supporting_facts": item.get("supporting_facts", ""),
                    "evidences": item.get("evidences", ""),
                })
            with open(wiki2_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False)
            print(f"  Saved {len(data)} items to {wiki2_path}")
        except Exception as e:
            print(f"  Failed: {e}")
    else:
        print(f"  Already exists: {wiki2_path}")

    print("[Done] Dataset download complete.")


# ============================================================
# Oracle Evaluation
# ============================================================
def run_oracle_evaluation(
    model_name: str,
    dataset_name: str = "hotpotqa",
    max_samples: int = 200,
):
    """Run evaluation with oracle (ground-truth) context."""
    results_dir = TOPIC6_DIR / "experiments" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    result_file = results_dir / f"{model_name}_{dataset_name}_oracle.json"
    if result_file.exists():
        print(f"[Skip] {result_file}")
        with open(result_file, encoding="utf-8") as f:
            return json.load(f)

    # Load full dataset with context
    ds_path = DATASETS_DIR / f"{dataset_name}_full_validation.json"
    if not ds_path.exists():
        print(f"[Error] Full dataset not found: {ds_path}")
        print("Run: python setup_evaluation.py --download")
        return {}

    with open(ds_path, encoding="utf-8") as f:
        dataset = json.load(f)[:max_samples]

    # Load model
    model_info = GGUF_MODELS.get(model_name)
    if not model_info:
        print(f"[Error] Unknown model: {model_name}")
        return {}
    model_path = str(MODELS_DIR / model_info["file"])
    print(f"[Oracle] Loading {model_name}...")
    generator = LLMGenerator(model_path, n_ctx=2048, n_threads=4, n_gpu_layers=-1)

    results = []
    metrics = defaultdict(list)

    for item in tqdm(dataset, desc=f"Oracle/{model_name}/{dataset_name}"):
        question = item["question"]

        # Build oracle context from supporting facts
        oracle_ctx = _build_oracle_context(item)
        if not oracle_ctx:
            continue

        # Build prompt with oracle context
        context_str = "\n\n".join(
            f"[Document {i+1}]: {ctx}" for i, ctx in enumerate(oracle_ctx)
        )
        prompt = (
            f"Answer the following question based on the provided documents. "
            f"Give a short, direct answer.\n\n"
            f"{context_str}\n\n"
            f"Question: {question}\n"
            f"Answer:"
        )

        try:
            answer, prefill_ms, decode_ms, n_tokens = generator.generate(prompt)
            ev = evaluate_answer(answer, item)

            results.append({
                "question": question,
                "prediction": answer,
                "ground_truth": item["answer"],
                "em": ev["em"],
                "f1": ev["f1"],
                "n_oracle_docs": len(oracle_ctx),
                "oracle_tokens": sum(len(c.split()) for c in oracle_ctx),
            })
            metrics["em"].append(ev["em"])
            metrics["f1"].append(ev["f1"])
        except Exception as e:
            print(f"[Error] {e}")
            continue

    em_pct = round(sum(metrics["em"]) / len(metrics["em"]) * 100, 2) if metrics["em"] else 0
    f1_pct = round(sum(metrics["f1"]) / len(metrics["f1"]) * 100, 2) if metrics["f1"] else 0

    summary = {
        "model": model_name,
        "dataset": dataset_name,
        "mode": "oracle",
        "n_samples": len(results),
        "metrics": {"em": em_pct, "f1": f1_pct},
        "results": results,
    }

    with open(result_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"  EM: {em_pct}% | F1: {f1_pct}%")
    return summary


def _build_oracle_context(item: dict) -> list[str]:
    """Extract oracle (supporting fact) context from dataset item."""
    context = item.get("context", {})
    supporting = item.get("supporting_facts", {})

    if not context or not supporting:
        return []

    titles = context.get("title", [])
    sentences = context.get("sentences", [])
    sup_titles = supporting.get("title", [])
    sup_sent_ids = supporting.get("sent_id", [])

    # Build title -> sentences map
    title_to_sents = {}
    for i, title in enumerate(titles):
        if i < len(sentences):
            title_to_sents[title] = sentences[i]

    # Extract supporting sentences
    oracle_docs = []
    for sup_title, sent_id in zip(sup_titles, sup_sent_ids):
        sents = title_to_sents.get(sup_title, [])
        if sent_id < len(sents):
            oracle_docs.append(f"{sup_title}: {sents[sent_id]}")

    if not oracle_docs:
        # Fallback: use all context
        for title, sents in zip(titles, sentences):
            oracle_docs.append(f"{title}: {' '.join(sents[:3])}")

    return oracle_docs


# ============================================================
# Full Matrix Runner
# ============================================================
def run_full_matrix(max_samples: int = 200):
    """Run the full experiment matrix."""
    from run_baseline import run_benchmark

    models = ["lfm2.5-1.2b-instruct", "qwen3-0.6b", "qwen3-1.7b"]
    datasets = ["hotpotqa", "triviaqa"]
    modes = ["rag", "entity", "con", "compress"]

    # Check if 2wikimqa exists
    if (DATASETS_DIR / "2wikimqa_validation.json").exists():
        datasets.append("2wikimqa")

    results_summary = []

    # Regular modes
    for model in models:
        for dataset in datasets:
            for mode in modes:
                print(f"\n>>> {model} / {dataset} / {mode}")
                try:
                    result = run_benchmark(
                        model_name=model,
                        dataset_name=dataset,
                        mode=mode,
                        max_samples=max_samples,
                    )
                    if result:
                        m = result.get("metrics", {})
                        results_summary.append({
                            "model": model, "dataset": dataset, "mode": mode,
                            "em": m.get("em", 0), "f1": m.get("f1", 0),
                        })
                except Exception as e:
                    print(f"[Error] {e}")

    # Oracle (HotpotQA only, needs full context)
    for model in models:
        print(f"\n>>> {model} / hotpotqa / oracle")
        try:
            result = run_oracle_evaluation(model, "hotpotqa", max_samples)
            if result:
                m = result.get("metrics", {})
                results_summary.append({
                    "model": model, "dataset": "hotpotqa", "mode": "oracle",
                    "em": m.get("em", 0), "f1": m.get("f1", 0),
                })
        except Exception as e:
            print(f"[Error] {e}")

    # Print full table
    print(f"\n{'='*80}")
    print(f"{'Model':<25} {'Dataset':<12} {'Mode':<10} {'EM%':>6} {'F1%':>6}")
    print(f"{'='*80}")
    for r in sorted(results_summary, key=lambda x: (x["dataset"], x["model"], x["mode"])):
        print(f"{r['model']:<25} {r['dataset']:<12} {r['mode']:<10} "
              f"{r['em']:>6.1f} {r['f1']:>6.1f}")

    # Save summary
    summary_path = TOPIC6_DIR / "experiments" / "results" / "full_matrix_summary.json"
    with open(summary_path, "w") as f:
        json.dump(results_summary, f, indent=2)
    print(f"\n[Saved] {summary_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--download", action="store_true",
                        help="Download datasets with oracle context")
    parser.add_argument("--oracle-only", action="store_true",
                        help="Run oracle evaluation only")
    parser.add_argument("--run-all", action="store_true",
                        help="Run full experiment matrix")
    parser.add_argument("--max-samples", type=int, default=200)
    args = parser.parse_args()

    if args.download:
        download_datasets()
    elif args.oracle_only:
        for model in ["lfm2.5-1.2b-instruct", "qwen3-0.6b", "qwen3-1.7b"]:
            run_oracle_evaluation(model, "hotpotqa", args.max_samples)
    elif args.run_all:
        run_full_matrix(args.max_samples)
    else:
        parser.print_help()
