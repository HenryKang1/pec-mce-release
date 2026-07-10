"""
Topic 6 Baseline Experiment:
Run all 5 SLMs on QA benchmarks with and without RAG.
Measure accuracy (EM, F1) and latency (TTFT, decode speed, total).

Usage:
  python run_baseline.py --model qwen3-0.6b --dataset popqa --mode rag
  python run_baseline.py --model all --dataset all --mode all
"""
import argparse
import json
import re
import string
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Optional

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "shared"))
from utils.config import (
    GGUF_MODELS, MODELS_DIR, DATASETS_DIR,
    RAG_CONFIG, TOPIC6_DIR,
)

from rag_pipeline import ChunkIndex, LLMGenerator, RAGPipeline


# ============================================================
# Evaluation Metrics
# ============================================================
def normalize_answer(s: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace."""
    s = s.lower().strip()
    # Remove articles
    s = re.sub(r'\b(a|an|the)\b', ' ', s)
    # Remove punctuation
    s = s.translate(str.maketrans('', '', string.punctuation))
    # Remove extra whitespace
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
    """Evaluate a single prediction against ground truth."""
    answer = item.get("answer", "")
    aliases = item.get("aliases", [answer])

    # Check against all valid answers
    best_em = False
    best_f1 = 0.0
    for ans in aliases:
        if exact_match(prediction, ans):
            best_em = True
        best_f1 = max(best_f1, f1_score(prediction, ans))

    return {"em": best_em, "f1": best_f1}


# ============================================================
# Wikipedia Corpus Loader (simplified for initial experiments)
# ============================================================
def load_or_build_wiki_index(index_path: Path, max_chunks: int = 100000) -> ChunkIndex:
    """Load existing index or build a lightweight one from dataset contexts."""
    idx = ChunkIndex()

    if (index_path / "index.faiss").exists():
        idx.load(index_path)
        return idx

    # For initial experiments, build index from HotpotQA contexts
    # (contains Wikipedia paragraphs as supporting documents)
    print("[Wiki Index] Building from HotpotQA contexts...")
    hotpot_path = DATASETS_DIR / "hotpotqa_validation.json"
    chunks = set()

    if hotpot_path.exists():
        with open(hotpot_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        # Note: We need the full HotpotQA dataset with contexts
        # For now, use a simple approach
        print(f"[Wiki Index] HotpotQA has {len(data)} items (questions only)")

    # Fallback: build a small test index
    # In full experiment, use Wikipedia dump
    print("[Wiki Index] Using Wikipedia dump is recommended for full experiments.")
    print("[Wiki Index] For now, building a minimal test index...")

    # Download a small Wikipedia subset for testing
    try:
        from datasets import load_dataset
        print("[Wiki Index] Loading wikipedia subset...")
        wiki = load_dataset("wikipedia", "20220301.en",
                            split="train", streaming=True,
                            trust_remote_code=True)

        for i, article in enumerate(wiki):
            if i >= 5000:  # 5000 articles for initial testing
                break
            text = article.get("text", "")
            # Simple chunking
            words = text.split()
            for j in range(0, len(words), 80):
                chunk = " ".join(words[j:j+100])
                if len(chunk) > 50:
                    chunks.add(chunk)
                    if len(chunks) >= max_chunks:
                        break
            if len(chunks) >= max_chunks:
                break

        chunk_list = list(chunks)[:max_chunks]
        print(f"[Wiki Index] Got {len(chunk_list)} chunks from Wikipedia")
        idx.build_from_chunks(chunk_list)
        idx.save(index_path)

    except Exception as e:
        print(f"[Wiki Index] Wikipedia download failed: {e}")
        print("[Wiki Index] Creating minimal dummy index for testing...")
        dummy = ["This is a test document."] * 10
        idx.build_from_chunks(dummy)

    return idx


# ============================================================
# Main Benchmark Runner
# ============================================================
def run_benchmark(
    model_name: str,
    dataset_name: str,
    mode: str,  # "vanilla", "rag", "con", "compiled", "compiled_con"
    max_samples: int = 500,
    top_k: int = RAG_CONFIG["top_k"],
    output_dir: Optional[Path] = None,
    index_type: str = "raw",  # "raw" or "compiled"
    entity_dir_name: Optional[str] = None,  # e.g. "hotpotqa_entity_index_qwen06"
    raw_dir_name: Optional[str] = None,  # e.g. "hotpotqa_chunk_index" for granularity ablation
    result_tag: Optional[str] = None,  # e.g. "compileQwen06"; affects only filename
) -> dict:
    """Run benchmark for a single model-dataset-mode combination."""

    if output_dir is None:
        output_dir = TOPIC6_DIR / "experiments" / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    tag = f"_{result_tag}" if result_tag else ""
    result_file = output_dir / f"{model_name}_{dataset_name}_{mode}{tag}.json"
    if result_file.exists():
        print(f"[Skip] Already exists: {result_file}")
        with open(result_file, "r") as f:
            return json.load(f)

    # Load dataset
    ds_path = DATASETS_DIR / f"{dataset_name}_validation.json"
    if not ds_path.exists():
        ds_path = DATASETS_DIR / f"{dataset_name}_test.json"
    if not ds_path.exists():
        print(f"[Error] Dataset not found: {ds_path}")
        return {}

    with open(ds_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    if max_samples and len(dataset) > max_samples:
        dataset = dataset[:max_samples]

    print(f"\n{'='*60}")
    print(f"Model: {model_name} | Dataset: {dataset_name} | Mode: {mode}")
    print(f"Samples: {len(dataset)} | Top-k: {top_k}")
    print(f"{'='*60}")

    # Load model
    model_info = GGUF_MODELS.get(model_name)
    if not model_info:
        print(f"[Error] Unknown model: {model_name}")
        return {}

    model_path = MODELS_DIR / model_info["file"]
    if not model_path.exists():
        print(f"[Error] Model file not found: {model_path}")
        print(f"Run: python shared/scripts/download_models.py")
        return {}

    generator = LLMGenerator(
        str(model_path),
        n_ctx=2048,
        n_threads=4,
        n_gpu_layers=-1,  # all layers on GPU
    )

    # Build RAG pipeline if needed
    pipeline = None
    actual_mode = mode  # mode passed to pipeline.query()
    if mode in ("rag", "con", "compress"):
        # Default: hotpotqa_raw_index / 2wikimqa_raw_index if available,
        # else wiki_index. Allow override via --raw-dir.
        if raw_dir_name:
            index_path = TOPIC6_DIR / "experiments" / "cache" / raw_dir_name
            if not (index_path / "index.faiss").exists():
                print(f"[Error] Raw index not found: {index_path}")
                return {}
            print(f"[Raw] Using override index: {index_path}")
            index = ChunkIndex()
            index.load(index_path)
        else:
            dataset_raw_paths = {
                "hotpotqa": TOPIC6_DIR / "experiments" / "cache" / "hotpotqa_raw_index",
                "2wikimqa": TOPIC6_DIR / "experiments" / "cache" / "2wikimqa_raw_index",
            }
            ds_raw = dataset_raw_paths.get(dataset_name)
            if ds_raw and (ds_raw / "index.faiss").exists():
                print(f"[Raw] Using dataset-specific index: {ds_raw}")
                index = ChunkIndex()
                index.load(ds_raw)
            else:
                index_path = TOPIC6_DIR / "experiments" / "cache" / "wiki_index"
                index = load_or_build_wiki_index(index_path)
        pipeline = RAGPipeline(index, generator, top_k=top_k)
    elif mode in ("compiled", "compiled_con"):
        index_path = TOPIC6_DIR / "experiments" / "cache" / "compiled_wiki_index"
        if not (index_path / "index.faiss").exists():
            print(f"[Error] Compiled index not found at {index_path}")
            print("Run: python compile_wiki_index.py --max-chunks 400000")
            return {}
        index = ChunkIndex()
        index.load(index_path)
        pipeline = RAGPipeline(index, generator, top_k=top_k)
        actual_mode = "con" if mode == "compiled_con" else "rag"
    elif mode in ("entity", "entity_con"):
        # Allow overriding the entity index dir (for Matching Matrix experiments)
        if entity_dir_name:
            override_path = TOPIC6_DIR / "experiments" / "cache" / entity_dir_name
            if not (override_path / "index.faiss").exists():
                print(f"[Error] Entity index not found at override: {override_path}")
                return {}
            index_path = override_path
        else:
            # Prefer dataset-specific entity index (100% coverage) if available
            dataset_entity_paths = {
                "hotpotqa": TOPIC6_DIR / "experiments" / "cache" / "hotpotqa_entity_index",
                "2wikimqa": TOPIC6_DIR / "experiments" / "cache" / "2wikimqa_entity_index",
            }
            wiki_entity_path = TOPIC6_DIR / "experiments" / "cache" / "entity_wiki_index"
            ds_path = dataset_entity_paths.get(dataset_name)
            if ds_path and (ds_path / "index.faiss").exists():
                index_path = ds_path
            elif (wiki_entity_path / "index.faiss").exists():
                index_path = wiki_entity_path
            else:
                print(f"[Error] Entity index not found")
                print("Run: python compile_hotpotqa_entities.py or compile_2wikimqa_entities.py")
                return {}
        print(f"[Entity] Using index: {index_path}")
        index = ChunkIndex()
        index.load(index_path)
        pipeline = RAGPipeline(index, generator, top_k=top_k)
        actual_mode = "con" if mode == "entity_con" else "rag"

    # Run evaluation
    results = []
    metrics = defaultdict(list)

    for item in tqdm(dataset, desc=f"{model_name}/{dataset_name}/{mode}"):
        question = item["question"]

        try:
            if mode in ("rag", "con", "compress", "compiled", "compiled_con", "entity", "entity_con") and pipeline:
                answer, profile = pipeline.query(question, top_k=top_k, mode=actual_mode)
            else:
                if pipeline:
                    answer, profile = pipeline.query_vanilla(question)
                else:
                    # Create temp pipeline for vanilla
                    _idx = ChunkIndex()
                    _idx.build_from_chunks(["placeholder"])
                    _pipe = RAGPipeline(_idx, generator)
                    answer, profile = _pipe.query_vanilla(question)

            eval_result = evaluate_answer(answer, item)

            results.append({
                "question": question,
                "prediction": answer,
                "ground_truth": item["answer"],
                "em": eval_result["em"],
                "f1": eval_result["f1"],
                "latency": profile.to_dict(),
            })

            metrics["em"].append(eval_result["em"])
            metrics["f1"].append(eval_result["f1"])
            metrics["ttft_ms"].append(profile.ttft_ms)
            metrics["total_ms"].append(profile.total_time_ms)
            metrics["tps"].append(profile.tokens_per_sec)

        except Exception as e:
            print(f"[Error] {question[:50]}...: {e}")
            continue

    # Aggregate
    summary = {
        "model": model_name,
        "dataset": dataset_name,
        "mode": mode,
        "n_samples": len(results),
        "top_k": top_k,
        "metrics": {
            "em": round(sum(metrics["em"]) / len(metrics["em"]) * 100, 2) if metrics["em"] else 0,
            "f1": round(sum(metrics["f1"]) / len(metrics["f1"]) * 100, 2) if metrics["f1"] else 0,
            "avg_ttft_ms": round(sum(metrics["ttft_ms"]) / len(metrics["ttft_ms"]), 2) if metrics["ttft_ms"] else 0,
            "avg_total_ms": round(sum(metrics["total_ms"]) / len(metrics["total_ms"]), 2) if metrics["total_ms"] else 0,
            "avg_tps": round(sum(metrics["tps"]) / len(metrics["tps"]), 2) if metrics["tps"] else 0,
            "p50_ttft_ms": round(sorted(metrics["ttft_ms"])[len(metrics["ttft_ms"])//2], 2) if metrics["ttft_ms"] else 0,
            "p95_ttft_ms": round(sorted(metrics["ttft_ms"])[int(len(metrics["ttft_ms"])*0.95)], 2) if metrics["ttft_ms"] else 0,
        },
        "results": results,
    }

    # Save
    with open(result_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n[Saved] {result_file}")
    print(f"  EM: {summary['metrics']['em']}%")
    print(f"  F1: {summary['metrics']['f1']}%")
    print(f"  Avg TTFT: {summary['metrics']['avg_ttft_ms']}ms")
    print(f"  Avg TPS: {summary['metrics']['avg_tps']}")

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="qwen3-0.6b",
                        help="Model name or 'all'")
    parser.add_argument("--dataset", default="popqa",
                        help="Dataset name or 'all'")
    parser.add_argument("--mode", default="vanilla",
                        choices=["vanilla", "rag", "con", "compress", "compiled", "compiled_con",
                                 "entity", "entity_con", "all"])
    parser.add_argument("--max-samples", type=int, default=500)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--entity-dir", default=None,
                        help="Override entity index dir name (e.g. hotpotqa_entity_index_qwen06)")
    parser.add_argument("--raw-dir", default=None,
                        help="Override raw index dir name (e.g. hotpotqa_chunk_index for granularity)")
    parser.add_argument("--result-tag", default=None,
                        help="Tag appended to result filename to avoid collisions")
    args = parser.parse_args()

    models = list(GGUF_MODELS.keys()) if args.model == "all" else [args.model]
    datasets = ["popqa", "triviaqa", "hotpotqa"] if args.dataset == "all" else [args.dataset]
    modes = ["vanilla", "rag", "con", "compress", "compiled", "compiled_con", "entity", "entity_con"] if args.mode == "all" else [args.mode]

    all_results = []
    for model in models:
        for dataset in datasets:
            for mode in modes:
                result = run_benchmark(
                    model_name=model,
                    dataset_name=dataset,
                    mode=mode,
                    max_samples=args.max_samples,
                    top_k=args.top_k,
                    entity_dir_name=args.entity_dir,
                    raw_dir_name=args.raw_dir,
                    result_tag=args.result_tag,
                )
                if result:
                    all_results.append({
                        "model": model,
                        "dataset": dataset,
                        "mode": mode,
                        "em": result.get("metrics", {}).get("em", 0),
                        "f1": result.get("metrics", {}).get("f1", 0),
                        "avg_ttft_ms": result.get("metrics", {}).get("avg_ttft_ms", 0),
                    })

    # Print summary table
    if all_results:
        print(f"\n{'='*80}")
        print(f"{'Model':<25} {'Dataset':<12} {'Mode':<8} {'EM%':>6} {'F1%':>6} {'TTFT(ms)':>10}")
        print(f"{'='*80}")
        for r in all_results:
            print(f"{r['model']:<25} {r['dataset']:<12} {r['mode']:<8} "
                  f"{r['em']:>6.1f} {r['f1']:>6.1f} {r['avg_ttft_ms']:>10.1f}")
