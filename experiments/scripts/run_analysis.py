"""
논문 Analysis/Ablation 실험:
  1. Top-k 변화 실험 (k=1,3,5,10,15)
  2. Latency 분해 (embedding, retrieval, generation)
  3. Same-token-budget 비교
  4. Significance test (paired bootstrap)

Usage:
  python run_analysis.py --all                          # 전부 실행
  python run_analysis.py --topk                         # k 변화 실험
  python run_analysis.py --latency                      # latency 분해
  python run_analysis.py --budget                       # same-budget 비교
  python run_analysis.py --significance                 # significance test
  python run_analysis.py --model qwen3-1.7b --topk      # 특정 모델
"""
import argparse
import json
import sys
import time
import random
import re
import string
from collections import defaultdict
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "shared"))
from utils.config import (
    GGUF_MODELS, MODELS_DIR, DATASETS_DIR, TOPIC6_DIR,
)
from rag_pipeline import ChunkIndex, LLMGenerator, RAGPipeline, LatencyProfile

RESULTS_DIR = TOPIC6_DIR / "experiments" / "results"
CACHE_DIR = TOPIC6_DIR / "experiments" / "cache"


# ============================================================
# Evaluation helpers
# ============================================================
def normalize_answer(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r'\b(a|an|the)\b', ' ', s)
    s = s.translate(str.maketrans('', '', string.punctuation))
    s = ' '.join(s.split())
    return s

def exact_match(pred: str, gt: str) -> bool:
    return normalize_answer(pred) == normalize_answer(gt)

def f1_score(pred: str, gt: str) -> float:
    pt = normalize_answer(pred).split()
    gt_t = normalize_answer(gt).split()
    if not pt or not gt_t:
        return float(pt == gt_t)
    common = set(pt) & set(gt_t)
    if not common:
        return 0.0
    p = len(common) / len(pt)
    r = len(common) / len(gt_t)
    return 2 * p * r / (p + r)

def evaluate_answer(pred: str, item: dict) -> dict:
    answer = item.get("answer", "")
    aliases = item.get("aliases", [answer])
    best_em, best_f1 = False, 0.0
    for ans in aliases:
        if exact_match(pred, ans):
            best_em = True
        best_f1 = max(best_f1, f1_score(pred, ans))
    return {"em": best_em, "f1": best_f1}

def load_dataset(name, max_samples=200):
    for suffix in ["_validation.json", "_test.json", "_full_validation.json"]:
        p = DATASETS_DIR / f"{name}{suffix}"
        if p.exists():
            with open(p, encoding="utf-8") as f:
                data = json.load(f)
            return data[:max_samples]
    return []


# ============================================================
# 1. Top-k 변화 실험
# ============================================================
def run_topk_experiment(model_name="lfm2.5-1.2b-instruct", dataset="hotpotqa",
                        max_samples=200, ks=[1, 3, 5, 10, 15]):
    """k값에 따른 RAG vs CTKS 성능 비교."""
    print(f"\n{'='*60}")
    print(f"  TOP-K EXPERIMENT: {model_name} / {dataset}")
    print(f"{'='*60}")

    result_file = RESULTS_DIR / f"analysis_topk_{model_name}_{dataset}.json"
    if result_file.exists():
        print(f"[Skip] {result_file}")
        with open(result_file, encoding="utf-8") as f:
            return json.load(f)

    data = load_dataset(dataset, max_samples)
    if not data:
        print("[Error] Dataset not found")
        return {}

    model_info = GGUF_MODELS[model_name]
    model_path = str(MODELS_DIR / model_info["file"])
    generator = LLMGenerator(model_path, n_ctx=2048, n_threads=4, n_gpu_layers=-1)

    # Load both indexes
    rag_index_path = CACHE_DIR / f"{dataset}_raw_index"
    entity_index_path = CACHE_DIR / f"{dataset}_entity_index"

    if not (rag_index_path / "index.faiss").exists():
        rag_index_path = CACHE_DIR / "wiki_index"
    if not (entity_index_path / "index.faiss").exists():
        entity_index_path = CACHE_DIR / "entity_wiki_index"

    rag_index = ChunkIndex()
    rag_index.load(rag_index_path)
    entity_index = ChunkIndex()
    entity_index.load(entity_index_path)

    results = {"model": model_name, "dataset": dataset, "ks": {}}

    for k in ks:
        print(f"\n--- k={k} ---")
        rag_pipe = RAGPipeline(rag_index, generator, top_k=k)
        ent_pipe = RAGPipeline(entity_index, generator, top_k=k)

        for mode_name, pipeline in [("rag", rag_pipe), ("entity", ent_pipe)]:
            ems, f1s, latencies = [], [], []
            for item in tqdm(data, desc=f"k={k}/{mode_name}"):
                try:
                    ans, profile = pipeline.query(item["question"], top_k=k)
                    ev = evaluate_answer(ans, item)
                    ems.append(ev["em"])
                    f1s.append(ev["f1"])
                    latencies.append(profile.total_time_ms)
                except Exception:
                    continue

            em_pct = sum(ems) / len(ems) * 100 if ems else 0
            f1_pct = sum(f1s) / len(f1s) * 100 if f1s else 0
            avg_ms = sum(latencies) / len(latencies) if latencies else 0

            key = f"k{k}_{mode_name}"
            results["ks"][key] = {"k": k, "mode": mode_name, "em": round(em_pct, 2),
                                  "f1": round(f1_pct, 2), "avg_ms": round(avg_ms, 1),
                                  "n": len(ems)}
            print(f"  {mode_name:8s} EM: {em_pct:.1f}%  F1: {f1_pct:.1f}%  ms: {avg_ms:.0f}")

    with open(result_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n[Saved] {result_file}")

    # Print comparison table
    print(f"\n{'k':>4} {'RAG EM':>8} {'CTKS EM':>9} {'Delta':>7} {'RAG ms':>8} {'CTKS ms':>9}")
    print("-" * 50)
    for k in ks:
        rag_r = results["ks"].get(f"k{k}_rag", {})
        ent_r = results["ks"].get(f"k{k}_entity", {})
        delta = ent_r.get("em", 0) - rag_r.get("em", 0)
        print(f"{k:>4} {rag_r.get('em',0):>7.1f}% {ent_r.get('em',0):>8.1f}% {delta:>+6.1f} "
              f"{rag_r.get('avg_ms',0):>7.0f}ms {ent_r.get('avg_ms',0):>8.0f}ms")

    return results


# ============================================================
# 2. Latency 분해
# ============================================================
def run_latency_breakdown(model_name="lfm2.5-1.2b-instruct", dataset="hotpotqa",
                          max_samples=100):
    """Latency를 embedding/retrieval/generation으로 분해."""
    print(f"\n{'='*60}")
    print(f"  LATENCY BREAKDOWN: {model_name} / {dataset}")
    print(f"{'='*60}")

    result_file = RESULTS_DIR / f"analysis_latency_{model_name}_{dataset}.json"
    if result_file.exists():
        print(f"[Skip] {result_file}")
        with open(result_file, encoding="utf-8") as f:
            return json.load(f)

    data = load_dataset(dataset, max_samples)
    if not data:
        return {}

    model_info = GGUF_MODELS[model_name]
    model_path = str(MODELS_DIR / model_info["file"])
    generator = LLMGenerator(model_path, n_ctx=2048, n_threads=4, n_gpu_layers=-1)

    modes = {}

    # RAG
    rag_path = CACHE_DIR / "wiki_index"
    if dataset == "hotpotqa" and (CACHE_DIR / "hotpotqa_raw_index" / "index.faiss").exists():
        rag_path = CACHE_DIR / "hotpotqa_raw_index"
    rag_idx = ChunkIndex()
    rag_idx.load(rag_path)
    modes["rag"] = RAGPipeline(rag_idx, generator)

    # Entity
    ent_path = CACHE_DIR / f"{dataset}_entity_index"
    if not (ent_path / "index.faiss").exists():
        ent_path = CACHE_DIR / "entity_wiki_index"
    if (ent_path / "index.faiss").exists():
        ent_idx = ChunkIndex()
        ent_idx.load(ent_path)
        modes["entity"] = RAGPipeline(ent_idx, generator)

    results = {"model": model_name, "dataset": dataset, "breakdown": {}}

    for mode_name, pipeline in modes.items():
        embed_times, retr_times, gen_times, total_times = [], [], [], []
        ctx_tokens_list = []

        for item in tqdm(data[:max_samples], desc=f"Latency/{mode_name}"):
            try:
                ans, profile = pipeline.query(item["question"])
                embed_times.append(profile.embedding_time_ms)
                retr_times.append(profile.retrieval_time_ms)
                gen_times.append(profile.prefill_time_ms + profile.decode_time_ms)
                total_times.append(profile.total_time_ms)
                ctx_tokens_list.append(profile.context_tokens)
            except Exception:
                continue

        n = len(embed_times)
        if n == 0:
            continue

        breakdown = {
            "mode": mode_name,
            "n": n,
            "avg_embed_ms": round(sum(embed_times) / n, 2),
            "avg_retrieval_ms": round(sum(retr_times) / n, 2),
            "avg_generation_ms": round(sum(gen_times) / n, 2),
            "avg_total_ms": round(sum(total_times) / n, 2),
            "avg_ctx_tokens": round(sum(ctx_tokens_list) / n, 1) if any(ctx_tokens_list) else 0,
        }
        results["breakdown"][mode_name] = breakdown

        print(f"\n  [{mode_name.upper()}]")
        print(f"    Embedding:  {breakdown['avg_embed_ms']:>7.1f}ms")
        print(f"    Retrieval:  {breakdown['avg_retrieval_ms']:>7.1f}ms")
        print(f"    Generation: {breakdown['avg_generation_ms']:>7.1f}ms")
        print(f"    Total:      {breakdown['avg_total_ms']:>7.1f}ms")

    with open(result_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n[Saved] {result_file}")
    return results


# ============================================================
# 3. Same-token-budget 비교
# ============================================================
def run_budget_experiment(model_name="lfm2.5-1.2b-instruct", dataset="hotpotqa",
                          max_samples=200):
    """동일 토큰 예산에서 RAG vs CTKS 비교.
    RAG: k=5 → ~500 tokens context
    CTKS k=5 → ~350 tokens context
    Fair: RAG k=3 ≈ CTKS k=5 in tokens
    """
    print(f"\n{'='*60}")
    print(f"  SAME-BUDGET EXPERIMENT: {model_name} / {dataset}")
    print(f"{'='*60}")

    result_file = RESULTS_DIR / f"analysis_budget_{model_name}_{dataset}.json"
    if result_file.exists():
        print(f"[Skip] {result_file}")
        with open(result_file, encoding="utf-8") as f:
            return json.load(f)

    data = load_dataset(dataset, max_samples)
    if not data:
        return {}

    model_info = GGUF_MODELS[model_name]
    model_path = str(MODELS_DIR / model_info["file"])
    generator = LLMGenerator(model_path, n_ctx=2048, n_threads=4, n_gpu_layers=-1)

    # Load indexes
    rag_path = CACHE_DIR / "wiki_index"
    if dataset == "hotpotqa" and (CACHE_DIR / "hotpotqa_raw_index" / "index.faiss").exists():
        rag_path = CACHE_DIR / "hotpotqa_raw_index"
    rag_idx = ChunkIndex()
    rag_idx.load(rag_path)

    ent_path = CACHE_DIR / f"{dataset}_entity_index"
    if not (ent_path / "index.faiss").exists():
        ent_path = CACHE_DIR / "entity_wiki_index"
    ent_idx = ChunkIndex()
    ent_idx.load(ent_path)

    # Conditions: vary k to match approximate token budgets
    conditions = [
        ("rag_k3",    rag_idx, 3, "rag"),
        ("rag_k5",    rag_idx, 5, "rag"),
        ("rag_k10",   rag_idx, 10, "rag"),
        ("ctks_k3",   ent_idx, 3, "rag"),
        ("ctks_k5",   ent_idx, 5, "rag"),
        ("ctks_k10",  ent_idx, 10, "rag"),
    ]

    results = {"model": model_name, "dataset": dataset, "conditions": {}}

    for cond_name, index, k, mode in conditions:
        pipeline = RAGPipeline(index, generator, top_k=k)
        ems, f1s, token_counts, latencies = [], [], [], []

        for item in tqdm(data, desc=cond_name):
            try:
                # Get retrieved chunks to count tokens
                chunks, scores, _ = index.search(item["question"], top_k=k)
                total_tokens = sum(len(c.split()) for c in chunks)
                token_counts.append(total_tokens)

                ans, profile = pipeline.query(item["question"], top_k=k, mode=mode)
                ev = evaluate_answer(ans, item)
                ems.append(ev["em"])
                f1s.append(ev["f1"])
                latencies.append(profile.total_time_ms)
            except Exception:
                continue

        n = len(ems)
        if n == 0:
            continue

        cond_result = {
            "name": cond_name, "k": k, "n": n,
            "em": round(sum(ems) / n * 100, 2),
            "f1": round(sum(f1s) / n * 100, 2),
            "avg_tokens": round(sum(token_counts) / n, 1),
            "avg_ms": round(sum(latencies) / n, 1),
        }
        results["conditions"][cond_name] = cond_result
        print(f"  {cond_name:12s} k={k:>2} tokens={cond_result['avg_tokens']:>6.0f} "
              f"EM={cond_result['em']:>5.1f}% ms={cond_result['avg_ms']:>6.0f}")

    with open(result_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n[Saved] {result_file}")

    # Print comparison
    print(f"\n  Same-budget comparison:")
    print(f"  {'Condition':<14} {'k':>3} {'Tokens':>8} {'EM%':>6} {'F1%':>6} {'ms':>7}")
    print(f"  {'-'*50}")
    for name in ["rag_k3", "ctks_k5", "rag_k5", "ctks_k10", "rag_k10"]:
        c = results["conditions"].get(name, {})
        if c:
            print(f"  {name:<14} {c['k']:>3} {c['avg_tokens']:>7.0f} "
                  f"{c['em']:>5.1f}% {c['f1']:>5.1f}% {c['avg_ms']:>6.0f}")

    return results


# ============================================================
# 4. Significance test (paired bootstrap)
# ============================================================
def run_significance_test(model_name="lfm2.5-1.2b-instruct", dataset="hotpotqa",
                          n_bootstrap=10000):
    """Paired bootstrap test: CTKS vs RAG 차이가 통계적으로 유의한가."""
    print(f"\n{'='*60}")
    print(f"  SIGNIFICANCE TEST: {model_name} / {dataset}")
    print(f"{'='*60}")

    # Load result files
    rag_file = RESULTS_DIR / f"{model_name}_{dataset}_rag.json"
    entity_file = RESULTS_DIR / f"{model_name}_{dataset}_entity.json"

    if not rag_file.exists() or not entity_file.exists():
        print(f"[Error] Need both {rag_file.name} and {entity_file.name}")
        return {}

    with open(rag_file, encoding="utf-8") as f:
        rag_data = json.load(f)
    with open(entity_file, encoding="utf-8") as f:
        entity_data = json.load(f)

    rag_results = rag_data.get("results", [])
    ent_results = entity_data.get("results", [])

    # Match by question
    rag_by_q = {r["question"]: r for r in rag_results}
    ent_by_q = {r["question"]: r for r in ent_results}

    common_qs = set(rag_by_q.keys()) & set(ent_by_q.keys())
    print(f"  Common questions: {len(common_qs)}")

    if len(common_qs) < 50:
        print("[Error] Too few common questions for significance test")
        return {}

    rag_ems = [int(rag_by_q[q]["em"]) for q in common_qs]
    ent_ems = [int(ent_by_q[q]["em"]) for q in common_qs]
    rag_f1s = [rag_by_q[q]["f1"] for q in common_qs]
    ent_f1s = [ent_by_q[q]["f1"] for q in common_qs]

    n = len(rag_ems)
    observed_em_diff = sum(ent_ems) / n - sum(rag_ems) / n
    observed_f1_diff = sum(ent_f1s) / n - sum(rag_f1s) / n

    # Paired bootstrap
    random.seed(42)
    em_diffs, f1_diffs = [], []
    for _ in range(n_bootstrap):
        indices = [random.randint(0, n - 1) for _ in range(n)]
        boot_rag_em = sum(rag_ems[i] for i in indices) / n
        boot_ent_em = sum(ent_ems[i] for i in indices) / n
        boot_rag_f1 = sum(rag_f1s[i] for i in indices) / n
        boot_ent_f1 = sum(ent_f1s[i] for i in indices) / n
        em_diffs.append(boot_ent_em - boot_rag_em)
        f1_diffs.append(boot_ent_f1 - boot_rag_f1)

    em_diffs.sort()
    f1_diffs.sort()

    def ci(diffs, level=0.95):
        lo = diffs[int(len(diffs) * (1 - level) / 2)]
        hi = diffs[int(len(diffs) * (1 + level) / 2)]
        p_value = sum(1 for d in diffs if d <= 0) / len(diffs)
        return lo, hi, p_value

    em_lo, em_hi, em_p = ci(em_diffs)
    f1_lo, f1_hi, f1_p = ci(f1_diffs)

    results = {
        "model": model_name, "dataset": dataset, "n": n,
        "n_bootstrap": n_bootstrap,
        "em": {
            "rag": round(sum(rag_ems) / n * 100, 2),
            "ctks": round(sum(ent_ems) / n * 100, 2),
            "diff": round(observed_em_diff * 100, 2),
            "ci_95_lo": round(em_lo * 100, 2),
            "ci_95_hi": round(em_hi * 100, 2),
            "p_value": round(em_p, 4),
        },
        "f1": {
            "rag": round(sum(rag_f1s) / n * 100, 2),
            "ctks": round(sum(ent_f1s) / n * 100, 2),
            "diff": round(observed_f1_diff * 100, 2),
            "ci_95_lo": round(f1_lo * 100, 2),
            "ci_95_hi": round(f1_hi * 100, 2),
            "p_value": round(f1_p, 4),
        },
    }

    sig_em = "***" if em_p < 0.001 else "**" if em_p < 0.01 else "*" if em_p < 0.05 else "n.s."
    sig_f1 = "***" if f1_p < 0.001 else "**" if f1_p < 0.01 else "*" if f1_p < 0.05 else "n.s."

    print(f"\n  n={n}, bootstrap={n_bootstrap}")
    print(f"\n  EM:  RAG {results['em']['rag']:.1f}% → CTKS {results['em']['ctks']:.1f}% "
          f"(+{results['em']['diff']:.1f})")
    print(f"       95% CI: [{results['em']['ci_95_lo']:.2f}, {results['em']['ci_95_hi']:.2f}]")
    print(f"       p={em_p:.4f} {sig_em}")
    print(f"\n  F1:  RAG {results['f1']['rag']:.1f}% → CTKS {results['f1']['ctks']:.1f}% "
          f"(+{results['f1']['diff']:.1f})")
    print(f"       95% CI: [{results['f1']['ci_95_lo']:.2f}, {results['f1']['ci_95_hi']:.2f}]")
    print(f"       p={f1_p:.4f} {sig_f1}")

    result_file = RESULTS_DIR / f"analysis_significance_{model_name}_{dataset}.json"
    with open(result_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n[Saved] {result_file}")

    return results


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CTKS Analysis Experiments")
    parser.add_argument("--model", default="lfm2.5-1.2b-instruct")
    parser.add_argument("--dataset", default="hotpotqa")
    parser.add_argument("--max-samples", type=int, default=200)
    parser.add_argument("--topk", action="store_true", help="Run top-k experiment")
    parser.add_argument("--latency", action="store_true", help="Run latency breakdown")
    parser.add_argument("--budget", action="store_true", help="Run same-budget experiment")
    parser.add_argument("--significance", action="store_true", help="Run significance test")
    parser.add_argument("--all", action="store_true", help="Run all analyses")
    args = parser.parse_args()

    if args.all or args.significance:
        # Significance test uses existing full results (no GPU needed)
        run_significance_test(args.model, args.dataset)
        # Also test other models if results exist
        for m in ["qwen3-0.6b", "qwen3-1.7b", "lfm2.5-1.2b-instruct"]:
            if m != args.model:
                rag_f = RESULTS_DIR / f"{m}_{args.dataset}_rag.json"
                ent_f = RESULTS_DIR / f"{m}_{args.dataset}_entity.json"
                if rag_f.exists() and ent_f.exists():
                    run_significance_test(m, args.dataset)

    if args.all or args.latency:
        run_latency_breakdown(args.model, args.dataset, max_samples=min(args.max_samples, 100))

    if args.all or args.topk:
        run_topk_experiment(args.model, args.dataset, max_samples=args.max_samples)

    if args.all or args.budget:
        run_budget_experiment(args.model, args.dataset, max_samples=args.max_samples)

    if not any([args.all, args.topk, args.latency, args.budget, args.significance]):
        parser.print_help()
