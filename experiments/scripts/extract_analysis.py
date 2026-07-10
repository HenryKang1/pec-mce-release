"""
기존 결과 JSON 에서 분석 데이터 추출 (GPU 불필요):
1. Significance test (paired bootstrap) — 기존 full results 사용
2. Latency breakdown — 기존 per-query latency 사용
3. Context token 통계 — 기존 per-query context_tokens 사용

Usage:
  python extract_analysis.py
"""
import json
import random
import statistics
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"


# ============================================================
# 1. Paired bootstrap significance test
# ============================================================
def paired_bootstrap(model, dataset, n_bootstrap=10000):
    rag_file = RESULTS_DIR / f"{model}_{dataset}_rag.json"
    ent_file = RESULTS_DIR / f"{model}_{dataset}_entity.json"

    if not rag_file.exists() or not ent_file.exists():
        return None

    with open(rag_file, encoding="utf-8") as f:
        rag_data = json.load(f)
    with open(ent_file, encoding="utf-8") as f:
        ent_data = json.load(f)

    rag_by_q = {r["question"]: r for r in rag_data["results"]}
    ent_by_q = {r["question"]: r for r in ent_data["results"]}
    common = sorted(set(rag_by_q) & set(ent_by_q))

    if len(common) < 50:
        return None

    rag_ems = [int(rag_by_q[q]["em"]) for q in common]
    ent_ems = [int(ent_by_q[q]["em"]) for q in common]
    rag_f1s = [rag_by_q[q]["f1"] for q in common]
    ent_f1s = [ent_by_q[q]["f1"] for q in common]

    n = len(common)
    obs_em = sum(ent_ems) / n - sum(rag_ems) / n
    obs_f1 = sum(ent_f1s) / n - sum(rag_f1s) / n

    random.seed(42)
    em_diffs, f1_diffs = [], []
    for _ in range(n_bootstrap):
        idx = [random.randint(0, n - 1) for _ in range(n)]
        em_diffs.append(sum(ent_ems[i] for i in idx) / n - sum(rag_ems[i] for i in idx) / n)
        f1_diffs.append(sum(ent_f1s[i] for i in idx) / n - sum(rag_f1s[i] for i in idx) / n)

    em_diffs.sort()
    f1_diffs.sort()

    def ci(diffs):
        lo = diffs[int(len(diffs) * 0.025)]
        hi = diffs[int(len(diffs) * 0.975)]
        p = sum(1 for d in diffs if d <= 0) / len(diffs)
        return lo, hi, p

    em_lo, em_hi, em_p = ci(em_diffs)
    f1_lo, f1_hi, f1_p = ci(f1_diffs)

    return {
        "model": model, "dataset": dataset, "n": n,
        "em_rag": round(sum(rag_ems) / n * 100, 2),
        "em_ctks": round(sum(ent_ems) / n * 100, 2),
        "em_diff": round(obs_em * 100, 2),
        "em_ci": (round(em_lo * 100, 2), round(em_hi * 100, 2)),
        "em_p": em_p,
        "f1_rag": round(sum(rag_f1s) / n * 100, 2),
        "f1_ctks": round(sum(ent_f1s) / n * 100, 2),
        "f1_diff": round(obs_f1 * 100, 2),
        "f1_ci": (round(f1_lo * 100, 2), round(f1_hi * 100, 2)),
        "f1_p": f1_p,
    }


# ============================================================
# 2. Latency breakdown from existing results
# ============================================================
def latency_breakdown(model, dataset):
    results = {}
    for mode in ["rag", "entity", "compress"]:
        f = RESULTS_DIR / f"{model}_{dataset}_{mode}.json"
        if not f.exists():
            continue
        with open(f, encoding="utf-8") as fp:
            data = json.load(fp)

        embeds, retrs, prefills, decodes, totals, ctx_toks = [], [], [], [], [], []
        for r in data["results"]:
            lat = r.get("latency", {})
            if "embedding_time_ms" in lat:
                embeds.append(lat["embedding_time_ms"])
                retrs.append(lat["retrieval_time_ms"])
                prefills.append(lat["prefill_time_ms"])
                decodes.append(lat["decode_time_ms"])
                totals.append(lat["total_time_ms"])
            if "context_tokens" in lat:
                ctx_toks.append(lat["context_tokens"])

        n = len(embeds)
        if n == 0:
            continue

        results[mode] = {
            "n": n,
            "embed_ms": round(sum(embeds) / n, 1),
            "retrieval_ms": round(sum(retrs) / n, 1),
            "prefill_ms": round(sum(prefills) / n, 1),
            "decode_ms": round(sum(decodes) / n, 1),
            "total_ms": round(sum(totals) / n, 1),
            "ctx_tokens": round(sum(ctx_toks) / n, 0) if ctx_toks else "N/A",
        }
    return results


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    models = ["lfm2.5-1.2b-instruct", "qwen3-0.6b", "qwen3-1.7b"]
    datasets = ["hotpotqa", "2wikimqa"]

    # ---- Significance Tests ----
    print("=" * 70)
    print("  PAIRED BOOTSTRAP SIGNIFICANCE TESTS (n=10,000)")
    print("=" * 70)

    sig_results = []
    for ds in datasets:
        for m in models:
            r = paired_bootstrap(m, ds)
            if r:
                sig_results.append(r)
                sig_em = "***" if r["em_p"] < 0.001 else "**" if r["em_p"] < 0.01 else "*" if r["em_p"] < 0.05 else "n.s."
                sig_f1 = "***" if r["f1_p"] < 0.001 else "**" if r["f1_p"] < 0.01 else "*" if r["f1_p"] < 0.05 else "n.s."
                print(f"\n  {m} / {ds} (n={r['n']})")
                print(f"    EM:  {r['em_rag']:.1f} → {r['em_ctks']:.1f}  (+{r['em_diff']:.2f})  "
                      f"95% CI [{r['em_ci'][0]:.2f}, {r['em_ci'][1]:.2f}]  p={r['em_p']:.4f} {sig_em}")
                print(f"    F1:  {r['f1_rag']:.1f} → {r['f1_ctks']:.1f}  (+{r['f1_diff']:.2f})  "
                      f"95% CI [{r['f1_ci'][0]:.2f}, {r['f1_ci'][1]:.2f}]  p={r['f1_p']:.4f} {sig_f1}")

    # LaTeX table for significance
    print("\n\n% --- LaTeX: Significance Test Table ---")
    print(r"\begin{table}[t]")
    print(r"\centering")
    print(r"\caption{Paired bootstrap significance test (10,000 iterations). $\Delta$ EM and 95\% CI in percentage points.}")
    print(r"\label{tab:significance}")
    print(r"\begin{tabular}{llcccl}")
    print(r"\toprule")
    print(r"\textbf{Model} & \textbf{Dataset} & \textbf{$\Delta$EM} & \textbf{95\% CI} & \textbf{$p$} & \\")
    print(r"\midrule")
    for r in sig_results:
        sig = "***" if r["em_p"] < 0.001 else "**" if r["em_p"] < 0.01 else "*" if r["em_p"] < 0.05 else "n.s."
        model_short = r["model"].replace("lfm2.5-1.2b-instruct", "LFM-Inst 1.2B").replace("qwen3-0.6b", "Qwen3-0.6B").replace("qwen3-1.7b", "Qwen3-1.7B")
        ds_short = r["dataset"].replace("hotpotqa", "HotpotQA").replace("2wikimqa", "2WikiMQA")
        p_str = f"<0.001" if r["em_p"] < 0.001 else f"{r['em_p']:.3f}"
        print(f"{model_short} & {ds_short} & +{r['em_diff']:.2f} & [{r['em_ci'][0]:.2f}, {r['em_ci'][1]:.2f}] & ${p_str}$ & {sig} \\\\")
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")

    # ---- Latency Breakdown ----
    print("\n\n" + "=" * 70)
    print("  LATENCY BREAKDOWN (from existing results)")
    print("=" * 70)

    for ds in datasets:
        for m in models:
            lat = latency_breakdown(m, ds)
            if not lat:
                continue
            print(f"\n  {m} / {ds}")
            print(f"  {'Mode':<10} {'Embed':>8} {'Retr':>8} {'Prefill':>8} {'Decode':>8} {'Total':>8} {'CtxTok':>8}")
            print(f"  {'-'*10} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
            for mode, v in lat.items():
                print(f"  {mode:<10} {v['embed_ms']:>7.1f}ms {v['retrieval_ms']:>7.1f}ms "
                      f"{v['prefill_ms']:>7.1f}ms {v['decode_ms']:>7.1f}ms "
                      f"{v['total_ms']:>7.1f}ms {str(v['ctx_tokens']):>8}")

    # LaTeX table for latency breakdown
    print("\n\n% --- LaTeX: Latency Breakdown Table ---")
    lfm_lat = latency_breakdown("lfm2.5-1.2b-instruct", "hotpotqa")
    if lfm_lat:
        print(r"\begin{table}[t]")
        print(r"\centering")
        print(r"\caption{Latency decomposition for LFM-Inst 1.2B on HotpotQA (ms). Embed = query embedding, Retr = FAISS search, Prefill = context encoding, Decode = token generation.}")
        print(r"\label{tab:latency}")
        print(r"\begin{tabular}{lcccccc}")
        print(r"\toprule")
        print(r"\textbf{Method} & \textbf{Embed} & \textbf{Retr} & \textbf{Prefill} & \textbf{Decode} & \textbf{Total} & \textbf{Ctx Tok} \\")
        print(r"\midrule")
        mode_names = {"rag": "RAG", "entity": r"\ours{}", "compress": "Compress"}
        for mode in ["rag", "entity", "compress"]:
            if mode in lfm_lat:
                v = lfm_lat[mode]
                ctx = str(int(v['ctx_tokens'])) if v['ctx_tokens'] != "N/A" else "---"
                print(f"{mode_names[mode]} & {v['embed_ms']:.1f} & {v['retrieval_ms']:.1f} & "
                      f"{v['prefill_ms']:.1f} & {v['decode_ms']:.1f} & {v['total_ms']:.1f} & {ctx} \\\\")
        print(r"\bottomrule")
        print(r"\end{tabular}")
        print(r"\end{table}")

    # ---- Save all results ----
    all_results = {
        "significance": sig_results,
        "latency": {f"{m}_{ds}": latency_breakdown(m, ds) for m in models for ds in datasets},
    }
    out_file = RESULTS_DIR / "analysis_extracted.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n\n[Saved] {out_file}")
