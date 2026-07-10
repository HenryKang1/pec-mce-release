"""Compare PEC-RAG variants against baselines (raw_topk and pec_bridge).

Reads all longbench result JSONs, builds a wide table per (model, task)
and reports per-variant EM/F1/latency/context-token deltas vs raw_topk.
Intended for picking a winner from the pilot phase.

Usage:
  python compare_pec_variants.py
  python compare_pec_variants.py --n 50
  python compare_pec_variants.py --n 200
"""
import argparse
import json
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results" / "longbench"

BASELINE = "raw_topk"
PEC_BASELINE = "pec_bridge"
NEW_VARIANTS = ["pec_hop", "pec_bridge_k3", "pec_query_expand"]
ALL_VARIANTS = [BASELINE, PEC_BASELINE] + NEW_VARIANTS

MODELS = ["lfm2.5-1.2b-instruct", "qwen3-1.7b"]
TASKS = ["hotpotqa", "2wikimqa", "musique"]


def load_metrics(model: str, task: str, variant: str, n: int) -> dict | None:
    candidates = [
        RESULTS_DIR / f"{model}_{task}_{variant}_n{n}.json",
        RESULTS_DIR / f"{model}_{task}_{variant}.json",
    ]
    for p in candidates:
        if p.exists():
            d = json.load(open(p, encoding="utf-8"))
            m = d.get("metrics", {})
            return {
                "em": m.get("em", 0.0),
                "f1": m.get("f1", 0.0),
                "loose": m.get("loose", 0.0),
                "lat": m.get("avg_latency_ms", m.get("avg_total_ms", 0.0)),
                "ctx": m.get("avg_context_tokens", 0.0),
                "n": d.get("n_samples", n),
                "file": p.name,
            }
    return None


def fmt_delta(new: float, base: float, kind: str = "higher_better") -> str:
    if base == 0:
        return f"{new:+.2f}"
    delta = new - base
    sign = "+" if delta > 0 else ""
    return f"{sign}{delta:.2f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50, help="sample size to filter on")
    args = ap.parse_args()

    print(f"\n=== PEC-RAG variant comparison (n={args.n}) ===\n")

    # Per-cell wide table
    header = f"{'model':<22} {'task':<10} {'variant':<18} {'EM':>6} {'ΔEM':>7} {'F1':>6} {'ΔF1':>7} {'lat':>8} {'Δlat%':>8} {'ctx':>6} {'Δctx%':>8}"
    print(header)
    print("-" * len(header))

    cells = {}
    for model in MODELS:
        for task in TASKS:
            base = load_metrics(model, task, BASELINE, args.n)
            if not base:
                # try n=200 as raw_topk reference if pilot run is n=50
                base = load_metrics(model, task, BASELINE, 200)
            for variant in ALL_VARIANTS:
                m = load_metrics(model, task, variant, args.n)
                if not m and variant in (BASELINE, PEC_BASELINE):
                    m = load_metrics(model, task, variant, 200)
                cells[(model, task, variant)] = (m, base)
                if not m:
                    continue
                if base and base["em"] != 0:
                    dem = fmt_delta(m["em"], base["em"])
                    df1 = fmt_delta(m["f1"], base["f1"])
                    dlat = f"{(m['lat'] - base['lat']) / base['lat'] * 100:+.0f}%" if base["lat"] else "n/a"
                    dctx = f"{(m['ctx'] - base['ctx']) / base['ctx'] * 100:+.0f}%" if base["ctx"] else "n/a"
                else:
                    dem = df1 = dlat = dctx = "n/a"
                print(f"{model:<22} {task:<10} {variant:<18} {m['em']:>6.2f} {dem:>7} {m['f1']:>6.2f} {df1:>7} {m['lat']:>8.1f} {dlat:>8} {m['ctx']:>6.0f} {dctx:>8}")
            print()

    # Winner per (model, task) — variant with highest EM among new ones, tie-broken by F1
    print("\n=== Winner per cell (new variants only, vs raw_topk baseline) ===")
    print(f"{'model':<22} {'task':<10} {'winner':<18} {'EM':>6} {'ΔEM':>7} {'F1':>6} {'ΔF1':>7}")
    for model in MODELS:
        for task in TASKS:
            base = cells[(model, task, BASELINE)][1]
            if not base:
                continue
            cands = []
            for v in NEW_VARIANTS:
                m = cells.get((model, task, v), (None, None))[0]
                if m:
                    cands.append((v, m))
            if not cands:
                continue
            cands.sort(key=lambda x: (-x[1]["em"], -x[1]["f1"]))
            v, m = cands[0]
            dem = fmt_delta(m["em"], base["em"])
            df1 = fmt_delta(m["f1"], base["f1"])
            print(f"{model:<22} {task:<10} {v:<18} {m['em']:>6.2f} {dem:>7} {m['f1']:>6.2f} {df1:>7}")

    # Aggregate (mean across cells) per variant
    print("\n=== Aggregate (mean across all 6 model-task cells) ===")
    print(f"{'variant':<18} {'mean EM':>8} {'mean F1':>8} {'mean lat':>10} {'mean ctx':>10} {'cells':>6}")
    for v in ALL_VARIANTS:
        ems, f1s, lats, ctxs = [], [], [], []
        for model in MODELS:
            for task in TASKS:
                m = cells.get((model, task, v), (None, None))[0]
                if m:
                    ems.append(m["em"]); f1s.append(m["f1"])
                    lats.append(m["lat"]); ctxs.append(m["ctx"])
        if not ems:
            continue
        n = len(ems)
        print(f"{v:<18} {sum(ems)/n:>8.2f} {sum(f1s)/n:>8.2f} {sum(lats)/n:>10.1f} {sum(ctxs)/n:>10.0f} {n:>6}")


if __name__ == "__main__":
    main()
