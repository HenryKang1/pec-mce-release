"""Paired bootstrap significance test for LongBench variant comparisons.

Usage:
  python bootstrap_longbench.py --model lfm2.5-1.2b-instruct --task hotpotqa \
      --variant-a pec_hop --variant-b raw_topk --n 200
"""
import argparse
import json
import random
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results" / "longbench"


def load(model: str, task: str, variant: str, n: int):
    candidates = [
        RESULTS_DIR / f"{model}_{task}_{variant}_n{n}.json",
        RESULTS_DIR / f"{model}_{task}_{variant}.json",
    ]
    for p in candidates:
        if p.exists():
            return json.load(open(p, encoding="utf-8"))
    raise FileNotFoundError(f"No result file for {model}/{task}/{variant} n={n}")


def paired_bootstrap(rows_a, rows_b, n_bootstrap=10000, seed=42):
    """Both lists must be aligned by question (same order, same count)."""
    n = len(rows_a)
    assert len(rows_b) == n
    em_a = [int(r["em"]) for r in rows_a]
    em_b = [int(r["em"]) for r in rows_b]
    f1_a = [r["f1"] for r in rows_a]
    f1_b = [r["f1"] for r in rows_b]

    obs_em = sum(em_a) / n - sum(em_b) / n
    obs_f1 = sum(f1_a) / n - sum(f1_b) / n

    random.seed(seed)
    em_d, f1_d = [], []
    for _ in range(n_bootstrap):
        idx = [random.randint(0, n - 1) for _ in range(n)]
        em_d.append(sum(em_a[i] for i in idx) / n - sum(em_b[i] for i in idx) / n)
        f1_d.append(sum(f1_a[i] for i in idx) / n - sum(f1_b[i] for i in idx) / n)

    em_d.sort(); f1_d.sort()

    def ci(diffs):
        lo = diffs[int(len(diffs) * 0.025)]
        hi = diffs[int(len(diffs) * 0.975)]
        # Two-sided p — fraction of resamples on the "wrong" side of zero
        p = 2 * min(
            sum(1 for d in diffs if d <= 0) / len(diffs),
            sum(1 for d in diffs if d >= 0) / len(diffs),
        )
        return lo, hi, p

    em_lo, em_hi, em_p = ci(em_d)
    f1_lo, f1_hi, f1_p = ci(f1_d)

    return {
        "n": n,
        "em_a": round(sum(em_a) / n * 100, 2),
        "em_b": round(sum(em_b) / n * 100, 2),
        "em_diff": round(obs_em * 100, 2),
        "em_ci_95": (round(em_lo * 100, 2), round(em_hi * 100, 2)),
        "em_p": round(em_p, 4),
        "f1_a": round(sum(f1_a) / n * 100, 2),
        "f1_b": round(sum(f1_b) / n * 100, 2),
        "f1_diff": round(obs_f1 * 100, 2),
        "f1_ci_95": (round(f1_lo * 100, 2), round(f1_hi * 100, 2)),
        "f1_p": round(f1_p, 4),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--task", required=True)
    ap.add_argument("--variant-a", required=True, help="treatment variant")
    ap.add_argument("--variant-b", required=True, help="baseline variant")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--n-bootstrap", type=int, default=10000)
    args = ap.parse_args()

    da = load(args.model, args.task, args.variant_a, args.n)
    db = load(args.model, args.task, args.variant_b, args.n)
    rows_a = da["results"]
    rows_b = db["results"]

    # Align by question
    by_q_a = {r["question"]: r for r in rows_a}
    by_q_b = {r["question"]: r for r in rows_b}
    common = [q for q in [r["question"] for r in rows_a] if q in by_q_b]
    aligned_a = [by_q_a[q] for q in common]
    aligned_b = [by_q_b[q] for q in common]

    print(f"\n[{args.model} / {args.task}] {args.variant_a} vs {args.variant_b} "
          f"(aligned n={len(common)}, bootstrap={args.n_bootstrap})")
    r = paired_bootstrap(aligned_a, aligned_b, args.n_bootstrap)
    print(f"  EM: {args.variant_a}={r['em_a']:.2f}% vs {args.variant_b}={r['em_b']:.2f}%  "
          f"Δ={r['em_diff']:+.2f} 95%CI[{r['em_ci_95'][0]:+.2f}, {r['em_ci_95'][1]:+.2f}]  p={r['em_p']:.4f}")
    print(f"  F1: {args.variant_a}={r['f1_a']:.2f}% vs {args.variant_b}={r['f1_b']:.2f}%  "
          f"Δ={r['f1_diff']:+.2f} 95%CI[{r['f1_ci_95'][0]:+.2f}, {r['f1_ci_95'][1]:+.2f}]  p={r['f1_p']:.4f}")


if __name__ == "__main__":
    main()
