"""Non-inferiority test for PEC variants vs raw_topk on LongBench.

For each (model, task) cell, paired-bootstrap ΔEM and ΔF1, then check whether
the 95% one-sided lower bound of the difference clears a pre-specified
non-inferiority margin (e.g., -3 EM points). Concludes "non-inferior" if so.

This matches the ARR positioning: we do NOT claim improvement (because LongBench
n=200 lacks power to detect 2-3pp effects). We claim the variant is no worse
than raw_topk by more than `margin`, while delivering substantial efficiency
gains (latency, context tokens) shown separately.

Usage:
  python noninferiority_test.py --variant pec_hop --baseline raw_topk \
      --margin-em 3.0 --margin-f1 3.0 --n 200
"""
import argparse
import json
import random
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results" / "longbench"

MODELS = [
    ("lfm2.5-1.2b-instruct", "LFM2.5-1.2B"),
    ("qwen3-1.7b", "Qwen3-1.7B"),
]
TASKS = [
    ("hotpotqa", 200),
    ("2wikimqa", 200),
    ("musique", 200),
    ("multifieldqa_en", 150),
    ("qasper", 200),
]


def load(model: str, task: str, variant: str, n: int) -> list[dict]:
    candidates = [
        RESULTS_DIR / f"{model}_{task}_{variant}_n{n}.json",
        RESULTS_DIR / f"{model}_{task}_{variant}.json",
    ]
    for p in candidates:
        if p.exists():
            return json.load(open(p, encoding="utf-8")).get("results", [])
    return []


def paired_bootstrap_one_sided(rows_a, rows_b, n_bootstrap=10000, seed=42):
    """Returns lower bound of one-sided 95% CI for (a - b) on EM and F1.

    Aligned by question already.
    """
    n = len(rows_a)
    em_a = [int(r["em"]) for r in rows_a]
    em_b = [int(r["em"]) for r in rows_b]
    f1_a = [r["f1"] for r in rows_a]
    f1_b = [r["f1"] for r in rows_b]

    obs_em = (sum(em_a) - sum(em_b)) / n
    obs_f1 = (sum(f1_a) - sum(f1_b)) / n

    random.seed(seed)
    em_d, f1_d = [], []
    for _ in range(n_bootstrap):
        idx = [random.randint(0, n - 1) for _ in range(n)]
        em_d.append((sum(em_a[i] for i in idx) - sum(em_b[i] for i in idx)) / n)
        f1_d.append((sum(f1_a[i] for i in idx) - sum(f1_b[i] for i in idx)) / n)

    em_d.sort(); f1_d.sort()
    em_lb = em_d[int(len(em_d) * 0.05)]      # one-sided 95% lower bound
    f1_lb = f1_d[int(len(f1_d) * 0.05)]
    em_ub = em_d[int(len(em_d) * 0.95)]      # one-sided 95% upper (for context)
    f1_ub = f1_d[int(len(f1_d) * 0.95)]
    em_two_lo = em_d[int(len(em_d) * 0.025)]
    em_two_hi = em_d[int(len(em_d) * 0.975)]
    f1_two_lo = f1_d[int(len(f1_d) * 0.025)]
    f1_two_hi = f1_d[int(len(f1_d) * 0.975)]
    return {
        "n": n,
        "obs_em_diff": obs_em * 100,
        "obs_f1_diff": obs_f1 * 100,
        "em_lb95_one": em_lb * 100,
        "em_ub95_one": em_ub * 100,
        "f1_lb95_one": f1_lb * 100,
        "f1_ub95_one": f1_ub * 100,
        "em_ci_two": (em_two_lo * 100, em_two_hi * 100),
        "f1_ci_two": (f1_two_lo * 100, f1_two_hi * 100),
    }


def latency_ctx(model: str, task: str, variant: str, n: int):
    f = RESULTS_DIR / f"{model}_{task}_{variant}_n{n}.json"
    if not f.exists():
        f = RESULTS_DIR / f"{model}_{task}_{variant}.json"
    if not f.exists():
        return None, None
    d = json.load(open(f, encoding="utf-8"))
    m = d.get("metrics", {})
    return m.get("avg_latency_ms"), m.get("avg_context_tokens")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="pec_hop")
    ap.add_argument("--baseline", default="raw_topk")
    ap.add_argument("--margin-em", type=float, default=3.0,
                    help="Non-inferiority margin on EM in pp (positive number)")
    ap.add_argument("--margin-f1", type=float, default=3.0)
    ap.add_argument("--n-bootstrap", type=int, default=10000)
    args = ap.parse_args()

    print(f"\n=== Non-inferiority test: {args.variant} vs {args.baseline} ===")
    print(f"Margin: ΔEM ≥ -{args.margin_em:.1f} pp,  ΔF1 ≥ -{args.margin_f1:.1f} pp")
    print(f"Test: 95% one-sided lower bound of (variant - baseline) > -margin → non-inferior\n")

    header = (f"{'model':<14} {'task':<18} {'n':>4} "
              f"{'EM_v':>5} {'EM_b':>5} {'ΔEM':>6} {'lb95':>6} {'NI?':>4}  "
              f"{'F1_v':>5} {'F1_b':>5} {'ΔF1':>6} {'lb95':>6} {'NI?':>4}  "
              f"{'lat ratio':>10} {'ctx ratio':>10}")
    print(header)
    print("-" * len(header))

    pass_em_count = 0
    pass_f1_count = 0
    total = 0
    summary_rows = []

    for model, mlabel in MODELS:
        for task, n in TASKS:
            ra = load(model, task, args.variant, n)
            rb = load(model, task, args.baseline, n)
            if not ra or not rb:
                print(f"{mlabel:<14} {task:<18} {'?':>4} (missing data)")
                continue
            # align by question
            ba = {r["question"]: r for r in ra}
            bb = {r["question"]: r for r in rb}
            common = [q for q in [r["question"] for r in ra] if q in bb]
            ra_a = [ba[q] for q in common]
            rb_a = [bb[q] for q in common]
            r = paired_bootstrap_one_sided(ra_a, rb_a, args.n_bootstrap)

            em_a = sum(int(x["em"]) for x in ra_a) / len(ra_a) * 100
            em_b = sum(int(x["em"]) for x in rb_a) / len(rb_a) * 100
            f1_a = sum(x["f1"] for x in ra_a) / len(ra_a) * 100
            f1_b = sum(x["f1"] for x in rb_a) / len(rb_a) * 100

            em_ni = r["em_lb95_one"] > -args.margin_em
            f1_ni = r["f1_lb95_one"] > -args.margin_f1
            if em_ni: pass_em_count += 1
            if f1_ni: pass_f1_count += 1
            total += 1

            la, ca = latency_ctx(model, task, args.variant, n)
            lb, cb = latency_ctx(model, task, args.baseline, n)
            lat_ratio = la / lb if lb and la else 0
            ctx_ratio = ca / cb if cb and ca else 0

            print(f"{mlabel:<14} {task:<18} {len(common):>4} "
                  f"{em_a:>5.1f} {em_b:>5.1f} {r['obs_em_diff']:>+6.2f} {r['em_lb95_one']:>+6.2f} "
                  f"{'✓' if em_ni else '✗':>4}  "
                  f"{f1_a:>5.1f} {f1_b:>5.1f} {r['obs_f1_diff']:>+6.2f} {r['f1_lb95_one']:>+6.2f} "
                  f"{'✓' if f1_ni else '✗':>4}  "
                  f"{lat_ratio:>9.2f}x {ctx_ratio:>9.2f}x")

            summary_rows.append({
                "model": model, "task": task, "n": len(common),
                "em_diff": r["obs_em_diff"], "em_lb95": r["em_lb95_one"], "em_ni": em_ni,
                "f1_diff": r["obs_f1_diff"], "f1_lb95": r["f1_lb95_one"], "f1_ni": f1_ni,
                "lat_ratio": lat_ratio, "ctx_ratio": ctx_ratio,
            })

    print("\n=== Aggregate ===")
    print(f"Non-inferior on EM (margin -{args.margin_em:.1f} pp): {pass_em_count} / {total} cells")
    print(f"Non-inferior on F1 (margin -{args.margin_f1:.1f} pp): {pass_f1_count} / {total} cells")

    # Per-model breakdown
    print("\n=== Per-model breakdown ===")
    for model, mlabel in MODELS:
        m_rows = [r for r in summary_rows if r["model"] == model]
        if not m_rows:
            continue
        ne_em = sum(1 for r in m_rows if r["em_ni"])
        ne_f1 = sum(1 for r in m_rows if r["f1_ni"])
        mean_em_diff = sum(r["em_diff"] for r in m_rows) / len(m_rows)
        mean_f1_diff = sum(r["f1_diff"] for r in m_rows) / len(m_rows)
        mean_lat = sum(r["lat_ratio"] for r in m_rows) / len(m_rows)
        mean_ctx = sum(r["ctx_ratio"] for r in m_rows) / len(m_rows)
        print(f"  {mlabel}: NI on EM = {ne_em}/{len(m_rows)}, NI on F1 = {ne_f1}/{len(m_rows)}, "
              f"mean ΔEM={mean_em_diff:+.2f}, mean ΔF1={mean_f1_diff:+.2f}, "
              f"lat={mean_lat:.2f}x, ctx={mean_ctx:.2f}x")


if __name__ == "__main__":
    main()
