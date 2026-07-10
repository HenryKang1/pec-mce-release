"""Recompute metrics over the first N results of an existing JSON file.

This lets us compare a new n=50 pilot run against the first 50 of an
existing n=200 baseline run without re-running the GPU experiment.
"""
import argparse
import json
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results" / "longbench"


def recompute(file: Path, n: int) -> dict:
    d = json.load(open(file, encoding="utf-8"))
    rows = d["results"][:n]
    em = sum(r["em"] for r in rows) / len(rows) * 100
    f1 = sum(r["f1"] for r in rows) / len(rows) * 100
    loose = sum(r.get("loose", 0) for r in rows) / len(rows) * 100
    lat = sum(r["latency_ms"] for r in rows) / len(rows)
    ctx = sum(r.get("context_tokens", 0) for r in rows) / len(rows)
    return {
        "model": d["model"], "task": d["task"], "variant": d["variant"],
        "n": len(rows), "em": round(em, 2), "f1": round(f1, 2),
        "loose": round(loose, 2),
        "avg_latency_ms": round(lat, 1),
        "avg_context_tokens": round(ctx, 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--variants", nargs="+",
                    default=["raw_topk", "pec_bridge"],
                    help="variants to recompute (looks for *_n200.json files)")
    args = ap.parse_args()

    print(f"\n=== Recomputed metrics on first {args.n} samples ===\n")
    print(f"{'model':<22} {'task':<10} {'variant':<14} {'n':>4} {'EM':>6} {'F1':>6} {'loose':>6} {'lat':>8} {'ctx':>6}")
    print("-" * 90)

    rows = []
    for f in sorted(RESULTS_DIR.glob("*_n200.json")):
        d = json.load(open(f, encoding="utf-8"))
        if d["variant"] not in args.variants:
            continue
        m = recompute(f, args.n)
        rows.append(m)
        print(f"{m['model']:<22} {m['task']:<10} {m['variant']:<14} {m['n']:>4} {m['em']:>6.2f} {m['f1']:>6.2f} {m['loose']:>6.2f} {m['avg_latency_ms']:>8.1f} {m['avg_context_tokens']:>6.0f}")

    out = RESULTS_DIR / f"_first{args.n}_baselines.json"
    json.dump(rows, open(out, "w"), indent=2)
    print(f"\n[Saved] {out}")


if __name__ == "__main__":
    main()
