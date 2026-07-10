"""
Aggregate LongBench results across variants / tasks / models into a
human-readable table + JSON summary.

Reads from results/longbench/{model}_{task}_{variant}.json.

Usage:
  python summarize_longbench.py
  python summarize_longbench.py --models lfm2.5-1.2b-instruct qwen3-0.6b qwen3-1.7b
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "shared"))
from utils.config import TOPIC6_DIR


TASKS = ["hotpotqa", "2wikimqa", "musique"]
VARIANTS = [
    "raw_trunc", "raw_topk", "summary", "anchors", "anchored",
    "pec_card", "pec_hydrate", "pec_adaptive", "pec_bridge",
]
DEFAULT_MODELS = ["lfm2.5-1.2b-instruct", "qwen3-0.6b", "qwen3-1.7b"]


def run(models: list[str]):
    rd = TOPIC6_DIR / "experiments" / "results" / "longbench"
    table = {}

    for m in models:
        for t in TASKS:
            for v in VARIANTS:
                path = rd / f"{m}_{t}_{v}.json"
                if not path.exists():
                    continue
                with open(path, encoding="utf-8") as f:
                    d = json.load(f)
                table[(m, t, v)] = d["metrics"]

    print(f"{'Model':<24} {'Task':<12} {'Variant':<12} {'EM':>7} {'F1':>7} "
          f"{'Loose':>7} {'Lat(ms)':>9}")
    print("-" * 84)
    for m in models:
        for t in TASKS:
            row_exists = any((m, t, v) in table for v in VARIANTS)
            if not row_exists:
                continue
            for v in VARIANTS:
                metrics = table.get((m, t, v))
                if metrics is None:
                    print(f"{m:<24} {t:<12} {v:<12} {'--':>7} {'--':>7} {'--':>7} {'--':>9}")
                else:
                    print(f"{m:<24} {t:<12} {v:<12} "
                          f"{metrics['em']:>6.2f}% "
                          f"{metrics['f1']:>6.2f}% "
                          f"{metrics['loose']:>6.2f}% "
                          f"{metrics['avg_latency_ms']:>9}")
            print()

    # Deltas: PEC vs raw_topk and legacy anchored
    print("\n=== Deltas (percentage points) ===")
    print(f"{'Model':<24} {'Task':<12} {'PEC-Raw':>9} {'PEC-Anc':>9} {'PEC-Trnc':>9}")
    for m in models:
        for t in TASKS:
            a = (table.get((m, t, "pec_bridge")) or
                 table.get((m, t, "pec_hydrate")) or
                 table.get((m, t, "pec_card")))
            r = table.get((m, t, "raw_topk"))
            s = table.get((m, t, "anchored"))
            tr = table.get((m, t, "raw_trunc"))
            if not a:
                continue
            parts = [f"{m:<24} {t:<12}"]
            if r:
                parts.append(f"{a['em']-r['em']:+8.2f}")
            else:
                parts.append("       --")
            if s:
                parts.append(f"{a['em']-s['em']:+8.2f}")
            else:
                parts.append("       --")
            if tr:
                parts.append(f"{a['em']-tr['em']:+8.2f}")
            else:
                parts.append("       --")
            print(" ".join(parts))

    # Save structured
    out = {"results": {f"{m}|{t}|{v}": metrics
                       for (m, t, v), metrics in table.items()}}
    out_path = rd.parent / "SUMMARY_longbench.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\n[Saved] {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    args = parser.parse_args()
    run(args.models)
