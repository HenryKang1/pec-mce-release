"""Aggregate the ARR-defense variant runs into a single comparison table.

Reads existing JSON files in experiments/results/longbench/ for the new variants
plus the canonical baselines (raw_topk, pec_hop, pec_bridge_k3) and emits a
markdown summary highlighting Pareto winners.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "experiments" / "results" / "longbench"

MODELS = ["lfm2.5-1.2b-instruct", "qwen3-1.7b"]
TASKS = ["hotpotqa", "2wikimqa", "musique", "qasper", "multifieldqa_en"]
NS_BY_TASK = {
    "hotpotqa": 200,
    "2wikimqa": 200,
    "musique": 200,
    "qasper": 200,
    "multifieldqa_en": 150,
}
BASELINES = ["raw_topk", "raw_topk_b840", "pec_hop", "pec_bridge", "pec_bridge_k3"]
NEW = ["sentence_only", "pec_router",
       "pec_hop_extractive", "pec_hop_short15", "pec_hop_concise",
       "pec_hop_relations"]
CORRUPT = ["pec_hop_shuffle_ptr", "pec_hop_random_anchor"]
ALL = BASELINES + NEW + CORRUPT


def load(model: str, task: str, variant: str):
    n = NS_BY_TASK[task]
    candidates = [
        RES / f"{model}_{task}_{variant}_n{n}.json",
        RES / f"{model}_{task}_{variant}.json",
    ]
    for p in candidates:
        if p.exists():
            with open(p, encoding="utf-8") as f:
                d = json.load(f)
            return d
    return None


def get_metrics(d):
    if not d:
        return None
    m = d["metrics"]
    return {
        "em": m["em"],
        "f1": m["f1"],
        "lat": m["avg_latency_ms"],
        "ctx": m.get("avg_context_tokens", 0),
        "n": d["n_samples"],
    }


def render_table(model: str) -> str:
    lines = [f"### {model}\n"]
    header = "| Task | Variant | n | EM | F1 | Lat ms | Ctx tok |"
    sep = "|---|---|---:|---:|---:|---:|---:|"
    lines.append(header)
    lines.append(sep)
    for task in TASKS:
        for v in ALL:
            d = load(model, task, v)
            m = get_metrics(d)
            if not m:
                lines.append(f"| {task} | {v} | - | - | - | - | - |")
                continue
            lines.append(
                f"| {task} | {v} | {m['n']} | {m['em']:.2f} | {m['f1']:.2f} | "
                f"{m['lat']:.1f} | {m['ctx']:.0f} |"
            )
        lines.append("")
    return "\n".join(lines)


def render_winners(model: str) -> str:
    """Per task, show pec_hop vs new variants delta."""
    lines = [f"### {model} — deltas vs pec_hop\n"]
    header = "| Task | metric | pec_hop | sentence_only | pec_router | pec_hop_extractive | pec_hop_relations |"
    sep = "|---|---|---:|---:|---:|---:|---:|"
    lines.append(header)
    lines.append(sep)
    for task in TASKS:
        base = get_metrics(load(model, task, "pec_hop"))
        if not base:
            continue
        for metric in ("em", "f1"):
            row = [task, metric.upper(), f"{base[metric]:.2f}"]
            for v in NEW:
                d = get_metrics(load(model, task, v))
                if not d:
                    row.append("-")
                else:
                    delta = d[metric] - base[metric]
                    sign = "+" if delta >= 0 else ""
                    row.append(f"{d[metric]:.2f} ({sign}{delta:.2f})")
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")
    return "\n".join(lines)


def render_corruption(model: str = "lfm2.5-1.2b-instruct") -> str:
    lines = [f"### {model} — corruption stress (negative controls)\n"]
    header = "| Task | metric | pec_hop | shuffle_ptr | random_anchor |"
    sep = "|---|---|---:|---:|---:|"
    lines.append(header)
    lines.append(sep)
    for task in ("hotpotqa", "2wikimqa", "musique"):
        base = get_metrics(load(model, task, "pec_hop"))
        if not base:
            continue
        for metric in ("em", "f1"):
            row = [task, metric.upper(), f"{base[metric]:.2f}"]
            for v in CORRUPT:
                d = get_metrics(load(model, task, v))
                if not d:
                    row.append("-")
                else:
                    delta = d[metric] - base[metric]
                    sign = "+" if delta >= 0 else ""
                    row.append(f"{d[metric]:.2f} ({sign}{delta:.2f})")
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    parts = ["# ARR-defense variant comparison\n"]
    for m in MODELS:
        parts.append(render_table(m))
        parts.append(render_winners(m))
    parts.append(render_corruption())
    out = "\n".join(parts)
    out_path = RES.parent / "ARR_DEFENSE_COMPARISON.md"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(out)
    print(out)
    print(f"\n[Saved] {out_path}")
