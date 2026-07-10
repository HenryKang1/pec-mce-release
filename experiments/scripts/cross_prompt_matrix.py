"""Aggregate prompt x baseline x task x reader matrix from existing JSONs.

Produces a single markdown report that lets us directly read off:
- Pure representation effect (best PEC variant - best non-PEC variant) at fixed prompt.
- Pure prompt effect (best prompt - default prompt) at fixed retrieval.
- Best variant overall per (reader, task) cell.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "experiments" / "results" / "longbench"

MODELS = ["lfm2.5-1.2b-instruct", "qwen3-4b", "gemma-4-e4b"]
TASKS = ["hotpotqa", "2wikimqa", "musique", "qasper", "multifieldqa_en"]
NS_BY_TASK = {
    "hotpotqa": 200, "2wikimqa": 200, "musique": 200,
    "qasper": 200, "multifieldqa_en": 150,
}
BASELINES = ["raw_topk", "raw_topk_b840", "sentence_only", "pec_hop"]
PROMPTS = ["default", "extractive", "short15", "concise"]


def variant_name(base: str, prompt: str) -> str:
    if prompt == "default":
        return base
    return f"{base}_{prompt}"


def load_metrics(model: str, task: str, base: str, prompt: str):
    n = NS_BY_TASK[task]
    v = variant_name(base, prompt)
    p = RES / f"{model}_{task}_{v}_n{n}.json"
    if not p.exists():
        return None
    with open(p, encoding="utf-8") as f:
        d = json.load(f)
    return d["metrics"]


def render_matrix(model: str, metric: str = "em") -> str:
    out = [f"### {model}  --  {metric.upper()}\n"]
    header = "| Task / Base \\ Prompt | " + " | ".join(PROMPTS) + " |"
    sep = "|---|" + "|".join(["---:"] * len(PROMPTS)) + "|"
    for task in TASKS:
        out.append(f"\n**{task}**\n")
        out.append(header)
        out.append(sep)
        # Track best PEC and best non-PEC per prompt for delta row
        best_pec_per_prompt = {p: None for p in PROMPTS}
        best_other_per_prompt = {p: None for p in PROMPTS}
        for base in BASELINES:
            row = [f"{base}"]
            for prompt in PROMPTS:
                m = load_metrics(model, task, base, prompt)
                if m is None:
                    row.append("-")
                else:
                    val = m[metric]
                    row.append(f"{val:.2f}")
                    if base == "pec_hop":
                        if best_pec_per_prompt[prompt] is None or val > best_pec_per_prompt[prompt]:
                            best_pec_per_prompt[prompt] = val
                    else:
                        if best_other_per_prompt[prompt] is None or val > best_other_per_prompt[prompt]:
                            best_other_per_prompt[prompt] = val
            out.append("| " + " | ".join(row) + " |")
        # Delta row: PEC-Hop minus best non-PEC (Raw / Raw840 / Sentence)
        delta_row = ["**Δ (PEC-best non-PEC)**"]
        for prompt in PROMPTS:
            p = best_pec_per_prompt[prompt]
            o = best_other_per_prompt[prompt]
            if p is None or o is None:
                delta_row.append("-")
            else:
                d = p - o
                sign = "+" if d >= 0 else ""
                delta_row.append(f"{sign}{d:.2f}")
        out.append("| " + " | ".join(delta_row) + " |")
    return "\n".join(out)


def render_oracle(model: str) -> str:
    """For each (task), pick the best (base, prompt) combination and compute macro EM."""
    out = [f"\n### {model} -- best (base, prompt) per task\n"]
    out.append("| Task | best base | best prompt | EM | F1 |")
    out.append("|---|---|---|---:|---:|")
    macro_em = 0.0
    macro_f1 = 0.0
    n_tasks = 0
    for task in TASKS:
        best = None
        for base in BASELINES:
            for prompt in PROMPTS:
                m = load_metrics(model, task, base, prompt)
                if m is None:
                    continue
                if best is None or m["em"] > best[2]["em"]:
                    best = (base, prompt, m)
        if best:
            base, prompt, m = best
            out.append(f"| {task} | {base} | {prompt} | {m['em']:.2f} | {m['f1']:.2f} |")
            macro_em += m["em"]
            macro_f1 += m["f1"]
            n_tasks += 1
    if n_tasks:
        out.append(f"| **macro** | -- | -- | **{macro_em/n_tasks:.2f}** | **{macro_f1/n_tasks:.2f}** |")
    return "\n".join(out)


if __name__ == "__main__":
    parts = ["# Cross-prompt x baseline x task x reader matrix\n",
             "Reads existing result JSONs from experiments/results/longbench/.",
             "Δ row = PEC-Hop EM/F1 minus the best non-PEC baseline at the same prompt.",
             ""]
    for model in MODELS:
        parts.append(render_matrix(model, "em"))
        parts.append("")
        parts.append(render_matrix(model, "f1"))
        parts.append(render_oracle(model))
    out = "\n".join(parts)
    out_path = RES.parent / "ARR_CROSSPROMPT_MATRIX.md"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(out)
    print(out)
    print(f"\n[Saved] {out_path}")
