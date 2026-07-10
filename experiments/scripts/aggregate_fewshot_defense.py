"""Build the reviewer-defense table: each baseline at extractive vs fewextractive,
across LFM2.5/Gemma/Qwen3, macro EM over hotpotqa+2wikimqa+musique (n=200)."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "experiments" / "results" / "longbench"

READERS = [
    ("lfm2.5-1.2b-instruct", "LFM"),
    ("gemma-4-e4b", "Gemma"),
    ("qwen3-4b", "Qwen3"),
]
TASKS = ["hotpotqa", "2wikimqa", "musique"]
BASES = [
    ("raw_topk",       "Raw RAG"),
    ("sentence_only",  "Sentence-only"),
    ("llmlingua2",     "LLMLingua-2"),
    ("provence",       "Provence"),
    ("pec_hop_rerank", "PEC-Hop (ours)"),
]
PROMPTS = [("extractive", "extractive"), ("fewextractive", "fewextractive")]


def load_em(model, task, base, prompt):
    p = RES / f"{model}_{task}_{base}_{prompt}_n200.json"
    if not p.exists():
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)["metrics"]["em"]


def macro(model, base, prompt):
    vals = [load_em(model, t, base, prompt) for t in TASKS]
    if any(v is None for v in vals):
        return None, vals
    return sum(vals) / len(vals), vals


rows = []
rows.append("# Reviewer-defense ablation: matched-prompt baselines")
rows.append("")
rows.append("Macro EM across HotpotQA, 2WikiMQA, MuSiQue (n=200 each).")
rows.append("All methods use the **same** reader and decoding prompt;")
rows.append("only the retrieval/compression component differs.")
rows.append("")
for model, label in READERS:
    rows.append(f"## {label} ({model})")
    rows.append("")
    rows.append("| Method | extractive (no demo) | fewextractive (2-shot) | Δ (demo lift) |")
    rows.append("|---|---:|---:|---:|")
    ext_em_pec = None
    fs_em_pec = None
    for base, name in BASES:
        ext, _ = macro(model, base, "extractive")
        fs, _  = macro(model, base, "fewextractive")
        ext_s = "—" if ext is None else f"{ext:.2f}"
        fs_s  = "—" if fs  is None else f"{fs:.2f}"
        if ext is None or fs is None:
            d_s = "—"
        else:
            d = fs - ext
            sign = "+" if d >= 0 else ""
            d_s = f"{sign}{d:.2f}"
        bold = "**" if base == "pec_hop_rerank" else ""
        rows.append(f"| {bold}{name}{bold} | {bold}{ext_s}{bold} | {bold}{fs_s}{bold} | {bold}{d_s}{bold} |")
        if base == "pec_hop_rerank":
            ext_em_pec, fs_em_pec = ext, fs
    # PEC-Hop gap rows
    rows.append("")
    rows.append("**PEC-Hop gap vs each baseline (under matched prompts):**")
    rows.append("")
    rows.append("| vs | extractive gap | fewextractive gap |")
    rows.append("|---|---:|---:|")
    for base, name in BASES[:-1]:
        ext, _ = macro(model, base, "extractive")
        fs, _  = macro(model, base, "fewextractive")
        e_gap = "—" if ext is None or ext_em_pec is None else f"{ext_em_pec-ext:+.2f}"
        f_gap = "—" if fs  is None or fs_em_pec  is None else f"{fs_em_pec-fs:+.2f}"
        rows.append(f"| PEC-Hop − {name} | {e_gap} | {f_gap} |")
    rows.append("")

# Per-cell appendix
rows.append("## Per-cell EM (appendix)")
rows.append("")
for model, label in READERS:
    rows.append(f"### {label}")
    rows.append("")
    rows.append("| Method | prompt | hotpot | 2wiki | musique | macro |")
    rows.append("|---|---|---:|---:|---:|---:|")
    for base, name in BASES:
        for pk, plabel in PROMPTS:
            m, vals = macro(model, base, pk)
            cells = ["—" if v is None else f"{v:.2f}" for v in vals]
            ms = "—" if m is None else f"{m:.2f}"
            rows.append(f"| {name} | {plabel} | {cells[0]} | {cells[1]} | {cells[2]} | {ms} |")
    rows.append("")

out = "\n".join(rows)
out_path = ROOT / "MCE_COMPASS_EMNLP_main_draft" / "ARR_FEWSHOT_DEFENSE.md"
out_path.write_text(out, encoding="utf-8")
print(out)
print(f"\n[Saved] {out_path}")
