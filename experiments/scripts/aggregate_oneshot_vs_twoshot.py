"""Compare 0-shot / 1-shot / 2-shot extractive prompts across all baselines
on 3 readers x 3 multi-hop tasks (n=200). Report macro EM and median latency
to find the Pareto-optimal demo count."""
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
    ("pec_hop_rerank", "PEC-Hop"),
]
PROMPTS = [
    ("extractive",     "0-shot"),
    ("oneextractive",  "1-shot"),
    ("fewextractive",  "2-shot"),
]


def load_metrics(model, task, base, prompt):
    p = RES / f"{model}_{task}_{base}_{prompt}_n200.json"
    if not p.exists():
        return None
    with open(p, encoding="utf-8") as f:
        d = json.load(f)
    m = d["metrics"]
    return {"em": m["em"], "lat": m["avg_latency_ms"]}


def macro_em(model, base, prompt):
    vals = [load_metrics(model, t, base, prompt) for t in TASKS]
    if any(v is None for v in vals):
        return None, None
    em = sum(v["em"] for v in vals) / len(vals)
    lat = sum(v["lat"] for v in vals) / len(vals)
    return em, lat


lines = []
def emit(s=""):
    lines.append(s)
    print(s)


emit("# 0-shot vs 1-shot vs 2-shot extractive prompts")
emit("")
emit("Macro EM and mean latency across HotpotQA + 2WikiMQA + MuSiQue (n=200).")
emit("All cells use the SAME extractive instruction; only the number of demos differs.")
emit("")

# === Per-reader EM table ===
for model, label in READERS:
    emit(f"## {label} ({model}) — macro EM / mean latency (ms)")
    emit("")
    emit("| Method | 0-shot EM | 1-shot EM | 2-shot EM | Δ(1−2) | 0-shot lat | 1-shot lat | 2-shot lat |")
    emit("|---|---:|---:|---:|---:|---:|---:|---:|")
    for base, name in BASES:
        cells = []
        for pk, _ in PROMPTS:
            em, lat = macro_em(model, base, pk)
            cells.append((em, lat))
        em0, lat0 = cells[0]
        em1, lat1 = cells[1]
        em2, lat2 = cells[2]
        d12 = (em1 - em2) if (em1 is not None and em2 is not None) else None
        def f(x): return "—" if x is None else f"{x:.2f}"
        def fl(x): return "—" if x is None else f"{x:.0f}"
        d_s = "—" if d12 is None else (f"+{d12:.2f}" if d12 >= 0 else f"{d12:.2f}")
        bold = "**" if base == "pec_hop_rerank" else ""
        emit(f"| {bold}{name}{bold} | {bold}{f(em0)}{bold} | {bold}{f(em1)}{bold} | "
             f"{bold}{f(em2)}{bold} | {bold}{d_s}{bold} | "
             f"{fl(lat0)} | {fl(lat1)} | {fl(lat2)} |")
    emit("")

# === Cross-reader 1-shot vs 2-shot comparison (Δ EM) ===
emit("## Cross-reader summary: does 1-shot match 2-shot?")
emit("")
emit("| Reader | Method | EM(1-shot) − EM(2-shot) | Lat saved (ms) |")
emit("|---|---|---:|---:|")
for model, label in READERS:
    for base, name in BASES:
        em1, lat1 = macro_em(model, base, "oneextractive")
        em2, lat2 = macro_em(model, base, "fewextractive")
        if em1 is None or em2 is None:
            continue
        dem = em1 - em2
        dlat = lat2 - lat1
        bold = "**" if base == "pec_hop_rerank" else ""
        emit(f"| {bold}{label}{bold} | {bold}{name}{bold} | {bold}{dem:+.2f}{bold} | {bold}{dlat:+.0f}{bold} |")
emit("")

# === Per-cell appendix ===
emit("## Per-cell EM (appendix)")
emit("")
for model, label in READERS:
    emit(f"### {label}")
    emit("")
    emit("| Method | prompt | hotpot EM | 2wiki EM | musique EM | macro EM | mean lat (ms) |")
    emit("|---|---|---:|---:|---:|---:|---:|")
    for base, name in BASES:
        for pk, plabel in PROMPTS:
            vals = [load_metrics(model, t, base, pk) for t in TASKS]
            cells = ["—" if v is None else f"{v['em']:.2f}" for v in vals]
            if any(v is None for v in vals):
                emit(f"| {name} | {plabel} | {cells[0]} | {cells[1]} | {cells[2]} | — | — |")
            else:
                m = sum(v["em"] for v in vals) / 3
                l = sum(v["lat"] for v in vals) / 3
                emit(f"| {name} | {plabel} | {cells[0]} | {cells[1]} | {cells[2]} | {m:.2f} | {l:.0f} |")
    emit("")

out_path = ROOT / "MCE_COMPASS_EMNLP_main_draft" / "ARR_ONESHOT_VS_TWOSHOT.md"
out_path.write_text("\n".join(lines), encoding="utf-8")
emit(f"[Saved] {out_path}")
