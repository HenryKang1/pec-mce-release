"""Generate publication-ready LaTeX tables for the PEC-RAG ARR paper.

Outputs 4 tables to stdout (paste into paper):
  Table 1: LFM-1.2B main results — 5 tasks x 4 variants
  Table 2: Non-inferiority test for LFM (pec_hop vs raw_topk, margin -3pp)
  Table 3: Qwen3-1.7B stress test — same layout as Table 1
  Table 4: Aggregate Pareto efficiency per model

Usage:
  python make_paper_tables.py > paper/tables_pec.tex
"""
import json
import random
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results" / "longbench"

MODELS = [
    ("lfm2.5-1.2b-instruct", "LFM2.5-1.2B"),
    ("qwen3-1.7b", "Qwen3-1.7B"),
]
TASKS = [
    ("hotpotqa", "HotpotQA", 200),
    ("2wikimqa", "2WikiMQA", 200),
    ("musique", "MuSiQue", 200),
    ("multifieldqa_en", "MFQA-en", 150),
    ("qasper", "Qasper", 200),
]
VARIANTS = [
    ("raw_topk", "Raw RAG"),
    ("raw_topk_b840", "Raw RAG$_{840}$"),
    ("pec_bridge", "PEC-Bridge"),
    ("pec_bridge_k3", "PEC-Bridge$_{k=3}$"),
    ("pec_hop", "\\textbf{PEC-Hop}"),
]


def load_metrics(model: str, task: str, variant: str, n: int) -> dict | None:
    """Load metrics; recompute latency as MEDIAN over per-query latencies to be
    robust against rare llama.cpp prefill hangs (e.g., w=2 had 1 query at 39min
    on HotpotQA and 1 at 57min on MuSiQue, which destroy raw means)."""
    for p in [
        RESULTS_DIR / f"{model}_{task}_{variant}_n{n}.json",
        RESULTS_DIR / f"{model}_{task}_{variant}.json",
    ]:
        if p.exists():
            d = json.load(open(p, encoding="utf-8"))
            m = dict(d.get("metrics", {}))
            rows = d.get("results", [])
            if rows:
                lats = sorted(r.get("latency_ms", 0) for r in rows)
                m["avg_latency_ms"] = lats[len(lats) // 2]  # median
            return m
    return None


def load_records(model: str, task: str, variant: str, n: int) -> list:
    for p in [
        RESULTS_DIR / f"{model}_{task}_{variant}_n{n}.json",
        RESULTS_DIR / f"{model}_{task}_{variant}.json",
    ]:
        if p.exists():
            return json.load(open(p, encoding="utf-8")).get("results", [])
    return []


def bf(s: str, cond: bool) -> str:
    return f"\\textbf{{{s}}}" if cond else s


def paired_bootstrap(rows_a, rows_b, n_bootstrap=10000, seed=42):
    n = len(rows_a)
    em_a = [int(r["em"]) for r in rows_a]; em_b = [int(r["em"]) for r in rows_b]
    f1_a = [r["f1"] for r in rows_a];      f1_b = [r["f1"] for r in rows_b]
    obs_em = (sum(em_a) - sum(em_b)) / n
    obs_f1 = (sum(f1_a) - sum(f1_b)) / n
    random.seed(seed)
    em_d, f1_d = [], []
    for _ in range(n_bootstrap):
        idx = [random.randint(0, n - 1) for _ in range(n)]
        em_d.append((sum(em_a[i] for i in idx) - sum(em_b[i] for i in idx)) / n)
        f1_d.append((sum(f1_a[i] for i in idx) - sum(f1_b[i] for i in idx)) / n)
    em_d.sort(); f1_d.sort()
    return {
        "obs_em": obs_em * 100,
        "obs_f1": obs_f1 * 100,
        "em_lb95": em_d[int(len(em_d)*0.05)] * 100,
        "f1_lb95": f1_d[int(len(f1_d)*0.05)] * 100,
        "em_ci": (em_d[int(len(em_d)*0.025)]*100, em_d[int(len(em_d)*0.975)]*100),
        "f1_ci": (f1_d[int(len(f1_d)*0.025)]*100, f1_d[int(len(f1_d)*0.975)]*100),
    }


def main_results_table(model: str, model_label: str, table_id: str, caption: str,
                         variants_override: list | None = None):
    """Per-model: 5 tasks x N variants x (EM, F1). variants_override lets us
    drop variants (e.g., Raw RAG_840 not run on Qwen3) so we don't print empty
    rows."""
    variants = variants_override if variants_override is not None else VARIANTS
    print(f"% =====  {model_label} main results table  =====")
    print(r"\begin{table*}[t]")
    print(r"\centering")
    print(r"\small")
    print(r"\begin{tabular}{l" + "rr" * len(TASKS) + r"rr}")
    print(r"\toprule")
    print(r"\multirow{2}{*}{Method}" +
          "".join(f" & \\multicolumn{{2}}{{c}}{{{label}}}" for _, label, _ in TASKS) +
          r" & \multicolumn{2}{c}{Mean} \\")
    cm_idx = 2
    cm_str = "".join([f" \\cmidrule(lr){{{cm_idx + 2*i}-{cm_idx + 2*i + 1}}}"
                       for i in range(len(TASKS) + 1)])
    print(cm_str)
    print(r"  " + "".join(f" & EM & F1" for _ in range(len(TASKS) + 1)) + r" \\")
    print(r"\midrule")

    # Find best per task per metric for bolding
    cell_data = {}
    for v, _ in variants:
        cell_data[v] = {}
        for t, _, n in TASKS:
            m = load_metrics(model, t, v, n)
            cell_data[v][t] = m

    best = {}
    for t, _, _ in TASKS:
        best[t] = {"em": -1, "f1": -1}
        for v, _ in variants:
            m = cell_data[v][t]
            if m:
                if m.get("em", 0) > best[t]["em"]: best[t]["em"] = m["em"]
                if m.get("f1", 0) > best[t]["f1"]: best[t]["f1"] = m["f1"]

    for v, vlabel in variants:
        cells = [vlabel]
        ems, f1s = [], []
        for t, _, _ in TASKS:
            m = cell_data[v][t]
            if m is None:
                cells.extend(["--", "--"])
                continue
            em_str = f"{m['em']:.1f}"; f1_str = f"{m['f1']:.1f}"
            cells.append(bf(em_str, abs(m['em'] - best[t]['em']) < 0.05))
            cells.append(bf(f1_str, abs(m['f1'] - best[t]['f1']) < 0.05))
            ems.append(m['em']); f1s.append(m['f1'])
        if ems:
            cells.append(f"{sum(ems)/len(ems):.1f}")
            cells.append(f"{sum(f1s)/len(f1s):.1f}")
        print(" & ".join(cells), r"\\")
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(rf"\caption{{{caption}}}")
    print(rf"\label{{tab:{table_id}}}")
    print(r"\end{table*}")
    print()


def noninferiority_table(model: str, model_label: str, margin: float = 3.0):
    """LFM-only: pec_hop vs raw_topk paired bootstrap with NI test."""
    print(f"% =====  Non-inferiority test for {model_label}  =====")
    print(r"\begin{table}[t]")
    print(r"\centering")
    print(r"\small")
    print(r"\resizebox{\columnwidth}{!}{%")
    print(r"\begin{tabular}{lrrcrrc}")
    print(r"\toprule")
    print(r"\multirow{2}{*}{Task} & \multicolumn{3}{c}{$\Delta$EM (pp)} & \multicolumn{3}{c}{$\Delta$F1 (pp)} \\")
    print(r"\cmidrule(lr){2-4} \cmidrule(lr){5-7}")
    print(r" & obs & 95\% LB & NI? & obs & 95\% LB & NI? \\")
    print(r"\midrule")
    n_em_pass, n_f1_pass, total = 0, 0, 0
    for t, tlabel, n in TASKS:
        ra = load_records(model, t, "pec_hop", n)
        rb = load_records(model, t, "raw_topk", n)
        if not ra or not rb:
            continue
        ba = {r["question"]: r for r in ra}; bb = {r["question"]: r for r in rb}
        common = [q for q in [r["question"] for r in ra] if q in bb]
        ra_a = [ba[q] for q in common]; rb_a = [bb[q] for q in common]
        r = paired_bootstrap(ra_a, rb_a)
        em_ni = r["em_lb95"] > -margin
        f1_ni = r["f1_lb95"] > -margin
        if em_ni: n_em_pass += 1
        if f1_ni: n_f1_pass += 1
        total += 1
        em_obs = r["obs_em"]; em_lb = r["em_lb95"]
        f1_obs = r["obs_f1"]; f1_lb = r["f1_lb95"]
        em_obs_str = f"$+${em_obs:.2f}" if em_obs >= 0 else f"$-${abs(em_obs):.2f}"
        em_lb_str  = f"$+${em_lb:.2f}" if em_lb >= 0 else f"$-${abs(em_lb):.2f}"
        f1_obs_str = f"$+${f1_obs:.2f}" if f1_obs >= 0 else f"$-${abs(f1_obs):.2f}"
        f1_lb_str  = f"$+${f1_lb:.2f}" if f1_lb >= 0 else f"$-${abs(f1_lb):.2f}"
        em_mark = r"\checkmark" if em_ni else r"$\times$"
        f1_mark = r"\checkmark" if f1_ni else r"$\times$"
        print(f"{tlabel} & {em_obs_str} & {em_lb_str} & {em_mark} & "
              f"{f1_obs_str} & {f1_lb_str} & {f1_mark} \\\\")
    print(r"\midrule")
    print(rf"\textbf{{NI pass}} & \multicolumn{{3}}{{c}}{{{n_em_pass}/{total}}} "
          rf"& \multicolumn{{3}}{{c}}{{{n_f1_pass}/{total}}} \\")
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"}")
    print(rf"\caption{{Non-inferiority test ({model_label}): paired bootstrap "
          rf"(10k iterations) of PEC-Hop vs Raw RAG. ``95\% LB'' is the lower "
          rf"bound of the one-sided 95\% confidence interval on the difference. "
          rf"Non-inferiority (NI) at margin $-{margin:.1f}$ pp is concluded when "
          rf"95\% LB $> -{margin:.1f}$.}}")
    print(rf"\label{{tab:ni-{model.replace('.', '').replace('-', '')}}}")
    print(r"\end{table}")
    print()


def efficiency_table():
    """Aggregate latency / context tokens per variant per model."""
    print(f"% =====  Aggregate efficiency table  =====")
    print(r"\begin{table}[t]")
    print(r"\centering")
    print(r"\small")
    print(r"\resizebox{\columnwidth}{!}{%")
    print(r"\begin{tabular}{llrrrr}")
    print(r"\toprule")
    print(r"Model & Method & EM & F1 & Lat (ms) & Ctx tok \\")
    print(r"\midrule")
    for m, mlabel in MODELS:
        local_variants = (VARIANTS if m == "lfm2.5-1.2b-instruct"
                          else [v for v in VARIANTS if v[0] != "raw_topk_b840"])
        for i, (v, vlabel) in enumerate(local_variants):
            ems, f1s, lats, ctxs = [], [], [], []
            for t, _, n in TASKS:
                d = load_metrics(m, t, v, n)
                if d:
                    ems.append(d["em"]); f1s.append(d["f1"])
                    lats.append(d.get("avg_latency_ms", 0))
                    ctxs.append(d.get("avg_context_tokens", 0))
            if not ems:
                continue
            n_ = len(ems)
            mlab_cell = f"\\multirow{{{len(local_variants)}}}{{*}}{{{mlabel}}}" if i == 0 else ""
            print(f"{mlab_cell} & {vlabel} & {sum(ems)/n_:.1f} & {sum(f1s)/n_:.1f} & "
                  f"{sum(lats)/n_:.0f} & {sum(ctxs)/n_:.0f} \\\\")
        print(r"\midrule" if m != MODELS[-1][0] else r"\bottomrule")
    print(r"\end{tabular}")
    print(r"}")
    print(r"\caption{Aggregate accuracy and efficiency across 5 LongBench tasks. "
          r"Lat = \emph{median} end-to-end query latency (robust to rare prefill "
          r"hangs); Ctx tok = mean reader-input context tokens.}")
    print(r"\label{tab:efficiency}")
    print(r"\end{table}")


def ablation_table(model: str = "lfm2.5-1.2b-instruct",
                    model_label: str = "LFM2.5-1.2B"):
    """Novelty-defense ablation: PEC-Hop vs three single-axis ablations."""
    abl_variants = [
        ("pec_hop", r"\textbf{PEC-Hop} (full)"),
        ("pec_hop_no_anchor", "\\quad $-$ anchor expansion"),
        ("pec_hop_fact_only", "\\quad $-$ card schema (sentence-only)"),
        ("pec_hop_no_hydration", "\\quad $-$ hydration"),
    ]
    print(f"% =====  {model_label} novelty-defense ablation  =====")
    print(r"\begin{table*}[t]")
    print(r"\centering")
    print(r"\small")
    print(r"\begin{tabular}{l" + "rr" * len(TASKS) + r"rr}")
    print(r"\toprule")
    print(r"\multirow{2}{*}{Method}" +
          "".join(f" & \\multicolumn{{2}}{{c}}{{{label}}}" for _, label, _ in TASKS) +
          r" & \multicolumn{2}{c}{Mean} \\")
    cm_idx = 2
    cm_str = "".join([f" \\cmidrule(lr){{{cm_idx + 2*i}-{cm_idx + 2*i + 1}}}"
                       for i in range(len(TASKS) + 1)])
    print(cm_str)
    print(r"  " + "".join(f" & EM & F1" for _ in range(len(TASKS) + 1)) + r" \\")
    print(r"\midrule")
    for v, vlabel in abl_variants:
        cells = [vlabel]
        ems, f1s = [], []
        for t, _, n in TASKS:
            m = load_metrics(model, t, v, n)
            if m is None:
                cells.extend(["--", "--"])
                continue
            cells.append(f"{m['em']:.1f}"); cells.append(f"{m['f1']:.1f}")
            ems.append(m['em']); f1s.append(m['f1'])
        if ems:
            cells.append(f"{sum(ems)/len(ems):.1f}")
            cells.append(f"{sum(f1s)/len(f1s):.1f}")
        print(" & ".join(cells), r"\\")
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\caption{Novelty-defense ablation on " + model_label +
          r". Each row removes one design choice from \textbf{PEC-Hop} while "
          r"holding the others fixed. ``$-$ anchor expansion'' replaces the "
          r"step-2 query with the original question; ``$-$ card schema'' uses "
          r"bare-sentence cards with no TITLE/ANCHORS/PTR markup; ``$-$ hydration'' "
          r"presents the verbatim card text without recovering the $\pm 1$-sentence "
          r"window. EM and F1 in percentage points.}")
    print(r"\label{tab:ablation}")
    print(r"\end{table*}")
    print()


def combined_ablation_table(model: str = "lfm2.5-1.2b-instruct",
                              model_label: str = "LFM2.5-1.2B"):
    """Combined ablation + hydration sweep + coverage diagnostic in ONE
    compact float to save the page-budget."""
    cov_path = RESULTS_DIR / "_coverage_diagnostic.json"
    cov_data = json.load(open(cov_path)) if cov_path.exists() else []
    def cov_for(v):
        rows = [r for r in cov_data if r["variant"] == v]
        if not rows:
            return None
        return sum(r["gold_in_ctx"] for r in rows) / len(rows)

    groups = [
        ("\\textit{Component ablations}", [
            ("pec_hop", r"\textbf{PEC-Hop} (full)"),
            ("pec_hop_no_anchor", "\\quad $-$ anchor expansion"),
            ("pec_hop_fact_only", "\\quad $-$ card schema"),
            ("pec_hop_no_hydration", "\\quad $-$ hydration"),
        ]),
        ("\\textit{Hydration window sweep}", [
            ("pec_hop_w0", "\\quad $w{=}0$ (sentence)"),
            ("pec_hop_w2", "\\quad $w{=}2$ (wider)"),
        ]),
    ]
    print(f"% =====  Combined ablation + hydration + coverage ({model_label})  =====")
    print(r"\begin{table}[t]")
    print(r"\centering")
    print(r"\footnotesize")
    print(r"\setlength{\tabcolsep}{4pt}")
    print(r"\begin{tabular}{lrrrrr}")
    print(r"\toprule")
    print(r"Variant & EM & F1 & Lat & Ctx & gold-in-ctx \\")
    print(r"        &    &    & (ms)& tok& (\%) \\")
    for group_label, variants in groups:
        print(r"\midrule")
        print(f"\\multicolumn{{6}}{{l}}{{{group_label}}} \\\\")
        for v, vlabel in variants:
            ems, f1s, lats, ctxs = [], [], [], []
            for t, _, n in TASKS:
                d = load_metrics(model, t, v, n)
                if d:
                    ems.append(d["em"]); f1s.append(d["f1"])
                    lats.append(d.get("avg_latency_ms", 0))
                    ctxs.append(d.get("avg_context_tokens", 0))
            if not ems:
                continue
            n_ = len(ems)
            cov = cov_for(v)
            cov_str = f"{cov:.1f}" if cov is not None else "--"
            print(f"{vlabel} & {sum(ems)/n_:.1f} & {sum(f1s)/n_:.1f} & "
                  f"{sum(lats)/n_:.0f} & {sum(ctxs)/n_:.0f} & {cov_str} \\\\")
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\caption{Single-axis ablations on " + model_label +
          r" (means of EM/F1/ctx, \emph{median} latency, and gold-answer-in-"
          r"context rate across the five LongBench tasks). The component block "
          r"isolates each \family{} design choice; the hydration block sweeps "
          r"$w \in \{0,1,2\}$ around the verbatim source sentence. $w{=}1$ is "
          r"the default; $w{=}2$ adds $\sim 75\%$ median latency without a "
          r"quality gain. Note that bare-sentence cards ($-$ schema) lose "
          r"$10$~pp of gold coverage and removing hydration loses another "
          r"$6$~pp, indicating that the schema and the local sentence window "
          r"jointly do retrieval work.}")
    print(r"\label{tab:ablation}")
    print(r"\end{table}")
    print()


def hydration_sweep_table(model: str = "lfm2.5-1.2b-instruct",
                            model_label: str = "LFM2.5-1.2B"):
    """Hydration window sweep: w=0 (sentence-only), w=1 (default), w=2 (wider)."""
    rows = [
        ("pec_hop_w0", "$w{=}0$ (sentence)"),
        ("pec_hop", "$w{=}1$ (default)"),
        ("pec_hop_w2", "$w{=}2$ (wider)"),
    ]
    print(f"% =====  Hydration window sweep ({model_label})  =====")
    print(r"\begin{table}[t]")
    print(r"\centering")
    print(r"\small")
    print(r"\begin{tabular}{lrrrr}")
    print(r"\toprule")
    print(r"Window & EM & F1 & Lat (ms) & Ctx tok \\")
    print(r"\midrule")
    for v, vlabel in rows:
        ems, f1s, lats, ctxs = [], [], [], []
        for t, _, n in TASKS:
            d = load_metrics(model, t, v, n)
            if d:
                ems.append(d["em"]); f1s.append(d["f1"])
                lats.append(d.get("avg_latency_ms", 0))
                ctxs.append(d.get("avg_context_tokens", 0))
        if not ems:
            continue
        n_ = len(ems)
        print(f"{vlabel} & {sum(ems)/n_:.1f} & {sum(f1s)/n_:.1f} & "
              f"{sum(lats)/n_:.0f} & {sum(ctxs)/n_:.0f} \\\\")
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\caption{Hydration window sweep on " + model_label +
          r" (means across the five LongBench tasks). Window~0 keeps only the "
          r"verbatim source sentence; window~1 is the \ours{} default and adds "
          r"the immediate sentence neighbours; window~2 widens to two sentences "
          r"on each side.}")
    print(r"\label{tab:hydration-sweep}")
    print(r"\end{table}")
    print()


def dynamic_table():
    """PEC-Hop-Dynamic vs static PEC-Hop and Raw RAG. Reports per-cell EM/F1/lat
    so the rescue effect on Qwen3 multi-hop is visible."""
    rows = [
        ("raw_topk", "Raw RAG"),
        ("pec_hop", "PEC-Hop"),
        ("pec_hop_dynamic", r"\textbf{PEC-Hop-Dyn}"),
    ]
    cells = [
        ("lfm2.5-1.2b-instruct", "LFM2.5-1.2B", "hotpotqa", 200, "HotpotQA"),
        ("lfm2.5-1.2b-instruct", "LFM2.5-1.2B", "2wikimqa", 200, "2WikiMQA"),
        ("lfm2.5-1.2b-instruct", "LFM2.5-1.2B", "musique", 200, "MuSiQue"),
        ("qwen3-1.7b", "Qwen3-1.7B", "hotpotqa", 200, "HotpotQA"),
        ("qwen3-1.7b", "Qwen3-1.7B", "2wikimqa", 200, "2WikiMQA"),
        ("qwen3-1.7b", "Qwen3-1.7B", "musique", 200, "MuSiQue"),
    ]
    print(f"% =====  Dynamic fallback table  =====")
    print(r"\begin{table}[t]")
    print(r"\centering")
    print(r"\small")
    print(r"\begin{tabular}{llrr}")
    print(r"\toprule")
    print(r"Reader / Task & Method & EM & F1 \\")
    print(r"\midrule")
    last_cell = None
    for model, mlabel, task, n, tlabel in cells:
        for v, vlabel in rows:
            d = load_metrics(model, task, v, n)
            if not d:
                continue
            cell = f"{mlabel} / {tlabel}" if (model, task) != last_cell else ""
            print(f"{cell} & {vlabel} & {d['em']:.1f} & {d['f1']:.1f} \\\\")
            last_cell = (model, task)
        print(r"\addlinespace[2pt]")
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\caption{Dynamic fallback. At step~1 we test \emph{evidence\_is\_weak} "
          r"(low title diversity, low top-card score, or low question--anchor "
          r"overlap) and switch to the \bridge{} prompt; otherwise we use \ours{}. "
          r"This simple heuristic gates too aggressively on LFM (where \ours{} "
          r"already wins) and produces only marginal gains on Qwen3-1.7B. We "
          r"report it as a \emph{negative} preliminary result: the capacity "
          r"threshold visible in \S\ref{sec:stress} cannot be cheaply rescued by "
          r"this particular gate; principled gating is left to future work.}")
    print(r"\label{tab:dynamic}")
    print(r"\end{table}")
    print()


def coverage_table():
    """Inline coverage diagnostic into the ablation table to save a float slot.
    The actual table is now folded into combined_ablation_table; this stub is
    kept for backward compatibility but emits no float."""
    return


if __name__ == "__main__":
    main_results_table(
        "lfm2.5-1.2b-instruct", "LFM2.5-1.2B", "main-lfm",
        "Main results on five LongBench tasks for LFM2.5-Inst 1.2B "
        "(n=200 each task except MFQA-en n=150). EM and F1 in percentage points; "
        "best per (task, metric) in bold."
    )
    noninferiority_table("lfm2.5-1.2b-instruct", "LFM2.5-1.2B", margin=3.0)
    combined_ablation_table()
    coverage_table()
    # Qwen3 stress test: skip Raw RAG_840 (not run on Qwen3).
    qwen_variants = [v for v in VARIANTS if v[0] != "raw_topk_b840"]
    main_results_table(
        "qwen3-1.7b", "Qwen3-1.7B", "stress-qwen",
        "Stress test on Qwen3-1.7B across the same five LongBench tasks. "
        "PEC-Hop loses 2--5 EM points on multi-hop tasks, revealing a capacity "
        "threshold beyond which compressed evidence loses bridge facts; "
        "single-doc tasks remain comparable.",
        variants_override=qwen_variants,
    )
    efficiency_table()
