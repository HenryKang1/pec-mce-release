"""
Summarize all experimental results into a single paper-ready JSON + Markdown.

Reads from results/:
  - {model}_{dataset}_{mode}.json       -- base EM numbers
  - conversion_{model}_{dataset}.json    -- retrieval/extraction split
  - coverage_{dataset}.json              -- coverage metrics
  - matching_matrix_{dataset}.json       -- 3x3 matrix (if done)
  - granularity_{model}_{dataset}.json   -- 3-cell ablation (if done)
  - analysis_extracted.json              -- significance + latency

Outputs:
  - results/SUMMARY.json        -- machine-readable aggregate
  - results/SUMMARY.md          -- human-readable tables
"""
import json
from pathlib import Path

RESULTS = Path(__file__).resolve().parents[1] / "results"

MODELS = ["lfm2.5-1.2b-instruct", "qwen3-0.6b", "qwen3-1.7b"]
DATASETS = ["hotpotqa", "2wikimqa"]
MODES = ["rag", "entity", "compress", "con", "oracle"]


def load_json(path: Path, default=None):
    if not path.exists():
        return default
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def main():
    summary = {"main_results": {}, "conversion": {}, "coverage": {},
               "matching_matrix": {}, "granularity": {}}

    # Main EM results
    for m in MODELS:
        for d in DATASETS:
            for mode in MODES:
                path = RESULTS / f"{m}_{d}_{mode}.json"
                data = load_json(path)
                if data and "metrics" in data:
                    key = f"{m}|{d}|{mode}"
                    summary["main_results"][key] = {
                        "em": data["metrics"].get("em"),
                        "f1": data["metrics"].get("f1"),
                        "avg_total_ms": data["metrics"].get("avg_total_ms"),
                        "n": data.get("n_samples"),
                    }

    # Conversion
    for m in MODELS:
        for d in DATASETS:
            path = RESULTS / f"conversion_{m}_{d}.json"
            data = load_json(path)
            if data:
                summary["conversion"][f"{m}|{d}"] = {
                    "rag":  data["rag"],
                    "ctks": data["ctks"],
                    "delta": data["delta"],
                }

    # Coverage
    for d in DATASETS:
        path = RESULTS / f"coverage_{d}.json"
        data = load_json(path)
        if data:
            summary["coverage"][d] = data

    # Matching matrix
    for d in DATASETS:
        path = RESULTS / f"matching_matrix_{d}.json"
        data = load_json(path)
        if data:
            summary["matching_matrix"][d] = data

    # Granularity
    for m in MODELS:
        for d in DATASETS:
            path = RESULTS / f"granularity_{m}_{d}.json"
            data = load_json(path)
            if data:
                summary["granularity"][f"{m}|{d}"] = data

    # Write JSON
    with open(RESULTS / "SUMMARY.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # Write Markdown
    lines = ["# CTKS Experiments Summary", ""]
    lines.append("## Main EM/F1 Results (full datasets)")
    lines.append("")
    lines.append("| Model | Dataset | Mode | n | EM | F1 | Latency (ms) |")
    lines.append("|-------|---------|------|---|-----|-----|--------------|")
    for key, v in summary["main_results"].items():
        m, d, mode = key.split("|")
        lines.append(f"| {m} | {d} | {mode} | {v['n']} | {v['em']} | {v['f1']} | {v['avg_total_ms']} |")

    if summary["conversion"]:
        lines.append("")
        lines.append("## Retrieval→Extraction Conversion (KEY FINDING)")
        lines.append("")
        lines.append("| Model | Dataset | Method | P(ans∈retrieval) | EM | Conv Rate |")
        lines.append("|-------|---------|--------|-----------------|-----|-----------|")
        for key, v in summary["conversion"].items():
            m, d = key.split("|")
            r, c = v["rag"], v["ctks"]
            lines.append(f"| {m} | {d} | RAG  | {r['retrieval_hit_rate']*100:.1f}% | {r['em_rate']*100:.1f}% | {r['conversion_rate']*100:.1f}% |")
            lines.append(f"| {m} | {d} | CTKS | {c['retrieval_hit_rate']*100:.1f}% | {c['em_rate']*100:.1f}% | **{c['conversion_rate']*100:.1f}%** |")

    if summary["coverage"]:
        lines.append("")
        lines.append("## Retrieval Coverage (gold supporting facts)")
        lines.append("")
        for d, v in summary["coverage"].items():
            lines.append(f"### {d} (n={v.get('n_evaluated')})")
            r, e = v["raw"], v["entity"]
            lines.append("| Metric | RAG (raw) | CTKS (entity) |")
            lines.append("|--------|-----------|---------------|")
            for k in ["retrieved_tokens", "title_recall", "answer_recall",
                      "entity_recall", "token_f1"]:
                lines.append(f"| {k} | {r[k]:.4f} | {e[k]:.4f} |")
            lines.append("")

    if summary["matching_matrix"]:
        lines.append("## Matching Matrix (compiler × reader)")
        for d, mat in summary["matching_matrix"].items():
            lines.append(f"### {d}")
            lines.append("| Reader\\Compiler | " + " | ".join(["lfm", "qwen06", "qwen17"]) + " |")
            lines.append("|-----------|------|-------|-------|")
            for reader in ["lfm", "qwen06", "qwen17"]:
                row = mat.get(reader, {})
                cells = []
                for compiler in ["lfm", "qwen06", "qwen17"]:
                    c = row.get(compiler)
                    if c is None:
                        cells.append("--")
                    else:
                        marker = "**" if reader == compiler else ""
                        cells.append(f"{marker}{c['em']:.1f}{marker}")
                lines.append(f"| {reader} | " + " | ".join(cells) + " |")
            lines.append("")

    if summary["granularity"]:
        lines.append("## Granularity Ablation")
        for key, g in summary["granularity"].items():
            m, d = key.split("|")
            lines.append(f"### {m} / {d}")
            lines.append("| Cell | EM | F1 | Latency |")
            lines.append("|------|-----|-----|---------|")
            for row in g["cells"]:
                if row.get("status") == "missing":
                    lines.append(f"| {row['cell']} | (missing) |||")
                else:
                    lines.append(f"| {row['cell']} | {row.get('em',0):.2f} | {row.get('f1',0):.2f} | {row.get('avg_total_ms',0):.1f} |")
            lines.append("")

    out_md = RESULTS / "SUMMARY.md"
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"[Saved] {RESULTS / 'SUMMARY.json'}")
    print(f"[Saved] {out_md}")


if __name__ == "__main__":
    main()
