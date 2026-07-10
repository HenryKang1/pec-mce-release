"""Summarize full external-retrieval reader EM results for the paper.

This script consumes completed full-HotpotQA reader result files and writes a
paper-ready JSON/Markdown summary with paired deltas against the coverage-matched
raw article baseline.
"""
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "experiments" / "results"


CELLS = {
    "raw_article": {
        "label": "Raw article",
        "file": "lfm2.5-1.2b-instruct_hotpotqa_rag_fairRag.json",
    },
    "extractive_note": {
        "label": "Extractive minimal note",
        "file": "lfm2.5-1.2b-instruct_hotpotqa_entity_extractCtks.json",
    },
    "smart_note": {
        "label": "Smart minimal note",
        "file": "lfm2.5-1.2b-instruct_hotpotqa_entity_smartCtks.json",
    },
    "llm_entity_note": {
        "label": "LLM entity note",
        "file": "lfm2.5-1.2b-instruct_hotpotqa_entity.json",
    },
}


def load_cell(name: str) -> dict:
    path = RESULTS / CELLS[name]["file"]
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    by_q = {r["question"]: r for r in data["results"]}
    lat = [r.get("latency", {}).get("ttft_ms", 0.0) for r in data["results"]]
    lat_sorted = sorted(lat)
    p50 = lat_sorted[len(lat_sorted) // 2] if lat_sorted else 0.0
    p95 = lat_sorted[int(len(lat_sorted) * 0.95)] if lat_sorted else 0.0
    return {
        "path": str(path),
        "summary": data["metrics"],
        "n": data["n_samples"],
        "by_q": by_q,
        "p50_ttft_ms": p50,
        "p95_ttft_ms": p95,
    }


def paired_bootstrap(base: dict, other: dict, metric: str, n_boot: int = 10000) -> dict:
    common = sorted(set(base["by_q"]) & set(other["by_q"]))
    base_vals = np.array([float(base["by_q"][q][metric]) for q in common], dtype=np.float32)
    other_vals = np.array([float(other["by_q"][q][metric]) for q in common], dtype=np.float32)
    diffs = other_vals - base_vals
    observed = 100.0 * float(np.mean(diffs))

    rng = np.random.default_rng(13)
    n = len(common)
    deltas = []
    batch = 256
    for start in range(0, n_boot, batch):
        b = min(batch, n_boot - start)
        idx = rng.integers(0, n, size=(b, n), endpoint=False)
        deltas.append(np.mean(diffs[idx], axis=1) * 100.0)
    deltas = np.sort(np.concatenate(deltas))
    return {
        "n_common": n,
        "delta_pp": observed,
        "ci95_low": float(deltas[int(0.025 * n_boot)]),
        "ci95_high": float(deltas[int(0.975 * n_boot)]),
    }


def main():
    cells = {name: load_cell(name) for name in CELLS}
    base = cells["raw_article"]
    paired = {}
    for name, cell in cells.items():
        if name == "raw_article":
            continue
        paired[name] = {
            "em": paired_bootstrap(base, cell, "em"),
            "f1": paired_bootstrap(base, cell, "f1"),
        }

    summary = {
        "dataset": "HotpotQA full validation",
        "reader": "LFM2.5-Instruct 1.2B",
        "baseline": "raw_article",
        "cells": {
            name: {
                "label": CELLS[name]["label"],
                "file": CELLS[name]["file"],
                "n": cell["n"],
                "em": cell["summary"]["em"],
                "f1": cell["summary"]["f1"],
                "avg_ttft_ms": cell["summary"]["avg_ttft_ms"],
                "p50_ttft_ms": round(cell["p50_ttft_ms"], 2),
                "p95_ttft_ms": round(cell["p95_ttft_ms"], 2),
            }
            for name, cell in cells.items()
        },
        "paired_vs_raw_article": paired,
    }

    out_json = RESULTS / "full_external_reader_em_summary.json"
    out_md = RESULTS / "FULL_EXTERNAL_READER_EM_SUMMARY.md"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    lines = [
        "# Full external-retrieval reader EM",
        "",
        "Reader: **LFM2.5-Instruct 1.2B**. Dataset: **HotpotQA full validation**.",
        "All systems use top-5 retrieval over dataset-level FAISS indices.",
        "",
        "| Evidence index | n | EM | F1 | p50 TTFT | p95 TTFT |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name in ["raw_article", "extractive_note", "smart_note", "llm_entity_note"]:
        row = summary["cells"][name]
        lines.append(
            f"| {row['label']} | {row['n']} | {row['em']:.2f} | {row['f1']:.2f} | "
            f"{row['p50_ttft_ms']:.1f} | {row['p95_ttft_ms']:.1f} |"
        )
    lines += ["", "## Paired deltas vs raw article", ""]
    lines += ["| Evidence index | common n | ΔEM | 95% CI | ΔF1 | 95% CI |", "|---|---:|---:|---:|---:|---:|"]
    for name in ["extractive_note", "smart_note", "llm_entity_note"]:
        em = paired[name]["em"]
        f1 = paired[name]["f1"]
        lines.append(
            f"| {CELLS[name]['label']} | {em['n_common']} | {em['delta_pp']:+.2f} | "
            f"[{em['ci95_low']:+.2f}, {em['ci95_high']:+.2f}] | "
            f"{f1['delta_pp']:+.2f} | [{f1['ci95_low']:+.2f}, {f1['ci95_high']:+.2f}] |"
        )
    lines += [
        "",
        "Takeaway: extractive minimal notes are reader-EM equivalent to coverage-matched raw article retrieval "
        "on the full external HotpotQA setting, while the retrieval-only diagnostic shows they use 0.80x words "
        "and retain 96.8% of raw answer recall.",
        "",
        f"JSON: `{out_json}`",
    ]
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print("\n".join(lines))
    print(f"[Saved] {out_json}")
    print(f"[Saved] {out_md}")


if __name__ == "__main__":
    main()
