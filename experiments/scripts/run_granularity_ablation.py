"""
Granularity Ablation: isolate aggregation effect vs summarization effect.

Four eval conditions on the same reader model (some may be skipped):
  1. chunk40_raw  — fine-grained raw chunks (~40 words, ~2.8/article, traditional RAG)
  2. chunk100_raw — coarse raw chunks (~100 words, ~1.2/article; near article-level)
  3. article_raw  — article-level raw text (current 'rag' mode)
  4. article_note — article-level compiled summary (current 'entity' mode, our CTKS)

Interpretation:
  (2) - (1) = AGGREGATION effect (whole-article context better than fragmented chunks)
  (3) - (2) = SUMMARIZATION effect (compilation beats aggregated raw)
  (3) - (1) = FULL CTKS gap

Prerequisites:
  - cache/hotpotqa_chunk_index/        (built by build_chunk_level_index.py)
  - cache/hotpotqa_raw_index/          (already exists, article-level)
  - cache/hotpotqa_entity_index/       (already exists)

Usage:
  python run_granularity_ablation.py --model lfm2.5-1.2b-instruct --dataset hotpotqa --max-samples 500
"""
import argparse
import subprocess
import sys
import json
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "shared"))
from utils.config import TOPIC6_DIR


def run_cell(model: str, dataset: str, mode: str, max_samples: int,
             raw_dir: str = None, entity_dir: str = None,
             result_tag: str = None, python_exe: str = None):
    script = Path(__file__).parent / "run_baseline.py"
    cmd = [python_exe, "-u", str(script),
           "--model", model,
           "--dataset", dataset,
           "--mode", mode,
           "--max-samples", str(max_samples)]
    if raw_dir:     cmd += ["--raw-dir", raw_dir]
    if entity_dir:  cmd += ["--entity-dir", entity_dir]
    if result_tag:  cmd += ["--result-tag", result_tag]
    print(f"\n[Cell] {result_tag}: {' '.join(cmd)}")
    t0 = time.time()
    subprocess.run(cmd, capture_output=False)
    print(f"[Cell] {result_tag} done in {time.time()-t0:.0f}s")


def collect(model: str, dataset: str):
    rd = TOPIC6_DIR / "experiments" / "results"
    rows = []
    cells = [
        ("chunk40_raw",  f"{model}_{dataset}_rag_granChunk40.json"),
        ("chunk100_raw", f"{model}_{dataset}_rag_granChunk100.json"),
        ("article_raw",  f"{model}_{dataset}_rag_granArticle.json"),
        ("article_note", f"{model}_{dataset}_entity_granNote.json"),
    ]
    for name, fname in cells:
        p = rd / fname
        if not p.exists():
            rows.append({"cell": name, "status": "missing"})
            continue
        with open(p, encoding="utf-8") as f:
            data = json.load(f)
        m = data.get("metrics", {})
        rows.append({
            "cell": name,
            "n": data.get("n_samples"),
            "em": m.get("em"),
            "f1": m.get("f1"),
            "avg_total_ms": m.get("avg_total_ms"),
        })
    summary = {"model": model, "dataset": dataset, "cells": rows}
    out = TOPIC6_DIR / "experiments" / "results" / f"granularity_{model}_{dataset}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"Granularity Ablation — {model} / {dataset}")
    print(f"{'='*60}")
    print(f"{'Cell':<14} {'EM':>6} {'F1':>6} {'Lat(ms)':>10}")
    for r in rows:
        if r.get("status") == "missing":
            print(f"{r['cell']:<14} (missing)")
        else:
            print(f"{r['cell']:<14} {r.get('em',0):>6.2f} {r.get('f1',0):>6.2f} "
                  f"{r.get('avg_total_ms',0):>10.1f}")
    # Derived
    try:
        ems = {r["cell"]: r.get("em", 0) for r in rows if r.get("em") is not None}
        ref_chunk = ems.get("chunk40_raw", ems.get("chunk100_raw"))
        if ref_chunk is not None and "article_raw" in ems and "article_note" in ems:
            agg_effect = ems["article_raw"] - ref_chunk
            sum_effect = ems["article_note"] - ems["article_raw"]
            full_gap  = ems["article_note"] - ref_chunk
            print(f"\nAggregation effect   (article_raw - chunk_raw) : {agg_effect:+.2f} EM")
            print(f"Summarization effect (article_note - article_raw): {sum_effect:+.2f} EM")
            print(f"Full CTKS gap        (article_note - chunk_raw)  : {full_gap:+.2f} EM")
    except Exception:
        pass
    print(f"\n[Saved] {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="lfm2.5-1.2b-instruct")
    parser.add_argument("--dataset", default="hotpotqa")
    parser.add_argument("--max-samples", type=int, default=500)
    parser.add_argument("--python",
                        default=r"python")
    parser.add_argument("--collect-only", action="store_true")
    args = parser.parse_args()

    if args.collect_only:
        collect(args.model, args.dataset)
        sys.exit(0)

    # Cell 1: fine-grained chunk-level raw (~40w, ~2.8/article)
    run_cell(args.model, args.dataset, "rag", args.max_samples,
             raw_dir=f"{args.dataset}_chunk_index_c40",
             result_tag="granChunk40", python_exe=args.python)

    # Cell 2: coarse chunk-level raw (~100w, ~1.2/article)
    run_cell(args.model, args.dataset, "rag", args.max_samples,
             raw_dir=f"{args.dataset}_chunk_index",
             result_tag="granChunk100", python_exe=args.python)

    # Cell 3: article-level raw (uses default *_raw_index)
    run_cell(args.model, args.dataset, "rag", args.max_samples,
             raw_dir=f"{args.dataset}_raw_index",
             result_tag="granArticle", python_exe=args.python)

    # Cell 4: article-level summary (CTKS entity, default index)
    run_cell(args.model, args.dataset, "entity", args.max_samples,
             entity_dir=f"{args.dataset}_entity_index",
             result_tag="granNote", python_exe=args.python)

    collect(args.model, args.dataset)
