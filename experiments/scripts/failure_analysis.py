"""Find failure cases of PEC-Hop relative to Raw RAG and dump them for
manual categorization.

For each (model, task) cell, identifies questions where Raw RAG gets
EM=1 but PEC-Hop gets EM=0 (and vice versa for completeness). Saves
the question, gold answer, both predictions, and selected_titles to
a JSONL file the human can label.

Usage:
  python failure_analysis.py --model lfm2.5-1.2b-instruct --task 2wikimqa --n 200
  python failure_analysis.py --all  # all cells
"""
import argparse
import json
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results" / "longbench"

CELLS_DEFAULT = [
    ("lfm2.5-1.2b-instruct", "hotpotqa", 200),
    ("lfm2.5-1.2b-instruct", "2wikimqa", 200),
    ("lfm2.5-1.2b-instruct", "musique", 200),
    ("lfm2.5-1.2b-instruct", "multifieldqa_en", 150),
    ("lfm2.5-1.2b-instruct", "qasper", 200),
    ("qwen3-1.7b", "hotpotqa", 200),
    ("qwen3-1.7b", "2wikimqa", 200),
    ("qwen3-1.7b", "musique", 200),
]


def load(model: str, task: str, variant: str, n: int) -> list:
    for p in [
        RESULTS_DIR / f"{model}_{task}_{variant}_n{n}.json",
        RESULTS_DIR / f"{model}_{task}_{variant}.json",
    ]:
        if p.exists():
            return json.load(open(p, encoding="utf-8")).get("results", [])
    return []


def diff_cell(model: str, task: str, n: int, out_dir: Path):
    raw = load(model, task, "raw_topk", n)
    hop = load(model, task, "pec_hop", n)
    if not raw or not hop:
        print(f"[skip] {model}/{task}: missing files")
        return
    raw_by = {r["question"]: r for r in raw}
    hop_by = {r["question"]: r for r in hop}
    common = [q for q in [r["question"] for r in raw] if q in hop_by]

    pec_loses = []   # raw correct, pec wrong
    pec_wins = []    # pec correct, raw wrong
    both_right = both_wrong = 0
    for q in common:
        r = raw_by[q]; h = hop_by[q]
        re_ = int(r["em"]); he = int(h["em"])
        rec = {
            "question": q,
            "gold": r["answers"],
            "raw_pred": r["prediction"],
            "raw_em": re_, "raw_f1": r["f1"], "raw_loose": r.get("loose", -1),
            "raw_titles": r.get("selected_titles", []),
            "hop_pred": h["prediction"],
            "hop_em": he, "hop_f1": h["f1"], "hop_loose": h.get("loose", -1),
            "hop_titles": h.get("selected_titles", []),
            "category": "TODO",  # human to fill
        }
        if re_ == 1 and he == 0:
            pec_loses.append(rec)
        elif re_ == 0 and he == 1:
            pec_wins.append(rec)
        elif re_ == 1 and he == 1:
            both_right += 1
        else:
            both_wrong += 1

    print(f"\n[{model}/{task}]  n={len(common)}  "
          f"both_right={both_right}  both_wrong={both_wrong}  "
          f"pec_wins={len(pec_wins)}  pec_loses={len(pec_loses)}")

    out_dir.mkdir(parents=True, exist_ok=True)
    out_loses = out_dir / f"failures_{model}_{task}_pec_loses.jsonl"
    out_wins = out_dir / f"failures_{model}_{task}_pec_wins.jsonl"
    with open(out_loses, "w", encoding="utf-8") as f:
        for r in pec_loses:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(out_wins, "w", encoding="utf-8") as f:
        for r in pec_wins:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  -> {out_loses.name}  {out_wins.name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model")
    ap.add_argument("--task")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()

    out_dir = RESULTS_DIR.parent / "failure_analysis"
    if args.all:
        for m, t, n in CELLS_DEFAULT:
            diff_cell(m, t, n, out_dir)
    elif args.model and args.task:
        diff_cell(args.model, args.task, args.n, out_dir)
    else:
        ap.error("Specify --model and --task, or --all")


if __name__ == "__main__":
    main()
