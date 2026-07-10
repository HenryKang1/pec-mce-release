"""MCE-Compass benchmark with the full PEC-Hop in the candidate family.

Restricted candidate set per (reader, task) cell:
  - raw_topk × {default, extractive, short15, concise}      (4)
  - sentence_only × {default, extractive, short15, concise} (4)
  - pec_hop_rerank × {fewextractive}                        (1)  ← full PEC-Hop

Computes all 5 policies used in the paper's headline table:
  - Raw RAG (raw_topk+default)
  - Single dev-selected (global best single across all readers/tasks)
  - Dev-Argmax (per-cell best on dev)
  - MCE-Compass
  - Oracle (per-question max EM upper bound)
"""
from __future__ import annotations

import json
from pathlib import Path
from statistics import mean

import mce_select_cost_benchmark as base


FULL_PEC_BASE = "pec_hop_rerank"
FULL_PEC_PROMPT = "fewextractive"
RESTRICTED_BASES = ["raw_topk", "sentence_only"]
# Include fewextractive as a fifth prompt for Raw RAG and Sentence-only baselines.
RESTRICTED_PROMPTS = base.PROMPTS + ["fewextractive"]  # 5 prompts


def candidates_for_full(reader: str, task: str) -> dict[base.Candidate, dict]:
    out: dict[base.Candidate, dict] = {}
    for b in RESTRICTED_BASES:
        for prompt in RESTRICTED_PROMPTS:
            cand = base.Candidate(b, prompt)
            data = base.load_json(reader, task, cand)
            if data is not None:
                out[cand] = data
    pec_cand = base.Candidate(FULL_PEC_BASE, FULL_PEC_PROMPT)
    pec_data = base.load_json(reader, task, pec_cand)
    if pec_data is not None:
        out[pec_cand] = pec_data
    return out


def choose_global_best_single_restricted(all_data, readers):
    best = None
    best_dev = -1.0
    full_cands = [base.Candidate(b, p) for b in RESTRICTED_BASES for p in RESTRICTED_PROMPTS]
    full_cands.append(base.Candidate(FULL_PEC_BASE, FULL_PEC_PROMPT))
    for cand in full_cands:
        cell_scores = []
        complete = True
        for reader in readers:
            for task in base.TASKS:
                data = all_data.get((reader, task), {}).get(cand)
                if data is None:
                    complete = False
                    break
                dev_idx, _ = base.split_indices(task, None)
                cell_scores.append(base.avg_em(data, dev_idx))
            if not complete:
                break
        if not complete:
            continue
        dev = sum(cell_scores) / len(cell_scores)
        if dev > best_dev:
            best = cand
            best_dev = dev
    return best


def main():
    all_data = {
        (reader, task): candidates_for_full(reader, task)
        for reader in base.READERS for task in base.TASKS
    }
    active_readers = [
        r for r in base.READERS
        if all(len(all_data[(r, t)]) >= 1 for t in base.TASKS)
    ]
    best_single = choose_global_best_single_restricted(all_data, active_readers)
    print(f"Global single dev-selected best: {best_single.label if best_single else 'NONE'}")

    per_reader = {}
    per_cell = []
    for reader in active_readers:
        rows_raw, rows_best, rows_argmax, rows_compass, rows_oracle = [], [], [], [], []
        for task in base.TASKS:
            cands = all_data[(reader, task)]
            if not cands:
                continue
            dev_idx, test_idx = base.split_indices(task, None)
            raw = base.Candidate("raw_topk", "default")
            if raw not in cands:
                print(f"[Skip] {reader}/{task}: no Raw RAG")
                continue
            argmax = base.choose_argmax(cands, dev_idx)
            compass = base.choose_compass(cands, dev_idx)

            raw_m = base.eval_candidate(cands[raw], test_idx)
            best_m = base.eval_candidate(cands[best_single], test_idx) if best_single in cands else raw_m
            argmax_m = base.eval_candidate(cands[argmax], test_idx)
            compass_m = base.eval_candidate(cands[compass], test_idx)

            oracle_em = []
            oracle_f1 = []
            for i in test_idx:
                oracle_em.append(max(float(d["results"][i]["em"]) for d in cands.values()))
                oracle_f1.append(max(float(d["results"][i]["f1"]) for d in cands.values()))
            oracle_m = {"em": 100.0*mean(oracle_em), "f1": 100.0*mean(oracle_f1), "latency": 0.0, "context": 0.0}

            rows_raw.append(raw_m); rows_best.append(best_m); rows_argmax.append(argmax_m)
            rows_compass.append(compass_m); rows_oracle.append(oracle_m)

            per_cell.append({
                "reader": reader, "task": task,
                "compass_choice": compass.label, "argmax_choice": argmax.label,
                "raw_em": raw_m["em"], "raw_lat": raw_m["latency"], "raw_ctx": raw_m["context"],
                "best_em": best_m["em"], "best_lat": best_m["latency"], "best_ctx": best_m["context"],
                "argmax_em": argmax_m["em"], "argmax_lat": argmax_m["latency"], "argmax_ctx": argmax_m["context"],
                "compass_em": compass_m["em"], "compass_f1": compass_m["f1"],
                "compass_lat": compass_m["latency"], "compass_ctx": compass_m["context"],
                "oracle_em": oracle_m["em"], "oracle_f1": oracle_m["f1"],
            })

        per_reader[reader] = {
            "raw": {k: mean(r[k] for r in rows_raw) for k in ["em","f1","latency","context"]},
            "best": {k: mean(r[k] for r in rows_best) for k in ["em","f1","latency","context"]},
            "argmax": {k: mean(r[k] for r in rows_argmax) for k in ["em","f1","latency","context"]},
            "compass": {k: mean(r[k] for r in rows_compass) for k in ["em","f1","latency","context"]},
            "oracle": {k: mean(r[k] for r in rows_oracle) for k in ["em","f1","latency","context"]},
        }

    # Per-cell printout
    print()
    print(f"{'reader':<26s} {'task':<16s} {'compass_choice':<35s} {'raw_em':>7s} {'compass_em':>11s} {'argmax_em':>10s}")
    for r in per_cell:
        print(f"{r['reader']:<26s} {r['task']:<16s} {r['compass_choice']:<35s} {r['raw_em']:>7.2f} {r['compass_em']:>11.2f} {r['argmax_em']:>10.2f}")

    # Macro per-reader for the headline table
    print()
    print("=" * 110)
    print("HEADLINE TABLE (macro over 5 tasks per reader)")
    print("=" * 110)
    print(f"{'reader':<26s} {'policy':<24s} {'EM':>7s} {'F1':>7s} {'Lat(ms)':>9s} {'Ctx':>6s} {'dEM':>7s} {'speed':>7s}")
    for reader, m in per_reader.items():
        raw_em = m["raw"]["em"]
        raw_lat = m["raw"]["latency"]
        print(f"--- {reader} ---")
        for pol_name, pol_key in [("Raw RAG", "raw"), ("Single dev-selected", "best"), ("Dev-Argmax", "argmax"), ("MCE-Compass", "compass"), ("Oracle", "oracle")]:
            pm = m[pol_key]
            em = pm["em"]; f1 = pm["f1"]; lat = pm["latency"]; ctx = pm["context"]
            d_em = em - raw_em
            speed = raw_lat / lat if lat > 0 else 0.0
            speed_str = f"{speed:>6.2f}x" if pol_key != "oracle" else "  --   "
            lat_str = f"{lat:>9.1f}" if pol_key != "oracle" else "      -- "
            ctx_str = f"{int(ctx):>6d}" if pol_key != "oracle" else "    --"
            print(f"{'':<26s} {pol_name:<24s} {em:>7.2f} {f1:>7.2f} {lat_str} {ctx_str} {d_em:>+7.2f} {speed_str}")

    out = Path(__file__).resolve().parents[1] / "results" / "mce_full_pec_hop_benchmark.json"
    json.dump({"per_cell": per_cell, "per_reader": per_reader,
               "best_single": best_single.label if best_single else None},
              open(out, "w", encoding="utf-8"), indent=2)
    print(f"\n[Saved] {out}")


if __name__ == "__main__":
    main()
