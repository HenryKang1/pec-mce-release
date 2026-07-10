"""Cost-aware MCE-Compass benchmark from completed LongBench JSONs.

This script evaluates three deployment-time selection protocols on the
completed result grid:

  1. MCE-Argmax: pick the (interface, decoding prompt) with the best dev EM.
  2. MCE-Light: pick the fastest candidate whose dev EM is within one
     dev-set question of the best candidate.
  3. MCE-Compass: bootstrap an accuracy-noninferior set, take the
     nondominated accuracy/cost frontier, and select the minimum-burden
     evidence interface with conversion-aware tie-breaking.

MCE-Compass is the paper-facing algorithm: accuracy-constrained minimal
evidence selection over a small, fixed family of evidence interfaces.
"""
from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, stdev


ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "experiments" / "results" / "longbench"
OUT = ROOT / "experiments" / "results"

READERS = ["lfm2.5-1.2b-instruct", "qwen3-4b", "gemma-4-e4b"]
TASKS = ["hotpotqa", "2wikimqa", "musique", "qasper", "multifieldqa_en"]
NS = {
    "hotpotqa": 200,
    "2wikimqa": 200,
    "musique": 200,
    "qasper": 200,
    "multifieldqa_en": 150,
}
DEV_SIZE = {
    "hotpotqa": 50,
    "2wikimqa": 50,
    "musique": 50,
    "qasper": 50,
    "multifieldqa_en": 40,
}

# Paper-facing candidate family: uncompressed raw evidence, minimal sentence
# evidence, and structured multi-hop evidence. raw_topk_b840 remains an
# appendix control, not a deployment interface.
BASES = ["raw_topk", "sentence_only", "pec_hop"]
PROMPTS = ["default", "extractive", "short15", "concise"]

# Tie tolerance for MCE-Light: at most one dev example below the best dev EM.
ONE_QUESTION_TIE = True

# MCE-Compass hyperparameters.  These are deliberately small and interpretable:
# one dev example of EM regret, plus bootstrap noninferiority, then a burden score
# that values latency more than token count.
BOOTSTRAP_ITERS = 1000
BOOTSTRAP_ALPHA = 0.05
F1_TIE_MARGIN = 3.0
TOKEN_BURDEN_WEIGHT = 0.35
NEAR_MIN_BURDEN = 0.10


@dataclass(frozen=True)
class Candidate:
    base: str
    prompt: str

    @property
    def variant(self) -> str:
        return self.base if self.prompt == "default" else f"{self.base}_{self.prompt}"

    @property
    def label(self) -> str:
        return f"{self.base}+{self.prompt}"


def load_json(reader: str, task: str, cand: Candidate) -> dict | None:
    # multifieldqa_en has 150 instances; some bg runs saved as _n200.
    # Try the canonical NS[task] first, then alternatives.
    for n_tag in [NS[task], 150, 200]:
        path = RES / f"{reader}_{task}_{cand.variant}_n{n_tag}.json"
        if path.exists():
            with open(path, encoding="utf-8") as f:
                return json.load(f)
    return None


def candidates_for(reader: str, task: str) -> dict[Candidate, dict]:
    out = {}
    for base in BASES:
        for prompt in PROMPTS:
            cand = Candidate(base, prompt)
            data = load_json(reader, task, cand)
            if data is not None:
                out[cand] = data
    return out


def split_indices(task: str, seed: int | None = None) -> tuple[list[int], list[int]]:
    n = NS[task]
    d = DEV_SIZE[task]
    idx = list(range(n))
    if seed is None:
        return idx[:d], idx[d:]
    rng = random.Random(seed)
    rng.shuffle(idx)
    return idx[:d], idx[d:]


def values(data: dict, field: str, idxs: list[int]) -> list[float]:
    return [float(data["results"][i][field]) for i in idxs]


def avg(data: dict, field: str, idxs: list[int]) -> float:
    vals = values(data, field, idxs)
    return sum(vals) / len(vals) if vals else 0.0


def avg_em(data: dict, idxs: list[int]) -> float:
    return 100.0 * avg(data, "em", idxs)


def avg_f1(data: dict, idxs: list[int]) -> float:
    return 100.0 * avg(data, "f1", idxs)


def avg_latency(data: dict, idxs: list[int]) -> float:
    return avg(data, "latency_ms", idxs)


def avg_context(data: dict, idxs: list[int]) -> float:
    return avg(data, "context_tokens", idxs)


def conversion(data: dict, idxs: list[int]) -> float:
    """Exact-match conversion among answer-bearing predictions."""
    loose = sum(float(data["results"][i].get("loose", 0.0)) for i in idxs)
    if loose <= 0:
        return 0.0
    em = sum(float(data["results"][i]["em"]) for i in idxs)
    return em / loose


def bootstrap_lb_diff(a: dict, b: dict, idxs: list[int], seed: int = 0) -> float:
    """One-sided lower bound for mean(EM_a - EM_b) in percentage points."""
    rng = random.Random(seed)
    diffs = [
        100.0 * (float(a["results"][i]["em"]) - float(b["results"][i]["em"]))
        for i in idxs
    ]
    if not diffs:
        return 0.0
    samples = []
    n = len(diffs)
    for _ in range(BOOTSTRAP_ITERS):
        samples.append(sum(diffs[rng.randrange(n)] for _ in range(n)) / n)
    samples.sort()
    return samples[int(BOOTSTRAP_ALPHA * BOOTSTRAP_ITERS)]


def burden(data: dict, raw_data: dict, idxs: list[int]) -> float:
    """Normalized reader burden relative to Raw RAG on the same dev split."""
    lat = max(avg_latency(data, idxs), 1e-6)
    raw_lat = max(avg_latency(raw_data, idxs), 1e-6)
    ctx = max(avg_context(data, idxs), 1e-6)
    raw_ctx = max(avg_context(raw_data, idxs), 1e-6)
    return math.log(lat / raw_lat) + TOKEN_BURDEN_WEIGHT * math.log(ctx / raw_ctx)


def choose_argmax(cands: dict[Candidate, dict], dev_idx: list[int]) -> Candidate:
    return max(
        cands,
        key=lambda c: (
            avg_em(cands[c], dev_idx),
            avg_f1(cands[c], dev_idx),
            -avg_latency(cands[c], dev_idx),
        ),
    )


def choose_light(cands: dict[Candidate, dict], dev_idx: list[int]) -> Candidate:
    best_em = max(avg_em(data, dev_idx) for data in cands.values())
    eps = 100.0 / len(dev_idx) if ONE_QUESTION_TIE else 0.0
    tied = [
        cand for cand, data in cands.items()
        if avg_em(data, dev_idx) >= best_em - eps - 1e-9
    ]
    # Fastest first; if tied, prefer fewer context tokens, then simpler interface.
    simplicity = {"sentence_only": 0, "pec_hop": 1, "raw_topk": 2}
    return min(
        tied,
        key=lambda c: (
            avg_latency(cands[c], dev_idx),
            avg_context(cands[c], dev_idx),
            simplicity.get(c.base, 99),
            c.label,
        ),
    )


def choose_compass(cands: dict[Candidate, dict], dev_idx: list[int]) -> Candidate:
    """MCE-Compass: accuracy-preserving minimal-evidence selector.

    Step 1: pick the dev-EM leader.
    Step 2: keep candidates that are EM-noninferior to that leader within one
            dev question, using a paired bootstrap lower bound, and F1-noninferior
            within a small fixed margin.  The F1 guard prevents a short-answer
            prompt from winning single-document QA purely by being cheap.
    Step 3: remove candidates dominated in dev EM, burden, and conversion.
    Step 4: select the minimum-burden candidate; within a small burden band,
            prefer higher conversion.
    """
    raw = Candidate("raw_topk", "default")
    raw_data = cands[raw]
    leader = choose_argmax(cands, dev_idx)
    leader_data = cands[leader]
    margin = 100.0 / len(dev_idx)

    admissible = []
    for cand, data in cands.items():
        point_diff = avg_em(data, dev_idx) - avg_em(leader_data, dev_idx)
        lb = bootstrap_lb_diff(data, leader_data, dev_idx, seed=17)
        f1_diff = avg_f1(data, dev_idx) - avg_f1(leader_data, dev_idx)
        em_ok = point_diff >= -margin - 1e-9 or lb >= -margin - 1e-9
        f1_ok = f1_diff >= -F1_TIE_MARGIN - 1e-9
        if em_ok and f1_ok:
            admissible.append(cand)
    if not admissible:
        admissible = [leader]

    def dominates(a: Candidate, b: Candidate) -> bool:
        da, db = cands[a], cands[b]
        a_em, b_em = avg_em(da, dev_idx), avg_em(db, dev_idx)
        a_burden, b_burden = burden(da, raw_data, dev_idx), burden(db, raw_data, dev_idx)
        a_conv, b_conv = conversion(da, dev_idx), conversion(db, dev_idx)
        no_worse = (
            a_em >= b_em - 1e-9
            and a_burden <= b_burden + 1e-9
            and a_conv >= b_conv - 1e-9
        )
        strictly_better = (
            a_em > b_em + 1e-9
            or a_burden < b_burden - 1e-9
            or a_conv > b_conv + 1e-9
        )
        return no_worse and strictly_better

    frontier = [
        cand for cand in admissible
        if not any(dominates(other, cand) for other in admissible if other != cand)
    ]
    if not frontier:
        frontier = admissible

    min_burden = min(burden(cands[c], raw_data, dev_idx) for c in frontier)
    near_min = [
        c for c in frontier
        if burden(cands[c], raw_data, dev_idx) <= min_burden + NEAR_MIN_BURDEN
    ]
    simplicity = {"sentence_only": 0, "pec_hop": 1, "raw_topk": 2}
    return max(
        near_min,
        key=lambda c: (
            conversion(cands[c], dev_idx),
            avg_f1(cands[c], dev_idx),
            -avg_context(cands[c], dev_idx),
            -simplicity.get(c.base, 99),
            c.label,
        ),
    )


def choose_global_best_single(all_data: dict, readers: list[str], seed: int | None) -> Candidate:
    best = None
    best_dev = -1.0
    for base in BASES:
        for prompt in PROMPTS:
            cand = Candidate(base, prompt)
            cell_scores = []
            for reader in readers:
                for task in TASKS:
                    data = all_data.get((reader, task), {}).get(cand)
                    if data is None:
                        cell_scores = []
                        break
                    dev_idx, _ = split_indices(task, seed)
                    cell_scores.append(avg_em(data, dev_idx))
                if not cell_scores:
                    break
            if len(cell_scores) != len(readers) * len(TASKS):
                continue
            dev = sum(cell_scores) / len(cell_scores)
            if dev > best_dev:
                best = cand
                best_dev = dev
    if best is None:
        raise RuntimeError("No complete global best-single candidate.")
    return best


def eval_candidate(data: dict, test_idx: list[int]) -> dict[str, float]:
    return {
        "em": avg_em(data, test_idx),
        "f1": avg_f1(data, test_idx),
        "latency": avg_latency(data, test_idx),
        "context": avg_context(data, test_idx),
    }


def macro(rows: list[dict[str, float]]) -> dict[str, float]:
    return {k: mean(row[k] for row in rows) for k in ["em", "f1", "latency", "context"]}


def benchmark(seed: int | None = None) -> tuple[dict, list[dict]]:
    all_data = {
        (reader, task): candidates_for(reader, task)
        for reader in READERS
        for task in TASKS
    }
    active_readers = [
        reader for reader in READERS
        if all(len(all_data[(reader, task)]) == len(BASES) * len(PROMPTS) for task in TASKS)
    ]
    best_single = choose_global_best_single(all_data, active_readers, seed)

    per_cell = []
    per_reader = {}
    for reader in active_readers:
        rows_raw, rows_best = [], []
        rows_argmax, rows_light, rows_compass, rows_oracle = [], [], [], []
        for task in TASKS:
            cands = all_data[(reader, task)]
            dev_idx, test_idx = split_indices(task, seed)
            raw = Candidate("raw_topk", "default")
            argmax = choose_argmax(cands, dev_idx)
            light = choose_light(cands, dev_idx)
            compass = choose_compass(cands, dev_idx)

            raw_m = eval_candidate(cands[raw], test_idx)
            best_m = eval_candidate(cands[best_single], test_idx)
            argmax_m = eval_candidate(cands[argmax], test_idx)
            light_m = eval_candidate(cands[light], test_idx)
            compass_m = eval_candidate(cands[compass], test_idx)

            oracle_em = []
            oracle_f1 = []
            for i in test_idx:
                oracle_em.append(max(float(data["results"][i]["em"]) for data in cands.values()))
                oracle_f1.append(max(float(data["results"][i]["f1"]) for data in cands.values()))
            oracle_m = {
                "em": 100.0 * mean(oracle_em),
                "f1": 100.0 * mean(oracle_f1),
                "latency": 0.0,
                "context": 0.0,
            }

            rows_raw.append(raw_m)
            rows_best.append(best_m)
            rows_argmax.append(argmax_m)
            rows_light.append(light_m)
            rows_compass.append(compass_m)
            rows_oracle.append(oracle_m)

            per_cell.append({
                "reader": reader,
                "task": task,
                "argmax_choice": argmax.label,
                "light_choice": light.label,
                "compass_choice": compass.label,
                "raw_em": raw_m["em"],
                "raw_latency": raw_m["latency"],
                "best_single_em": best_m["em"],
                "argmax_em": argmax_m["em"],
                "argmax_latency": argmax_m["latency"],
                "light_em": light_m["em"],
                "light_f1": light_m["f1"],
                "light_latency": light_m["latency"],
                "light_context": light_m["context"],
                "compass_em": compass_m["em"],
                "compass_f1": compass_m["f1"],
                "compass_latency": compass_m["latency"],
                "compass_context": compass_m["context"],
                "compass_conversion_dev": conversion(cands[compass], dev_idx),
                "oracle_em": oracle_m["em"],
            })

        per_reader[reader] = {
            "raw": macro(rows_raw),
            "best_single": macro(rows_best),
            "argmax": macro(rows_argmax),
            "light": macro(rows_light),
            "compass": macro(rows_compass),
            "oracle": macro(rows_oracle),
        }

    summary = {
        "seed": seed,
        "readers": active_readers,
        "candidate_bases": BASES,
        "candidate_prompts": PROMPTS,
        "tie_tolerance": "one dev question",
        "best_single": best_single.label,
        "per_reader": per_reader,
    }
    return summary, per_cell


def render(summary: dict, per_cell: list[dict]) -> str:
    lines = []
    lines.append("# Cost-aware MCE-Compass benchmark\n")
    lines.append(f"Candidate bases: {', '.join(summary['candidate_bases'])}")
    lines.append(f"Candidate prompts: {', '.join(summary['candidate_prompts'])}")
    lines.append(f"Tie tolerance: {summary['tie_tolerance']}")
    lines.append(f"Best fixed config by dev macro: **{summary['best_single']}**\n")

    lines.append("## Deterministic first-dev split\n")
    lines.append("| reader | policy | EM | F1 | latency ms | ctx tok | EM vs raw | speed vs raw | EM vs best | speed vs best |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for reader, policies in summary["per_reader"].items():
        raw = policies["raw"]
        best = policies["best_single"]
        for name in ["raw", "best_single", "argmax", "light", "compass", "oracle"]:
            m = policies[name]
            if name == "oracle":
                lines.append(
                    f"| {reader} | {name} | {m['em']:.2f} | {m['f1']:.2f} | - | - | "
                    f"{m['em'] - raw['em']:+.2f} | - | {m['em'] - best['em']:+.2f} | - |"
                )
                continue
            speed_raw = raw["latency"] / m["latency"] if m["latency"] > 0 else 0.0
            speed_best = best["latency"] / m["latency"] if m["latency"] > 0 else 0.0
            lines.append(
                f"| {reader} | {name} | {m['em']:.2f} | {m['f1']:.2f} | "
                f"{m['latency']:.1f} | {m['context']:.0f} | "
                f"{m['em'] - raw['em']:+.2f} | {speed_raw:.2f}x | "
                f"{m['em'] - best['em']:+.2f} | {speed_best:.2f}x |"
            )

    lines.append("\n## Per-cell choices\n")
    lines.append("| reader | task | Argmax | Light | Compass | compass EM | compass F1 | compass ms | compass ctx |")
    lines.append("|---|---|---|---|---|---:|---:|---:|---:|")
    for row in per_cell:
        lines.append(
            f"| {row['reader']} | {row['task']} | {row['argmax_choice']} | "
            f"{row['light_choice']} | {row['compass_choice']} | "
            f"{row['compass_em']:.2f} | {row['compass_f1']:.2f} | "
            f"{row['compass_latency']:.1f} | {row['compass_context']:.0f} |"
        )
    return "\n".join(lines)


def stability(n_seeds: int = 10) -> list[dict]:
    rows = []
    for seed in range(n_seeds):
        summary, _ = benchmark(seed=seed)
        for reader, policies in summary["per_reader"].items():
            raw = policies["raw"]
            best = policies["best_single"]
            light = policies["light"]
            rows.append({
                "seed": seed,
                "reader": reader,
                "best_single": summary["best_single"],
                "light_em": light["em"],
                "compass_em": policies["compass"]["em"],
                "best_em": best["em"],
                "raw_em": raw["em"],
                "light_latency": light["latency"],
                "compass_latency": policies["compass"]["latency"],
                "best_latency": best["latency"],
                "raw_latency": raw["latency"],
            })
    return rows


def render_stability(rows: list[dict]) -> str:
    lines = ["\n## 10-split stability for MCE-Compass\n"]
    lines.append("| reader | EM mean±std | Δ vs best mean±std | Δ vs raw mean±std | speed vs raw | #Δbest>0 |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for reader in READERS:
        rr = [r for r in rows if r["reader"] == reader]
        if not rr:
            continue
        em = [r["compass_em"] for r in rr]
        d_best = [r["compass_em"] - r["best_em"] for r in rr]
        d_raw = [r["compass_em"] - r["raw_em"] for r in rr]
        speed = [r["raw_latency"] / r["compass_latency"] for r in rr if r["compass_latency"] > 0]
        n_pos = sum(1 for d in d_best if d > 0)
        lines.append(
            f"| {reader} | {mean(em):.2f}±{stdev(em):.2f} | "
            f"{mean(d_best):+.2f}±{stdev(d_best):.2f} | "
            f"{mean(d_raw):+.2f}±{stdev(d_raw):.2f} | "
            f"{mean(speed):.2f}x | {n_pos}/10 |"
        )
    return "\n".join(lines)


def main() -> None:
    summary, per_cell = benchmark(seed=None)
    rows = stability(n_seeds=10)
    report = render(summary, per_cell) + "\n" + render_stability(rows)
    out_md = OUT / "MCE_COMPASS_COST_BENCHMARK.md"
    out_json = OUT / "mce_compass_cost_benchmark.json"
    with open(out_md, "w", encoding="utf-8") as f:
        f.write(report)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({"deterministic": summary, "per_cell": per_cell, "stability": rows}, f, indent=2)
    print(report)
    print(f"\nSaved -> {out_md}")
    print(f"Saved -> {out_json}")


if __name__ == "__main__":
    main()
