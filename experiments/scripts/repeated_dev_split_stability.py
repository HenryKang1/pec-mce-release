"""
Repeated dev/test split stability for MCE-Select.

For each random seed in [0..9], partition the n questions per task into
a 50-question dev split (40 for multifieldqa_en) and a held-out test
split. For each (reader, task), pick the (base, prompt) with the
highest dev EM. Compare it to the paper's best-single baseline: one
(base, prompt) chosen by dev macro across all reader-task cells and
then applied uniformly. Report macro test EM mean +/- std across the 10 seeds.
"""
import json
import random
from pathlib import Path
from statistics import mean, stdev

ROOT = Path(__file__).resolve().parents[1] / "results" / "longbench"

READERS = ["lfm2.5-1.2b-instruct", "qwen3-4b", "gemma-4-e4b"]
TASKS = ["hotpotqa", "2wikimqa", "musique", "qasper", "multifieldqa_en"]
BASES = ["raw_topk", "raw_topk_b840", "sentence_only", "pec_hop"]
PROMPTS = ["", "_extractive", "_short15", "_concise"]
MIN_CANDIDATES_PER_TASK = 12
N_PER_TASK = {
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
N_SEEDS = 10


def load_results(reader: str, task: str, base: str, prompt: str):
    n = N_PER_TASK[task]
    fname = f"{reader}_{task}_{base}{prompt}_n{n}.json"
    fpath = ROOT / fname
    if not fpath.exists():
        return None
    with open(fpath) as f:
        d = json.load(f)
    em_per_q = [int(r["em"]) for r in d["results"]]
    if len(em_per_q) != n:
        # truncate/pad defensively
        em_per_q = em_per_q[:n]
    return em_per_q


def split_indices(seed: int, task: str):
    rng = random.Random(seed)
    n = N_PER_TASK[task]
    d = DEV_SIZE[task]
    idxs = list(range(n))
    rng.shuffle(idxs)
    return set(idxs[:d]), idxs[d:]


def load_reader_cache(reader: str):
    """Load all available (task, base, prompt) EM vectors for a reader."""
    cache = {}
    for task in TASKS:
        for base in BASES:
            for prompt in PROMPTS:
                ems = load_results(reader, task, base, prompt)
                if ems is not None:
                    cache[(task, base, prompt)] = ems
    return cache


def active_readers(all_caches: dict):
    """Use only readers with a complete-enough policy grid.

    Gemma can be present as a downloaded model or a smoke-test result before
    the full 5-task x candidate matrix has finished.  Including that partial
    reader would make the global best-single baseline undefined, so we skip it
    until every task has enough candidates to support the same comparison.
    """
    active, skipped = [], []
    for reader, cache in all_caches.items():
        counts = {
            task: sum(1 for base in BASES for prompt in PROMPTS
                      if (task, base, prompt) in cache)
            for task in TASKS
        }
        if all(counts[task] >= MIN_CANDIDATES_PER_TASK for task in TASKS):
            active.append(reader)
        else:
            skipped.append((reader, counts))
    return active, skipped


def choose_global_best_single(all_caches: dict, readers: list[str], seed: int):
    """Match mce_policy.py: one (base, prompt) chosen across all reader-task cells."""
    best_global_dev = -1.0
    best_global_bp = None
    n_active_cells = len(readers) * len(TASKS)
    for base in BASES:
        for prompt in PROMPTS:
            cell_devs = []
            for reader in readers:
                cache = all_caches[reader]
                for task in TASKS:
                    key = (task, base, prompt)
                    if key not in cache:
                        continue
                    dev_idx, _ = split_indices(seed, task)
                    ems = cache[key]
                    cell_devs.append(sum(ems[i] for i in dev_idx) / len(dev_idx))
            if len(cell_devs) != n_active_cells:
                continue
            dev_macro = sum(cell_devs) / len(cell_devs)
            if dev_macro > best_global_dev:
                best_global_dev = dev_macro
                best_global_bp = (base, prompt)
    return best_global_bp


def reader_macros(reader: str, seed: int, cache: dict, best_global_bp: tuple):
    """Returns (mce_select_macro, paper_best_single_macro) for the seed."""

    # MCE-Select: per-(reader, task) dev-best (base, prompt)
    test_em_select = []
    for task in TASKS:
        dev_idx, test_idx = split_indices(seed, task)
        best_dev_em = -1.0
        best_bp = None
        for base in BASES:
            for prompt in PROMPTS:
                key = (task, base, prompt)
                if key not in cache:
                    continue
                ems = cache[key]
                dev_em = sum(ems[i] for i in dev_idx) / len(dev_idx)
                if dev_em > best_dev_em:
                    best_dev_em = dev_em
                    best_bp = (base, prompt)
        ems = cache[(task, *best_bp)]
        test_em_select.append(
            100.0 * sum(ems[i] for i in test_idx) / len(test_idx)
        )

    test_em_single = []
    for task in TASKS:
        _, test_idx = split_indices(seed, task)
        key = (task, *best_global_bp)
        if key not in cache:
            continue
        ems = cache[key]
        test_em_single.append(
            100.0 * sum(ems[i] for i in test_idx) / len(test_idx)
        )

    return (
        sum(test_em_select) / len(test_em_select),
        sum(test_em_single) / len(test_em_single),
        best_global_bp,
    )


def main():
    print(f"Repeated dev/test split stability ({N_SEEDS} seeds)\n")
    print(
        f"{'reader':<28s}  {'MCE-Sel mean+/-std':>22s}  "
        f"{'BestSingle mean+/-std':>22s}  {'Delta mean+/-std':>20s}  "
        f"{'#Delta>0':>10s}"
    )
    print("-" * 110)

    all_caches = {reader: load_reader_cache(reader) for reader in READERS}
    readers, skipped = active_readers(all_caches)
    if skipped:
        print("Skipped readers without a complete-enough candidate grid:")
        for reader, counts in skipped:
            count_str = ", ".join(f"{task}={counts[task]}" for task in TASKS)
            print(f"  - {reader}: {count_str}")
        print()
    if not readers:
        raise RuntimeError("No reader has enough completed candidates for stability analysis.")

    global_bps = [choose_global_best_single(all_caches, readers, seed) for seed in range(N_SEEDS)]
    if any(bp is None for bp in global_bps):
        raise RuntimeError("Could not choose a global best-single baseline for every split.")

    out_rows = []
    for reader in readers:
        sels, sings, bps = [], [], []
        for seed in range(N_SEEDS):
            bp = global_bps[seed]
            sel, sing, _ = reader_macros(reader, seed, all_caches[reader], bp)
            sels.append(sel)
            sings.append(sing)
            bps.append(bp)
        deltas = [s - g for s, g in zip(sels, sings)]
        mu_s, sd_s = mean(sels), stdev(sels)
        mu_g, sd_g = mean(sings), stdev(sings)
        mu_d, sd_d = mean(deltas), stdev(deltas)
        n_pos = sum(1 for d in deltas if d > 0)
        print(
            f"{reader:<28s}  {mu_s:7.2f} +/- {sd_s:4.2f}      "
            f"{mu_g:7.2f} +/- {sd_g:4.2f}      "
            f"{mu_d:+7.2f} +/- {sd_d:4.2f}    "
            f"{n_pos}/{N_SEEDS:>3d}"
        )
        out_rows.append(
            {
                "reader": reader,
                "mce_select": {"mean": mu_s, "std": sd_s, "per_seed": sels},
                "best_single": {
                    "mean": mu_g,
                    "std": sd_g,
                    "per_seed": sings,
                    "best_global_bp_per_seed": bps,
                },
                "delta": {
                    "mean": mu_d,
                    "std": sd_d,
                    "per_seed": deltas,
                    "n_positive": n_pos,
                },
            }
        )

    out_path = ROOT.parent / "mce_select_stability.json"
    with open(out_path, "w") as f:
        json.dump({"n_seeds": N_SEEDS, "results": out_rows}, f, indent=2)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
