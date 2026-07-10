"""
Matching Matrix experiment: 3 compilers x 3 readers.

For the CTKS self-compilation claim, we want to test whether
  self-compile (compiler=reader) > cross-compile (compiler!=reader).

This script evaluates each of the 9 (compiler, reader) combinations on
HotpotQA using the entity-level indices compiled by each SLM.

Prerequisites:
  - cache/hotpotqa_entity_index/               -- LFM (legacy dir)
  - cache/hotpotqa_entity_index_qwen06/        -- Qwen3-0.6B
  - cache/hotpotqa_entity_index_qwen17/        -- Qwen3-1.7B

Usage:
  python run_matching_matrix.py --dataset hotpotqa --max-samples 500
  python run_matching_matrix.py --only lfm_to_lfm  # run one cell
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "shared"))
from utils.config import TOPIC6_DIR

# (short_name, reader_model_name, compiler_dir_suffix_or_None_for_legacy)
READERS = [
    ("lfm",    "lfm2.5-1.2b-instruct", None),
    ("qwen06", "qwen3-0.6b",           "qwen06"),
    ("qwen17", "qwen3-1.7b",           "qwen17"),
]

COMPILERS = [
    ("lfm",    None),        # legacy dir = hotpotqa_entity_index/
    ("qwen06", "qwen06"),    # hotpotqa_entity_index_qwen06/
    ("qwen17", "qwen17"),    # hotpotqa_entity_index_qwen17/
]


def entity_dir_for(dataset: str, compiler_suffix):
    if compiler_suffix is None:
        return f"{dataset}_entity_index"
    return f"{dataset}_entity_index_{compiler_suffix}"


def run_cell(reader_short, reader_model, compiler_short, compiler_suffix,
             dataset: str, max_samples: int, top_k: int, python_exe: str):
    """Run one (compiler, reader) combination."""
    entity_dir = entity_dir_for(dataset, compiler_suffix)
    dir_path = TOPIC6_DIR / "experiments" / "cache" / entity_dir
    if not (dir_path / "index.faiss").exists():
        print(f"[Skip] Compiler index not ready: {dir_path}")
        return None

    tag = f"matrix_c{compiler_short}"
    script = Path(__file__).parent / "run_baseline.py"

    cmd = [
        python_exe, "-u", str(script),
        "--model", reader_model,
        "--dataset", dataset,
        "--mode", "entity",
        "--max-samples", str(max_samples),
        "--top-k", str(top_k),
        "--entity-dir", entity_dir,
        "--result-tag", tag,
    ]
    print(f"\n{'='*70}")
    print(f"[Cell] compiler={compiler_short} -> reader={reader_short}")
    print(f"[Cmd ] {' '.join(cmd)}")
    print(f"{'='*70}")

    t0 = time.time()
    p = subprocess.run(cmd, capture_output=False)
    print(f"[Cell] done in {time.time()-t0:.0f}s (rc={p.returncode})")
    return p.returncode == 0


def collect_results(dataset: str, output_file: Path):
    """Aggregate the 9 result files into a single matrix summary."""
    results_dir = TOPIC6_DIR / "experiments" / "results"
    matrix = {}
    for reader_short, reader_model, _ in READERS:
        matrix[reader_short] = {}
        for compiler_short, _ in COMPILERS:
            tag = f"matrix_c{compiler_short}"
            f = results_dir / f"{reader_model}_{dataset}_entity_{tag}.json"
            if f.exists():
                with open(f, encoding="utf-8") as fp:
                    data = json.load(fp)
                m = data.get("metrics", {})
                matrix[reader_short][compiler_short] = {
                    "em": m.get("em", 0),
                    "f1": m.get("f1", 0),
                    "n": data.get("n_samples", 0),
                    "file": str(f.name),
                }
            else:
                matrix[reader_short][compiler_short] = None

    with open(output_file, "w", encoding="utf-8") as fp:
        json.dump(matrix, fp, indent=2, ensure_ascii=False)

    # Pretty print
    print(f"\n{'='*70}")
    print(f"Matching Matrix for {dataset} (EM%)")
    print(f"{'='*70}")
    print(f"{'Reader\\Compiler':<16}", end="")
    for c, _ in COMPILERS:
        print(f"{c:>10}", end="")
    print()
    for r, _, _ in READERS:
        print(f"{r:<16}", end="")
        for c, _ in COMPILERS:
            cell = matrix[r][c]
            if cell is None:
                print(f"{'--':>10}", end="")
            else:
                marker = "*" if r == c else " "
                print(f"{cell['em']:>9.1f}{marker}", end="")
        print()
    print("* = self-compile (diagonal)")
    print(f"\n[Saved] {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="hotpotqa")
    parser.add_argument("--max-samples", type=int, default=500,
                        help="Samples per cell (keep modest; 9 cells total)")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--only", default=None,
                        help="Run only one cell, e.g. 'qwen06_to_qwen06' (compiler_to_reader)")
    parser.add_argument("--python",
                        default=r"python")
    parser.add_argument("--collect-only", action="store_true",
                        help="Skip running; just aggregate existing result files")
    args = parser.parse_args()

    output_file = TOPIC6_DIR / "experiments" / "results" / f"matching_matrix_{args.dataset}.json"

    if args.collect_only:
        collect_results(args.dataset, output_file)
        sys.exit(0)

    # Plan cells
    cells = []
    for reader_short, reader_model, _ in READERS:
        for compiler_short, compiler_suffix in COMPILERS:
            key = f"{compiler_short}_to_{reader_short}"
            if args.only and args.only != key:
                continue
            cells.append((reader_short, reader_model, compiler_short, compiler_suffix, key))

    print(f"\n[Plan] Will run {len(cells)} cells:")
    for _, _, _, _, key in cells:
        print(f"   - {key}")

    for reader_short, reader_model, compiler_short, compiler_suffix, key in cells:
        run_cell(
            reader_short, reader_model,
            compiler_short, compiler_suffix,
            dataset=args.dataset,
            max_samples=args.max_samples,
            top_k=args.top_k,
            python_exe=args.python,
        )

    collect_results(args.dataset, output_file)
