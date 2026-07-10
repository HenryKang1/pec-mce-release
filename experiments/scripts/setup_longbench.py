"""
Download LongBench QA subsets and inspect structure.
Saves per-task JSON for downstream use.

Subsets of interest (English, multi-hop or long-passage QA):
  - hotpotqa     : 200 Q, avg context ~9k tokens
  - 2wikimqa     : 200 Q, avg context ~5k tokens
  - musique      : 200 Q, avg context ~11k tokens
  - qasper       : NLP papers, ~3k tokens
  - narrativeqa  : stories, very long

Usage:
  python setup_longbench.py --subsets hotpotqa 2wikimqa musique
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "shared"))
from utils.config import DATASETS_DIR


def run(subsets: list[str]):
    import requests
    from io import BytesIO
    out_dir = DATASETS_DIR / "longbench"
    out_dir.mkdir(parents=True, exist_ok=True)

    for name in subsets:
        print(f"\n[Load] {name}")
        # Download raw JSONL directly from the HF repo
        url = f"https://huggingface.co/datasets/THUDM/LongBench/resolve/main/data/{name}.jsonl"
        print(f"  [GET] {url}")
        r = requests.get(url, timeout=120)
        r.raise_for_status()
        items = []
        for line in r.text.splitlines():
            line = line.strip()
            if line:
                items.append(json.loads(line))
        ds = items
        print(f"  n = {len(ds)}")
        print(f"  keys = {list(ds[0].keys())}")

        sample = ds[0]
        for k, v in sample.items():
            if isinstance(v, str):
                print(f"  {k:<16} ({len(v)} chars): {v[:200]}...")
            else:
                print(f"  {k:<16}: {v}")

        # Save as list of dicts
        out_path = out_dir / f"{name}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(list(ds), f, ensure_ascii=False)
        print(f"  [Saved] {out_path} ({out_path.stat().st_size//1024} KB)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--subsets", nargs="+",
                        default=["hotpotqa", "2wikimqa", "musique"])
    args = parser.parse_args()
    run(args.subsets)
