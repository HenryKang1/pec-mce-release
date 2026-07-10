# From Evidence to Exact Answers — Anonymous Code Release

Code, per-question outputs, and reproduction scripts for ARR 2026 May submission **#11474**:
*"From Evidence to Exact Answers: Copy-Preserving Compression and Conversion-Aware Calibration for Frozen SLM RAG."*

## Contents

```
experiments/scripts/     evaluation pipeline, MCE-Compass selector, analysis scripts
experiments/results/     per-question outputs (JSON) for the three paper readers
                         + script-generated aggregate reports (JSON/MD)
shared/                  config, model/dataset download helpers
MODELS.md                exact GGUF artifacts with SHA-256 hashes
requirements.txt         Python dependencies
```

Key entry points:

| File | Role |
|---|---|
| `experiments/scripts/longbench_pipeline.py` | Main pipeline: card construction, PEC-HOP retrieval (Algorithm 1), all evidence-interface variants, all decoding prompts, EM/F1/latency measurement |
| `experiments/scripts/rag_pipeline.py` | FAISS index, gte-small encoder, llama.cpp reader wrapper |
| `experiments/scripts/mce_policy_expanded.py` | MCE-Compass calibration rule (Algorithm 2) over the candidate grid |
| `experiments/scripts/mce_full_pec_hop_benchmark.py` | Compressor head-to-head (Table 1) |
| `experiments/scripts/conversion_decomp.py` | Conversion diagnostic P(EM \| loose) (Table 11) |
| `experiments/scripts/mce_dev_size_sweep.py` | 20-partition dev-budget sweep (Figure 2) |
| `experiments/scripts/full_retrieval_evidence_recall.py` | 66,581-article HotpotQA external-retrieval stress test (Table 4) |
| `experiments/scripts/make_paper_tables.py` | Regenerates LaTeX tables from `experiments/results/` |

The 4-line card schema (TITLE/ANCHORS/FACT/PTR), the anchor regex (Appendix D), and all five
decoding prompt templates (Appendix E, including the fixed 2-shot extractive demonstrations)
are defined in `longbench_pipeline.py`.

## Setup

```bash
pip install -r requirements.txt
python shared/scripts/download_models.py      # GGUF readers + gte-small (see MODELS.md)
```

For GPU offload, install llama-cpp-python with CUDA (the paper uses a single RTX 3090, CUDA 12.1,
all layers offloaded, `n_ctx=2048`, Q4_K_M GGUF):

```bash
CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python --no-cache-dir
```

**LongBench data:** download the LongBench (v1) English task files
(`hotpotqa.jsonl`, `2wikimqa.jsonl`, `musique.jsonl`, `qasper.jsonl`, `multifieldqa_en.jsonl`)
from the THUDM/LongBench release and place them under `shared/datasets/longbench/data/`.
The pipeline takes the first 200 instances per task (150 for multifieldqa_en); the first 50
(40 for multifieldqa_en) form the deterministic dev split — no shuffling, so re-runs reproduce
identical dev/test partitions.

## Reproducing the paper

Runner scripts assume the repository root as working directory.

```bash
# Single cell: one (reader, task, variant) run
python experiments/scripts/longbench_pipeline.py \
  --task hotpotqa --model lfm2.5-1.2b-instruct --variant pec_hop_fewextractive

# Table 1 + Table 7 (compressor head-to-head, matched 2-shot extractive prompt)
bash experiments/scripts/run_h2h_comprehensive.sh
python experiments/scripts/aggregate_h2h_comprehensive.py
python experiments/scripts/mce_full_pec_hop_benchmark.py

# Table 2 (MCE-Compass headline) + Table 3 (policy ablation)
python experiments/scripts/mce_policy_expanded.py
python experiments/scripts/mce_select_cost_benchmark.py

# Table 9 (matched 0-shot vs 2-shot prompts)
bash experiments/scripts/run_baseline_fewshot_ablation.sh
bash experiments/scripts/run_baseline_fewshot_ablation_gemma_qwen.sh
python experiments/scripts/aggregate_fewshot_defense.py

# Table 11 (conversion diagnostic)
python experiments/scripts/conversion_decomp.py

# Figure 2 (20-partition dev-size sweep)
python experiments/scripts/mce_dev_size_sweep.py

# Table 4 / Appendix G (external-retrieval stress test)
python experiments/scripts/full_retrieval_evidence_recall.py
python experiments/scripts/full_external_reader_em_summary.py

# Corruption controls (Table 17): PTR shuffle / random anchors
python experiments/scripts/longbench_pipeline.py \
  --task hotpotqa --model lfm2.5-1.2b-instruct --variant pec_hop_shuffle_ptr
python experiments/scripts/longbench_pipeline.py \
  --task musique --model lfm2.5-1.2b-instruct --variant pec_hop_random_anchor

# LaTeX tables from shipped result JSONs (no GPU needed)
python experiments/scripts/make_paper_tables.py
```

`experiments/results/` already contains the per-question outputs for the three paper readers
(LFM2.5-Instruct-1.2B, Qwen3-4B, Gemma-4-E4B) and the script-generated aggregate reports, so the
analysis and table scripts run without re-executing the readers. Evidence cards are constructed
deterministically at run time from the LongBench contexts (per-question FAISS index); they are
not shipped as separate artifacts.

## Determinism and measurement

Greedy decoding everywhere (temperature 0, top-p 1, top-k 0); deterministic first-N splits;
seed 0 for any sampling-related operation. Latency is wall-clock per question
(`time.perf_counter`) including encoder pass, FAISS retrieval, reader prefill, and decode,
after a 5-question warmup. Paired bootstrap (10,000 resamples) for significance.

## License

Code: MIT (see LICENSE). LongBench is MIT; LFM2.5-1.2B-Instruct is under the LFM Open License
Agreement; Qwen3-4B and Gemma-4-E4B-it are Apache 2.0; gte-small is MIT.
