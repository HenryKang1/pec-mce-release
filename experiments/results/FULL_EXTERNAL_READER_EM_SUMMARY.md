# Full external-retrieval reader EM

Reader: **LFM2.5-Instruct 1.2B**. Dataset: **HotpotQA full validation**.
All systems use top-5 retrieval over dataset-level FAISS indices.

| Evidence index | n | EM | F1 | p50 TTFT | p95 TTFT |
|---|---:|---:|---:|---:|---:|
| Raw article | 7404 | 17.76 | 29.68 | 178.2 | 330.6 |
| Extractive minimal note | 7405 | 17.68 | 29.57 | 263.7 | 486.3 |
| Smart minimal note | 7405 | 17.57 | 29.28 | 265.8 | 508.9 |
| LLM entity note | 7405 | 13.32 | 24.30 | 79.4 | 160.3 |

## Paired deltas vs raw article

| Evidence index | common n | ΔEM | 95% CI | ΔF1 | 95% CI |
|---|---:|---:|---:|---:|---:|
| Extractive minimal note | 7404 | -0.08 | [-0.62, +0.47] | -0.11 | [-0.65, +0.44] |
| Smart minimal note | 7404 | -0.19 | [-0.81, +0.43] | -0.39 | [-1.01, +0.21] |
| LLM entity note | 7404 | -4.44 | [-5.25, -3.63] | -5.37 | [-6.19, -4.58] |

Takeaway: extractive minimal notes are reader-EM equivalent to coverage-matched raw article retrieval on the full external HotpotQA setting, while the retrieval-only diagnostic shows they use 0.80x words and retain 96.8% of raw answer recall.

JSON: `.\experiments\results\full_external_reader_em_summary.json`
