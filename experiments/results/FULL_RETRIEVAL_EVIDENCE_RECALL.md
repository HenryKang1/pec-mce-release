# Full-retrieval evidence recall

Retrieval-only diagnostic on HotpotQA full validation. No reader LLM is called.
Dataset size: **7405**. Top-k: **5**.

| Variant | Ans recall | Support-title recall | All titles | Support-sentence recall | Words | Ans density/1k |
|---|---:|---:|---:|---:|---:|---:|
| Raw article | 66.06% | 88.42% | 80.95% | 74.34% | 280 | 2.587 |
| Raw chunk-100 | 65.06% | 88.19% | 80.43% | 72.09% | 240 | 2.843 |
| Raw chunk-40 | 58.68% | 86.47% | 76.93% | 53.40% | 137 | 4.332 |
| Extractive minimal note | 63.93% | 87.43% | 79.08% | 70.60% | 223 | 2.981 |
| Smart minimal note | 61.73% | 87.04% | 78.35% | 61.11% | 228 | 2.829 |
| LLM entity note | 46.14% | 82.78% | 70.93% | 0.45% | 263 | 1.765 |

## Key comparison

Extractive minimal evidence uses **0.80x** the words of raw article retrieval while retaining **96.8%** of its answer recall and increasing answer density by **1.15x**.

JSON: `.\experiments\results\full_retrieval_evidence_recall.json`
