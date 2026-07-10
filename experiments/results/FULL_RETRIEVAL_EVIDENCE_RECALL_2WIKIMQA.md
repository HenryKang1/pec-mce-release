# Full-retrieval evidence recall

Retrieval-only diagnostic on 2WikiMQA validation. No reader LLM is called.
Dataset size: **12576**. Top-k: **5**.

| Variant | Ans recall | Support-title recall | All titles | Support-sentence recall | Words | Ans density/1k |
|---|---:|---:|---:|---:|---:|---:|
| Raw article | 54.64% | 94.47% | 87.25% | 65.38% | 292 | 2.330 |
| Extractive minimal note | 51.34% | 93.14% | 84.37% | 59.74% | 196 | 2.806 |
| LLM entity note | 46.60% | 88.07% | 72.67% | 1.12% | 263 | 1.790 |

## Key comparison

Extractive minimal evidence uses **0.67x** the words of raw article retrieval while retaining **94.0%** of its answer recall and increasing answer density by **1.20x**.

JSON: `.\experiments\results\full_retrieval_evidence_recall_2wikimqa.json`
