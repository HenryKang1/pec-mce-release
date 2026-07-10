# Conversion decomposition

Per (reader, task, interface, prompt):
  EM = exact match
  loose = answer-in-prediction (gold contained anywhere)
  empty = prediction was empty/whitespace
  verbose_fail = fraction of loose-correct predictions that missed EM (model knew but verbose)
  conversion = EM / loose (given answer was produced, did it match exactly)

## lfm2.5-1.2b-instruct

| task | base | prompt | EM | loose | empty | verbose_fail | conversion |
|---|---|---|---:|---:|---:|---:|---:|
| hotpotqa | raw_topk | default | 18.00% | 34.50% | 0.00% | 47.83% | 52.17% |
| hotpotqa | raw_topk | concise | 20.00% | 32.00% | 0.00% | 37.50% | 62.50% |
| hotpotqa | sentence_only | extractive | 22.00% | 26.00% | 0.00% | 15.38% | 84.62% |
| hotpotqa | sentence_only | short15 | 22.00% | 27.50% | 0.00% | 20.00% | 80.00% |
| hotpotqa | pec_hop | extractive | 25.50% | 29.50% | 0.50% | 13.56% | 86.44% |
| hotpotqa | pec_hop | concise | 22.50% | 34.00% | 0.00% | 33.82% | 66.18% |
| 2wikimqa | raw_topk | default | 18.00% | 26.50% | 0.00% | 32.08% | 67.92% |
| 2wikimqa | raw_topk | concise | 20.50% | 28.50% | 0.00% | 28.07% | 71.93% |
| 2wikimqa | sentence_only | extractive | 23.00% | 24.00% | 0.00% | 4.17% | 95.83% |
| 2wikimqa | sentence_only | short15 | 22.50% | 24.00% | 0.00% | 6.25% | 93.75% |
| 2wikimqa | pec_hop | extractive | 30.00% | 32.00% | 0.00% | 6.25% | 93.75% |
| 2wikimqa | pec_hop | concise | 21.00% | 30.00% | 0.00% | 30.00% | 70.00% |
| musique | raw_topk | default | 4.50% | 13.50% | 0.00% | 66.67% | 33.33% |
| musique | raw_topk | concise | 6.50% | 10.50% | 0.50% | 38.10% | 61.90% |
| musique | sentence_only | extractive | 7.50% | 8.00% | 0.00% | 6.25% | 93.75% |
| musique | sentence_only | short15 | 6.00% | 7.00% | 0.00% | 14.29% | 85.71% |
| musique | pec_hop | extractive | 9.50% | 10.00% | 0.00% | 5.00% | 95.00% |
| musique | pec_hop | concise | 7.50% | 12.50% | 0.00% | 40.00% | 60.00% |
| qasper | raw_topk | default | 0.00% | 7.00% | 0.00% | 100.00% | 0.00% |
| qasper | raw_topk | concise | 1.00% | 7.00% | 0.00% | 85.71% | 14.29% |
| qasper | sentence_only | extractive | 13.00% | 14.50% | 0.00% | 10.34% | 89.66% |
| qasper | sentence_only | short15 | 9.00% | 15.50% | 0.50% | 41.94% | 58.06% |
| qasper | pec_hop | extractive | 8.50% | 10.00% | 0.00% | 15.00% | 85.00% |
| qasper | pec_hop | concise | 1.50% | 9.50% | 0.00% | 84.21% | 15.79% |
| multifieldqa_en | raw_topk | default | 4.67% | 7.33% | 0.00% | 36.36% | 63.64% |
| multifieldqa_en | raw_topk | concise | 2.67% | 4.67% | 0.00% | 42.86% | 57.14% |
| multifieldqa_en | sentence_only | extractive | 9.33% | 9.33% | 0.00% | 0.00% | 100.00% |
| multifieldqa_en | sentence_only | short15 | 10.67% | 12.00% | 0.67% | 11.11% | 88.89% |
| multifieldqa_en | pec_hop | extractive | 5.33% | 6.67% | 1.33% | 20.00% | 80.00% |
| multifieldqa_en | pec_hop | concise | 6.67% | 10.00% | 0.00% | 33.33% | 66.67% |

### lfm2.5-1.2b-instruct -- macro across tasks

| base | prompt | EM | loose | empty | verbose_fail | conversion |
|---|---|---:|---:|---:|---:|---:|
| raw_topk | default | 9.03% | 17.77% | 0.00% | 56.59% | 43.41% |
| raw_topk | concise | 10.13% | 16.53% | 0.10% | 46.45% | 53.55% |
| sentence_only | extractive | 14.97% | 16.37% | 0.00% | 7.23% | 92.77% |
| sentence_only | short15 | 14.03% | 17.20% | 0.23% | 18.72% | 81.28% |
| pec_hop | extractive | 15.77% | 17.63% | 0.37% | 11.96% | 88.04% |
| pec_hop | concise | 11.83% | 19.20% | 0.00% | 44.27% | 55.73% |

## qwen3-4b

| task | base | prompt | EM | loose | empty | verbose_fail | conversion |
|---|---|---|---:|---:|---:|---:|---:|
| hotpotqa | raw_topk | default | 15.00% | 53.50% | 0.00% | 71.96% | 28.04% |
| hotpotqa | raw_topk | concise | 29.00% | 52.50% | 0.00% | 44.76% | 55.24% |
| hotpotqa | sentence_only | extractive | 19.00% | 24.00% | 34.50% | 20.83% | 79.17% |
| hotpotqa | sentence_only | short15 | 13.00% | 27.50% | 31.00% | 52.73% | 47.27% |
| hotpotqa | pec_hop | extractive | 20.50% | 26.00% | 31.00% | 21.15% | 78.85% |
| hotpotqa | pec_hop | concise | 31.00% | 44.50% | 0.00% | 30.34% | 69.66% |
| 2wikimqa | raw_topk | default | 8.50% | 61.50% | 0.00% | 86.18% | 13.82% |
| 2wikimqa | raw_topk | concise | 16.50% | 52.50% | 0.00% | 68.57% | 31.43% |
| 2wikimqa | sentence_only | extractive | 28.00% | 28.50% | 42.00% | 1.75% | 98.25% |
| 2wikimqa | sentence_only | short15 | 9.00% | 37.00% | 26.50% | 75.68% | 24.32% |
| 2wikimqa | pec_hop | extractive | 23.00% | 28.00% | 38.50% | 17.86% | 82.14% |
| 2wikimqa | pec_hop | concise | 22.50% | 45.50% | 0.00% | 50.55% | 49.45% |
| musique | raw_topk | default | 7.50% | 34.00% | 0.00% | 77.94% | 22.06% |
| musique | raw_topk | concise | 13.00% | 29.00% | 0.00% | 55.17% | 44.83% |
| musique | sentence_only | extractive | 5.50% | 7.00% | 50.50% | 21.43% | 78.57% |
| musique | sentence_only | short15 | 4.00% | 12.00% | 24.50% | 66.67% | 33.33% |
| musique | pec_hop | extractive | 4.50% | 8.00% | 50.00% | 43.75% | 56.25% |
| musique | pec_hop | concise | 11.50% | 22.50% | 0.00% | 48.89% | 51.11% |
| qasper | raw_topk | default | 2.50% | 8.00% | 0.00% | 68.75% | 31.25% |
| qasper | raw_topk | concise | 5.50% | 8.50% | 0.00% | 35.29% | 64.71% |
| qasper | sentence_only | extractive | 10.50% | 13.00% | 35.50% | 19.23% | 80.77% |
| qasper | sentence_only | short15 | 4.00% | 16.00% | 11.00% | 75.00% | 25.00% |
| qasper | pec_hop | extractive | 6.00% | 6.50% | 63.00% | 7.69% | 92.31% |
| qasper | pec_hop | concise | 6.50% | 9.00% | 0.00% | 27.78% | 72.22% |
| multifieldqa_en | raw_topk | default | 4.00% | 11.33% | 0.00% | 64.71% | 35.29% |
| multifieldqa_en | raw_topk | concise | 4.67% | 10.00% | 0.00% | 53.33% | 46.67% |
| multifieldqa_en | sentence_only | extractive | 7.33% | 8.67% | 32.00% | 15.38% | 84.62% |
| multifieldqa_en | sentence_only | short15 | 7.33% | 18.67% | 14.67% | 60.71% | 39.29% |
| multifieldqa_en | pec_hop | extractive | 5.33% | 7.33% | 48.00% | 27.27% | 72.73% |
| multifieldqa_en | pec_hop | concise | 9.33% | 14.67% | 0.00% | 36.36% | 63.64% |

### qwen3-4b -- macro across tasks

| base | prompt | EM | loose | empty | verbose_fail | conversion |
|---|---|---:|---:|---:|---:|---:|
| raw_topk | default | 7.50% | 33.67% | 0.00% | 73.91% | 26.09% |
| raw_topk | concise | 13.73% | 30.50% | 0.00% | 51.43% | 48.57% |
| sentence_only | extractive | 14.07% | 16.23% | 38.90% | 15.73% | 84.27% |
| sentence_only | short15 | 7.47% | 22.23% | 21.53% | 66.16% | 33.84% |
| pec_hop | extractive | 11.87% | 15.17% | 46.10% | 23.55% | 76.45% |
| pec_hop | concise | 16.17% | 27.23% | 0.00% | 38.78% | 61.22% |

