# Cross-prompt x baseline x task x reader matrix

Reads existing result JSONs from experiments/results/longbench/.
Δ row = PEC-Hop EM/F1 minus the best non-PEC baseline at the same prompt.

### lfm2.5-1.2b-instruct  --  EM


**hotpotqa**

| Task / Base \ Prompt | default | extractive | short15 | concise |
|---|---:|---:|---:|---:|
| raw_topk | 18.00 | 24.00 | 22.00 | 20.00 |
| raw_topk_b840 | 16.00 | 24.50 | 22.00 | 19.00 |
| sentence_only | 18.00 | 22.00 | 22.00 | 19.00 |
| pec_hop | 20.50 | 25.50 | 26.00 | 22.50 |
| **Δ (PEC-best non-PEC)** | +2.50 | +1.00 | +4.00 | +2.50 |

**2wikimqa**

| Task / Base \ Prompt | default | extractive | short15 | concise |
|---|---:|---:|---:|---:|
| raw_topk | 18.00 | 25.00 | 24.50 | 20.50 |
| raw_topk_b840 | 15.50 | 25.00 | 25.00 | 20.50 |
| sentence_only | 17.00 | 23.00 | 22.50 | 18.00 |
| pec_hop | 17.00 | 30.00 | 29.50 | 21.00 |
| **Δ (PEC-best non-PEC)** | -1.00 | +5.00 | +4.50 | +0.50 |

**musique**

| Task / Base \ Prompt | default | extractive | short15 | concise |
|---|---:|---:|---:|---:|
| raw_topk | 4.50 | 8.00 | 8.00 | 6.50 |
| raw_topk_b840 | 6.50 | 6.00 | 7.00 | 7.50 |
| sentence_only | 6.00 | 7.50 | 6.00 | 6.50 |
| pec_hop | 7.00 | 9.50 | 9.50 | 7.50 |
| **Δ (PEC-best non-PEC)** | +0.50 | +1.50 | +1.50 | +0.00 |

**qasper**

| Task / Base \ Prompt | default | extractive | short15 | concise |
|---|---:|---:|---:|---:|
| raw_topk | 0.00 | 8.50 | 8.00 | 1.00 |
| raw_topk_b840 | 1.00 | 7.00 | 5.50 | 1.50 |
| sentence_only | 3.00 | 13.00 | 9.00 | 4.00 |
| pec_hop | 2.00 | 8.50 | 6.00 | 1.50 |
| **Δ (PEC-best non-PEC)** | -1.00 | -4.50 | -3.00 | -2.50 |

**multifieldqa_en**

| Task / Base \ Prompt | default | extractive | short15 | concise |
|---|---:|---:|---:|---:|
| raw_topk | 4.67 | 3.33 | 2.00 | 2.67 |
| raw_topk_b840 | 4.00 | 4.00 | 3.33 | 3.33 |
| sentence_only | 8.67 | 9.33 | 10.67 | 8.00 |
| pec_hop | 6.00 | 5.33 | 6.00 | 6.67 |
| **Δ (PEC-best non-PEC)** | -2.67 | -4.00 | -4.67 | -1.33 |

### lfm2.5-1.2b-instruct  --  F1


**hotpotqa**

| Task / Base \ Prompt | default | extractive | short15 | concise |
|---|---:|---:|---:|---:|
| raw_topk | 27.71 | 31.74 | 29.86 | 29.25 |
| raw_topk_b840 | 27.88 | 33.24 | 31.38 | 30.56 |
| sentence_only | 28.44 | 31.24 | 30.16 | 28.34 |
| pec_hop | 31.14 | 35.31 | 34.55 | 32.18 |
| **Δ (PEC-best non-PEC)** | +2.70 | +2.07 | +3.17 | +1.62 |

**2wikimqa**

| Task / Base \ Prompt | default | extractive | short15 | concise |
|---|---:|---:|---:|---:|
| raw_topk | 24.96 | 31.60 | 30.81 | 27.80 |
| raw_topk_b840 | 25.98 | 31.84 | 31.22 | 28.23 |
| sentence_only | 24.06 | 28.13 | 28.40 | 25.13 |
| pec_hop | 26.51 | 35.55 | 35.53 | 29.11 |
| **Δ (PEC-best non-PEC)** | +0.53 | +3.71 | +4.31 | +0.88 |

**musique**

| Task / Base \ Prompt | default | extractive | short15 | concise |
|---|---:|---:|---:|---:|
| raw_topk | 11.74 | 14.32 | 12.84 | 13.92 |
| raw_topk_b840 | 14.68 | 12.74 | 14.00 | 16.01 |
| sentence_only | 12.27 | 12.34 | 12.12 | 13.69 |
| pec_hop | 14.26 | 14.49 | 15.83 | 14.01 |
| **Δ (PEC-best non-PEC)** | -0.42 | +0.17 | +1.83 | -2.00 |

**qasper**

| Task / Base \ Prompt | default | extractive | short15 | concise |
|---|---:|---:|---:|---:|
| raw_topk | 9.04 | 16.27 | 16.49 | 9.74 |
| raw_topk_b840 | 10.77 | 14.86 | 14.38 | 9.98 |
| sentence_only | 22.30 | 28.04 | 26.55 | 22.14 |
| pec_hop | 12.64 | 17.68 | 16.88 | 12.24 |
| **Δ (PEC-best non-PEC)** | -9.66 | -10.36 | -9.67 | -9.90 |

**multifieldqa_en**

| Task / Base \ Prompt | default | extractive | short15 | concise |
|---|---:|---:|---:|---:|
| raw_topk | 26.87 | 21.21 | 21.79 | 23.07 |
| raw_topk_b840 | 27.30 | 19.40 | 20.17 | 24.21 |
| sentence_only | 36.40 | 33.28 | 34.34 | 33.55 |
| pec_hop | 27.21 | 21.24 | 21.96 | 27.09 |
| **Δ (PEC-best non-PEC)** | -9.19 | -12.04 | -12.38 | -6.46 |

### lfm2.5-1.2b-instruct -- best (base, prompt) per task

| Task | best base | best prompt | EM | F1 |
|---|---|---|---:|---:|
| hotpotqa | pec_hop | short15 | 26.00 | 34.55 |
| 2wikimqa | pec_hop | extractive | 30.00 | 35.55 |
| musique | pec_hop | extractive | 9.50 | 14.49 |
| qasper | sentence_only | extractive | 13.00 | 28.04 |
| multifieldqa_en | sentence_only | short15 | 10.67 | 34.34 |
| **macro** | -- | -- | **17.83** | **29.39** |
### qwen3-1.7b  --  EM


**hotpotqa**

| Task / Base \ Prompt | default | extractive | short15 | concise |
|---|---:|---:|---:|---:|
| raw_topk | 10.50 | 3.00 | 6.50 | 13.00 |
| raw_topk_b840 | - | 3.50 | 6.00 | 13.00 |
| sentence_only | 6.00 | 3.00 | 5.00 | 9.00 |
| pec_hop | 8.50 | 2.50 | 4.50 | 9.00 |
| **Δ (PEC-best non-PEC)** | -2.00 | -1.00 | -2.00 | -4.00 |

**2wikimqa**

| Task / Base \ Prompt | default | extractive | short15 | concise |
|---|---:|---:|---:|---:|
| raw_topk | 12.00 | 3.50 | 12.00 | 19.00 |
| raw_topk_b840 | - | 4.00 | 8.50 | 16.50 |
| sentence_only | 6.00 | 4.00 | 4.50 | 7.50 |
| pec_hop | 7.00 | 2.50 | 3.50 | 12.00 |
| **Δ (PEC-best non-PEC)** | -5.00 | -1.50 | -8.50 | -7.00 |

**musique**

| Task / Base \ Prompt | default | extractive | short15 | concise |
|---|---:|---:|---:|---:|
| raw_topk | 0.50 | 0.00 | 0.00 | 1.50 |
| raw_topk_b840 | - | 0.00 | 0.00 | 1.00 |
| sentence_only | 0.00 | 0.00 | 0.00 | 0.00 |
| pec_hop | 0.00 | 0.00 | 0.00 | 0.00 |
| **Δ (PEC-best non-PEC)** | -0.50 | +0.00 | +0.00 | -1.50 |

**qasper**

| Task / Base \ Prompt | default | extractive | short15 | concise |
|---|---:|---:|---:|---:|
| raw_topk | 6.00 | 4.50 | 5.00 | 6.50 |
| raw_topk_b840 | - | 3.50 | 5.50 | 4.00 |
| sentence_only | 5.00 | 2.00 | 5.00 | 4.00 |
| pec_hop | 4.50 | 0.50 | 3.00 | 4.00 |
| **Δ (PEC-best non-PEC)** | -1.50 | -4.00 | -2.50 | -2.50 |

**multifieldqa_en**

| Task / Base \ Prompt | default | extractive | short15 | concise |
|---|---:|---:|---:|---:|
| raw_topk | 2.67 | 2.00 | 2.00 | 2.00 |
| raw_topk_b840 | - | 2.00 | 1.33 | 3.33 |
| sentence_only | 2.00 | 0.67 | 1.33 | 2.00 |
| pec_hop | 2.00 | 0.67 | 2.00 | 1.33 |
| **Δ (PEC-best non-PEC)** | -0.67 | -1.33 | +0.00 | -2.00 |

### qwen3-1.7b  --  F1


**hotpotqa**

| Task / Base \ Prompt | default | extractive | short15 | concise |
|---|---:|---:|---:|---:|
| raw_topk | 23.05 | 8.45 | 9.70 | 26.56 |
| raw_topk_b840 | - | 6.77 | 9.37 | 24.93 |
| sentence_only | 14.45 | 7.25 | 7.26 | 19.16 |
| pec_hop | 19.02 | 7.29 | 8.04 | 21.43 |
| **Δ (PEC-best non-PEC)** | -4.03 | -1.16 | -1.66 | -5.13 |

**2wikimqa**

| Task / Base \ Prompt | default | extractive | short15 | concise |
|---|---:|---:|---:|---:|
| raw_topk | 20.74 | 8.23 | 13.92 | 25.56 |
| raw_topk_b840 | - | 7.07 | 10.26 | 23.83 |
| sentence_only | 15.71 | 9.52 | 7.41 | 16.53 |
| pec_hop | 17.19 | 8.71 | 9.62 | 21.32 |
| **Δ (PEC-best non-PEC)** | -3.55 | -0.81 | -4.30 | -4.24 |

**musique**

| Task / Base \ Prompt | default | extractive | short15 | concise |
|---|---:|---:|---:|---:|
| raw_topk | 8.00 | 1.17 | 1.10 | 9.81 |
| raw_topk_b840 | - | 0.81 | 1.39 | 10.14 |
| sentence_only | 5.53 | 2.65 | 2.02 | 6.29 |
| pec_hop | 7.32 | 2.34 | 2.30 | 7.66 |
| **Δ (PEC-best non-PEC)** | -0.68 | -0.31 | +0.28 | -2.48 |

**qasper**

| Task / Base \ Prompt | default | extractive | short15 | concise |
|---|---:|---:|---:|---:|
| raw_topk | 12.45 | 8.41 | 8.86 | 14.54 |
| raw_topk_b840 | - | 7.14 | 8.87 | 11.64 |
| sentence_only | 24.57 | 16.81 | 13.34 | 20.16 |
| pec_hop | 14.32 | 8.78 | 8.55 | 11.94 |
| **Δ (PEC-best non-PEC)** | -10.25 | -8.03 | -4.79 | -8.22 |

**multifieldqa_en**

| Task / Base \ Prompt | default | extractive | short15 | concise |
|---|---:|---:|---:|---:|
| raw_topk | 26.06 | 11.78 | 16.76 | 22.85 |
| raw_topk_b840 | - | 10.80 | 12.55 | 23.22 |
| sentence_only | 34.43 | 17.10 | 12.95 | 30.99 |
| pec_hop | 24.84 | 12.91 | 13.81 | 23.66 |
| **Δ (PEC-best non-PEC)** | -9.59 | -4.19 | -2.95 | -7.33 |

### qwen3-1.7b -- best (base, prompt) per task

| Task | best base | best prompt | EM | F1 |
|---|---|---|---:|---:|
| hotpotqa | raw_topk | concise | 13.00 | 26.56 |
| 2wikimqa | raw_topk | concise | 19.00 | 25.56 |
| musique | raw_topk | concise | 1.50 | 9.81 |
| qasper | raw_topk | concise | 6.50 | 14.54 |
| multifieldqa_en | raw_topk_b840 | concise | 3.33 | 23.22 |
| **macro** | -- | -- | **8.67** | **19.94** |
### qwen3-4b  --  EM


**hotpotqa**

| Task / Base \ Prompt | default | extractive | short15 | concise |
|---|---:|---:|---:|---:|
| raw_topk | 15.00 | 20.50 | 12.50 | 29.00 |
| raw_topk_b840 | 14.00 | 18.50 | 11.50 | 25.00 |
| sentence_only | 13.50 | 19.00 | 13.00 | 14.50 |
| pec_hop | 19.00 | 20.50 | 12.00 | 31.00 |
| **Δ (PEC-best non-PEC)** | +4.00 | +0.00 | -1.00 | +2.00 |

**2wikimqa**

| Task / Base \ Prompt | default | extractive | short15 | concise |
|---|---:|---:|---:|---:|
| raw_topk | 8.50 | 25.50 | 8.00 | 16.50 |
| raw_topk_b840 | - | - | - | - |
| sentence_only | 10.50 | 28.00 | 9.00 | 14.00 |
| pec_hop | 6.50 | 23.00 | 8.50 | 22.50 |
| **Δ (PEC-best non-PEC)** | -4.00 | -5.00 | -0.50 | +6.00 |

**musique**

| Task / Base \ Prompt | default | extractive | short15 | concise |
|---|---:|---:|---:|---:|
| raw_topk | 7.50 | 4.00 | 1.00 | 13.00 |
| raw_topk_b840 | - | - | - | - |
| sentence_only | 2.50 | 5.50 | 4.00 | 5.00 |
| pec_hop | 7.50 | 4.50 | 3.50 | 11.50 |
| **Δ (PEC-best non-PEC)** | +0.00 | -1.00 | -0.50 | -1.50 |

**qasper**

| Task / Base \ Prompt | default | extractive | short15 | concise |
|---|---:|---:|---:|---:|
| raw_topk | 2.50 | 6.00 | 3.00 | 5.50 |
| raw_topk_b840 | - | - | - | - |
| sentence_only | 5.00 | 10.50 | 4.00 | 4.00 |
| pec_hop | 3.50 | 6.00 | 4.00 | 6.50 |
| **Δ (PEC-best non-PEC)** | -1.50 | -4.50 | +0.00 | +1.00 |

**multifieldqa_en**

| Task / Base \ Prompt | default | extractive | short15 | concise |
|---|---:|---:|---:|---:|
| raw_topk | 4.00 | 4.00 | 3.33 | 4.67 |
| raw_topk_b840 | - | - | - | - |
| sentence_only | 3.33 | 7.33 | 7.33 | 8.00 |
| pec_hop | 4.00 | 5.33 | 6.00 | 9.33 |
| **Δ (PEC-best non-PEC)** | +0.00 | -2.00 | -1.33 | +1.33 |

### qwen3-4b  --  F1


**hotpotqa**

| Task / Base \ Prompt | default | extractive | short15 | concise |
|---|---:|---:|---:|---:|
| raw_topk | 29.06 | 27.00 | 23.30 | 45.21 |
| raw_topk_b840 | 28.10 | 26.00 | 23.35 | 41.11 |
| sentence_only | 23.64 | 25.56 | 20.68 | 23.60 |
| pec_hop | 32.78 | 28.30 | 24.70 | 44.62 |
| **Δ (PEC-best non-PEC)** | +3.72 | +1.30 | +1.35 | -0.59 |

**2wikimqa**

| Task / Base \ Prompt | default | extractive | short15 | concise |
|---|---:|---:|---:|---:|
| raw_topk | 19.91 | 28.16 | 16.66 | 30.48 |
| raw_topk_b840 | - | - | - | - |
| sentence_only | 21.17 | 30.42 | 18.35 | 22.78 |
| pec_hop | 16.70 | 25.94 | 18.31 | 32.42 |
| **Δ (PEC-best non-PEC)** | -4.47 | -4.48 | -0.04 | +1.94 |

**musique**

| Task / Base \ Prompt | default | extractive | short15 | concise |
|---|---:|---:|---:|---:|
| raw_topk | 17.28 | 9.37 | 8.99 | 22.91 |
| raw_topk_b840 | - | - | - | - |
| sentence_only | 9.16 | 8.59 | 8.69 | 10.06 |
| pec_hop | 15.49 | 7.27 | 10.54 | 20.35 |
| **Δ (PEC-best non-PEC)** | -1.79 | -2.10 | +1.55 | -2.56 |

**qasper**

| Task / Base \ Prompt | default | extractive | short15 | concise |
|---|---:|---:|---:|---:|
| raw_topk | 10.25 | 9.34 | 9.45 | 12.93 |
| raw_topk_b840 | - | - | - | - |
| sentence_only | 24.38 | 18.68 | 19.89 | 21.43 |
| pec_hop | 13.12 | 8.23 | 9.64 | 15.64 |
| **Δ (PEC-best non-PEC)** | -11.26 | -10.45 | -10.25 | -5.79 |

**multifieldqa_en**

| Task / Base \ Prompt | default | extractive | short15 | concise |
|---|---:|---:|---:|---:|
| raw_topk | 24.65 | 16.49 | 23.03 | 27.39 |
| raw_topk_b840 | - | - | - | - |
| sentence_only | 35.24 | 25.76 | 35.29 | 36.77 |
| pec_hop | 27.08 | 15.70 | 24.84 | 32.06 |
| **Δ (PEC-best non-PEC)** | -8.16 | -10.06 | -10.45 | -4.71 |

### qwen3-4b -- best (base, prompt) per task

| Task | best base | best prompt | EM | F1 |
|---|---|---|---:|---:|
| hotpotqa | pec_hop | concise | 31.00 | 44.62 |
| 2wikimqa | sentence_only | extractive | 28.00 | 30.42 |
| musique | raw_topk | concise | 13.00 | 22.91 |
| qasper | sentence_only | extractive | 10.50 | 18.68 |
| multifieldqa_en | pec_hop | concise | 9.33 | 32.06 |
| **macro** | -- | -- | **18.37** | **29.74** |