# Cost-aware MCE-Compass benchmark

Candidate bases: raw_topk, sentence_only, pec_hop
Candidate prompts: default, extractive, short15, concise
Tie tolerance: one dev question
Best fixed config by dev macro: **raw_topk+concise**

## Deterministic first-dev split

| reader | policy | EM | F1 | latency ms | ctx tok | EM vs raw | speed vs raw | EM vs best | speed vs best |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| lfm2.5-1.2b-instruct | raw | 9.39 | 20.88 | 286.7 | 1357 | +0.00 | 1.00x | -0.79 | 0.68x |
| lfm2.5-1.2b-instruct | best_single | 10.18 | 21.49 | 195.0 | 1357 | +0.79 | 1.47x | +0.00 | 1.00x |
| lfm2.5-1.2b-instruct | argmax | 17.02 | 29.53 | 186.9 | 677 | +7.62 | 1.53x | +6.84 | 1.04x |
| lfm2.5-1.2b-instruct | light | 16.70 | 28.78 | 148.0 | 459 | +7.31 | 1.94x | +6.52 | 1.32x |
| lfm2.5-1.2b-instruct | compass | 16.22 | 28.07 | 154.3 | 348 | +6.82 | 1.86x | +6.04 | 1.26x |
| lfm2.5-1.2b-instruct | oracle | 31.39 | 52.54 | - | - | +22.00 | - | +21.21 | - |
| qwen3-4b | raw | 7.48 | 20.47 | 1410.6 | 1369 | +0.00 | 1.00x | -6.32 | 1.91x |
| qwen3-4b | best_single | 13.79 | 28.29 | 2690.5 | 1369 | +6.32 | 0.52x | +0.00 | 1.00x |
| qwen3-4b | argmax | 16.21 | 30.99 | 2406.3 | 800 | +8.73 | 0.59x | +2.41 | 1.12x |
| qwen3-4b | light | 17.67 | 30.47 | 1935.2 | 686 | +10.19 | 0.73x | +3.88 | 1.39x |
| qwen3-4b | compass | 16.21 | 30.66 | 2392.2 | 800 | +8.73 | 0.59x | +2.41 | 1.12x |
| qwen3-4b | oracle | 32.72 | 55.31 | - | - | +25.24 | - | +18.92 | - |
| gemma-4-e4b | raw | 19.19 | 29.98 | 385.3 | 1345 | +0.00 | 1.00x | +4.58 | 0.94x |
| gemma-4-e4b | best_single | 14.61 | 23.53 | 360.8 | 1345 | -4.58 | 1.07x | +0.00 | 1.00x |
| gemma-4-e4b | argmax | 22.11 | 35.39 | 315.9 | 874 | +2.92 | 1.22x | +7.50 | 1.14x |
| gemma-4-e4b | light | 19.31 | 30.09 | 283.5 | 874 | +0.12 | 1.36x | +4.70 | 1.27x |
| gemma-4-e4b | compass | 22.11 | 35.39 | 315.9 | 874 | +2.92 | 1.22x | +7.50 | 1.14x |
| gemma-4-e4b | oracle | 31.25 | 50.11 | - | - | +12.06 | - | +16.64 | - |

## Per-cell choices

| reader | task | Argmax | Light | Compass | compass EM | compass F1 | compass ms | compass ctx |
|---|---|---|---|---|---:|---:|---:|---:|
| lfm2.5-1.2b-instruct | hotpotqa | pec_hop+short15 | sentence_only+short15 | sentence_only+short15 | 23.33 | 31.51 | 158.3 | 224 |
| lfm2.5-1.2b-instruct | 2wikimqa | pec_hop+extractive | pec_hop+extractive | pec_hop+extractive | 27.33 | 33.24 | 128.0 | 790 |
| lfm2.5-1.2b-instruct | musique | raw_topk+concise | pec_hop+extractive | sentence_only+extractive | 8.00 | 14.11 | 192.7 | 238 |
| lfm2.5-1.2b-instruct | qasper | sentence_only+extractive | sentence_only+extractive | sentence_only+extractive | 13.33 | 28.11 | 143.6 | 197 |
| lfm2.5-1.2b-instruct | multifieldqa_en | sentence_only+short15 | sentence_only+extractive | sentence_only+short15 | 9.09 | 33.40 | 148.9 | 290 |
| qwen3-4b | hotpotqa | raw_topk+concise | raw_topk+concise | raw_topk+concise | 28.00 | 43.92 | 8713.1 | 1382 |
| qwen3-4b | 2wikimqa | pec_hop+concise | sentence_only+extractive | pec_hop+concise | 22.67 | 33.27 | 2430.3 | 813 |
| qwen3-4b | musique | raw_topk+concise | raw_topk+concise | raw_topk+concise | 12.67 | 23.36 | 434.4 | 1333 |
| qwen3-4b | qasper | sentence_only+extractive | sentence_only+extractive | sentence_only+extractive | 11.33 | 19.74 | 145.1 | 185 |
| qwen3-4b | multifieldqa_en | sentence_only+concise | sentence_only+short15 | sentence_only+short15 | 6.36 | 33.01 | 238.1 | 287 |
| gemma-4-e4b | hotpotqa | raw_topk+default | raw_topk+default | raw_topk+default | 37.33 | 49.34 | 365.6 | 1358 |
| gemma-4-e4b | 2wikimqa | raw_topk+default | raw_topk+default | raw_topk+default | 33.33 | 41.70 | 338.2 | 1241 |
| gemma-4-e4b | musique | raw_topk+concise | raw_topk+extractive | raw_topk+concise | 14.67 | 20.38 | 347.8 | 1308 |
| gemma-4-e4b | qasper | sentence_only+default | sentence_only+short15 | sentence_only+default | 10.67 | 29.05 | 259.8 | 183 |
| gemma-4-e4b | multifieldqa_en | sentence_only+concise | sentence_only+concise | sentence_only+concise | 14.55 | 36.47 | 267.9 | 279 |

## 10-split stability for MCE-Compass

| reader | EM mean±std | Δ vs best mean±std | Δ vs raw mean±std | speed vs raw | #Δbest>0 |
|---|---:|---:|---:|---:|---:|
| lfm2.5-1.2b-instruct | 15.93±1.00 | +5.14±2.01 | +7.05±0.90 | 1.83x | 10/10 |
| qwen3-4b | 16.24±1.47 | +2.11±2.20 | +8.60±1.42 | 1.99x | 7/10 |
| gemma-4-e4b | 19.95±1.14 | +8.04±3.25 | +1.82±1.25 | 1.23x | 10/10 |