# MCE-COMPASS rule policy with EXPANDED candidate pool

Bases (7): raw_topk, raw_topk_b840, sentence_only, llmlingua2, provence, pec_hop, pec_hop_rerank
Prompts (5): default, extractive, short15, concise, fewextractive

## Per-(reader, task) rule choice on dev (first 50/40 questions)

| reader | task | n_candidates | dev winner | dev EM | test EM |
|---|---|---:|---|---:|---:|
| lfm2.5-1.2b-instruct | hotpotqa | 28 | provence + fewextractive | 22.00 | 30.67 |
| lfm2.5-1.2b-instruct | 2wikimqa | 28 | pec_hop + fewextractive | 40.00 | 27.33 |
| lfm2.5-1.2b-instruct | musique | 28 | provence + extractive | 8.00 | 12.67 |
| lfm2.5-1.2b-instruct | qasper | 26 | sentence_only + extractive | 12.00 | 13.33 |
| lfm2.5-1.2b-instruct | multifieldqa_en | 16 | sentence_only + short15 | 15.00 | 9.09 |
| qwen3-4b | hotpotqa | 28 | raw_topk + fewextractive | 44.00 | 42.67 |
| qwen3-4b | 2wikimqa | 24 | raw_topk + fewextractive | 32.00 | 40.67 |
| qwen3-4b | musique | 24 | pec_hop_rerank + fewextractive | 22.00 | 19.33 |
| qwen3-4b | qasper | 22 | sentence_only + extractive | 8.00 | 11.33 |
| qwen3-4b | multifieldqa_en | 12 | sentence_only + concise | 12.50 | 6.36 |
| gemma-4-e4b | hotpotqa | 28 | provence + fewextractive | 28.00 | 27.33 |
| gemma-4-e4b | 2wikimqa | 28 | pec_hop + fewextractive | 36.00 | 25.33 |
| gemma-4-e4b | musique | 28 | pec_hop_rerank + default | 14.00 | 12.67 |
| gemma-4-e4b | qasper | 26 | sentence_only + default | 10.00 | 10.67 |
| gemma-4-e4b | multifieldqa_en | 16 | sentence_only + default | 15.00 | 13.64 |

## Selection frequency per reader (over 5 tasks)

| reader | bases picked | prompts picked |
|---|---|---|
| lfm2.5-1.2b-instruct | provence(2), sentence_only(2), pec_hop(1) | fewextractive(2), extractive(2), short15(1) |
| qwen3-4b | raw_topk(2), sentence_only(2), pec_hop_rerank(1) | fewextractive(3), extractive(1), concise(1) |
| gemma-4-e4b | sentence_only(2), provence(1), pec_hop(1), pec_hop_rerank(1) | default(3), fewextractive(2) |

## Best-fixed-single config (same (base, prompt) across all reader×task)

Top 10 single configs by dev macro EM (over all complete cells):

| rank | (base, prompt) | dev macro EM |
|---:|---|---:|
| 1 | raw_topk + concise | 11.60 |
| 2 | pec_hop + concise | 10.90 |
| 3 | pec_hop + extractive | 10.27 |
| 4 | raw_topk + default | 10.03 |
| 5 | sentence_only + default | 9.60 |
| 6 | sentence_only + extractive | 9.60 |
| 7 | sentence_only + concise | 9.43 |
| 8 | pec_hop + default | 9.30 |
| 9 | raw_topk + extractive | 9.17 |
| 10 | sentence_only + short15 | 7.77 |

**Best single = raw_topk + concise**

| reader | task | test EM (best single) |
|---|---|---:|
| lfm2.5-1.2b-instruct | hotpotqa | 22.00 |
| lfm2.5-1.2b-instruct | 2wikimqa | 20.00 |
| lfm2.5-1.2b-instruct | musique | 6.67 |
| lfm2.5-1.2b-instruct | qasper | 1.33 |
| lfm2.5-1.2b-instruct | multifieldqa_en | 0.91 |
| qwen3-4b | hotpotqa | 28.00 |
| qwen3-4b | 2wikimqa | 17.33 |
| qwen3-4b | musique | 12.67 |
| qwen3-4b | qasper | 7.33 |
| qwen3-4b | multifieldqa_en | 3.64 |
| gemma-4-e4b | hotpotqa | 28.67 |
| gemma-4-e4b | 2wikimqa | 22.67 |
| gemma-4-e4b | musique | 14.67 |
| gemma-4-e4b | qasper | 0.67 |
| gemma-4-e4b | multifieldqa_en | 6.36 |

| reader | macro test EM (best single) | macro test EM (rule) | Δ |
|---|---:|---:|---:|
| lfm2.5-1.2b-instruct | 10.18 | 18.62 | +8.44 |
| qwen3-4b | 13.79 | 24.07 | +10.28 |
| gemma-4-e4b | 14.61 | 17.93 | +3.32 |

## What wins under fewextractive (matched-prompt sub-pool)

| reader | task | best base (fewextractive only) | dev EM |
|---|---|---|---:|
| lfm2.5-1.2b-instruct | hotpotqa | provence | 22.00 |
| lfm2.5-1.2b-instruct | 2wikimqa | pec_hop | 40.00 |
| lfm2.5-1.2b-instruct | musique | pec_hop | 6.00 |
| lfm2.5-1.2b-instruct | qasper | pec_hop | 8.00 |
| qwen3-4b | hotpotqa | raw_topk | 44.00 |
| qwen3-4b | 2wikimqa | raw_topk | 32.00 |
| qwen3-4b | musique | pec_hop_rerank | 22.00 |
| qwen3-4b | qasper | sentence_only | 8.00 |
| gemma-4-e4b | hotpotqa | provence | 28.00 |
| gemma-4-e4b | 2wikimqa | pec_hop | 36.00 |
| gemma-4-e4b | musique | pec_hop_rerank | 12.00 |
| gemma-4-e4b | qasper | sentence_only | 4.00 |
