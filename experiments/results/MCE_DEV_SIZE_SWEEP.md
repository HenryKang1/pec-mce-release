# MCE-Compass dev-size sensitivity

Completed LongBench result grids are re-scored with different calibration-set sizes.
No reader model is called. Budgets are shown as `multi-doc/single-doc` because
multifieldqa_en has 150 examples and uses the same 0.8 ratio as the main 50/40 split.

| Dev budget | Reader | Delta vs best fixed EM | Delta vs Raw EM | Speed vs Raw | Positive splits |
|---|---|---:|---:|---:|---:|
| 10/8 | lfm2.5-1.2b-instruct | +3.48+-2.04 | +4.85 | 1.70x | 19/20 |
| 10/8 | qwen3-4b | +4.59+-3.49 | +7.34 | 1.92x | 18/20 |
| 10/8 | gemma-4-e4b | +3.47+-3.07 | -0.48 | 1.32x | 19/20 |
| 25/20 | lfm2.5-1.2b-instruct | +3.82+-1.76 | +6.05 | 1.73x | 19/20 |
| 25/20 | qwen3-4b | +2.13+-3.24 | +7.93 | 2.23x | 15/20 |
| 25/20 | gemma-4-e4b | +7.46+-2.48 | +1.19 | 1.25x | 20/20 |
| 50/40 | lfm2.5-1.2b-instruct | +4.66+-1.74 | +6.61 | 1.81x | 20/20 |
| 50/40 | qwen3-4b | +1.92+-1.76 | +8.88 | 2.28x | 17/20 |
| 50/40 | gemma-4-e4b | +8.13+-2.42 | +2.05 | 1.23x | 20/20 |
| 75/60 | lfm2.5-1.2b-instruct | +5.11+-1.31 | +7.09 | 1.83x | 20/20 |
| 75/60 | qwen3-4b | +1.53+-1.77 | +9.13 | 2.68x | 15/20 |
| 75/60 | gemma-4-e4b | +8.30+-1.18 | +2.31 | 1.21x | 20/20 |

Takeaway: the selector is not only a first-50 artifact. At the paper's
50/40 dev budget it is positive over the best fixed interface on
20/20 LFM and Gemma splits and 15/20 Qwen splits; the same qualitative
pattern holds at 25/20 and 75/60. Even the small 10/8 budget is
mostly positive, but the main paper should use 50/40 as the
reproducible operating point.
