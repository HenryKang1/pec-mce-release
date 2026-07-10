# ARR-defense variant comparison

### lfm2.5-1.2b-instruct

| Task | Variant | n | EM | F1 | Lat ms | Ctx tok |
|---|---|---:|---:|---:|---:|---:|
| hotpotqa | raw_topk | 200 | 18.00 | 27.71 | 372.2 | 1361 |
| hotpotqa | raw_topk_b840 | 200 | 16.00 | 27.88 | 113.1 | 839 |
| hotpotqa | pec_hop | 200 | 20.50 | 31.14 | 150.6 | 808 |
| hotpotqa | pec_bridge | 200 | 18.00 | 30.33 | 148.4 | 673 |
| hotpotqa | pec_bridge_k3 | 200 | 16.50 | 27.77 | 146.7 | 897 |
| hotpotqa | sentence_only | 200 | 18.00 | 28.44 | 165.2 | 226 |
| hotpotqa | pec_router | 200 | 17.50 | 30.23 | 196.6 | 760 |
| hotpotqa | pec_hop_extractive | 200 | 25.50 | 35.31 | 163.5 | 808 |
| hotpotqa | pec_hop_short15 | 200 | 26.00 | 34.55 | 320.4 | 808 |
| hotpotqa | pec_hop_concise | 200 | 22.50 | 32.18 | 365.6 | 808 |
| hotpotqa | pec_hop_relations | 200 | 22.50 | 33.57 | 171.3 | 806 |
| hotpotqa | pec_hop_shuffle_ptr | 200 | 14.00 | 22.71 | 181.4 | 809 |
| hotpotqa | pec_hop_random_anchor | 200 | 18.00 | 26.62 | 155.8 | 806 |

| 2wikimqa | raw_topk | 200 | 18.00 | 24.96 | 359.1 | 1248 |
| 2wikimqa | raw_topk_b840 | 200 | 15.50 | 25.98 | 108.1 | 832 |
| 2wikimqa | pec_hop | 200 | 17.00 | 26.51 | 144.1 | 787 |
| 2wikimqa | pec_bridge | 200 | 15.50 | 24.99 | 233.0 | 609 |
| 2wikimqa | pec_bridge_k3 | 200 | 16.50 | 26.37 | 138.8 | 817 |
| 2wikimqa | sentence_only | 200 | 17.00 | 24.06 | 111.8 | 241 |
| 2wikimqa | pec_router | 200 | 15.50 | 25.36 | 156.6 | 760 |
| 2wikimqa | pec_hop_extractive | 200 | 30.00 | 35.55 | 129.1 | 787 |
| 2wikimqa | pec_hop_short15 | 200 | 29.50 | 35.53 | 348.9 | 787 |
| 2wikimqa | pec_hop_concise | 200 | 21.00 | 29.11 | 374.4 | 787 |
| 2wikimqa | pec_hop_relations | 200 | 16.00 | 25.66 | 139.1 | 789 |
| 2wikimqa | pec_hop_shuffle_ptr | 200 | 18.00 | 26.37 | 148.9 | 736 |
| 2wikimqa | pec_hop_random_anchor | 200 | 15.50 | 24.12 | 155.7 | 795 |

| musique | raw_topk | 200 | 4.50 | 11.74 | 372.3 | 1314 |
| musique | raw_topk_b840 | 200 | 6.50 | 14.68 | 125.5 | 839 |
| musique | pec_hop | 200 | 7.00 | 14.26 | 159.3 | 796 |
| musique | pec_bridge | 200 | 5.00 | 12.36 | 256.7 | 627 |
| musique | pec_bridge_k3 | 200 | 5.50 | 13.13 | 157.5 | 844 |
| musique | sentence_only | 200 | 6.00 | 12.27 | 158.2 | 237 |
| musique | pec_router | 200 | 4.50 | 11.69 | 199.2 | 742 |
| musique | pec_hop_extractive | 200 | 9.50 | 14.49 | 162.9 | 796 |
| musique | pec_hop_short15 | 200 | 9.50 | 15.83 | 367.2 | 796 |
| musique | pec_hop_concise | 200 | 7.50 | 14.01 | 395.5 | 796 |
| musique | pec_hop_relations | 200 | 6.50 | 14.42 | 201.9 | 805 |
| musique | pec_hop_shuffle_ptr | 200 | 6.50 | 12.08 | 165.5 | 812 |
| musique | pec_hop_random_anchor | 200 | 3.50 | 10.14 | 174.6 | 791 |

| qasper | raw_topk | 200 | 0.00 | 9.04 | 163.7 | 1392 |
| qasper | raw_topk_b840 | 200 | 1.00 | 10.77 | 154.2 | 840 |
| qasper | pec_hop | 200 | 2.00 | 12.64 | 135.7 | 796 |
| qasper | pec_bridge | 200 | 1.00 | 11.12 | 129.1 | 570 |
| qasper | pec_bridge_k3 | 200 | 0.50 | 10.78 | 150.3 | 856 |
| qasper | sentence_only | 200 | 3.00 | 22.30 | 153.4 | 198 |
| qasper | pec_router | 200 | 1.00 | 11.21 | 167.7 | 582 |
| qasper | pec_hop_extractive | 200 | 8.50 | 17.68 | 128.9 | 796 |
| qasper | pec_hop_short15 | 200 | 6.00 | 16.88 | 235.6 | 796 |
| qasper | pec_hop_concise | 200 | 1.50 | 12.24 | 298.5 | 796 |
| qasper | pec_hop_relations | 200 | 2.00 | 12.15 | 156.9 | 797 |
| qasper | pec_hop_shuffle_ptr | - | - | - | - | - |
| qasper | pec_hop_random_anchor | - | - | - | - | - |

| multifieldqa_en | raw_topk | 150 | 4.67 | 26.87 | 176.6 | 1467 |
| multifieldqa_en | raw_topk_b840 | 150 | 4.00 | 27.30 | 149.1 | 840 |
| multifieldqa_en | pec_hop | 150 | 6.00 | 27.21 | 148.2 | 1000 |
| multifieldqa_en | pec_bridge | 150 | 3.33 | 26.82 | 134.8 | 685 |
| multifieldqa_en | pec_bridge_k3 | 150 | 3.33 | 25.32 | 156.6 | 1001 |
| multifieldqa_en | sentence_only | 150 | 8.67 | 36.40 | 145.4 | 286 |
| multifieldqa_en | pec_router | 150 | 3.33 | 26.41 | 162.6 | 743 |
| multifieldqa_en | pec_hop_extractive | 150 | 5.33 | 21.24 | 127.6 | 1000 |
| multifieldqa_en | pec_hop_short15 | 150 | 6.00 | 21.96 | 267.1 | 1000 |
| multifieldqa_en | pec_hop_concise | 150 | 6.67 | 27.09 | 346.1 | 1000 |
| multifieldqa_en | pec_hop_relations | 150 | 4.67 | 25.99 | 162.7 | 1001 |
| multifieldqa_en | pec_hop_shuffle_ptr | - | - | - | - | - |
| multifieldqa_en | pec_hop_random_anchor | - | - | - | - | - |

### lfm2.5-1.2b-instruct — deltas vs pec_hop

| Task | metric | pec_hop | sentence_only | pec_router | pec_hop_extractive | pec_hop_relations |
|---|---|---:|---:|---:|---:|---:|
| hotpotqa | EM | 20.50 | 18.00 (-2.50) | 17.50 (-3.00) | 25.50 (+5.00) | 26.00 (+5.50) | 22.50 (+2.00) | 22.50 (+2.00) |
| hotpotqa | F1 | 31.14 | 28.44 (-2.70) | 30.23 (-0.91) | 35.31 (+4.17) | 34.55 (+3.41) | 32.18 (+1.04) | 33.57 (+2.43) |

| 2wikimqa | EM | 17.00 | 17.00 (+0.00) | 15.50 (-1.50) | 30.00 (+13.00) | 29.50 (+12.50) | 21.00 (+4.00) | 16.00 (-1.00) |
| 2wikimqa | F1 | 26.51 | 24.06 (-2.45) | 25.36 (-1.15) | 35.55 (+9.04) | 35.53 (+9.02) | 29.11 (+2.60) | 25.66 (-0.85) |

| musique | EM | 7.00 | 6.00 (-1.00) | 4.50 (-2.50) | 9.50 (+2.50) | 9.50 (+2.50) | 7.50 (+0.50) | 6.50 (-0.50) |
| musique | F1 | 14.26 | 12.27 (-1.99) | 11.69 (-2.57) | 14.49 (+0.23) | 15.83 (+1.57) | 14.01 (-0.25) | 14.42 (+0.16) |

| qasper | EM | 2.00 | 3.00 (+1.00) | 1.00 (-1.00) | 8.50 (+6.50) | 6.00 (+4.00) | 1.50 (-0.50) | 2.00 (+0.00) |
| qasper | F1 | 12.64 | 22.30 (+9.66) | 11.21 (-1.43) | 17.68 (+5.04) | 16.88 (+4.24) | 12.24 (-0.40) | 12.15 (-0.49) |

| multifieldqa_en | EM | 6.00 | 8.67 (+2.67) | 3.33 (-2.67) | 5.33 (-0.67) | 6.00 (+0.00) | 6.67 (+0.67) | 4.67 (-1.33) |
| multifieldqa_en | F1 | 27.21 | 36.40 (+9.19) | 26.41 (-0.80) | 21.24 (-5.97) | 21.96 (-5.25) | 27.09 (-0.12) | 25.99 (-1.22) |

### qwen3-1.7b

| Task | Variant | n | EM | F1 | Lat ms | Ctx tok |
|---|---|---:|---:|---:|---:|---:|
| hotpotqa | raw_topk | 200 | 10.50 | 23.05 | 1455.0 | 1385 |
| hotpotqa | raw_topk_b840 | - | - | - | - | - |
| hotpotqa | pec_hop | 200 | 8.50 | 19.02 | 213.3 | 820 |
| hotpotqa | pec_bridge | 200 | 11.00 | 22.49 | 1313.7 | 685 |
| hotpotqa | pec_bridge_k3 | 200 | 8.00 | 19.63 | 246.2 | 915 |
| hotpotqa | sentence_only | 200 | 6.00 | 14.45 | 212.4 | 222 |
| hotpotqa | pec_router | 200 | 10.00 | 22.34 | 264.1 | 775 |
| hotpotqa | pec_hop_extractive | 200 | 2.50 | 7.29 | 238.9 | 820 |
| hotpotqa | pec_hop_short15 | 200 | 4.50 | 8.04 | 654.3 | 820 |
| hotpotqa | pec_hop_concise | 200 | 9.00 | 21.43 | 669.2 | 820 |
| hotpotqa | pec_hop_relations | 200 | 8.00 | 19.62 | 252.3 | 818 |
| hotpotqa | pec_hop_shuffle_ptr | - | - | - | - | - |
| hotpotqa | pec_hop_random_anchor | - | - | - | - | - |

| 2wikimqa | raw_topk | 200 | 12.00 | 20.74 | 1503.3 | 1277 |
| 2wikimqa | raw_topk_b840 | - | - | - | - | - |
| 2wikimqa | pec_hop | 200 | 7.00 | 17.19 | 244.0 | 811 |
| 2wikimqa | pec_bridge | 200 | 8.00 | 18.66 | 1364.9 | 624 |
| 2wikimqa | pec_bridge_k3 | 200 | 9.50 | 19.19 | 253.8 | 840 |
| 2wikimqa | sentence_only | 200 | 6.00 | 15.71 | 220.5 | 241 |
| 2wikimqa | pec_router | 200 | 8.00 | 18.80 | 285.5 | 781 |
| 2wikimqa | pec_hop_extractive | 200 | 2.50 | 8.71 | 228.0 | 811 |
| 2wikimqa | pec_hop_short15 | 200 | 3.50 | 9.62 | 690.5 | 811 |
| 2wikimqa | pec_hop_concise | 200 | 12.00 | 21.32 | 779.8 | 811 |
| 2wikimqa | pec_hop_relations | 200 | 7.50 | 18.53 | 272.8 | 812 |
| 2wikimqa | pec_hop_shuffle_ptr | - | - | - | - | - |
| 2wikimqa | pec_hop_random_anchor | - | - | - | - | - |

| musique | raw_topk | 200 | 0.50 | 8.00 | 1510.9 | 1342 |
| musique | raw_topk_b840 | - | - | - | - | - |
| musique | pec_hop | 200 | 0.00 | 7.32 | 243.1 | 808 |
| musique | pec_bridge | 200 | 1.00 | 7.09 | 1459.3 | 639 |
| musique | pec_bridge_k3 | 200 | 0.00 | 7.50 | 265.7 | 863 |
| musique | sentence_only | 200 | 0.00 | 5.53 | 249.9 | 232 |
| musique | pec_router | 200 | 0.50 | 7.20 | 297.4 | 757 |
| musique | pec_hop_extractive | 200 | 0.00 | 2.34 | 271.1 | 808 |
| musique | pec_hop_short15 | 200 | 0.00 | 2.30 | 748.2 | 808 |
| musique | pec_hop_concise | 200 | 0.00 | 7.66 | 776.6 | 808 |
| musique | pec_hop_relations | 200 | 0.00 | 7.53 | 278.9 | 817 |
| musique | pec_hop_shuffle_ptr | - | - | - | - | - |
| musique | pec_hop_random_anchor | - | - | - | - | - |

| qasper | raw_topk | 200 | 6.00 | 12.45 | 216.9 | 1373 |
| qasper | raw_topk_b840 | - | - | - | - | - |
| qasper | pec_hop | 200 | 4.50 | 14.32 | 194.0 | 780 |
| qasper | pec_bridge | 200 | 4.00 | 11.81 | 161.8 | 559 |
| qasper | pec_bridge_k3 | 200 | 4.00 | 10.69 | 183.1 | 840 |
| qasper | sentence_only | 200 | 5.00 | 24.57 | 181.5 | 186 |
| qasper | pec_router | 200 | 4.00 | 11.80 | 190.5 | 571 |
| qasper | pec_hop_extractive | 200 | 0.50 | 8.78 | 241.4 | 780 |
| qasper | pec_hop_short15 | 200 | 3.00 | 8.55 | 477.3 | 780 |
| qasper | pec_hop_concise | 200 | 4.00 | 11.94 | 541.9 | 780 |
| qasper | pec_hop_relations | 200 | 4.00 | 14.10 | 273.5 | 780 |
| qasper | pec_hop_shuffle_ptr | - | - | - | - | - |
| qasper | pec_hop_random_anchor | - | - | - | - | - |

| multifieldqa_en | raw_topk | 150 | 2.67 | 26.06 | 265.6 | 1468 |
| multifieldqa_en | raw_topk_b840 | - | - | - | - | - |
| multifieldqa_en | pec_hop | 150 | 2.00 | 24.84 | 233.0 | 1011 |
| multifieldqa_en | pec_bridge | 150 | 2.00 | 26.08 | 206.4 | 698 |
| multifieldqa_en | pec_bridge_k3 | 150 | 1.33 | 23.71 | 237.1 | 1019 |
| multifieldqa_en | sentence_only | 150 | 2.00 | 34.43 | 298.1 | 283 |
| multifieldqa_en | pec_router | 150 | 2.00 | 25.17 | 331.3 | 757 |
| multifieldqa_en | pec_hop_extractive | 150 | 0.67 | 12.91 | 310.1 | 1011 |
| multifieldqa_en | pec_hop_short15 | 150 | 2.00 | 13.81 | 557.9 | 1011 |
| multifieldqa_en | pec_hop_concise | 150 | 1.33 | 23.66 | 661.2 | 1011 |
| multifieldqa_en | pec_hop_relations | 150 | 1.33 | 25.60 | 358.7 | 1009 |
| multifieldqa_en | pec_hop_shuffle_ptr | - | - | - | - | - |
| multifieldqa_en | pec_hop_random_anchor | - | - | - | - | - |

### qwen3-1.7b — deltas vs pec_hop

| Task | metric | pec_hop | sentence_only | pec_router | pec_hop_extractive | pec_hop_relations |
|---|---|---:|---:|---:|---:|---:|
| hotpotqa | EM | 8.50 | 6.00 (-2.50) | 10.00 (+1.50) | 2.50 (-6.00) | 4.50 (-4.00) | 9.00 (+0.50) | 8.00 (-0.50) |
| hotpotqa | F1 | 19.02 | 14.45 (-4.57) | 22.34 (+3.32) | 7.29 (-11.73) | 8.04 (-10.98) | 21.43 (+2.41) | 19.62 (+0.60) |

| 2wikimqa | EM | 7.00 | 6.00 (-1.00) | 8.00 (+1.00) | 2.50 (-4.50) | 3.50 (-3.50) | 12.00 (+5.00) | 7.50 (+0.50) |
| 2wikimqa | F1 | 17.19 | 15.71 (-1.48) | 18.80 (+1.61) | 8.71 (-8.48) | 9.62 (-7.57) | 21.32 (+4.13) | 18.53 (+1.34) |

| musique | EM | 0.00 | 0.00 (+0.00) | 0.50 (+0.50) | 0.00 (+0.00) | 0.00 (+0.00) | 0.00 (+0.00) | 0.00 (+0.00) |
| musique | F1 | 7.32 | 5.53 (-1.79) | 7.20 (-0.12) | 2.34 (-4.98) | 2.30 (-5.02) | 7.66 (+0.34) | 7.53 (+0.21) |

| qasper | EM | 4.50 | 5.00 (+0.50) | 4.00 (-0.50) | 0.50 (-4.00) | 3.00 (-1.50) | 4.00 (-0.50) | 4.00 (-0.50) |
| qasper | F1 | 14.32 | 24.57 (+10.25) | 11.80 (-2.52) | 8.78 (-5.54) | 8.55 (-5.77) | 11.94 (-2.38) | 14.10 (-0.22) |

| multifieldqa_en | EM | 2.00 | 2.00 (+0.00) | 2.00 (+0.00) | 0.67 (-1.33) | 2.00 (+0.00) | 1.33 (-0.67) | 1.33 (-0.67) |
| multifieldqa_en | F1 | 24.84 | 34.43 (+9.59) | 25.17 (+0.33) | 12.91 (-11.93) | 13.81 (-11.03) | 23.66 (-1.18) | 25.60 (+0.76) |

### lfm2.5-1.2b-instruct — corruption stress (negative controls)

| Task | metric | pec_hop | shuffle_ptr | random_anchor |
|---|---|---:|---:|---:|
| hotpotqa | EM | 20.50 | 14.00 (-6.50) | 18.00 (-2.50) |
| hotpotqa | F1 | 31.14 | 22.71 (-8.43) | 26.62 (-4.52) |

| 2wikimqa | EM | 17.00 | 18.00 (+1.00) | 15.50 (-1.50) |
| 2wikimqa | F1 | 26.51 | 26.37 (-0.14) | 24.12 (-2.39) |

| musique | EM | 7.00 | 6.50 (-0.50) | 3.50 (-3.50) |
| musique | F1 | 14.26 | 12.08 (-2.18) | 10.14 (-4.12) |
