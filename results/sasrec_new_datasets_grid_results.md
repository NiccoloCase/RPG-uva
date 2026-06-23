# SASRec New-Dataset Grid Results

Source sweep:

- Job array definition: `jobs/new_datasets/sasrec/grid/run_sasrec_train_grid.sh`
- Slurm array job: `24119085`
- Selection metric: best validation `NDCG@20`
- Seed: `2024`

Each row reports the metrics from the epoch where that run achieved its best validation `NDCG@20`.

## Video Games

| Rank | lr | dropout | blocks | best epoch | NDCG@10 | NDCG@20 | Recall@10 | Recall@20 | Log |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.0003 | 0.5 | 2 | 126 | 0.0627 | 0.0789 | 0.1200 | 0.1843 | `sasrec_new_train_grid-24119085_1.out` |
| 2 | 0.0003 | 0.2 | 2 | 61 | 0.0609 | 0.0768 | 0.1188 | 0.1821 | `sasrec_new_train_grid-24119085_0.out` |
| 3 | 0.001 | 0.5 | 3 | 53 | 0.0577 | 0.0730 | 0.1119 | 0.1724 | `sasrec_new_train_grid-24119085_10.out` |
| 4 | 0.001 | 0.5 | 2 | 45 | 0.0578 | 0.0723 | 0.1124 | 0.1701 | `sasrec_new_train_grid-24119085_4.out` |
| 5 | 0.001 | 0.5 | 1 | 49 | 0.0570 | 0.0713 | 0.1120 | 0.1689 | `sasrec_new_train_grid-24119085_9.out` |
| 6 | 0.001 | 0.2 | 2 | 28 | 0.0566 | 0.0710 | 0.1100 | 0.1675 | `sasrec_new_train_grid-24119085_3.out` |
| 7 | 0.0003 | 0.8 | 2 | 174 | 0.0527 | 0.0673 | 0.1017 | 0.1595 | `sasrec_new_train_grid-24119085_2.out` |
| 8 | 0.003 | 0.5 | 2 | 28 | 0.0532 | 0.0668 | 0.1028 | 0.1573 | `sasrec_new_train_grid-24119085_7.out` |
| 9 | 0.003 | 0.2 | 2 | 56 | 0.0504 | 0.0643 | 0.0996 | 0.1550 | `sasrec_new_train_grid-24119085_6.out` |
| 10 | 0.001 | 0.8 | 2 | 61 | 0.0490 | 0.0618 | 0.0950 | 0.1461 | `sasrec_new_train_grid-24119085_5.out` |
| 11 | 0.003 | 0.8 | 2 | 82 | 0.0414 | 0.0531 | 0.0809 | 0.1273 | `sasrec_new_train_grid-24119085_8.out` |

Best setting:

- `lr=0.0003`, `dropout=0.5`, `blocks=2`

## Pet Supplies

| Rank | lr | dropout | blocks | best epoch | NDCG@10 | NDCG@20 | Recall@10 | Recall@20 | Log |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.0003 | 0.2 | 2 | 95 | 0.0360 | 0.0452 | 0.0683 | 0.1047 | `sasrec_new_train_grid-24119085_11.out` |
| 2 | 0.0003 | 0.5 | 2 | 94 | 0.0356 | 0.0449 | 0.0677 | 0.1049 | `sasrec_new_train_grid-24119085_12.out` |
| 3 | 0.001 | 0.5 | 2 | 50 | 0.0333 | 0.0418 | 0.0637 | 0.0973 | `sasrec_new_train_grid-24119085_15.out` |
| 4 | 0.001 | 0.5 | 1 | 36 | 0.0321 | 0.0416 | 0.0627 | 0.1005 | `sasrec_new_train_grid-24119085_20.out` |
| 5 | 0.001 | 0.2 | 2 | 29 | 0.0310 | 0.0398 | 0.0610 | 0.0958 | `sasrec_new_train_grid-24119085_14.out` |
| 6 | 0.001 | 0.5 | 3 | 18 | 0.0305 | 0.0396 | 0.0588 | 0.0948 | `sasrec_new_train_grid-24119085_21.out` |
| 7 | 0.003 | 0.5 | 2 | 46 | 0.0311 | 0.0389 | 0.0571 | 0.0881 | `sasrec_new_train_grid-24119085_18.out` |
| 8 | 0.001 | 0.8 | 2 | 54 | 0.0294 | 0.0386 | 0.0567 | 0.0930 | `sasrec_new_train_grid-24119085_16.out` |
| 9 | 0.003 | 0.2 | 2 | 24 | 0.0255 | 0.0331 | 0.0506 | 0.0805 | `sasrec_new_train_grid-24119085_17.out` |
| 10 | 0.003 | 0.8 | 2 | 74 | 0.0236 | 0.0308 | 0.0471 | 0.0758 | `sasrec_new_train_grid-24119085_19.out` |
| 11 | 0.0003 | 0.8 | 2 | 2 | 0.0106 | 0.0140 | 0.0212 | 0.0346 | `sasrec_new_train_grid-24119085_13.out` |

Best setting:

- `lr=0.0003`, `dropout=0.2`, `blocks=2`
