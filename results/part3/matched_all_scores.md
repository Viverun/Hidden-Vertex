# Part 3C: all scores under complexity matching

Test set: 10000 events (5000 signal). Background weighted to the signal distribution: multiplicity (20 quantile bins) or (multiplicity, HT) (8x8 quantile bins). Weights clipped at the 99th percentile. Point estimates only, no confidence intervals. Model rows: AUC averaged over seeds. The oracle row uses the known signal mass. Matched columns use signal information and are diagnostics.

Background effective sample size (of 5000): mult-matched 2969, (mult, HT)-matched 831.

| Score | seeds | plain | mult-matched | (mult, HT)-matched |
|---|---|---|---|---|
| -|mjj-3500 GeV| (oracle) | 1 | 0.900 | 0.899 | 0.817 |
| -tau21 (leading jet) | 1 | 0.801 | 0.812 | 0.805 |
| kNN, jet4 | 3 | 0.761 | 0.754 | 0.729 |
| GMM, jet4 | 3 | 0.753 | 0.744 | 0.718 |
| WNAE pilot, jet5 | 3 | 0.721 | 0.723 | 0.680 |
| kNN, jet5 | 3 | 0.717 | 0.710 | 0.651 |
| GMM, jet5 | 3 | 0.694 | 0.689 | 0.630 |
| WNAE long, jet5 | 3 | 0.637 | 0.631 | 0.571 |
| Node-level GAE (Part 1 AE, mean) | 1 | 0.781 | 0.670 | 0.570 |
| mjj | 1 | 0.788 | 0.776 | 0.540 |
| mean pT per particle | 1 | 0.811 | 0.670 | 0.509 |
| HT (sum pT) | 1 | 0.792 | 0.833 | 0.510 |
| -multiplicity | 1 | 0.715 | 0.499 | 0.514 |
| MLP autoencoder, jet4 | 3 | 0.539 | 0.534 | 0.526 |
| MLP autoencoder, jet5 | 3 | 0.505 | 0.508 | 0.491 |
| Event-level GAE: Chamfer | 3 | 0.428 | 0.457 | 0.438 |
| Event-level GAE: kNN | 3 | 0.401 | 0.461 | 0.448 |
| Event-level GAE: GMM | 3 | 0.436 | 0.495 | 0.470 |

Command: `python scripts/part3c_all_scores_matched.py`
