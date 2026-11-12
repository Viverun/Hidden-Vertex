# Part 3C / C1: complexity-matched AUC (run tag multmatch)

Test set: 10000 events (5000 signal). Background weighted to the signal distribution: (b) 20 quantile bins of multiplicity; (c) 8x8 quantile bins of (multiplicity, HT). Weights clipped at the 99th percentile. 95% CI: percentile bootstrap, 1000 resamples of events, weights recomputed in each resample. Model rows: AUC averaged over seeds.

| Score | seeds | (a) plain AUC [95% CI] | (b) mult-matched [95% CI] | (c) (mult, HT)-matched [95% CI] |
|---|---|---|---|---|
| EventAE chamfer [multmatch] | 3 | 0.415 [0.405, 0.427] | 0.472 [0.458, 0.485] | 0.448 [0.424, 0.470] |
| EventAE knn [multmatch] | 3 | 0.399 [0.389, 0.408] | 0.482 [0.471, 0.495] | 0.467 [0.447, 0.483] |
| EventAE gmm [multmatch] | 3 | 0.424 [0.415, 0.433] | 0.503 [0.491, 0.515] | 0.480 [0.461, 0.498] |
| mean pT per particle | 1 | 0.811 [0.802, 0.820] | 0.670 [0.663, 0.676] | 0.509 [0.499, 0.531] |
| HT (sum pT) | 1 | 0.792 [0.783, 0.801] | 0.833 [0.823, 0.842] | 0.510 [0.502, 0.522] |
| -multiplicity | 1 | 0.715 [0.703, 0.725] | 0.499 [0.499, 0.500] | 0.514 [0.505, 0.532] |

## Weight diagnostics

| Matching | cells | clip value | bg ESS (of 5000) | bg with weight 0 | signal in cells without bg | weighted bg mean mult | weighted bg mean HT |
|---|---|---|---|---|---|---|---|
| mult | 20 | 2.836 | 2969 | 0.0% | 0.0% | 139.7 | 2825 |
| mult_ht | 64 | 13.643 | 831 | 4.2% | 0.0% | 141.9 | 3338 |

Unweighted means: background multiplicity 181.8, signal 139.9; background HT 2900, signal 3309.

## Fixed prediction (kNN and GMM)

If complexity alone explains the kNN/GMM inversion, their matched AUC is within 0.5 +/- 0.03; if it stays below 0.47, complexity (as measured by multiplicity and HT) does not fully explain it.

| Score | (b) mult-matched | (c) (mult, HT)-matched |
|---|---|---|
| EventAE knn [multmatch] | 0.482: within 0.5 +/- 0.03 | 0.467: below 0.47 |
| EventAE gmm [multmatch] | 0.503: within 0.5 +/- 0.03 | 0.480: within 0.5 +/- 0.03 |
