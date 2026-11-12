# Part 3B: WNAE on jet5 (with the Part 3A rows)

| Score | Group | AUC | CI | CI level | min within-bin AUC | Notes |
|---|---|---|---|---|---|---|
| -|mjj-3500| (oracle) | oracle | 0.900 | 0.894–0.906 | 95.0% | 0.777 | needs the signal mass; not model independent |
| mean pT per particle | particle | 0.811 | 0.803–0.821 | 95.0% | 0.599 | = HT / multiplicity |
| -tau21 (leading jet) | jet | 0.801 | 0.792–0.810 | 95.0% | 0.664 | two-prong substructure |
| HT (sum pT) | particle | 0.792 | 0.783–0.802 | 95.0% | 0.337 | more energy = anomalous |
| mjj | jet | 0.788 | 0.778–0.799 | 95.0% | 0.322 | dijet invariant mass |
| Part 1 AE (mean) | external | 0.781 | 0.772–0.790 | 95.0% | 0.609 | from scores.npz |
| kNN dist [jet4] | variant | 0.761 | 0.751–0.774 | 98.3% | 0.678 | 3 seed(s) |
| GMM NLL [jet4] | variant | 0.753 | 0.742–0.765 | 98.3% | 0.670 | 3 seed(s) |
| WNAE recon [jet5] | variant | 0.721 | 0.713–0.730 | 95.0% | 0.582 | 3 seed(s) |
| kNN dist [jet5] | model | 0.717 | 0.705–0.729 | 98.3% | 0.569 | 3 seed(s) |
| -multiplicity | particle | 0.715 | 0.704–0.726 | 95.0% | 0.399 | fewer particles = anomalous (sign seen on labelled data in Part 1) |
| GMM NLL [jet5] | model | 0.694 | 0.683–0.708 | 98.3% | 0.546 | 3 seed(s) |
| mj2 | jet | 0.637 | 0.626–0.649 | 95.0% | 0.566 | second jet mass |
| mj1 | jet | 0.634 | 0.623–0.645 | 95.0% | 0.541 | leading jet mass |
| leading pT | particle | 0.554 | 0.543–0.565 | 95.0% | 0.388 |  |
| AE recon [jet4] | variant | 0.539 | 0.530–0.548 | 98.3% | 0.486 | 3 seed(s) |
| m_all | particle | 0.516 | 0.504–0.527 | 95.0% | 0.182 | invariant mass of all particles |
| AE recon [jet5] | model | 0.505 | 0.495–0.516 | 98.3% | 0.443 | 3 seed(s) |
| Part 1 AE (max) | external | 0.478 | 0.467–0.489 | 95.0% | 0.376 | from scores.npz |

## Gate

Bar: mean pT per particle (AUC 0.811). Result: **NOT PASSED**

| Model score | AUC | CI lower | beats bar | min within-bin | within-bin ok | passes |
|---|---|---|---|---|---|---|
| GMM NLL [jet5] | 0.694 | 0.683 | False | 0.546 | False | False |
| kNN dist [jet5] | 0.717 | 0.705 | False | 0.569 | False | False |
| AE recon [jet5] | 0.505 | 0.495 | False | 0.443 | False | False |

## Per-seed AUC and within-bin AUC ranges (seed-averaged scores)

| Score | per-seed AUC | HT bins | multiplicity bins | leading-pT bins |
|---|---|---|---|---|
| GMM NLL [jet5] | 0.694 / 0.694 / 0.695 | 0.546–0.743 | 0.632–0.734 | 0.619–0.737 |
| kNN dist [jet5] | 0.717 / 0.717 / 0.717 | 0.569–0.767 | 0.658–0.744 | 0.654–0.762 |
| AE recon [jet5] | 0.529 / 0.453 / 0.533 | 0.465–0.557 | 0.474–0.563 | 0.443–0.537 |
| GMM NLL [jet4] | 0.753 / 0.751 / 0.755 | 0.670–0.782 | 0.695–0.781 | 0.711–0.785 |
| kNN dist [jet4] | 0.761 / 0.761 / 0.761 | 0.678–0.786 | 0.712–0.784 | 0.717–0.803 |
| AE recon [jet4] | 0.528 / 0.543 / 0.545 | 0.486–0.576 | 0.504–0.627 | 0.515–0.556 |
| WNAE recon [jet5] | 0.660 / 0.757 / 0.747 | 0.582–0.745 | 0.685–0.749 | 0.667–0.753 |

## Spearman correlation of the seed-0 score (all | background | signal)

| Score | multiplicity | HT | mean pT (= HT / multiplicity) |
|---|---|---|---|
| GMM NLL [jet5] | -0.26 \| -0.04 \| -0.30 | +0.23 \| +0.08 \| +0.02 | +0.34 \| +0.09 \| +0.30 |
| kNN dist [jet5] | -0.27 \| -0.05 \| -0.28 | +0.25 \| +0.07 \| +0.02 | +0.36 \| +0.10 \| +0.29 |
| AE recon [jet5] | -0.07 \| +0.02 \| -0.14 | +0.02 \| +0.03 \| -0.03 | +0.08 \| -0.00 \| +0.12 |
| GMM NLL [jet4] | -0.31 \| -0.07 \| -0.30 | +0.25 \| +0.05 \| -0.01 | +0.39 \| +0.11 \| +0.30 |
| kNN dist [jet4] | -0.31 \| -0.07 \| -0.27 | +0.26 \| +0.05 \| -0.01 | +0.39 \| +0.11 \| +0.27 |
| AE recon [jet4] | -0.01 \| +0.01 \| +0.00 | +0.04 \| +0.03 \| +0.00 | +0.04 \| +0.01 \| -0.00 |
| WNAE recon [jet5] | -0.19 \| -0.02 \| -0.22 | +0.22 \| +0.07 \| +0.08 | +0.27 \| +0.07 \| +0.25 |

## WNAE training

| Seed | epochs run / max | best epoch | best val W | final val W | diverged | minutes |
|---|---|---|---|---|---|---|
| 0 | 6 / 6 | 6 | 1.0087 | 1.0087 | no | 23.5 |
| 1 | 6 / 6 | 6 | 1.0713 | 1.0713 | no | 32.0 |
| 2 | 6 / 6 | 6 | 1.0416 | 1.0416 | no | 24.8 |

### Negative samples vs positives (validation set, selected epoch): mean / std per feature

| Seed | feature | positive mean | negative mean | positive std | negative std |
|---|---|---|---|---|---|
| 0 | mj1 | -0.012 | -0.069 | 0.986 | 0.568 |
| 0 | mj2 | -0.011 | +0.043 | 1.015 | 0.807 |
| 0 | tau21_j1 | -0.002 | +0.039 | 1.003 | 0.622 |
| 0 | tau21_j2 | +0.007 | +0.014 | 1.012 | 0.771 |
| 0 | mjj | +0.009 | +0.045 | 0.987 | 0.458 |
| 1 | mj1 | -0.012 | +0.009 | 0.986 | 0.578 |
| 1 | mj2 | -0.011 | -0.076 | 1.015 | 0.791 |
| 1 | tau21_j1 | -0.002 | +0.050 | 1.003 | 0.513 |
| 1 | tau21_j2 | +0.007 | +0.074 | 1.012 | 0.749 |
| 1 | mjj | +0.009 | +0.075 | 0.987 | 0.494 |
| 2 | mj1 | -0.012 | -0.016 | 0.986 | 0.821 |
| 2 | mj2 | -0.011 | -0.017 | 1.015 | 0.544 |
| 2 | tau21_j1 | -0.002 | +0.015 | 1.003 | 0.777 |
| 2 | tau21_j2 | +0.007 | +0.037 | 1.012 | 0.580 |
| 2 | mjj | +0.009 | -0.008 | 0.987 | 0.488 |
