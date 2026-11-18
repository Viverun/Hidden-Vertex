# Part 3d: CWoLa bump hunt (weak supervision)

This study uses the protocol in `configs/part3_cwola.yaml`. The protocol was fixed before we computed a test score. Training uses no labels. Each classifier learns to separate events in an mjj signal region from events in the sidebands. Labels choose the injected signal events and evaluate the frozen test set (5,000 background + 5,000 signal). Test events never enter the pool, the training data or the reference sample.

Score: the mean of two quantiles (grid A and grid B). Higher = more anomalous. Bar: mean pT per particle (AUC 0.811). CI: 95% percentile bootstrap of the seed-mean AUC (1000 resamples). Within-bin: 10 quantile bins of HT, multiplicity and leading pT. Gate: CI lower bound > 0.811 and min within-bin AUC >= 0.6. The gate is judged at fraction 0.001.

## Main result (model-independent windows)

| Fraction S/B | injected S (per seed) | per-seed AUC | seed-mean AUC | 95% CI | min within-bin AUC | grid A / grid B AUC | max SIC | gate |
|---|---|---|---|---|---|---|---|---|
| 0 | 0 / 0 / 0 | 0.527 / 0.534 / 0.536 | 0.532 | 0.522–0.542 | 0.461 | 0.555 / 0.492 | 1.00 | NOT PASSED |
| 0.0005 | 478 / 478 / 478 | 0.660 / 0.650 / 0.687 | 0.666 | 0.657–0.675 | 0.605 | 0.639 / 0.633 | 1.06 | NOT PASSED |
| 0.001 | 955 / 955 / 955 | 0.721 / 0.708 / 0.719 | 0.716 | 0.707–0.725 | 0.628 | 0.678 / 0.688 | 1.19 | NOT PASSED |
| 0.002 | 1910 / 1910 / 1910 | 0.733 / 0.749 / 0.742 | 0.741 | 0.732–0.750 | 0.655 | 0.711 / 0.721 | 3.22 | NOT PASSED |
| 0.005 | 4775 / 4775 / 4775 | 0.732 / 0.744 / 0.729 | 0.735 | 0.725–0.744 | 0.645 | 0.714 / 0.718 | 3.70 | NOT PASSED |
| 0.01 | 9550 / 9550 / 9550 | 0.740 / 0.739 / 0.737 | 0.739 | 0.729–0.749 | 0.652 | 0.714 / 0.720 | 2.77 | NOT PASSED |

## Oracle window (uses the known signal mass: SR 3.3–3.7 TeV; not model independent)

One classifier per run. It scores all test events. The score is the raw ensemble output.

| Fraction S/B | per-seed AUC | seed-mean AUC | 95% CI | min within-bin AUC | max SIC | gate (info only) |
|---|---|---|---|---|---|---|
| 0 | 0.381 / 0.605 / 0.448 | 0.478 | 0.469–0.486 | 0.437 | 1.00 | not passed |
| 0.0005 | 0.617 / 0.786 / 0.639 | 0.681 | 0.673–0.689 | 0.602 | 1.22 | not passed |
| 0.001 | 0.808 / 0.842 / 0.855 | 0.835 | 0.828–0.842 | 0.733 | 2.51 | passed |
| 0.002 | 0.881 / 0.897 / 0.883 | 0.887 | 0.881–0.894 | 0.760 | 10.01 | passed |
| 0.005 | 0.906 / 0.920 / 0.908 | 0.912 | 0.906–0.917 | 0.796 | 13.23 | passed |
| 0.01 | 0.922 / 0.934 / 0.926 | 0.928 | 0.923–0.933 | 0.817 | 13.32 | passed |

## Within-bin AUC ranges (main score)

| Fraction S/B | HT bins | multiplicity bins | leading-pT bins |
|---|---|---|---|
| 0 | 0.461–0.563 | 0.535–0.574 | 0.504–0.570 |
| 0.0005 | 0.605–0.679 | 0.613–0.734 | 0.637–0.697 |
| 0.001 | 0.649–0.752 | 0.628–0.785 | 0.687–0.742 |
| 0.002 | 0.655–0.788 | 0.664–0.795 | 0.721–0.770 |
| 0.005 | 0.645–0.801 | 0.656–0.787 | 0.706–0.764 |
| 0.01 | 0.652–0.804 | 0.658–0.793 | 0.709–0.766 |

## Signal and background in the pool near 3.5 TeV (for information; uses pool labels)

Counts are for the whole pool (training part + reference sample), seed-mean.

| Fraction S/B | region (TeV) | S | B | S/B | S/sqrt(B) |
|---|---|---|---|---|---|
| 0 | grid_a_3.2-3.6 | 0 | 133813 | 0.0000 | 0.0 |
| 0 | grid_b_3.4-3.8 | 0 | 99834 | 0.0000 | 0.0 |
| 0 | oracle_3.3-3.7 | 0 | 115867 | 0.0000 | 0.0 |
| 0.0005 | grid_a_3.2-3.6 | 323 | 133813 | 0.0024 | 0.9 |
| 0.0005 | grid_b_3.4-3.8 | 331 | 99834 | 0.0033 | 1.0 |
| 0.0005 | oracle_3.3-3.7 | 362 | 115867 | 0.0031 | 1.1 |
| 0.001 | grid_a_3.2-3.6 | 647 | 133813 | 0.0048 | 1.8 |
| 0.001 | grid_b_3.4-3.8 | 649 | 99834 | 0.0065 | 2.1 |
| 0.001 | oracle_3.3-3.7 | 722 | 115867 | 0.0062 | 2.1 |
| 0.002 | grid_a_3.2-3.6 | 1297 | 133813 | 0.0097 | 3.5 |
| 0.002 | grid_b_3.4-3.8 | 1290 | 99834 | 0.0129 | 4.1 |
| 0.002 | oracle_3.3-3.7 | 1442 | 115867 | 0.0124 | 4.2 |
| 0.005 | grid_a_3.2-3.6 | 3230 | 133813 | 0.0241 | 8.8 |
| 0.005 | grid_b_3.4-3.8 | 3220 | 99834 | 0.0323 | 10.2 |
| 0.005 | oracle_3.3-3.7 | 3597 | 115867 | 0.0310 | 10.6 |
| 0.01 | grid_a_3.2-3.6 | 6507 | 133813 | 0.0486 | 17.8 |
| 0.01 | grid_b_3.4-3.8 | 6418 | 99834 | 0.0643 | 20.3 |
| 0.01 | oracle_3.3-3.7 | 7207 | 115867 | 0.0622 | 21.2 |

## Windows after the merge rule (training counts of fraction 0, seed 0)

Grid A:

| SR (TeV) | sidebands (TeV) | train SR events | train SB events |
|---|---|---|---|
| (-inf, 2.4) | [2.4, 2.8) | 82460 | 218873 |
| [2.4, 2.8) | [2.2, 2.4) + [2.8, 3.0) | 218873 | 139930 |
| [2.8, 3.2) | [2.6, 2.8) + [3.2, 3.4) | 161157 | 164821 |
| [3.2, 3.6) | [3.0, 3.2) + [3.6, 3.8) | 93796 | 100150 |
| [3.6, 4.0) | [3.4, 3.6) + [4.0, 4.2) | 51972 | 56252 |
| [4.0, 4.4) | [3.8, 4.0) + [4.4, 4.6) | 27968 | 30936 |
| [4.4, 4.8) | [4.2, 4.4) + [4.8, 5.0) | 15149 | 16753 |
| [4.8, +inf) | [4.4, 4.8) | 17125 | 15149 |

Grid B:

| SR (TeV) | sidebands (TeV) | train SR events | train SB events |
|---|---|---|---|
| (-inf, 2.6) | [2.6, 3.0) | 190099 | 201985 |
| [2.6, 3.0) | [2.4, 2.6) + [3.0, 3.2) | 201985 | 178045 |
| [3.0, 3.4) | [2.8, 3.0) + [3.4, 3.6) | 123993 | 130960 |
| [3.4, 3.8) | [3.2, 3.4) + [3.8, 4.0) | 69953 | 75815 |
| [3.8, 4.2) | [3.6, 3.8) + [4.2, 4.4) | 38271 | 41669 |
| [4.2, 4.6) | [4.0, 4.2) + [4.6, 4.8) | 20633 | 22484 |
| [4.6, 5.0) | [4.4, 4.6) + [5.0, 5.2) | 11269 | 12128 |
| [5.0, +inf) | [4.6, 5.0) | 12297 | 11269 |

Merges in all runs: 0.

## Pre-registered predictions

1. fraction 0.0: seed-mean AUC inside [0.47, 0.53]: **FAIL** (value: 0.532)
2. seed-mean AUC does not decrease as the fraction increases (tolerance 0.01): **PASS** (value: worst_step: 0.002–0.005, change: -0.006)
3. fraction 0.001 (primary): gate passed (CI lower bound > 0.811 and min within-bin >= 0.6): **FAIL** (value: ci_lo: 0.707, within_bin_min: 0.628)
4. fraction >= 0.005: seed-mean AUC > 0.90: **FAIL** (value: 0.005: 0.735, 0.01: 0.739)
