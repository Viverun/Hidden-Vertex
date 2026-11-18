# Part 3 Results: Jet Features, WNAE, Complexity Matching and a Bump Hunt (2026-11-18)

**Result: no Part 3 method passed the pre-registered gate.** No label-free score is better than the
baseline mean pT per particle (AUC 0.811). The weakly supervised bump hunt passes only when it uses the
known signal mass. After Part 3, the project stopped. See the section "Project status" in the README.

- Configurations: `configs/part3_jet.yaml` (studies A, B, D) and `configs/part3_cwola.yaml` (bump hunt).
- Code: `src/pipeline/part3.py`, `src/pipeline/part3_cwola.py`, `src/data/jet_features.py`,
  `src/model/mlp_ae.py`, `src/model/wnae.py`.
- Raw results: `results/part3/` and `results/part2/part3c_multmatch.md`.
- All numbers in this document come from these result files.

## 1. Protocol rules

The rules are the same as in Part 2 (`docs/PART2_RESULTS.md`, Section 1).

1. The test set is the frozen Part 2 test set: 5,000 background and 5,000 signal events.
2. Test events do not go into a training set, a reference set or a fit of a transform.
3. The configuration of each study was fixed before we calculated a test score. Section 8 gives the
   exceptions.
4. A model score passes the gate if two conditions are true:
   - The lower limit of its seed-averaged AUC interval is higher than 0.811.
   - Its minimum AUC in bins of HT, multiplicity and leading pT is 0.6 or higher.
5. Each model uses 3 seeds (0, 1, 2). The table rows use the seed-averaged score.
6. Studies that use signal information (matching, reverse direction, oracle window) are diagnostics.
   They are not detectors, and the gate does not apply to them.

## 2. Study A: density scores on jet features

The study fits three scorers on background events only:

- the mean distance to the nearest training events (kNN),
- the negative log-likelihood under a Gaussian mixture model (GMM),
- the reconstruction error of an MLP autoencoder (AE).

There are two feature sets:

- jet5: mj1, mj2, tau21_j1, tau21_j2 and mjj. This set is the gate set.
- jet4: the same features without mjj. This set is a variant.

Command: `python scripts/part3_run.py density`, then `python scripts/part3_run.py report`.
Result file: `results/part3/jet_density.md`.

| Score | Features | AUC | 98.3% CI | Minimum AUC in a bin |
|---|---|---|---|---|
| kNN distance | jet4 | 0.761 | 0.751–0.774 | 0.678 |
| GMM likelihood | jet4 | 0.753 | 0.742–0.765 | 0.670 |
| MLP AE | jet4 | 0.539 | 0.530–0.548 | 0.486 |
| kNN distance | jet5 | 0.717 | 0.705–0.729 | 0.569 |
| GMM likelihood | jet5 | 0.694 | 0.683–0.708 | 0.546 |
| MLP AE | jet5 | 0.505 | 0.495–0.516 | 0.443 |

For comparison: mean pT per particle 0.811, −tau21 (leading jet) 0.801, oracle mass window 0.900.

- The density scores are better than the Part 2 scores (0.40 to 0.44).
- All scores are below 0.811. **The gate failed.**
- Without mjj, the AUC of the density scores increases. The jet4 density scores meet the requirement in bins.
- The MLP AE is near 0.5 on both feature sets.

## 3. Study B: Wasserstein-normalized autoencoder (WNAE)

The WNAE is a re-implementation on the jet5 features. It is not the CMS code.

- Pilot run: batch size 4096, 6 epochs.
- Long run: batch size 1024, up to 150 epochs, early stop after 30 epochs without improvement.

Commands:

- Pilot: `python scripts/part3_run.py wnae --seed N` (N = 0, 1, 2).
- Long run: `python scripts/part3_run.py wnae --run wnae_long --seed N` (N = 0, 1, 2).
- Report: `python scripts/part3_run.py report`.

Result file: `results/part3/wnae.md`.

| Run | AUC | 95% CI | Minimum AUC in a bin | AUC of seeds 0 / 1 / 2 | Best epoch |
|---|---|---|---|---|---|
| Pilot | 0.721 | 0.713–0.730 | 0.582 | 0.660 / 0.757 / 0.747 | 6 / 6 / 6 |
| Long | 0.637 | 0.627–0.647 | 0.496 | 0.682 / 0.611 / 0.616 | 141 / 39 / 92 |

- Neither run passes the gate. We report the two runs, and we select neither run.
- The long run has a lower test AUC than the pilot.
- The two runs use different batch sizes. Thus, their validation distances are not directly comparable.
- The seeds differ by up to 0.1 in AUC. Thus, the two settings cannot be ranked from this evidence.

## 4. Study C: complexity matching

### 4.1 Matched evaluation of the Part 2 scores

The background test events get weights. With the weights, the background has the same distribution as the
signal:

- (b) in 20 quantile bins of multiplicity,
- (c) in 8 × 8 quantile bins of multiplicity and HT.

The weights are clipped at the 99th percentile. The 95% bootstrap intervals calculate the weights again in
each resample.

Command: `python scripts/part3c_matched_eval.py`. Result file: `results/part3/c1_matched.md`.

| Score | Plain AUC | Multiplicity-matched | (Multiplicity, HT)-matched |
|---|---|---|---|
| Event-level GAE: Chamfer | 0.428 | 0.457 [0.445, 0.470] | 0.438 [0.415, 0.459] |
| Event-level GAE: kNN | 0.401 | 0.461 [0.450, 0.472] | 0.448 [0.430, 0.463] |
| Event-level GAE: GMM | 0.436 | 0.495 [0.484, 0.506] | 0.470 [0.451, 0.485] |
| mean pT per particle | 0.811 | 0.670 [0.663, 0.676] | 0.509 [0.499, 0.531] |
| HT | 0.792 | 0.833 [0.823, 0.842] | 0.510 [0.502, 0.522] |
| −multiplicity | 0.715 | 0.499 [0.499, 0.500] | 0.514 [0.505, 0.532] |

Background effective sample size (of 5,000): 2,969 for multiplicity matching, 831 for (multiplicity, HT)
matching.

The matching reduces the inversion. But no Part 2 score becomes better than 0.5.

### 4.2 Retraining on a multiplicity-matched background

The Part 2 model is trained again on a background sample. The sample is drawn with replacement from the
training events. The weights make its multiplicity follow 16,200 signal events outside the test set
(30 quantile bins, weights clipped at the 99.5th percentile). The validation set has no weights.

Commands:

1. Train on Kaggle with `scripts/kaggle/part3_reweight.py` (run tag `multmatch`, seeds 0, 1, 2).
2. Report: `python scripts/part2_run.py report --config configs/part2_main.yaml --tags main,multmatch --name part3c_multmatch`.
3. Matched evaluation: `python scripts/part3c_matched_eval.py --tag multmatch --name c1_matched_multmatch`.

Result files: `results/part2/part3c_multmatch.md` and `results/part3/c1_matched_multmatch.md`.

| Score | Plain AUC (normal training) | Plain AUC (matched training) | Multiplicity-matched | (Multiplicity, HT)-matched |
|---|---|---|---|---|
| Chamfer | 0.428 | 0.415 | 0.472 | 0.448 |
| kNN | 0.401 | 0.399 | 0.482 | 0.467 |
| GMM | 0.436 | 0.424 | 0.503 | 0.480 |

Retraining on a multiplicity-matched background does not recover performance.

### 4.3 All scores under complexity matching

This table applies the same weights to all saved scores. It gives point estimates only, without intervals.
The oracle row uses the known signal mass.

Command: `python scripts/part3c_all_scores_matched.py`. Result file: `results/part3/matched_all_scores.md`.

| Score | Plain | Multiplicity-matched | (Multiplicity, HT)-matched |
|---|---|---|---|
| −\|mjj − 3500 GeV\| (oracle) | 0.900 | 0.899 | 0.817 |
| −tau21 (leading jet) | 0.801 | 0.812 | 0.805 |
| kNN, jet4 | 0.761 | 0.754 | 0.729 |
| GMM, jet4 | 0.753 | 0.744 | 0.718 |
| WNAE pilot, jet5 | 0.721 | 0.723 | 0.680 |
| kNN, jet5 | 0.717 | 0.710 | 0.651 |
| GMM, jet5 | 0.694 | 0.689 | 0.630 |
| WNAE long, jet5 | 0.637 | 0.631 | 0.571 |
| Node-level GAE (Part 1 AE, mean) | 0.781 | 0.670 | 0.570 |
| MLP AE, jet4 | 0.539 | 0.534 | 0.526 |
| MLP AE, jet5 | 0.505 | 0.508 | 0.491 |
| mjj | 0.788 | 0.776 | 0.540 |
| mean pT per particle | 0.811 | 0.670 | 0.509 |
| HT | 0.792 | 0.833 | 0.510 |
| −multiplicity | 0.715 | 0.499 | 0.514 |
| Event-level GAE: Chamfer | 0.428 | 0.457 | 0.438 |
| Event-level GAE: kNN | 0.401 | 0.461 | 0.448 |
| Event-level GAE: GMM | 0.436 | 0.495 | 0.470 |

- The jet-feature scores keep some information after (multiplicity, HT) matching (0.57 to 0.73).
- No learned score is near −tau21 (0.805) or the oracle mass window (0.817).
- The Part 2 scores stay at 0.44 to 0.47.

## 5. Study D: reverse direction (diagnostic)

The kNN, GMM and MLP AE scorers are fitted on 76,000 signal events. The AE validation set has 9,000 more
signal events. The frozen test set is scored with the background as the positive class. This study uses
signal labels. It is not an anomaly-detection result.

Command: `python scripts/part3_run.py reverse`. Result file: `results/part3/reverse.md`.

| Score | Forward AUC (background-trained) | Reverse AUC (signal-trained) | Reverse 95% CI |
|---|---|---|---|
| GMM, jet5 | 0.694 | 0.903 | 0.898–0.909 |
| kNN, jet5 | 0.717 | 0.905 | 0.900–0.911 |
| MLP AE, jet5 | 0.505 | 0.845 | 0.838–0.851 |
| GMM, jet4 | 0.753 | 0.810 | 0.802–0.818 |
| kNN, jet4 | 0.761 | 0.791 | 0.782–0.799 |
| MLP AE, jet4 | 0.539 | 0.635 | 0.627–0.644 |

- All scores are better in the reverse direction.
- The difference is large for jet5 and small for jet4. Thus, the asymmetry comes with mjj.
- The jet features contain the information. The forward weakness is not a lack of input information.

## 6. Study E: CWoLa bump hunt (weak supervision)

Signal events are injected into an unlabeled pool at a fraction S/B. For each mjj window, a classifier
learns to separate the signal region from the sidebands. Two window grids (A and B) give two quantiles, and
the score is their mean. The gate is judged at S/B = 0.001.

Commands: `python scripts/part3_cwola.py all`, then `python scripts/part3_cwola.py report`.
Result file: `results/part3/cwola.md`.

| S/B | Seed-mean AUC | 95% CI | Minimum AUC in a bin | Gate | Oracle window AUC | Oracle 95% CI | Oracle gate (information only) |
|---|---|---|---|---|---|---|---|
| 0 | 0.532 | 0.522–0.542 | 0.461 | not passed | 0.478 | 0.469–0.486 | not passed |
| 0.0005 | 0.666 | 0.657–0.675 | 0.605 | not passed | 0.681 | 0.673–0.689 | not passed |
| 0.001 | 0.716 | 0.707–0.725 | 0.628 | not passed | 0.835 | 0.828–0.842 | passed |
| 0.002 | 0.741 | 0.732–0.750 | 0.655 | not passed | 0.887 | 0.881–0.894 | passed |
| 0.005 | 0.735 | 0.725–0.744 | 0.645 | not passed | 0.912 | 0.906–0.917 | passed |
| 0.01 | 0.739 | 0.729–0.749 | 0.652 | not passed | 0.928 | 0.923–0.933 | passed |

- The window-agnostic score does not pass the gate at any fraction. It stops near 0.74.
- The oracle window (signal region 3.3 to 3.7 TeV) passes from S/B = 0.001. It uses the known signal mass.
  Thus, it is not a model-independent result.

## 7. Pre-registered predictions and verdicts

| Study | Rule or prediction (fixed before the result) | Value | Verdict |
|---|---|---|---|
| A, B | Gate: a jet-feature or WNAE score passes the gate. | best 0.761 (kNN, jet4) | failed |
| C1 | If complexity alone explains the inversion, the matched kNN and GMM AUC is in 0.5 ± 0.03. | kNN 0.461 / 0.448; GMM 0.495 / 0.470 | kNN below 0.47 in both; GMM in range for multiplicity only |
| C2 | If multiplicity causes the inversion, matched training gives kNN and GMM an AUC of 0.47 or more. | kNN 0.399, GMM 0.424 | failed |
| E1 | At S/B = 0, the seed-mean AUC is in [0.47, 0.53]. | 0.532 | failed |
| E2 | The seed-mean AUC does not decrease as S/B increases (tolerance 0.01). | largest decrease 0.006 | passed |
| E3 | At S/B = 0.001, the gate is passed. | CI lower limit 0.707 | failed |
| E4 | At S/B ≥ 0.005, the seed-mean AUC is more than 0.90. | 0.735 and 0.739 | failed |

Thus, complexity (multiplicity and HT) is at most a partial explanation of the Part 2 inversion.

## 8. Disclosures

- **The WNAE long-run settings were fixed after we saw the pilot test AUCs.** The pilot settings were fixed
  before. We report both runs, and we select neither run.
- **The WNAE uses jet features, not the graph model.** Its result does not apply to a WNAE on the particle
  graph.
- **The matched AUCs in Section 4.3 have no confidence intervals.** The (multiplicity, HT) matching leaves a
  background effective sample size of approximately 830 (831).
- **The multiplicity-matched training set has approximately 48% duplicate events.** It has 92,938 unique events
  in 180,000 draws. It matches multiplicity only, not HT.
- **Matching and the reverse direction use signal information.** They are diagnostics.
- **The bump hunt uses injected signal in training.** It is a weakly supervised method, not an unsupervised
  method. It also assumes a resonance in mjj.
- **The oracle window uses the known signal mass.** Its gate results are for information only.
- **The cause of the AUC of 0.532 at S/B = 0 is not known.** The two grids differ (0.555 for grid A, 0.492 for
  grid B).
