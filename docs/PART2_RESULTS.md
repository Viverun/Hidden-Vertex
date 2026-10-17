# Part 2 Results: Event-Level Graph Autoencoder and Simple Baselines (2026-10-13)

**Result: the model did not pass the pre-registered gate.** The unsupervised event-level autoencoder is not
better than the simple particle-level baselines. All three of its anomaly scores are inverted: their AUC is
0.40 to 0.44, which is below 0.5. Thus, the model finds signal events more typical than background events.

- Design: `docs/superpowers/specs/2026-10-13-part2-event-level-ae-design.md`
- Plan: `docs/superpowers/plans/2026-10-13-part2-event-level-ae.md`
- Code: branch `part2-event-level-ae`. The entry point is `scripts/part2_run.py`. The configuration is
  `configs/part2_main.yaml`.
- Raw results: `results/part2/` (git ignores this directory): `stage0.md`, `stage1.md`, `final.md|json`,
  `runs/<tag>/`.
- Trained checkpoints: [huggingface.co/Viverun/lhco-rnd-event-gae](https://huggingface.co/Viverun/lhco-rnd-event-gae).

## 1. Setup

| Item | Value |
|---|---|
| Data | `events_anomalydetection_v2.h5`. The file is sorted: rows 0 to 999,999 are background, and rows 1,000,000 and higher are signal. Kaggle dataset: `atulchikku/lhc-olympics-2020-ad-r-and-d`. |
| Test set (frozen, the same as Part 1) | 5,000 background events (rows 40,000 to 44,999) and 5,000 signal events (rows 1,000,000 to 1,004,999) |
| Training set | 180,000 background events (rows 45,000 and higher). The validation set is 20,000 background events (the last 10% of the 200,000). |
| Model | `EventAE`: 3 neighbour-averaging layers (width 64, ΔR ≤ 0.4, residual), mean, max and pT-weighted pooling, a latent space of 16 dimensions, a decoder for the 48 particles with the highest pT, and a symmetric Chamfer loss. The model has 95,888 parameters. |
| Scores | (1) Chamfer reconstruction error. (2) Mean distance to the 20 nearest training latents (kNN). (3) Negative log-likelihood of the latent under a Gaussian mixture model with 8 components (GMM). The kNN and GMM scorers use the training latents only. |
| Training | 30 epochs, AdamW, learning rate 1e-3, batch size 256, patience 5. The early stop did not occur in the main runs. One epoch takes approximately 34 s on one T4 GPU. One seed takes approximately 18 min. |
| Main runs | 3 seeds (0, 1, 2). The configuration was fixed before any test-set result was available. |
| Kaggle kernels (private, user `viveriun`) | `hidden-vertex-part2-stage0`, `-stage1`, `-stage2`, `-stage3`, `-smoke` (timing), `-mult` (multiplicity quantiles), `-release` (checkpoints for Hugging Face) |

**The gate.** The function `evaluate_gate` contains the gate. Its definition was fixed before the results
(design, Section 1). A model score passes if two conditions are true:

1. The lower limit of its seed-averaged AUC (98.3% interval with the Bonferroni correction) is higher than
   the best particle-level baseline.
2. Its minimum AUC in bins of HT, multiplicity and leading pT is 0.6 or higher.

## 2. Stage 0: the harness reproduces Part 1

The new harness reproduces the four verified Part 1 values exactly. Each value is in its confidence interval.

| Score | Expected | Measured | CI |
|---|---|---|---|
| HT (sum of pT) | 0.792 | 0.792 | 0.783–0.802 |
| mjj | 0.788 | 0.788 | 0.778–0.799 |
| Part 1 AE, maximum error | 0.478 | 0.478 | 0.467–0.489 |
| Part 1 AE, mean error | 0.781 | 0.781 | 0.772–0.790 |

This table shows the baselines on the frozen test set. A higher score means a more anomalous event. The CI is
the 95% interval.

| Score | Group | AUC | CI |
|---|---|---|---|
| −\|mjj − 3500\| (this score needs the signal mass) | oracle | 0.900 | 0.894–0.906 |
| **mean pT per particle (= HT / multiplicity)** | particle | **0.811** | 0.803–0.821 |
| −τ21 (leading jet) | jet | 0.801 | 0.792–0.810 |
| HT (Σ pT) | particle | 0.792 | 0.783–0.802 |
| mjj | jet | 0.788 | 0.778–0.799 |
| Part 1 AE (mean error) | external | 0.781 | 0.772–0.790 |
| −multiplicity | particle | 0.715 | 0.704–0.726 |
| mj2 / mj1 | jet | 0.637 / 0.634 | |
| leading pT | particle | 0.554 | 0.543–0.565 |
| m_all | particle | 0.516 | 0.504–0.527 |
| Part 1 AE (maximum error) | external | 0.478 | 0.467–0.489 |

**Answer to the open question from Part 1.** The Part 1 score with the mean error (AUC 0.781) is not better
than a simple variable. The mean pT per particle gets an AUC of **0.811**. This value is higher than HT (0.792)
and higher than the Part 1 score.

The gate uses the best particle-level baseline as its limit. Thus, the limit increased from approximately 0.79
(the value in the design) to **0.811**.

## 3. Stage 1: main result (the gate failed)

This table shows the seed-averaged AUC of 3 seeds. The CI is a 98.3% bootstrap interval over the 10,000 test
events.

| Model score | AUC | CI | Minimum AUC in a bin | Higher than 0.811 | Passes |
|---|---|---|---|---|---|
| Chamfer reconstruction error | 0.428 | 0.415–0.442 | 0.311 | no | no |
| kNN distance in the latent space | 0.401 | 0.389–0.412 | 0.281 | no | no |
| Gaussian-mixture likelihood | 0.436 | 0.426–0.448 | 0.308 | no | no |

AUC for each seed (Chamfer / kNN / GMM):

| Seed | Chamfer | kNN | GMM |
|---|---|---|---|
| 0 | 0.419 | 0.405 | 0.448 |
| 1 | 0.433 | 0.397 | 0.427 |
| 2 | 0.432 | 0.401 | 0.434 |

The difference between seeds is approximately 0.01 to 0.02. This difference is much smaller than the distance
to 0.5. Thus, the inversion is systematic. It is not noise.

The training was correct:

- The training loss and the validation loss agree. The best validation loss is 0.12 to 0.13.
- The latent space did not collapse. No NaN values occurred.
- At the end of the training, the validation loss changed randomly and was almost constant. The best epochs
  were 26, 29 and 30 for seeds 0, 1 and 2. For seed 0, the validation loss in the last five epochs was 0.131
  to 0.142.

## 4. Diagnosis: what the scores follow

This table shows the Spearman correlation of the seed-0 scores with simple event variables. Each cell has
three values: all test events | background only | signal only. The table does not include jet-level variables
such as jet mass, τ21 and mjj. The mean pT per particle is a particle-level variable.

| Score | HT | multiplicity | leading pT | mean pT per particle |
|---|---|---|---|---|
| Chamfer | −0.01 \| +0.15 \| 0.00 | +0.08 \| +0.08 \| −0.01 | +0.17 \| +0.18 \| +0.19 | −0.07 \| −0.02 \| +0.01 |
| kNN | −0.02 \| +0.19 \| 0.00 | **+0.28** \| +0.30 \| +0.19 | −0.02 \| −0.05 \| +0.05 | **−0.27** \| −0.24 \| −0.20 |
| GMM | +0.02 \| +0.18 \| 0.00 | **+0.23** \| +0.27 \| +0.15 | +0.01 \| −0.04 \| +0.09 | **−0.20** \| −0.21 \| −0.16 |

### Findings

1. **The kNN and GMM scores probably follow the event complexity.**
   - These scores increase with multiplicity and decrease with mean pT. This is also true in the background
     events alone.
   - In bins of multiplicity, their AUC is near 0.5 (kNN 0.43 to 0.50, GMM 0.47 to 0.51).
   - Signal events have fewer particles, and their particles have higher pT. The mean multiplicity is 140 for
     signal and 182 for background. Thus, signal events possibly occupy dense regions of the latent space,
     where the model reconstructs events easily.
   - The sign of the score was fixed before the test: "far from the training data means anomalous". With this
     sign, signal events in dense regions give an AUC below 0.5.
   - The correlations suggest this explanation. No test of the cause was done.

   **The Chamfer score is different.** It has almost no correlation with multiplicity (ρ +0.08) or mean pT
   (ρ −0.07). It stays inverted in bins of multiplicity (0.41 to 0.48) and in bins of HT (as low as 0.31).
   Thus, multiplicity alone does not explain the Chamfer inversion. Its cause is not known.

2. **In bins of HT, multiplicity and leading pT, the scores do not find the signal.** The seed-averaged AUC
   ranges in the bins are:

   | Score | HT bins | Multiplicity bins | Leading-pT bins |
   |---|---|---|---|
   | Chamfer | 0.31–0.56 | 0.41–0.48 | 0.40–0.44 |
   | kNN | 0.28–0.54 | 0.43–0.50 | 0.38–0.45 |
   | GMM | 0.31–0.57 | 0.47–0.51 | 0.40–0.50 |

   In multiplicity bins, the GMM score is near 0.5. The kNN score is mostly near 0.5 (the lowest bins are
   0.43 to 0.45). The Chamfer score is slightly inverted (0.41 to 0.48). In HT bins, the scores have inverted
   information (AUC 0.28 to 0.31 in the lowest bins). Thus, at a fixed multiplicity or HT, the model does not
   separate the signal in the correct direction.

3. **The three scores do not agree.** For seed 0, the Spearman correlations are: Chamfer–kNN 0.53,
   Chamfer–GMM 0.42 and kNN–GMM 0.87. The density scores also change more between seeds than the Chamfer
   score. This table gives the Spearman correlation between seeds (pairs 0–1, 0–2 and 1–2):

   | Score | Correlation between seeds |
   |---|---|
   | kNN | 0.55–0.80 |
   | GMM | 0.51–0.71 |
   | Chamfer | 0.81–0.83 |

4. For information only: if you change the sign of the scores after the test, the AUC is 0.572 (Chamfer),
   0.599 (kNN) and 0.564 (GMM). This is **not** a result. The sign was fixed before the test. Also, the values
   with the changed sign are much lower than the 0.811 baseline and lower than the Part 1 value of 0.781.
   But these values show that the scores contain a weak complexity signal. They are not noise.

### Interpretation

The data agrees with this explanation for the kNN and GMM scores: signal events have fewer particles than QCD
background events. The autoencoder learns the dominant QCD events. Thus, it puts the signal events in dense
regions of the latent space and identifies the incorrect events as anomalous.

This explanation is a hypothesis. The correlations support it, but no test of the mechanism was done. It does
not explain the Chamfer inversion (see finding 1).

The baselines that find the signal use jet-level structure (jet mass, two-prong τ21, dijet mass) and the
particle-level mean pT. The encoder did not get these variables as explicit features.

## 5. Ablations (sensitivity analysis only; 2 seeds each; not used to select a model)

Seed-averaged test AUC:

| Run | Seeds | Chamfer | kNN | GMM |
|---|---|---|---|---|
| main (latent 16, K 48, depth 3, all pooling types, Chamfer) | 3 | 0.428 | 0.401 | 0.436 |
| latent 4 | 2 | 0.366 | 0.405 | 0.403 |
| latent 8 | 2 | 0.401 | 0.372 | 0.376 |
| latent 32 | 2 | 0.428 | 0.412 | 0.485 |
| K = 24 reconstructed particles | 2 | 0.502 | 0.447 | 0.475 |
| K = 96 reconstructed particles | 2 | 0.490 | 0.425 | 0.447 |
| depth 2 | 2 | 0.421 | 0.412 | 0.442 |
| depth 4 | 2 | 0.429 | 0.388 | 0.428 |
| no pT-weighted pooling | 2 | 0.424 | 0.408 | 0.451 |
| pT-weighted Chamfer loss | 2 | 0.475 | 0.454 | 0.468 |

No configuration gives a useful score. The best value is 0.502 (K = 24, Chamfer). This value is equal to a
random selection.

Three changes move the Chamfer score nearest to 0.5: K = 24, K = 96 and the pT-weighted loss. A possible
reason: the number of soft particles in the loss possibly changes how much the score depends on multiplicity.
This is speculation. K = 24 and K = 96 both move the score towards 0.5, which is not a monotonic change, and no
test was done. None of these changes gives a useful score.

A smaller latent space makes the Chamfer inversion stronger (latent 4: 0.366, latent 8: 0.401, latent 16:
0.428). The effect on kNN is not clear (latent 4: 0.405, latent 8: 0.372, latent 16: 0.401). The effect on GMM is
approximate (latent 4: 0.403, latent 8: 0.376, latent 16: 0.436).

These runs did not change the main configuration.

## 6. Contamination (one seed; compared with seed 0 of the pure run: 0.419 / 0.405 / 0.448)

| Training set | Chamfer | kNN | GMM |
|---|---|---|---|
| background only (seed 0) | 0.419 | 0.405 | 0.448 |
| 1% signal (approximately 1,800 events) | 0.424 | 0.420 | 0.461 |
| 9% signal (16,200 events) | 0.413 | 0.362 | 0.387 |

The R&D file contains 9% signal. With 9% contamination, the density scores are more inverted. Relative to
seed 0, kNN decreases by 0.043 and GMM decreases by 0.061. A possible reason: signal events in the
training set make the regions near the signal denser. No test of this reason was done. The Chamfer score
almost does not change.

In the 1% run, kNN (0.420) and GMM (0.461) are slightly higher than the ranges of the pure 3-seed runs. These
ranges are 0.397 to 0.405 for kNN and 0.427 to 0.448 for GMM. The difference is approximately +0.015. With one seed, this result is not
conclusive. Do not interpret differences smaller than approximately 0.02 as effects.

## 7. Conclusion and next steps

The event-level graph autoencoder, as the design specifies it, is not better than the simple baselines. On this
test set, it is worse than a random selection.

The evidence (Section 4) agrees with this explanation: the signal events have a lower multiplicity than the
background, and their particles have higher pT. The latent-density scores find these events typical. This
explanation does not apply to the Chamfer inversion.

Part 2 gives these results:

- A validated benchmark harness that can evaluate any model. It reproduces the Part 1 values exactly.
- The full baseline table. The mean pT per particle (0.811) is the particle-level limit that a model must exceed.
- A negative result with a partial explanation (for the latent-density scores only). The ablations and a
  contamination run support the result.

Next steps, in the sequence of expected value (the fallback in the design is "adding jet-substructure inputs"):

1. **Add jet-level inputs** to the encoder: jet masses, τ21 and mjj from the `.features.h5` file. As an
   alternative, make the reference autoencoder that uses jet features. The τ21 score alone, without labels,
   gets 0.801.
2. **Remove the complexity shortcut.** Condition the model on multiplicity and HT or mean pT, or remove their
   effect by regression. Then the remaining score measures the event structure at a fixed complexity. The
   results in bins show that the current model has no information in the correct direction at a fixed
   complexity. In the bins, the AUC is near 0.5 or inverted.
3. Select the sign of the score with background-only or simulated data. Do not select it after you see the test
   labels.

## 8. Differences from the design and the plan

- **`data.max_particles` changed from 300 to 350** before any model result was available. The first Stage 0
  run failed the 2% truncation check of the design: 3.7% of background events have more than 300 particles.
  The measured multiplicity quantiles in a sample of 20,000 events were: median 176, p99 342, maximum 450. With
  a limit of 350, 0.7% of background events and 0.02% of signal events are above the limit. The baselines use
  the full events without truncation.
- After the code review, the cache writes and the run writes became safer. The changes are atomic role saves
  and lazy cache metadata. Also, the code writes the scores file last, as the "run finished" marker. The summary
  uses this marker.
- The sweep has one more check than the plan: an unknown `training.loss` value causes an error. The sweep does
  not change silently to the plain Chamfer loss.
- The smoke run (3 epochs, 20,000 events) calculated and printed test AUCs. These values were ignored and did
  not change the configuration.
- Section 3 of the design specified a change to `src/data/oracle_split.py`. A new module,
  `src/data/event_store.py`, was used instead. Thus, the Part 1 code did not change.
- Context for the criterion in bins: the minimum AUC in a bin for the limit baseline itself (mean pT per
  particle) is 0.599. This value is below the 0.6 criterion.
- The reference autoencoder with jet features was not made. It needs a separate model, and it is not part of
  the gate.
- Small review notes that were not done are in the SDD ledger. The ledger is not in the repository.

### Known limits and tasks for Part 3 (not fixed)

- Since commit d70e008, each scores file contains the HDF5 row number of each test event. The benchmark bank
  does not contain row numbers yet. The comparison of a run with the bank only checks the label vectors. Thus,
  an incorrect row order in one class is not found. Add row numbers to the bank files, and make sure that the
  row numbers are equal.
- Provenance: record the resolved configuration and the git SHA (with a flag for uncommitted changes) in the
  metrics and in the Kaggle bundle.
- The `labels.npy` cache has no connection to the identity of the HDF5 file.
- `apply_overrides` accepts items that do not contain `=`.
