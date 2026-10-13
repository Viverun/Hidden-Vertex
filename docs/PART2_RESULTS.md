# Part 2 Results: Event-Level Graph Autoencoder vs. Simple Baselines (2026-10-13)

**Outcome: the pre-registered gate was NOT passed.** The unsupervised event-level autoencoder does not beat the
simple particle-level baselines. All three of its anomaly scores are *inverted* (AUC 0.40–0.44, below 0.5):
signal events look *more* typical to the model than background events.

- Spec: `docs/superpowers/specs/2026-10-13-part2-event-level-ae-design.md`
- Plan: `docs/superpowers/plans/2026-10-13-part2-event-level-ae.md`
- Code: branch `part2-event-level-ae`; entry point `scripts/part2_run.py`; configuration `configs/part2_main.yaml`
- Raw results: `results/part2/` (git-ignored): `stage0.md`, `stage1.md`, `final.md|json`, `runs/<tag>/`

## 1. Setup

| | |
|---|---|
| Data | `events_anomalydetection_v2.h5` (sorted: rows 0–999,999 background, 1,000,000+ signal), Kaggle dataset `atulchikku/lhc-olympics-2020-ad-r-and-d` |
| Test set (frozen, same as Part 1) | 5,000 background (rows 40,000–44,999) + 5,000 signal (rows 1,000,000–1,004,999) |
| Training set | 180,000 pure background events (rows 45,000+); validation slice 20,000 pure background events (last 10% of the 200k) |
| Model | `EventAE`: 3 neighbour-averaging layers (width 64, ΔR < 0.4, residual), mean/max/pT-weighted pooling, latent 16, decoder to the 48 hardest particles, symmetric Chamfer loss; 95,888 parameters |
| Scores | (1) Chamfer reconstruction error, (2) mean distance to the 20 nearest training latents, (3) negative Gaussian-mixture log-likelihood (8 components) of the latent; density scorers fit on training latents only |
| Training | 30 epochs, AdamW lr 1e-3, batch 256, patience 5 (never triggered in the headline runs), ~34 s/epoch on one T4, ~18 min per seed |
| Headline runs | 3 seeds (0, 1, 2), configuration fixed before any test-set result |
| Kaggle kernels (private, `viveriun`) | `hidden-vertex-part2-stage0`, `-stage1`, `-stage2`, `-stage3`, `-smoke` (timing), `-mult` (multiplicity quantiles) |

Gate (spec Section 1, fixed in `evaluate_gate` before results): a model score passes if the lower bound of its
seed-averaged AUC (Bonferroni-adjusted 98.3% interval) exceeds the **best particle-level baseline**, and its minimum
within-bin AUC (bins of HT, multiplicity, leading pT) is at least 0.6.

## 2. Stage 0: harness reproduces Part 1, and raises the bar

The new harness reproduces all four verified Part 1 numbers exactly (each inside its own CI):

| Score | Expected | Measured | CI |
|---|---|---|---|
| HT (sum pT) | 0.792 | 0.792 | 0.783–0.802 |
| mjj | 0.788 | 0.788 | 0.778–0.799 |
| Part 1 AE, max reduction | 0.478 | 0.478 | 0.467–0.489 |
| Part 1 AE, mean reduction | 0.781 | 0.781 | 0.772–0.790 |

Baselines on the frozen test set (higher = more anomalous; 95% CI):

| Score | Group | AUC | CI |
|---|---|---|---|
| −\|mjj − 3500\| (needs the signal mass) | oracle | 0.900 | 0.894–0.906 |
| **mean pT per particle (= HT / multiplicity)** | particle | **0.811** | 0.803–0.821 |
| −τ21 (leading jet) | jet | 0.801 | 0.792–0.810 |
| HT (Σ pT) | particle | 0.792 | 0.783–0.802 |
| mjj | jet | 0.788 | 0.778–0.799 |
| Part 1 AE (mean reduction) | external | 0.781 | 0.772–0.790 |
| −multiplicity | particle | 0.715 | 0.704–0.726 |
| mj2 / mj1 | jet | 0.637 / 0.634 | |
| leading pT | particle | 0.554 | 0.543–0.565 |
| m_all | particle | 0.516 | 0.504–0.527 |
| Part 1 AE (max reduction) | external | 0.478 | 0.467–0.489 |

**Answer to the open question from Part 1.** The Part 1 mean-reduction score (AUC 0.781) is no better than a
one-line quantity: mean pT per particle scores **0.811**, above both HT (0.792) and the Part 1 score. This also
raised the gate's bar from the ≈0.79 quoted in the spec to **0.811**, because the bar is defined as the best
particle-level baseline.

## 3. Stage 1: headline result (gate NOT PASSED)

Seed-averaged AUC (3 seeds), 98.3% bootstrap CI over the 10,000 test events:

| Model score | AUC | CI | Min within-bin AUC | Beats bar 0.811 | Passes |
|---|---|---|---|---|---|
| Chamfer reconstruction error | 0.428 | 0.415–0.442 | 0.311 | no | no |
| kNN distance in latent space | 0.401 | 0.389–0.412 | 0.281 | no | no |
| Gaussian-mixture likelihood | 0.436 | 0.426–0.448 | 0.308 | no | no |

Per seed (Chamfer / kNN / GMM): seed 0 = 0.419 / 0.405 / 0.448; seed 1 = 0.433 / 0.397 / 0.427;
seed 2 = 0.432 / 0.401 / 0.434. Seed-to-seed spread is about 0.01–0.02, much smaller than the distance to 0.5, so the
inversion is systematic, not noise. Training was healthy: train and validation loss agree (best validation loss
0.12–0.13), no latent collapse, no NaNs. Validation loss was noisy and nearly flat at the end of training (best epochs 26, 29 and 30 for
seeds 0, 1 and 2; seed 0's last five epochs ranged 0.131–0.142).

## 4. Diagnosis: what the scores track

Spearman correlation of the seed-0 scores with simple event quantities (jet-level quantities such as jet mass, τ21 and mjj are not in this table; mean pT per particle is particle-level) (all test events | background only | signal only):

| Score | HT | multiplicity | leading pT | mean pT per particle |
|---|---|---|---|---|
| Chamfer | −0.01 \| +0.15 \| 0.00 | +0.08 \| +0.08 \| −0.01 | +0.17 \| +0.18 \| +0.19 | −0.07 \| −0.02 \| +0.01 |
| kNN | −0.02 \| +0.19 \| 0.00 | **+0.28** \| +0.30 \| +0.19 | −0.02 \| −0.05 \| +0.05 | **−0.27** \| −0.24 \| −0.20 |
| GMM | +0.02 \| +0.18 \| 0.00 | **+0.23** \| +0.27 \| +0.15 | +0.01 \| −0.04 \| +0.09 | **−0.20** \| −0.21 \| −0.16 |

Findings:

1. **The latent-density scores appear to track event complexity.** kNN and GMM scores rise with multiplicity and fall
   with mean pT, within background alone as well, and their within-multiplicity-bin AUC is near chance (kNN
   0.43–0.50, GMM 0.47–0.51). Signal events have *fewer, harder* particles (mean multiplicity 140 vs. 182 for
   background), which is consistent with them sitting in dense, easy regions of the latent space; under the
   pre-declared sign ("far from training data = anomalous") that would produce AUC below 0.5. This is suggested by the
   correlations, not tested causally. **The Chamfer score is different:** it barely tracks multiplicity (ρ +0.08) or
   mean pT (ρ −0.07) and stays inverted within multiplicity bins (0.41–0.48) and HT bins (down to 0.31), so its
   inversion is not explained by multiplicity alone and its cause is not established.
2. **Within bins of HT, multiplicity and leading pT, the scores do not recover the signal.** Within-bin AUC ranges
   (seed-averaged): Chamfer HT 0.31–0.56, multiplicity 0.41–0.48, leading pT 0.40–0.44; kNN 0.28–0.54, 0.43–0.50,
   0.38–0.45; GMM 0.31–0.57, 0.47–0.51, 0.40–0.50. Within multiplicity bins the scores are near chance, and within HT
   bins they carry *inverted* information (AUC 0.28–0.31 at the low end), so at fixed multiplicity or HT the model
   does not separate signal in the right direction.
3. **The three scores disagree with each other** (seed 0 Spearman: Chamfer–kNN 0.53, Chamfer–GMM 0.42, kNN–GMM
   0.87) and the density scores are less stable across seeds (Spearman between seeds over all three pairs 0–1, 0–2,
   1–2: kNN 0.55–0.80, GMM 0.51–0.71) than Chamfer (0.81–0.83).
4. For information only: flipping the sign post hoc would give AUC 0.572 (Chamfer), 0.599 (kNN), 0.564 (GMM). That
   is *not* a result (the sign was fixed in advance and the flipped scores are still far below the 0.811 baseline and
   below Part 1's 0.781), but it shows the scores contain a weak complexity signal, not noise.

Interpretation: the evidence is consistent with the signal events being *less* complex than QCD background in terms
of particle count (for the latent-density scores), so that an autoencoder modelling the dominant QCD manifold places
them in dense regions and flags the wrong events. This is a hypothesis supported by correlations, not a tested
mechanism, and it does not account for the Chamfer inversion (see finding 1). What distinguishes the signal in the
baselines is jet-level structure (jet mass, two-prong τ21, dijet mass) and particle-level mean pT, which the
encoder had no direct access to as explicit features.

## 5. Ablations (sensitivity analysis only; 2 seeds each; not used to select a winner)

Seed-averaged test AUC:

| Run | Seeds | Chamfer | kNN | GMM |
|---|---|---|---|---|
| main (latent 16, K 48, depth 3, full pooling, Chamfer) | 3 | 0.428 | 0.401 | 0.436 |
| latent 4 | 2 | 0.366 | 0.405 | 0.403 |
| latent 8 | 2 | 0.401 | 0.372 | 0.376 |
| latent 32 | 2 | 0.428 | 0.412 | 0.485 |
| K = 24 reconstructed particles | 2 | 0.502 | 0.447 | 0.475 |
| K = 96 reconstructed particles | 2 | 0.490 | 0.425 | 0.447 |
| depth 2 | 2 | 0.421 | 0.412 | 0.442 |
| depth 4 | 2 | 0.429 | 0.388 | 0.428 |
| no pT-weighted pooling | 2 | 0.424 | 0.408 | 0.451 |
| pT-weighted Chamfer loss | 2 | 0.475 | 0.454 | 0.468 |

No configuration reaches a useful score; the best single value is 0.502 (K = 24, Chamfer), i.e. chance. Moving from
K = 48 to K = 24 or 96 and the pT-weighted loss move the Chamfer score closest to 0.5, which one might speculate
means that how many (soft) particles the loss counts changes how strongly the score is tied to multiplicity (speculative:
K = 24 and K = 96 both move toward 0.5 non-monotonically, and this was not tested), but none helps. Smaller latent
sizes make the Chamfer inversion stronger (z4 0.366, z8 0.401 vs. 0.428), but not clearly the kNN one (z4 0.405 vs.
0.401; z8 0.372) and only roughly the GMM one (z4 0.403, z8 0.376 vs. 0.436). These runs were not used to modify the headline configuration.

## 6. Contamination (single seed; compared with seed 0 of the pure run: 0.419 / 0.405 / 0.448)

| Training set | Chamfer | kNN | GMM |
|---|---|---|---|
| pure background (seed 0) | 0.419 | 0.405 | 0.448 |
| 1% signal (≈1,800 events) | 0.424 | 0.420 | 0.461 |
| 9% signal (16,200 events) | 0.413 | 0.362 | 0.387 |

9% contamination (the R&D file's signal fraction) makes the density scores *more* inverted (kNN −0.043, GMM −0.061 vs.
seed 0), plausibly because signal events in the training set make signal-like regions denser (not tested). The
Chamfer score is nearly unaffected. The 1% run's kNN (0.420) and GMM (0.461) are slightly above the pure 3-seed ranges
(kNN 0.397–0.405, GMM 0.427–0.448), by about +0.015; with a single seed this is inconclusive. Differences below ~0.02
should not be read as effects.

## 7. Conclusion and recommended next step

The unsupervised event-level graph autoencoder, as specified, **does not beat the simple baselines** and is below
chance on this test set; the evidence (Section 4) is consistent with signal events here being lower in multiplicity and
harder than background, which the latent-density scores treat as typical (the Chamfer inversion is not explained by this). The honest deliverables of Part 2 are therefore:

- a validated, model-agnostic benchmark harness (reproduces the Part 1 numbers exactly) and the full baseline
  table, including the finding that **mean pT per particle (0.811) is the particle-level bar to beat**;
- a negative result with a partially supported explanation (latent-density scores only), backed by ablations and a contamination run.

Recommended next steps, in order of expected value (spec fallback: "adding jet-substructure inputs"):

1. **Add jet-level inputs** (jet masses, τ21, mjj from the `.features.h5` file) to the encoder or build the
   jet-feature AE reference row. The label-free τ21 score alone reaches 0.801.
2. **Remove the complexity shortcut**: condition the model on, or regress out, multiplicity and HT/mean pT, so the
   remaining score measures structure at fixed complexity. The within-bin results show this is where the current
   model has no information.
3. Treat the score's sign as a design choice made on background-only or simulated data, not after seeing test labels.

## 8. Deviations from the spec and plan

- **`data.max_particles` 300 → 350** before any model result existed. Stage 0's first attempt failed the spec's 2%
  truncation guard (3.7% of background events have more than 300 particles); measured multiplicity quantiles
  (median 176, p99 342, max 450 in a 20k sample) gave 0.7% of background and 0.02% of signal above 350. Baselines use
  the full untruncated events.
- Cache and run writes were hardened after code review (atomic role saves, lazy cache metadata, scores file written
  last as the "run finished" marker, summary keyed on that marker).
- The sweep gets one more validation than the plan: unknown `training.loss` values raise instead of silently
  falling back to plain Chamfer.
- The smoke run (3 epochs, 20k events) computed and printed test AUCs; they were ignored and changed no configuration.
- Spec Section 3 called for extending `src/data/oracle_split.py`; a new module `src/data/event_store.py` was used
  instead so that Part 1 code stays untouched.
- Context for the within-bin criterion: the bar baseline itself (mean pT per particle) has a minimum within-bin AUC of
  0.599, below the 0.6 criterion.
- The jet-feature AE reference row was not built (it needs a separate model and is not part of the gate).
- Deferred minor review notes are listed in the SDD ledger (not part of the repository).

### Known limitations / follow-ups for Part 3 (deferred, not fixed)

- Runs are joined to the benchmark bank only by checking label vectors, so within-class row misalignment would go
  undetected; store test row ids in scores and bank files and assert equality.
- Provenance: record the resolved config and git SHA (plus dirty flag) in metrics and in the Kaggle bundle.
- The `labels.npy` cache is not tied to the HDF5 file identity.
- `apply_overrides` accepts items without `=`.
