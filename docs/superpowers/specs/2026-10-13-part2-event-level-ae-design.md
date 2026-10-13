# Part 2 Design: Event-Level Graph Autoencoder and Benchmark Harness

Date: 2026-10-13. Status: design approved section by section in chat; awaiting spec review.
Predecessor: `docs/PART1_VERIFICATION.md` (Part 1 baseline verified, AUC 0.483 ± 0.005 with max reduction, ~0.78 with mean reduction).

## 1. Goal and success criterion

Build an **unsupervised, event-level** anomaly detector for the LHC Olympics 2020 R&D data and measure it honestly against simple label-free baselines.

**Success:** the 3-seed mean ROC-AUC of at least one of the model's three scores has a bootstrap lower bound **above the best particle-level baseline** (HT, mean pT, leading pT, multiplicity, m_all; best is Σ pT ≈ 0.79) on the frozen test set. Jet-level baselines (mjj, τ21, jet masses) and the oracle −|mjj−3500| are reported but are not the bar, because the model never sees clustered jets. Because the best of three scores is taken, the bound uses a Bonferroni-adjusted 98.3% interval instead of 95%. In addition, within-bin AUC (bins of HT, multiplicity, leading pT) stays above ~0.6, so the score carries information beyond event composition. Approaching the jet-substructure scores (−τ21 0.80, −|mjj−3500| 0.90) is a stretch goal. Note that −|mjj−3500| needs the signal mass and is not a fair unsupervised target.

**If the gate fails,** the deliverable is the diagnosis (what each score correlates with, as in Part 1) and a recommendation for the next step (for example, jet-substructure inputs). This outcome counts as a result and is decided now, not after the fact.

Honest expectation: published unsupervised methods on this data reach roughly 0.6–0.8, so clearing 0.79 is uncertain. The harness and ablations exist so that a shortfall is explained.

## 2. Scope and non-goals

In scope: benchmark harness, event-level encoder and decoder, three score rules, contamination run, ablations, Kaggle runner.
Out of scope: supervised or weakly supervised methods (CWoLa), the LHCO black-box files, changes to the Part 1 code paths (Part 1 stays reproducible as is), a jet-feature AE beyond one reference row.

## 3. Components

Each unit has one purpose and a narrow interface. The benchmark harness knows nothing about models; scoring rules know nothing about the encoder.

| File | Purpose | Depends on |
|---|---|---|
| `src/benchmarks/baselines.py` | Label-free event scores: Σ pT (HT), mean pT per particle (= HT / multiplicity), leading-particle pT, multiplicity, invariant mass of all particles; jet scores (mjj, τ21, jet masses) read from `.features.h5` | particle arrays |
| `src/benchmarks/report.py` | Any per-event scores + labels → AUC, bootstrap 95% CI, AUC within HT / multiplicity / leading-pT bins; writes JSON and Markdown | numpy, sklearn |
| `src/data/event_features.py` | Per-particle features: log pT, η, φ, log E, pT/HT, Δη and Δφ to the leading particle (φ wrapped) | particle arrays |
| `src/data/oracle_split.py` (extend) | Frozen test set; disjoint train rows; contaminated-mix mode; larger train sizes; compact particle-array cache | existing |
| `src/model/event_ae.py` | `EventAE`; `forward(batch) -> (reconstruction, z)` on dense padded tensors | event_features |
| `src/training/set_loss.py` | Symmetric Chamfer loss with masking (and pT-weighted variant for ablation) | torch |
| `src/utils/latent_scores.py` | Fit on training latents only: kNN distance, Gaussian-mixture log-likelihood | sklearn |
| `scripts/part2_run.py` + `configs/part2_*.yaml` | One entry point: split, train, latents, all scores, benchmark; also the contamination run | all of the above |
| `src/pipeline/part2.py` | Library behind the entry point (prepare cache, train one run, build score bank, make report), so it can be tested without the CLI; `scripts/part2_run.py` is argparse only | all of the above |
| `scripts/kaggle_bundle.py` + `scripts/kaggle/part2_*.py` | Bundle the repo into a Kaggle script kernel; one small body file per stage | — |
| `scripts/part2_sweep.py` | Ablation runner with per-run result files, skip-if-exists | part2_run |

Reused from Part 1: `oracle_split.py`, `transforms.py`, seeding, the sweep-runner pattern, the Kaggle bundling pattern.

## 4. Data flow and model

**Data.**
- Test set frozen to Part 1's 5,000 background + 5,000 signal events (comparable numbers): background rows 40,000–44,999 (Part 1 trained on the first 40,000), signal rows 1,000,000–1,004,999. Part 1 skipped events with fewer than 2 particles, so its rows may differ by a handful; Stage 0 reproducing HT 0.792 confirms the sets match. Signal test rows are 1,000,000–1,004,999 in `events_anomalydetection_v2.h5` (sorted file; needs `scan_limit: null`).
- Train: 200k pure-background events from rows 45,000 onward (disjoint from the test rows). The last 10% of them (20k) is held out as a background-only validation set for early stopping; signal is never used for model selection.
- Contamination run: the 180k training events become background plus ~9% signal (and a second run at ~1%), with signal drawn from rows 1,005,000 onward (after the test rows, so no leakage). The 20k validation slice stays pure background in every run.
- Storage: compact particle arrays (pT-sorted, zero-padded to `max_particles`, default 300; roughly 0.7 GB for 200k events), not saved graphs (edge lists would be ~17 GB). Event-level baseline quantities (HT, multiplicity, m_all, ...) are computed from the full untruncated rows at prep time. Prep fails if more than 2% of events exceed `max_particles`. Dense padded tensors `[batch, particles, features]` plus a mask, held on the GPU. The ΔR < 0.4 adjacency (φ wrapped) is built on the GPU per batch. This replaces PyG for Part 2 and is the largest departure from Part 1.

**Encoder.** 3 message-passing layers, width 64, neighbour averaging over ΔR < 0.4, residual connections. Event pooling by mean, max and pT-weighted sum, then an MLP to z (default 16; configurable 4–32).

**Decoder.** MLP from z to the K = 48 hardest particles (log pT, η, cos φ, sin φ). Only hard particles are reconstructed, because soft radiation drives the multiplicity bias seen in Part 1. K is checked against the data; events with fewer than K particles are masked.

**Loss.** Symmetric Chamfer distance, φ represented as (cos, sin). pT-weighted Chamfer is an ablation only.

**Scores (one trained model, three rules).**
1. Chamfer reconstruction error.
2. Mean distance from z to its 20 nearest *training* latents.
3. Gaussian-mixture log-likelihood of z, fit on training latents only (negated so higher = more anomalous).

## 5. Evaluation plan

| Stage | What | Gate |
|---|---|---|
| 0. Harness | Benchmark table on the frozen test set: HT, mean pT (= HT / multiplicity), leading pT, multiplicity, m_all, mjj, τ21, mj1/mj2, Part 1 max and mean scores (the jet-feature AE reference row is added after Stage 1, since it needs a trained model) | Must reproduce verified Part 1 numbers (HT 0.792, mjj 0.788, Part 1 max ~0.48, mean ~0.78) within their CIs. If not, the harness is wrong and nothing else is trusted. |
| 1. Main run | Train `EventAE` on 200k pure background, 3 seeds; all 3 scores with bootstrap CIs; within-bin AUC | Section 1 success criterion. The within-bin check is the minimum over all bins of HT, multiplicity and leading pT (seed-averaged) and must be ≥ 0.6. |
| 2. Ablations | Latent size (4/8/16/32), reconstructed particles K (24/48/96), encoder depth, with/without pT-weighted pooling, plain vs pT-weighted loss; 2 seeds each | Sensitivity analysis only |
| 3. Contamination | Retrain with 9% and 1% signal in training | Report AUC drop vs the pure run |

Honesty rules:
- The headline configuration is the Section 4 defaults, fixed before anything touches the test set. Ablations are sensitivity analysis, never used to pick a winner.
- Early stopping uses only the background validation loss.
- The pass/fail result of the gate is written to the results JSON using the rule above.

## 6. Compute plan (Kaggle T4×2, 30 h/week quota)

| Step | Work | Estimate |
|---|---|---|
| Data prep | One scan of the sorted v2 file; cache compact arrays | ~10 min, CPU |
| Stage 0 | Baselines only | ~5 min |
| Stage 1 | 3 seeds, 2 at a time | ~1 h |
| Stage 2 | ~10 configs × 2 seeds | ~3 h |
| Stage 3 | 2 contamination runs × 1 seed | ~30 min |

Total about 5 GPU-hours. Each kernel gets a `-t` timeout. Each run writes metrics and scores to its own file and is skipped if that file exists, so a timeout loses one run, not the batch. A 10k-event smoke run measures the real epoch time first; if 200k events is too slow, epochs or train size are reduced and reported. Kernel pattern: base64 repo bundle, background polling, `kaggle kernels output`, as in Part 1. The Kaggle CLI cannot cancel runs.

## 7. Error handling

- Fail loudly: unmet split quotas, single-class test set, or train/test row overlap raise errors.
- NaN loss, or a collapsed latent (std below a threshold), aborts the run and records the reason.
- Density scores are fit on training latents only; the code asserts this.

## 8. Testing (local CPU, before any Kaggle run)

- Event features: φ wrapping, pT/HT fractions, padding mask.
- Adjacency: dense ΔR < 0.4 matches PyG `radius_graph` on the same events, including φ wrap.
- Chamfer: zero for identical sets; order invariant; handles events with fewer than K particles; matches a brute-force NumPy version.
- Encoder: permutation invariant; padding contributes nothing.
- Latent scores: on synthetic Gaussians an outlier ranks above inliers.
- Report: AUC and bootstrap CI match `sklearn` on random inputs; perfect and inverted scores give 1.0 and 0.0.
- Split: train rows never overlap test or contamination-signal rows.
- Stage 0 reproduction of Part 1 numbers is the end-to-end check on the harness.

## 9. Risks and open points

- Clearing 0.79 unsupervised is uncertain (see Section 1). The diagnosis path is the planned fallback.
- The mean-pT (= HT / multiplicity) baseline is untested; Stage 0 settles whether the Part 1 mean score is just average particle hardness.
- Epoch time for 200k events with dense adjacency is unmeasured. The smoke run decides the epoch budget.
- Chamfer with K = 48 ignores soft particles by design; the K ablation checks that this choice is not what limits performance.
