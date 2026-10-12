# Part 1 Verification Report (2026-10-12)

Independent re-run of the Part 1 baseline with the optimized repository code on Kaggle (2× Tesla T4),
plus diagnostics that test the claims made in the notebook and README.

- Kaggle run: `viveriun/hidden-vertex-part1-verification` (version 3, private), 546 s wall time.
- Data: `atulchikku/lhc-olympics-2020-ad-r-and-d` → `events_anomalydetection_v2.h5` (+ `.features.h5` for jet-level baselines).
- Config: `configs/part1_notebook_parity.yaml` (same model, loss, optimizer, epochs, batch size, split sizes and
  normalization constants as the notebook), seeds 0–4.
- Diagnostics script: `scripts/part1_diagnostics.py`.

---

## 1. Dataset: the notebook's file is not the public v2 file

| | Notebook (`events_anomalydetection.h5`, Kaggle dataset v1) | Public `events_anomalydetection_v2.h5` |
|---|---|---|
| Ordering | Shuffled, ~9% signal throughout | **Sorted**: rows 0–999,999 background, 1,000,000–1,099,999 signal |
| Event #0 | 109 particles, 1,750 edges | 74 particles, 758 edges |
| Oracle quotas filled after | ~56k rows | ~1,005,000 rows |

Consequences:

- The README's "Shuffled Data Plot Twist" is a property of the v1 file only. On v2, `scan_limit: 200000`
  yields **zero** signal events, so every AUC is undefined. The split now supports `scan_limit: null`
  (whole file, 75 s on Kaggle), warns when any quota is unmet, and `run_part1_eval.py` refuses a
  single-class test set.
- The v1 file is no longer attached to `viveriun/lhc-collider` (current version holds only
  `norm_stats.pt` and `model_architecture.py`), so the exact notebook number (0.4615) cannot be
  re-derived. All numbers below are on v2.

## 2. Headline result

| | AUC |
|---|---|
| Notebook (v1 file, 1 unseeded run) | 0.4615 (printed); the conclusion cell says 0.4650; `metrics.py` docstring says 0.4065 |
| **Repo pipeline (v2 file, 5 seeds)** | **0.483 ± 0.005** (min 0.477, max 0.491) |
| Bootstrap 95% CI, per seed | upper bounds 0.489–0.502 |

**Verdict: reproduced qualitatively.** The baseline sits slightly *below* 0.5 on both files. With 10,000 test
events this is a small but real inversion (the AE scores background as slightly more anomalous), not pure
randomness.

## 3. Claims vs. evidence

| Existing claim | Evidence | Status |
|---|---|---|
| "AUC ~0.46, random-level discrimination" | 0.483 ± 0.005, CI excludes 0.5 for 4/5 seeds | ⚠️ Mostly right; it is a weak *inverted* signal, not exactly random |
| "429 trainable parameters" | 429 total, but 9 belong to `gate_nn`, whose output is computed and discarded | ⚠️ Effective model: **420** parameters |
| "1D bottleneck forces irreducible per-particle reconstruction errors" | A per-particle MLP with the same 1D bottleneck reaches train MSE **0.0001** (identity); GCN with no edges reaches **0.005**. The notebook GCN plateaus at **~0.51** | ❌ With a pT-only loss the 1D latent can carry pT losslessly. All reconstruction error comes from **GCN neighbourhood averaging** (ΔR < 0.4), i.e. the score measures how much a particle's pT differs from its neighbours |
| "You cannot detect an event-level anomaly by examining particles in isolation" | Event scores built from particle lists alone separate well: HT **0.79**, inverted multiplicity **0.72** | ❌ Overclaim. What fails is *this reconstruction-error score*, not particle-level information |
| "Multiplicity trap: score responds primarily to multiplicity" | Spearman(score, multiplicity) = **−0.07** (bg −0.17, sig 0.00); AUC within multiplicity deciles 0.39–0.59 | ❌ Not supported |
| (implicit) what the score tracks | Spearman(score, leading-particle pT) = **0.70**; AUC within max-pT deciles **0.38–0.47** | ✅ New finding: the score is dominated by the hardest particle's pT; at fixed leading pT, signal events reconstruct *better* than QCD (likely because the hardest signal particles sit in denser high-pT neighbourhoods; hypothesis, not tested) |
| "Radius graph mimics anti-kT" (ΔR < 0.4) | Δφ is not wrapped at ±π; wrapping adds 1.9% edges; AUC unchanged (0.478/0.491/0.488) | ⚠️ Physics bug, negligible effect. Fixed behind `data.wrap_phi` (off for parity) |
| Notebook loss curve 0.644 → 0.526 | Repo: 0.62 → 0.51 (v2, 5 seeds) | ✅ Consistent |
| `norm_stats.pt` holds the normalization used | Keys are `feat_mean`/`feat_std`; values equal the notebook's hard-coded constants | ✅ Values match; old repo code read `stats['mean']` and crashed |

## 4. What a working anomaly score needs (context for Part 2)

Simple, label-free event quantities on the same test set:

| Score | AUC |
|---|---|
| −\|mjj − 3500 GeV\| (needs the signal mass; not model-independent) | 0.900 |
| −τ21 of the leading jet | 0.801 |
| HT = Σ pT | 0.792 |
| mjj | 0.788 |
| −multiplicity | 0.715 |
| mj1 / mj2 | 0.634 / 0.637 |
| **Part 1 AE** | **0.483** |

The signal (W′ → XY → 4 quarks) shows up in **jet mass, 2-prong substructure (τ21), total pT, and lower
multiplicity**. These are all event- or jet-level quantities. That supports the Part 2 direction (event-level
latent, jet/substructure-aware inputs). An unsupervised Part 2 model should be benchmarked against HT
(0.79), since a model that cannot beat Σ pT has not learned anything beyond the energy scale.

## 5. Engineering verification

| | Before | After |
|---|---|---|
| Unit tests | none | 20 (`tests/test_part1_pipeline.py`); graph construction, oracle routing and scoring are checked against verbatim copies of the notebook code |
| `run_part1_eval.py` from repo root | `ModuleNotFoundError: src`; then `KeyError: 'mean'` on the Kaggle stats file | Runs; seeded; reuses the cached split; saves metrics, CI, per-event scores and ROC |
| Oracle split on v2 | 0 signal, silent | Full-file scan in 75 s; warns on unmet quotas |
| `train-default` | Assumed rows 0–999,999 are background (wrong for v1), fed unstandardized inputs, ignored `weight_decay` | Background selected by label (`labels.npy`), standardized inputs, `weight_decay` honoured |
| `process_h5.py` | Loaded the full file (~18.5 GB) into RAM; fully connected graphs | Streams in chunks; radius graphs; writes `labels.npy` |
| Shuffled chunk loading | Random order + 20-chunk FIFO cache → a chunk reload on most items | Chunk-grouped shuffle + true LRU → one load per chunk per epoch |
| Train + eval, one seed | ~300 s on Kaggle CPU | 53–81 s on one T4 (two seeds run in parallel) |
| Seed reproducibility | unseeded | identical AUC across repeated seeded runs (verified locally on CPU) |

## 6. Recommended README corrections

1. Report **0.48 ± 0.01 (v2, 5 seeds)** next to the original 0.4615 (v1), and remove 0.4650 / 0.4065.
2. Replace "random-level" with "slightly below random (inverted)".
3. Replace "429 parameters" with "420 effective (+9 unused gate parameters)".
4. Rewrite the "Multiplicity Trap" section: the score tracks leading-particle pT, not multiplicity.
5. Reword the key result: node-level *reconstruction error* fails; particle-level information does not
   (HT alone reaches 0.79).
6. State that `events_anomalydetection_v2.h5` is sorted and needs `scan_limit: null`.
