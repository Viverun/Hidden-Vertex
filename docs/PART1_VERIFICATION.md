# Part 1 Verification Report (2026-10-12)

This report gives the results of an independent new run of the Part 1 baseline. The run used the optimized
repository code on Kaggle (2 Tesla T4 GPUs). The report also gives the results of diagnostic tests of the
statements in the notebook and the README.

- Kaggle run: `viveriun/hidden-vertex-part1-verification` (version 3, private). Wall time: 546 s.
- Data: `atulchikku/lhc-olympics-2020-ad-r-and-d` → `events_anomalydetection_v2.h5`. The jet-level baselines
  also use `.features.h5`.
- Configuration: `configs/part1_notebook_parity.yaml`. The model, loss, optimizer, epochs, batch size, split
  sizes and normalization constants are the same as in the notebook. Seeds: 0 to 4.
- Diagnostic script: `scripts/part1_diagnostics.py`.

---

## 1. Dataset: the notebook file is not the public v2 file

| Property | Notebook (`events_anomalydetection.h5`, Kaggle dataset v1) | Public `events_anomalydetection_v2.h5` |
|---|---|---|
| Row order | Shuffled. Approximately 9% of the events in each part of the file are signal. | **Sorted.** Rows 0 to 999,999 are background. Rows 1,000,000 to 1,099,999 are signal. |
| Event 0 | 109 particles, 1,750 edges | 74 particles, 758 edges |
| Rows read before the oracle quotas are full | approximately 56,000 | approximately 1,005,000 |

Results of this difference:

- The "Shuffled Data Plot Twist" in the old README applies to the v1 file only. On the v2 file,
  `scan_limit: 200000` gives **zero** signal events. Thus, no AUC can be calculated.
- The split now accepts `scan_limit: null`. This setting reads the full file (75 s on Kaggle).
- The split now gives a warning if a quota is not full.
- `run_part1_eval.py` now stops if the test set contains one class only.
- The dataset `viveriun/lhc-collider` does not contain the v1 file now. The current version contains only
  `norm_stats.pt` and `model_architecture.py`. Thus, the exact notebook value (0.4615) cannot be calculated
  again. All the values in this report use the v2 file.

## 2. Main result

| Run | AUC |
|---|---|
| Notebook (v1 file, 1 run without a seed) | 0.4615 (printed). The conclusion cell shows 0.4650. The `metrics.py` docstring shows 0.4065. |
| **Repository pipeline (v2 file, 5 seeds)** | **0.483 ± 0.005** (minimum 0.477, maximum 0.491) |
| Bootstrap 95% CI, each seed | Upper limits 0.489 to 0.502 |

**Decision: the result is reproduced qualitatively.** On both files, the baseline is slightly below 0.5. With
10,000 test events, this is a small but real inversion: the autoencoder gives background events slightly higher
scores. It is not random.

## 3. Statements and evidence

| Statement | Evidence | Status |
|---|---|---|
| "AUC ~0.46, random-level discrimination" | 0.483 ± 0.005. The CI does not include 0.5 for 4 of 5 seeds. | ⚠️ Almost correct. The signal is weak and inverted. It is not exactly random. |
| "429 trainable parameters" | The total is 429. But 9 parameters are in `gate_nn`. The model calculates the output of this gate but does not use it. | ⚠️ The effective model has **420** parameters. |
| "1D bottleneck forces irreducible per-particle reconstruction errors" | A per-particle MLP with the same 1D bottleneck gets a training MSE of **0.0001** (identity). A GCN without edges gets **0.005**. The notebook GCN stops at approximately **0.51**. | ❌ With a loss that uses pT only, the 1D latent can carry pT without loss. All the reconstruction error comes from the **GCN averaging over the neighbours** (ΔR ≤ 0.4). Thus, the score measures the difference between the pT of a particle and the pT of its neighbours. |
| "You cannot detect an event-level anomaly by examining particles in isolation" | Event scores from the particle lists alone separate well: HT **0.79**, inverted multiplicity **0.72**. | ❌ This statement is too strong. This reconstruction-error score fails. The particle-level information does not fail. |
| "Multiplicity trap: the score responds primarily to multiplicity" | Spearman(score, multiplicity) = **−0.07** (background −0.17, signal 0.00). The AUC in multiplicity deciles is 0.39 to 0.59. | ❌ The data does not support this statement. |
| (not stated) what the score follows | Spearman(score, leading-particle pT) = **0.70**. The AUC in deciles of the maximum pT is **0.38 to 0.47**. | ✅ New result: the pT of the particle with the highest pT controls the score. At a fixed leading pT, the model reconstructs signal events *better* than QCD events. A possible reason: the particles with the highest pT in signal events have denser high-pT neighbourhoods. This is a hypothesis. No test was done. |
| "The radius graph is similar to anti-kT" (ΔR ≤ 0.4) | Δφ does not wrap at ±π. With the wrap, the number of edges increases by 1.9%. The AUC does not change (0.478 / 0.491 / 0.488). | ⚠️ A physics error with almost no effect. The option `data.wrap_phi` corrects it. For parity, the option is off. |
| Notebook loss curve 0.644 → 0.526 | Repository: 0.62 → 0.51 (v2, 5 seeds) | ✅ The values agree. |
| `norm_stats.pt` contains the normalization that the model uses | The keys are `feat_mean` and `feat_std`. The values are equal to the constants in the notebook code. | ✅ The values agree. The old repository code read `stats['mean']` and stopped with an error. |

## 4. What a working anomaly score needs (context for Part 2)

This table shows simple event variables without labels on the same test set:

| Score | AUC |
|---|---|
| −\|mjj − 3500 GeV\| (needs the signal mass; this score depends on the signal model) | 0.900 |
| −τ21 of the leading jet | 0.801 |
| HT = Σ pT | 0.792 |
| mjj | 0.788 |
| −multiplicity | 0.715 |
| mj1 / mj2 | 0.634 / 0.637 |
| **Part 1 AE** | **0.483** |

The signal (W′ → XY → 4 quarks) is visible in four variables. These variables are the jet mass, the 2-prong
substructure (τ21), the total pT and a lower multiplicity. All these variables are at the event level or the jet level. This
supports the Part 2 direction: a latent space at the event level and inputs with jet and substructure
information.

Compare an unsupervised Part 2 model with HT (0.79). If a model is not better than Σ pT, it has learned only
the energy scale.

## 5. Capacity sweep and score reduction (follow-up, 2026-10-13)

Kaggle runs:

- `viveriun/hidden-vertex-part1-capacity-sweep`: 64 runs with `scripts/part1_capacity_sweep.py`.
- `viveriun/hidden-vertex-part1-mean-check`: diagnostics of the mean and the maximum.

The values are means over 2 seeds:

| Loss | Width | Parameters | Training loss | AUC (maximum) | AUC (mean) |
|---|---|---|---|---|---|
| pt_only | 16×8 (Part 1) | 420–539 | 0.500–0.518 | 0.476–0.485 | 0.765–0.786 |
| pt_only | 32×16 | 1.3k–1.6k | 0.496–0.503 | 0.484–0.490 | 0.789–0.792 |
| pt_only | 64×32 | 4.7k–5.2k | 0.493–0.495 | 0.481–0.490 | 0.788–0.793 |
| pt_only | 128×64 | 17.7k–18.6k | 0.493–0.504 | 0.474–0.494 | 0.779–0.795 |
| weighted, latent 1 | all widths | | 2.10–2.25 | 0.437–0.451 | 0.50–0.55 |
| weighted, latent 4–8 | all widths | | 1.66–1.78 | 0.474–0.490 | 0.748–0.787 |

Findings:

1. **A larger node-level autoencoder does not help.** With 44 times more parameters, the loss decreases by
   approximately 2%. The AUC does not change.
2. **The bottleneck has an effect only with the loss that uses 3 features.** With `weighted`, latent 1 gives
   poor results. A latent of 4 or more gives the same results as `pt_only`.
3. **The reduction method has a larger effect than the model.** The same trained model gets an AUC of 0.48
   with `max` and **0.78** with `mean`. An old comment in the configuration said: "'max' node error is most
   sensitive to W' signals". This test shows that the comment is not correct.

Is the mean score a real result, or does it only show the event composition? (seed 0, 10,000 test events)

| Property | Mean reduction | Maximum reduction |
|---|---|---|
| AUC | 0.781 [0.772–0.790] | 0.478 [0.467–0.488] |
| Spearman with multiplicity | −0.71 | −0.07 |
| Spearman with leading pT | 0.22 | 0.70 |
| AUC in multiplicity deciles | 0.68–0.81 | 0.39–0.59 |
| AUC in HT deciles | 0.61–0.82 | 0.38–0.45 |
| AUC in leading-pT deciles | 0.69–0.82 | 0.38–0.47 |

**Decision:** the mean score keeps an AUC of 0.6 to 0.8 at a fixed multiplicity, HT or leading pT. Thus, it has
more information than any one simple variable.

But it is not better than the HT baseline overall (0.78 and 0.79). Also, it has a strong negative correlation
with multiplicity. Thus, it possibly follows the mean pT of the particles (HT / multiplicity). This combined
baseline was not tested at the time of this report. Make it the first item in the Part 2 benchmark.

(Part 2 tested this baseline later: HT/N gets an AUC of 0.811. See `docs/PART2_RESULTS.md`.)

## 6. Engineering verification

| Item | Before | After |
|---|---|---|
| Unit tests | none | 20 (`tests/test_part1_pipeline.py`). The tests compare the graph construction, the oracle routing and the scoring with exact copies of the notebook code. |
| `run_part1_eval.py` from the repository root | `ModuleNotFoundError: src`, then `KeyError: 'mean'` on the Kaggle stats file | The script runs. It uses a seed and the cached split. It saves the metrics, the CI, the score of each event and the ROC curve. |
| Oracle split on v2 | 0 signal events, no warning | Reads the full file in 75 s. Gives a warning if a quota is not full. |
| `train-default` | Assumed that rows 0 to 999,999 are background (incorrect for v1). Used inputs without standardization. Ignored `weight_decay`. | Selects background by label (`labels.npy`). Uses standardized inputs. Uses `weight_decay`. |
| `process_h5.py` | Loaded the full file (approximately 18.5 GB) into memory. Made fully connected graphs. | Reads the file in chunks. Makes radius graphs. Writes `labels.npy`. |
| Shuffled chunk loading | Random order and a FIFO cache of 20 chunks. Most items caused a new chunk load. | Shuffle in chunk groups and a true LRU cache. Each chunk loads one time in each epoch. |
| Training and evaluation, one seed | approximately 300 s on a Kaggle CPU | 53 to 81 s on one T4 (two seeds run at the same time) |
| Seed reproducibility | no seed | Repeated runs with the same seed give the same AUC (verified locally on a CPU). |

## 7. Recommended README corrections

All these corrections are now in the README.

1. Give **0.48 ± 0.01 (v2, 5 seeds)** next to the original 0.4615 (v1). Remove 0.4650 and 0.4065.
2. Replace "random-level" with "slightly below random (inverted)".
3. Replace "429 parameters" with "420 effective parameters (and 9 gate parameters that the model does not use)".
4. Write the "Multiplicity Trap" section again. The score follows the leading-particle pT, not the multiplicity.
5. Change the main result. The node-level *reconstruction error* fails. The particle-level information does not
   fail: HT alone gets 0.79.
6. State that `events_anomalydetection_v2.h5` is sorted and that it needs `scan_limit: null`.
