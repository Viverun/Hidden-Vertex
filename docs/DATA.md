# Data: The Hidden Vertex

This document describes the dataset, the file versions and the data pipelines of Part 1 and Part 2.

## 1. Dataset

The project uses the LHC Olympics 2020 R&D dataset (G. Kasieczka, B. Nachman, D. Shih,
[doi:10.5281/zenodo.2629072](https://doi.org/10.5281/zenodo.2629072), license CC BY 4.0).

| Property | Value |
|---|---|
| Background | 1,000,000 QCD dijet events |
| Signal | 100,000 events: W′ → XY → (qq̄)(qq̄), m(W′) = 3.5 TeV, m(X) = 500 GeV, m(Y) = 100 GeV |
| Simulation | Pythia 8 and Delphes 3 |
| Particles per event | Up to 700 |
| Particle features | pT (GeV), η, φ |

The repository does not include the dataset because the file is large (approximately 18.5 GB uncompressed).

## 2. File format

The main file is an HDF5 file. The table is at `df/block0_values`. Each row is one event and has 2,101 columns:

| Columns | Contents |
|---|---|
| 0 to 2,099 | 700 particles × (pT, η, φ). Empty particle slots contain zeros. |
| 2,100 | Truth label: 0 = background (QCD), 1 = signal (W′) |

The helper `open_events()` in `src/data/dataset.py` opens the file. The constants `NUM_PARTICLE_COLUMNS = 2100`
and `LABEL_COLUMN = 2100` are in the same file.

A second file, `events_anomalydetection_v2.features.h5`, contains jet features for each event. These features are
the four-momenta and masses of the two leading jets (`pxj1`, `mj1`, …) and the N-subjettiness values. The jet-level baselines use
this file (`src/benchmarks/baselines.py`).

## 3. File versions

**Caution:** two versions of the event file exist, and they have different row orders.

| Property | `events_anomalydetection.h5` (v1) | `events_anomalydetection_v2.h5` (v2) |
|---|---|---|
| Row order | Shuffled. Approximately 9% of the events in each part of the file are signal. | **Sorted.** Rows 0 to 999,999 are background. Rows 1,000,000 to 1,099,999 are signal. |
| Used by | The Part 1 notebook | The repository pipelines (Part 1 and Part 2) |
| Location | Not available now | Kaggle dataset `atulchikku/lhc-olympics-2020-ad-r-and-d` |

The particles in the v2 file are sorted by pT in decreasing sequence.

Because the v2 file is sorted, a pipeline that reads only the first rows gets background events only. Set
`scan_limit: null` in the Part 1 configuration. Then the oracle split reads the full file.

## 4. Part 1 pipeline (graphs)

### 4.1 Oracle split

`build_oracle_purified_split()` in `src/data/oracle_split.py` reads the events in file order. It uses the truth
label to put each event in a set:

1. The first 40,000 background events go to the training set.
2. The next 5,000 background events go to the test set.
3. The first 5,000 signal events go to the test set.

The truth label is only used to make a training set without signal ("oracle purification"). The model does not
see the labels.

The command `python main.py prepare-part1 --config configs/part1_notebook_parity.yaml` makes the split and saves
it to `results/scores/part1_oracle_split.pt`. The evaluation script uses this saved split again.

### 4.2 Radius graphs

`create_radius_graph()` in `src/data/dataset.py` makes one graph for each event:

- **Nodes** are the particles with pT > 0. The node features are (pT, η, φ).
- **Edges** connect two particles if ΔR = √(Δη² + Δφ²) is 0.4 or less.
- Events with fewer than 2 particles do not give a graph.

The option `wrap_phi` controls Δφ. With `wrap_phi: false`, Δφ is not periodic. This setting is the same as in the
notebook. With `wrap_phi: true`, Δφ wraps at ±π. This setting is physically correct. The wrap adds 1.9% more edges
and does not change the AUC (see `PART1_VERIFICATION.md`).

### 4.3 Normalization

`standardize_features()` in `src/data/transforms.py`:

1. Replaces pT with log(pT).
2. Subtracts the mean and divides by the standard deviation of each feature.

With `stats_path: null`, the code uses the constants from the notebook:

| Feature | Mean | Standard deviation |
|---|---|---|
| log pT | 1.1831 | 1.5210 |
| η | 0.0012 | 1.4121 |
| φ | −0.0034 | 1.8150 |

The file `norm_stats.pt` on Kaggle contains the same values with the keys `feat_mean` and `feat_std`.

### 4.4 Graph chunks for larger training sets

`scripts/process_h5.py` changes the full HDF5 file into graph files:

- It reads 10,000 events at a time. Thus, the full file is never in memory.
- It writes `data_chunk_{c}.pt`. Event `i` is in chunk `i // 10000`.
- It writes `labels.npy` with the truth label of each event. The command `train-default` uses this file to select
  background events.

`RawChunkDataset` and `ChunkGroupedSampler` in `src/data/dataset.py` read these chunks. The sampler shuffles the
events in groups by chunk, and an LRU cache keeps the recent chunks. Thus, each chunk loads one time in each epoch.

## 5. Part 2 pipeline (padded arrays)

Part 2 does not use graph objects. It stores events as dense arrays. The code is in `src/data/event_store.py`.

### 5.1 Event sets

`plan_rows()` selects the HDF5 rows for each set. The configuration `configs/part2_main.yaml` gives the sizes:

| Set | Label | Rows | Events |
|---|---|---|---|
| `test_bg` | 0 | 40,000 to 44,999 | 5,000 |
| `test_sig` | 1 | 1,000,000 to 1,004,999 | 5,000 |
| `train_bg` | 0 | 45,000 to 244,999 | 200,000 (180,000 for training, 20,000 for validation) |
| `contam_sig` | 1 | after the test signal rows | 16,200 (for the contamination runs only) |

The test set is the same as in Part 1.

### 5.2 Event cache

`build_cache()` writes one file for each set to `<cache_dir>/<set>.npz`:

| Array | Shape | Contents |
|---|---|---|
| `particles` | [N, 350, 3] | (pT, η, φ), sorted by pT, with zeros in empty slots |
| `rows` | [N] | HDF5 row of each event |
| `labels` | [N] | Truth label |
| summary | [N] each | HT, multiplicity, leading pT and other event values |

The code calculates the summary values from the full 700-slot rows before the truncation to 350 particles. Thus,
the baselines use the full events.

The limit of 350 particles truncates 0.7% of background events and 0.02% of signal events. The pipeline stops if
more than 2% of the events of a set are truncated (`max_truncated_fraction`).

The file `cache_meta.json` records the split and the particle limit. If the configuration changes, the pipeline
stops. Then delete the cache or use a different `cache_dir`.

### 5.3 Model features

`build_event_features()` in `src/data/event_features.py` calculates 7 features for each particle:

| Feature | Definition |
|---|---|
| `log_pt` | log(pT) |
| `eta` | η |
| `phi` | φ |
| `log_e` | log(pT · cosh η) |
| `pt_frac` | pT / HT |
| `d_eta_lead` | η − η of the leading particle |
| `d_phi_lead` | φ − φ of the leading particle (periodic) |

The code standardizes these features with the mean and the standard deviation over all real particles of the
training set (`fit_feature_stats()`).

## 6. Data checks

The pipelines do these checks:

- The Part 1 split gives a warning if a quota is not full.
- `run_part1_eval.py` stops if the test set contains one class only.
- The Part 2 cache stops if the truncation is more than 2%.
- The Part 2 cache stops if the label of a row does not agree with the expected label of its set.
- The jet-feature loader stops if its labels do not agree with the event labels.
- Since commit d70e008, each Part 2 scores file contains the HDF5 row of each test event.

## 7. Quick reference

| Task | Command |
|---|---|
| Part 1 split | `python main.py prepare-part1 --config configs/part1_notebook_parity.yaml` |
| Part 1 graph chunks | `python scripts/process_h5.py --h5 <file> --out dataset/processed` (options: `--chunk-size`, `--radius`, `--wrap-phi`) |
| Part 2 cache | `python scripts/part2_run.py prepare --config configs/part2_main.yaml` |
| Change a value | `--set key=value`, for example `--set data.cache_dir=/tmp/cache` |
