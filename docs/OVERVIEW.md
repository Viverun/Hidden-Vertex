# Overview: The Hidden Vertex

This document gives a summary of the project: the purpose, the method, the parts, the results and the files.
For the verified values, see [`PART1_VERIFICATION.md`](PART1_VERIFICATION.md) and
[`PART2_RESULTS.md`](PART2_RESULTS.md).

## 1. Purpose

The project tests one question: can a graph autoencoder (GAE) find a new physics signal in LHC collision data
without information about the signal?

Most LHC searches are supervised. A theory predicts a signal, and the search looks for that signal. If a new
particle does not agree with a prediction, a supervised search can miss it.

Unsupervised anomaly detection uses a different method:

1. Train a model on Standard Model background events only.
2. Use the model to give each event an anomaly score.
3. Examine the events that have the highest scores.

A good anomaly score gives signal events higher scores than background events. The AUC of the ROC curve
measures this. An AUC of 1.0 is a perfect separation. An AUC of 0.5 is equal to a random selection. An AUC
below 0.5 means that the score gives background events higher scores than signal events.

## 2. Data

The project uses the LHC Olympics 2020 R&D dataset:

- 1,000,000 QCD dijet background events.
- 100,000 signal events: W′ → XY → (qq̄)(qq̄), with m(W′) = 3.5 TeV, m(X) = 500 GeV and m(Y) = 100 GeV.
- Each event has up to 700 particles. Each particle has pT, η and φ.

For the details, see [`DATA.md`](DATA.md).

## 3. Parts of the project

| Part | Model | Question | Result |
|---|---|---|---|
| 1 | Node-level GAE: GCN layers and a 1D latent for each particle | Can a bottleneck at the particle level find anomalies at the event level? | AUC 0.483 ± 0.005 (maximum error, 5 seeds). The score follows the pT of the leading particle. |
| 2 | Event-level GAE: neighbour-averaging layers, pooling, a 16D latent for each event, Chamfer loss | Is a model with an event-level bottleneck better than simple baselines? | No. The three scores have an AUC of 0.40 to 0.44. The baseline HT/N gets 0.811. |
| 3 | Jet-aware model with complexity control | Is a score that ignores particle multiplicity useful? | Not started. |

The main result is negative. In this dataset, signal events have fewer particles than background events, and
their particles have higher pT. The autoencoder scores follow the event complexity. Thus, the model finds signal
events more typical than background events.

Simple variables that do not use labels separate the signal well:

| Variable | AUC |
|---|---|
| Mean pT per particle (HT/N) | 0.811 |
| −τ21 of the leading jet | 0.801 |
| HT (Σ pT) | 0.792 |
| −multiplicity | 0.715 |

Thus, an unsupervised score must be better than these baselines before it is useful.

## 4. How to start

1. Read the [README](../README.md) for the quick start commands.
2. For Part 1, open `notebooks/lhc-anamoly-detector-part(1).ipynb` and run the cells in sequence.
3. For Part 2, run the commands in Section "Part 2" of the README, or load the trained checkpoints from
   [Hugging Face](https://huggingface.co/Viverun/lhco-rnd-event-gae).

### Part 1: notebook and repository code

| Notebook section | Repository code |
|---|---|
| Act 1: data pipeline | `src/data/dataset.py` (`create_radius_graph`, `LHCH5Dataset`) |
| Act 2: model with a strict bottleneck | `src/model/autoencoder.py` (`PhysicsAE`) |
| Act 3: oracle purification | `src/data/oracle_split.py` and `python main.py prepare-part1 --config configs/part1_notebook_parity.yaml` |
| Act 4: ROC evaluation | `python scripts/run_part1_eval.py --config configs/part1_notebook_parity.yaml` |

## 5. Repository structure

```
Hidden-Vertex/
├── README.md                  Project summary and quick start
├── main.py                    Part 1 commands: prepare-part1, train-default
├── configs/                   YAML configuration files
├── docs/                      Documents (this directory)
├── notebooks/                 Part 1 notebook
├── paper/                     Preprint (LaTeX source and PDF)
├── scripts/
│   ├── process_h5.py          HDF5 file → radius graphs in chunks (Part 1)
│   ├── run_part1_eval.py      Part 1 training and evaluation
│   ├── part1_diagnostics.py   Part 1 baselines, correlations and ablations
│   ├── part1_capacity_sweep.py
│   ├── part2_run.py           Part 2: prepare, bank, train, report
│   ├── part2_sweep.py         Part 2: seeds and ablations
│   ├── kaggle_bundle.py       Makes a Kaggle script kernel from the repository
│   └── kaggle/                Kaggle kernel bodies for each stage
├── src/
│   ├── data/                  Data loading, graphs, oracle split, event cache
│   ├── model/                 PhysicsAE (Part 1), EventAE (Part 2)
│   ├── training/              Training loops and the Chamfer loss
│   ├── benchmarks/            Baselines and the benchmark report
│   ├── pipeline/              Part 2 pipeline
│   └── utils/                 Metrics and latent-space scorers
├── tests/                     Unit tests (pytest)
└── results/                   Outputs (most files are not in git)
```

## 6. Documents

| Document | Contents |
|---|---|
| [`PHYSICS.md`](PHYSICS.md) | Physics background: the Standard Model, the signal, triggers, unsupervised search |
| [`DATA.md`](DATA.md) | Dataset format, file versions, data pipelines for Part 1 and Part 2 |
| [`ARCHITECTURE.md`](ARCHITECTURE.md) | The Part 1 and Part 2 models and the anomaly scores |
| [`TRAINING.md`](TRAINING.md) | Training procedures, Kaggle runs, reproducibility |
| [`PART1_VERIFICATION.md`](PART1_VERIFICATION.md) | Part 1 audit and results |
| [`PART2_RESULTS.md`](PART2_RESULTS.md) | Part 2 results, diagnosis and ablations |

## 7. Citation and license

Cite the archived release: [doi:10.5281/zenodo.23432779](https://doi.org/10.5281/zenodo.23432779). The README
has a BibTeX entry.

The code has the MIT license. See [`LICENSE`](../LICENSE).

The LHC Olympics 2020 organizers made the dataset public. The computations used Kaggle GPU notebooks.
