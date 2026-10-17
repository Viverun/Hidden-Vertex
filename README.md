
---
# The Hidden Vertex 🕵
### *Unsupervised Discovery of Long-Lived Particles using Graph Autoencoders*

---

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23432779.svg)](https://doi.org/10.5281/zenodo.23432779)

## What this project is

The Hidden Vertex is a research project in machine learning for particle physics.
It tests if a graph autoencoder (GAE) can find unknown particles at the Large Hadron Collider (LHC)
without information about the signal.

The method is unsupervised anomaly detection:

1. Train a model on Standard Model background events only.
2. Use the model to give each new event an anomaly score.
3. Examine the events that have the highest scores.

The data is the LHC Olympics 2020 R&D dataset. The background is QCD dijet events. The signal is a W′ → XY → (qq̄)(qq̄) resonance
(m(W′) = 3.5 TeV, m(X) = 500 GeV, m(Y) = 100 GeV).

## Project status (October 2026)

| Part | Model | Status | Result |
|---|---|---|---|
| 1 | Node-level GAE | Complete | AUC 0.483 ± 0.005 (max error). See `docs/PART1_VERIFICATION.md`. |
| 2 | Event-level GAE | Complete | The pre-registered gate failed. See `docs/PART2_RESULTS.md`. |
| 3 | Jet-aware model with complexity control | Next | Not started. |

In Part 2, all three anomaly scores have an AUC of 0.40 to 0.44. An AUC below 0.5 means that the model
gives background events higher scores than signal events. A simple variable without labels,
the mean transverse momentum per particle (HT/N), gets an AUC of 0.811 on the same events.

Thus, the results are negative. The project gives a validated benchmark harness and a set of baselines.
It does not give a final anomaly detector.

## Publications and files

| Item | Location |
|---|---|
| Preprint | `paper/main.pdf` (LaTeX source in `paper/`) |
| Archive (DOI) | [10.5281/zenodo.23432779](https://doi.org/10.5281/zenodo.23432779) |
| Trained Part 2 models | [huggingface.co/Viverun/lhco-rnd-event-gae](https://huggingface.co/Viverun/lhco-rnd-event-gae) |

The Hugging Face repository has the Part 2 checkpoints for three seeds. It also has the anomaly score
and the HDF5 row number of each test event.

## Quick start

### 1. Part 1 notebook

1. Open `notebooks/lhc-anamoly-detector-part(1).ipynb`.
2. Run the cells in sequence.
3. Use a Kaggle runtime. This runtime has all the necessary packages.

The notebook shows the full Part 1 procedure: the data pipeline, the model, the oracle purification and
the ROC analysis.

### 2. Kaggle (optional)

1. Attach the dataset [kaggle.com/datasets/viveriun/lhc-collider](https://www.kaggle.com/datasets/viveriun/lhc-collider).
2. Import the model with `model_architecture.py` and `norm_stats.pt`.

### 3. Local installation

```bash
git clone https://github.com/Viverun/Hidden-Vertex.git
cd Hidden-Vertex
pip install -r requirements.txt      # PyTorch Geometric is necessary
python -m pytest tests               # run the test suite
```

### 4. Part 1 with the repository scripts

These commands do the same work as the notebook:

```bash
# 1) Make the oracle-purified training and test split
python main.py prepare-part1 --config configs/part1_notebook_parity.yaml

# 2) Train and evaluate the Part 1 baseline
python scripts/run_part1_eval.py --config configs/part1_notebook_parity.yaml
```

The scripts write these files:

| File | Contents |
|---|---|
| `results/models/part1_purified_baseline.pt` | Trained model |
| `results/scores/part1_metrics.json` | AUC, bootstrap 95% confidence interval, seed |
| `results/scores/part1_scores.npz` | Score and label of each event |
| `results/figures/part1_roc.png` | ROC curve |

To set the random seed, use `training.seed` in the configuration file or the option `--seed N`.
The script `run_part1_eval.py` uses the split that `prepare-part1` saved.

To calculate the baselines, correlations and ablations, run this command:

```bash
python scripts/part1_diagnostics.py --features dataset/raw/events_anomalydetection_v2.features.h5
```

### 5. Part 2

```bash
python scripts/part2_run.py prepare --config configs/part2_main.yaml        # make the event cache
python scripts/part2_sweep.py --stage main --config configs/part2_main.yaml  # train 3 seeds
```

The `bank` step calculates the baselines on the frozen test set. For the Stage 1 table, this step also
needs the Part 1 seed-0 scores:

1. Run `scripts/run_part1_eval.py`. It writes `scores_max` and `scores_mean`.
2. Give both arrays to `bank`:
   `--external "Part 1 AE (max)=<scores.npz>:scores_max" --external "Part 1 AE (mean)=<scores.npz>:scores_mean"`.
3. Make the report: `python scripts/part2_run.py report --tags main --name stage1`.

The full Kaggle sequence is in `scripts/kaggle/part2_stage0.py`.

## Part 1: node-level autoencoder

### Graph representation

The model does not change the sparse particle data into images. It uses a graph for each event:

* Nodes are particles. The node features are `log pT`, `η` and `φ`.
* Edges connect two particles if their distance ΔR is 0.4 or less.
  This value is the same as the radius of the anti-kt jet algorithm.

### Architecture

The question for Part 1 was: can a bottleneck at the particle level find anomalies at the event level?

```
Input graph (radius graph, ΔR ≤ 0.4)
        ↓
GCN layers (message passing)
        ↓
Bottleneck: 1D latent space for each particle
        ↓
Decoder: reconstruction of (log pT, η, φ)
```

### Results

The data is the LHC Olympics 2020 R&D dataset. The training set has background events only
(oracle purification).

| Run | AUC |
|---|---|
| Notebook (original shuffled `events_anomalydetection.h5`, 1 run without a seed) | 0.4615 |
| Repository pipeline (`events_anomalydetection_v2.h5`, seeds 0 to 4) | **0.483 ± 0.005** |

* The model has 420 effective trainable parameters. The notebook shows 429. Nine parameters are in an
  attention gate, and the model does not use the output of this gate.
* For the full verification, the baselines and the ablations, see
  [`docs/PART1_VERIFICATION.md`](docs/PART1_VERIFICATION.md).

### Main result

The maximum particle reconstruction error gives an AUC slightly below 0.5. Thus, the model gives the
QCD background slightly higher scores than the W′ signal.

The method that combines the particle errors has a large effect. The model size has almost no effect:

* With the mean error instead of the maximum, the same model gets an AUC of 0.78. This value is not
  better than the Σ pT baseline.
* A model that is 44 times larger gives the same result (capacity sweep, `docs/PART1_VERIFICATION.md` §5).

Thus, the reconstruction-error score fails. The particle information is not the problem. Simple event
variables from the same particle lists separate signal from background. Σ pT has an AUC of 0.79.
Fewer particles has 0.72. The jet variable −τ21 has 0.80.

### Analysis: what the score measures

Two ablations explain the result. For the details, see `docs/PART1_VERIFICATION.md`.

* The 1D bottleneck does not limit the model. The loss uses pT only, and one latent number can carry pT
  without loss. A per-particle MLP with the same bottleneck gets a mean squared error of 0.0001.
* All the reconstruction error comes from the GCN, which averages each particle with its neighbours
  (ΔR ≤ 0.4). Thus, the score measures the difference between the pT of a particle and the pT of its
  neighbours.
* The score follows the pT of the leading particle (Spearman 0.70). It does not follow the multiplicity
  (−0.07). Thus, the data does not support the earlier "multiplicity trap" explanation.

At a fixed leading-particle pT, the model reconstructs signal events slightly better than QCD events.
This causes the AUC below 0.5. This result was the reason for Part 2: a bottleneck at the event level
and inputs with jet and substructure information.

### Limits of Part 1

- Node-level reconstruction does not find event-level anomalies well.
- The model does not encode the invariant mass or the global event shape.
- The pT of the leading particle controls the maximum node error.
- With a loss that uses pT only, the 1D latent is not an information bottleneck.
- Part 1 is a baseline. It is not a final discovery architecture.

### Part 1 freeze checklist

- The notebook is the main Part 1 workflow for users.
- The notebook and the repository scripts give the same baseline.
- The repository has the oracle purification and the evaluation scripts.
- The Quick Start commands reproduce the Part 1 results.
- Record all new model changes in Part 2 or later.

## Part 2: event-level autoencoder

The Part 2 model is an event-level graph autoencoder. It uses dense padded tensors and a Chamfer loss.
It gives three scores:

1. The reconstruction error.
2. The distance to the nearest neighbours in the latent space (kNN).
3. The likelihood under a Gaussian mixture model in the latent space (GMM).

The work also made a benchmark harness that can evaluate any model. On Kaggle, the project trained
3 seeds, 18 ablation runs and 2 runs with signal contamination. The results were compared with baselines
that do not use labels.

The model did not pass its pre-registered gate. The best particle-level baseline, HT/N, has an AUC of
0.811. The model scores have an AUC of 0.40 to 0.44.

The signal events in this dataset have fewer particles than the QCD background, and their particles have
higher pT. The latent-density scores find these events more typical than background events. This agrees
with the AUC below 0.5. The cause of the inversion of the reconstruction-error score is not known.

For the full report, see [`docs/PART2_RESULTS.md`](docs/PART2_RESULTS.md). For the design, see
`docs/superpowers/specs/`.

## Part 3: plan

* Add jet-level inputs (jet masses, τ21, mjj) to the particle features.
* Control the score for multiplicity and mean pT. Then the score measures the event structure at a
  fixed complexity.

## Data

The repository does not include the raw HDF5 dataset because the file is large. Get the full dataset of
1.1 million events from the official Zenodo record of the LHC Olympics 2020 R&D dataset.

**Caution:** the two versions of the data file have different row orders.

* The notebook used the original file `events_anomalydetection.h5`. Its rows are shuffled, and
  approximately 9% of the events in each part of the file are signal.
* The public file `events_anomalydetection_v2.h5` is sorted. It is on Kaggle as
  `atulchikku/lhc-olympics-2020-ad-r-and-d`. Rows 0 to 999,999 are background, and rows 1,000,000 and
  higher are signal.

Thus, the parity configuration sets `scan_limit: null`. With this setting, the oracle split reads the
full file.

## Documents

| Document | Contents |
|---|---|
| `notebooks/` | Part 1 training notebook, step by step |
| `docs/OVERVIEW.md` | Project overview |
| `docs/DATA.md` | Dataset and data pipeline |
| `docs/PHYSICS.md` | Physics background |
| `docs/ARCHITECTURE.md` | Model architecture |
| `docs/TRAINING.md` | Training procedure |
| `docs/PART1_VERIFICATION.md` | Part 1 audit: verified numbers, data-file problems, capacity sweep |
| `docs/PART2_RESULTS.md` | Part 2 results: benchmark table, diagnosis, ablations, next steps |

## Citation

Cite the archived release:

```bibtex
@misc{khan2026gae,
  author = {Khan, Jamil},
  title  = {When Graph Autoencoders Look the Wrong Way: Complexity Bias and Simple Baselines in
            Unsupervised Anomaly Detection on the LHC Olympics 2020 Dataset},
  year   = {2026},
  doi    = {10.5281/zenodo.23432779},
  url    = {https://github.com/Viverun/Hidden-Vertex}
}
```

## Purpose

Most LHC searches look for a signal that a theory predicts. This project tests a different method:
learn the background, and then look for events that do not agree with it. The Part 1 and Part 2 results
show that this method needs strong baselines and controls for event complexity.
