
---
# 🏛️ The Hidden Vertex 🕵️‍♂️⚛️
### *Unsupervised Discovery of Long-Lived Particles using Graph Autoencoders*

---

## 🎯 What Is This?

**The Hidden Vertex** is a deep learning research project aimed at discovering
unknown particles at the **Large Hadron Collider (LHC)** without being told what
to look for.

### The Problem
The LHC trigger and filtering systems discard **99.998% of collision data** using
hard-coded, theory-driven rules.  
If new physics does not resemble our predefined expectations, it is lost
forever.

### The Solution
Train AI systems to learn **Standard Model (SM) physics so well** that anything
unusual automatically stands out — enabling **model-independent discovery** of
new physics.

### Project Status (Oct 2026)

- **Part 1 (node-level GAE) is complete**: negative-but-useful result; max-reduction AUC 0.483 ± 0.005 (see `docs/PART1_VERIFICATION.md`).
- **Part 2 (event-level GAE) is complete**: pre-registered gate **not passed**; all three anomaly scores are inverted (AUC 0.40–0.44), while the label-free mean pT per particle (HT/N) reaches **0.811** (see `docs/PART2_RESULTS.md`).
- **Preprint**: `paper/main.pdf` (LaTeX source in `paper/`).
- **Part 3** (jet-aware, complexity-controlled model) is next.

For this reason, Part 1 should be interpreted as a validated infrastructure + physics baseline, not as a final anomaly detector.

---

## 🚀 Quick Start

### 1. Notebook-First Experience (Recommended)

For most users, the notebook is the primary and best way to use this project.

- Open: `notebooks/lhc-anamoly-detector-part(1).ipynb`
- Follow cells in order to reproduce the full Part 1 story (data pipeline → model → oracle purification → ROC analysis)
- Use Kaggle runtime for easiest dependency compatibility

This is the **canonical user-facing workflow** for Part 1.

---

### 2. Kaggle Infrastructure (Optional Setup Path)

The easiest way to explore this project is via the **Kaggle Model Hub**.

- Attach Dataset: [https://www.kaggle.com/datasets/viveriun/lhc-collider](https://www.kaggle.com/datasets/viveriun/lhc-collider)
- Import Model: Use the provided `model_architecture.py` and `norm_stats.pt`

This provides a fully reproducible, dependency-safe environment.

---

### 3. Local Installation (Developer / Repro Support)

```bash
# Clone repository
git clone https://github.com/Viverun/Hidden-Vertex.git
cd Hidden-Vertex

# Install dependencies (requires PyTorch Geometric)
pip install -r requirements.txt

# Run the test suite
python -m pytest tests
````

### 4. Part 1 Notebook Parity (Repository Workflow)

To reproduce notebook logic via local scripts/modules:

```bash
# 1) Build oracle-purified train/test split
python main.py prepare-part1 --config configs/part1_notebook_parity.yaml

# 2) Train + evaluate Part 1 purified baseline
python scripts/run_part1_eval.py --config configs/part1_notebook_parity.yaml
```

Outputs are saved to:

- `results/models/part1_purified_baseline.pt`
- `results/scores/part1_metrics.json` (AUC, bootstrap 95% CI, seed)
- `results/scores/part1_scores.npz` (per-event scores and labels)
- `results/figures/part1_roc.png`

Runs are seeded (`training.seed`, or `--seed N`); `run_part1_eval.py` reuses the split saved by
`prepare-part1`. Optional diagnostics (baselines, correlations, ablations):

```bash
python scripts/part1_diagnostics.py --features dataset/raw/events_anomalydetection_v2.features.h5
```

---

## 🏗️ Technical Highlights (Part 1 Baseline)

### Why Graph Neural Networks?

Traditional ML approaches rasterize sparse particle data into dense images
(CNNs).
This project instead treats collisions as **geometric graphs**:

* **Particles → Nodes**
  Features: `log pT`, `η`, `φ`

* **Topology → Radius Graphs**
  Edges connect particles within physical detector space
  `ΔR < 0.4`, mimicking the anti-`k_t` jet clustering algorithm

---

### Architecture: Node-Level Strict Autoencoder

**Phase 1 Hypothesis:**
Can a node-level bottleneck detect event-level anomalies?

```
Input Graph (Radius Graph, ΔR < 0.4)
        ↓
GCN Layers (Message Passing)
        ↓
Bottleneck: 1D Latent Space (per particle)
        ↓
Decoder: Reconstruction of (log pT, η, φ)
```

---

## 📊 Results & Scientific Findings

The baseline was evaluated using the **LHC Olympics 2020 R&D dataset**.
An *Oracle purification* procedure ensured strictly unsupervised learning on
pure background events only.

### Metrics

| | AUC |
|---|---|
| Notebook (original shuffled `events_anomalydetection.h5`, 1 unseeded run) | 0.4615 |
| Repository pipeline (`events_anomalydetection_v2.h5`, seeds 0–4) | **0.483 ± 0.005** |

* **Trainable Parameters**: 420 effective (the notebook reports 429; 9 belong to an attention gate whose output is never used)
* Full verification, baselines and ablations: [`docs/PART1_VERIFICATION.md`](docs/PART1_VERIFICATION.md)

### Key Result

The node-level reconstruction error is slightly **below** random (a weak inverted signal):
it scores QCD background as marginally more anomalous than the W′ signal.

How node errors are combined matters more than model size: the same trained model reaches
**AUC 0.78 with a mean** over particles instead of the max, which is still no better than the Σ pT baseline,
and a 44× larger model changes nothing (capacity sweep, `docs/PART1_VERIFICATION.md` §5).

The max-error result is a failure of the *reconstruction-error score*, not of particle-level information:
simple event quantities built from the same particle lists separate well (Σ pT: AUC 0.79,
fewer particles: 0.72; jet-level −τ21: 0.80).

---

## 🔬 Physics Context

This project is built using the **LHC Olympics 2020 Anomaly Detection Challenge**
dataset. The objective is to identify **Beyond Standard Model (BSM)** physics
signals hidden within a dominant background of **Standard Model QCD** radiation.

### Note on Data

The full raw HDF5 dataset is intentionally not stored here to keep this
repository lightweight.

**Two file versions behave differently.** The notebook used the original shuffled
`events_anomalydetection.h5` (~9% signal throughout). The public
`events_anomalydetection_v2.h5` (e.g. Kaggle `atulchikku/lhc-olympics-2020-ad-r-and-d`) is **sorted**:
rows 0–999,999 are background and 1,000,000+ are signal. The parity config therefore uses
`scan_limit: null` so the oracle split scans the whole file.

Please refer to the official **Zenodo release** for the complete
**1.1 million event dataset**.

---

## 🧠 The Discovery Logic: Why a 1D Bottleneck?

This study employs a **Strict Node-Level Autoencoder**.

By forcing each particle through a **1D latent bottleneck**, we directly test the
hypothesis that BSM physics manifests as **irreducible particle-level
reconstruction errors**.

### Key Finding

Results from **Part 1** show that while node-level GNNs are computationally
efficient, detecting **heavy resonances** requires sensitivity to **global event
structure**, such as:

* Invariant mass distributions
* Multi-jet correlations

These effects are explored further in **Part 2**, which transitions toward
**event-level latent representations**.

---

## 🔬 The Physics Autopsy: What the Score Actually Measures

Two ablations explain the result (details in `docs/PART1_VERIFICATION.md`):

* **The 1D bottleneck does not constrain anything.** The loss only scores pT, and a 1D latent can carry
  pT losslessly: a per-particle MLP with the same bottleneck reaches MSE 0.0001.
* **All reconstruction error comes from GCN neighbourhood averaging** (ΔR < 0.4). The score measures how
  far a particle's pT is from its neighbours'. It tracks the **leading-particle pT** (Spearman 0.70),
  not multiplicity (−0.07), so the earlier "multiplicity trap" explanation is not supported.

At fixed leading-particle pT, signal events reconstruct slightly *better* than QCD, which produces the
sub-0.5 AUC. This motivates Phase 2: an event-level bottleneck and jet/substructure-aware inputs.

---

## ⚠️ Known Limitations (Part 1)

- Node-level reconstruction is weak for event-level anomaly signatures.
- The model does not explicitly encode invariant mass or global event shape.
- Max-node error scoring is dominated by the leading particle's pT.
- With a pT-only loss, the 1D latent is not an information bottleneck.
- Part 1 is a baseline/proof stage, not a final discovery architecture.

---

## ✅ Part 1 Freeze Checklist

- Notebook is the canonical user-facing Part 1 workflow.
- Notebook narrative and repository workflow are aligned for baseline reproduction.
- Oracle purification + evaluation scripts are available in repository code.
- Part 1 outputs are reproducible via the commands in Quick Start.
- Further model changes should be tracked under Part 2.

---

## 📈 Roadmap

### ✅ Phase 1: Node-Level Baseline (Complete)

* ✓ Memory-safe HDF5 streaming pipeline
* ✓ Radius graph generation via `torch.cdist`
* ✓ Node-level 1D-bottleneck GNN autoencoder
* ✓ Oracle dataset purification

### ✅ Phase 2: Global Event Learning (Complete, negative result)

An event-level graph autoencoder (dense padded tensors, Chamfer loss, three scores: reconstruction error,
latent kNN distance, latent Gaussian-mixture likelihood) was built together with a model-agnostic benchmark
harness, run on Kaggle (3 seeds, 18 ablation runs, 2 contamination runs) and evaluated against label-free
baselines. **It did not pass its pre-registered gate:** the best particle-level baseline, mean pT per particle,
scores AUC 0.811, while the model's scores are inverted (AUC 0.40-0.44), consistent with signal events here being
lower-multiplicity and harder than QCD background and looking *more* typical to the latent-density scores (the
cause of the reconstruction-error inversion is not established). Full write-up:
[`docs/PART2_RESULTS.md`](docs/PART2_RESULTS.md); design: `docs/superpowers/specs/`.

```bash
python scripts/part2_run.py prepare --config configs/part2_main.yaml     # event cache
python scripts/part2_sweep.py --stage main --config configs/part2_main.yaml   # 3 seeds
```

The `bank` step (baselines on the frozen test set) also needs the Part 1 seed-0 scores for the Stage 1 table to
match. `scripts/run_part1_eval.py` writes `scores_max` and `scores_mean`; pass both to `bank` via
`--external "Part 1 AE (max)=<scores.npz>:scores_max" --external "Part 1 AE (mean)=<scores.npz>:scores_mean"`,
then build the report with `python scripts/part2_run.py report --tags main --name stage1`. The exact Kaggle
sequence is in `scripts/kaggle/part2_stage0.py`.

### 🔜 Phase 3: Jet-Aware, Complexity-Controlled Learning (Next)

* ⏳ Jet-level inputs (jet masses, tau21, mjj) alongside particle features
* ⏳ Condition on / regress out multiplicity and mean pT so the score measures structure at fixed complexity

---

## 📚 Documentation

| Resource   | Description                              |
| ---------- | ---------------------------------------- |
| Kaggle Hub | Access pretrained weights & architecture |
| Notebook   | Step-by-step training walkthrough        |
| `docs/PART1_VERIFICATION.md` | Part 1 audit: verified numbers, data-file gotchas, capacity sweep |
| `docs/PART2_RESULTS.md` | Part 2 results: benchmark table, diagnosis, ablations, next steps |

---

## 📜 Citation

```bibtex
@software{hidden_vertex_2026,
  title  = {The Hidden Vertex: Unsupervised Discovery of Long-Lived Particles using Graph Autoencoders},
  author = {Jamil Khan},
  year   = {2026},
  url    = {https://github.com/Viverun/Hidden-Vertex},
  note   = {Graph Neural Network baseline for particle physics anomaly detection}
}
```

---

## 🌟 Why This Matters

Particle physics discovery has stagnated since the Higgs boson in 2012.

**The Hidden Vertex** shifts the paradigm from:

*“Searching for what we predict”*
to
*“Discovering what we didn’t expect.”*
