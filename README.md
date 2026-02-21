
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

### Project Status (Feb 2026)

- **Part 1 is complete** and frozen as a baseline milestone.
- The Part 1 conclusion is a **negative-but-useful result**: node-level bottlenecks did not separate event-level anomalies reliably.
- **Part 2 is next** and focuses on global event representation learning.

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
- `results/scores/part1_metrics.json`
- `results/figures/part1_roc.png`

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

* **Purified Baseline AUC**: ~0.46 (random-level discrimination)
* **Trainable Parameters**: 429

### Key Result

An AUC near **0.5** mathematically demonstrates a fundamental principle:

> You cannot detect an event-level anomaly by examining particles in isolation.

---

## 🔬 Physics Context

This project is built using the **LHC Olympics 2020 Anomaly Detection Challenge**
dataset. The objective is to identify **Beyond Standard Model (BSM)** physics
signals hidden within a dominant background of **Standard Model QCD** radiation.

### Note on Data

The full raw HDF5 dataset (~23GB) is intentionally not stored here to keep this
repository lightweight.

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

## 🔬 The Physics Autopsy: The *Multiplicity Trap*

To a node-level autoencoder, a high-momentum particle from a standard QCD jet
looks identical to one from a BSM decay.

Because this architecture is blind to **global event shape**, it responds
primarily to **particle multiplicity**, not anomalous physics.

This rigorously establishes a scientifically meaningful baseline and motivates
Phase 2.

---

## ⚠️ Known Limitations (Part 1)

- Node-level reconstruction is weak for event-level anomaly signatures.
- The model does not explicitly encode invariant mass or global event shape.
- Max-node error scoring can be biased by multiplicity effects.
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

### 🔄 Phase 2: Global Event Learning (In Progress)

* ⏳ Global graph pooling (`global_mean_pool`) for invariant mass sensitivity
* ⏳ Event-level latent bottleneck replacing per-node-only logic
* ⏳ Richer kinematics for resonance-aware learning
* ⏳ Latent manifold analysis and 2D visualization

---

## 📚 Documentation

| Resource   | Description                              |
| ---------- | ---------------------------------------- |
| Kaggle Hub | Access pretrained weights & architecture |
| Notebook   | Step-by-step training walkthrough        |

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
