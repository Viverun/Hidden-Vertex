
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

---

## 🚀 Quick Start

### 1. Kaggle Infrastructure (Recommended)

The easiest way to explore this project is via the **Kaggle Model Hub**.

- Attach Dataset: [https://www.kaggle.com/datasets/viveriun/lhc-collider](https://www.kaggle.com/datasets/viveriun/lhc-collider)
- Import Model: Use the provided `model_architecture.py` and `norm_stats.pt`

This provides a fully reproducible, dependency-safe environment.

---

### 2. Local Installation

```bash
# Clone repository
git clone https://github.com/Viverun/Hidden-Vertex.git
cd Hidden-Vertex

# Install dependencies (requires PyTorch Geometric)
pip install -r requirements.txt
````

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

* **Purified Baseline AUC**: 0.4615
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

## 📈 Roadmap

### ✅ Phase 1: Node-Level Baseline (Complete)

* ✓ Memory-safe HDF5 streaming pipeline
* ✓ Radius graph generation via `torch.cdist`
* ✓ Node-level 1D-bottleneck GNN autoencoder
* ✓ Oracle dataset purification

### 🔄 Phase 2: Global Event Learning (In Progress)

* ⏳ Global graph pooling (`global_mean_pool`) for invariant mass sensitivity
* ⏳ Energy-weighted attention to suppress detector noise
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

```

---