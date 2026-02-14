# The Hidden Vertex 🕵️‍♂️⚛️
**Unsupervised Discovery of Long-Lived Particles using Graph Autoencoders**

![Python 3.11+](https://img.shields.io/badge/python-3.11+-3776AB?style=flat&logo=python&logoColor=white)
[![PyTorch Geometric](https://img.shields.io/badge/PyG-State--of--the--Art-red.svg)](https://pytorch-geometric.readthedocs.io/en/latest/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Dataset: LHC Olympics 2020](https://img.shields.io/badge/Dataset-LHC_Olympics_2020-orange)](https://zenodo.org/records/4536624)

## 📖 Executive Summary
This project addresses one of the most critical challenges in modern High-Energy Physics: the **"Lamppost Problem."** Traditional searches at the Large Hadron Collider (LHC) rely on supervised learning, which limits discovery to particles we already know to look for. 

**"The Hidden Vertex"** is a deep learning system designed to discover **New Physics** (such as Dark Matter or Long-Lived Particles) without prior knowledge of their specific signatures. By leveraging Geometric Deep Learning, specifically Graph Autoencoders (GAEs), the model learns the manifold of Standard Model physics and flags any events that deviate from this learned norm as anomalies. This approach allows for the model-independent discovery of rare physics events that would otherwise be discarded as noise or pile-up.

---

## 🔬 The Physics Challenge
* **The Problem:** The LHC produces 40 million collisions per second. Standard algorithms (Triggers) discard 99.998% of these based on pre-programmed rules. If "New Physics" particles behave unexpectedly—for example, if they travel a few millimeters before decaying (Long-Lived Particles/LLPs) or have unusual topological shapes—current algorithms delete them, mistaking them for background noise or pile-up.
* **The Goal:** To build a "General Purpose Anomaly Detector" that preserves these rare events without needing to be explicitly trained on what they look like.

---

## 🧠 The Solution: Geometric Deep Learning
Unlike traditional Computer Vision approaches that force sparse particle data into dense pixel grids (CNNs), this project treats particle jets as **Graphs**.

* **Data Representation:** Events are represented as Point Clouds in the detector space `(η, φ, pT)`.
* **Graph Construction:** Particles are treated as **Nodes**. Relationships between particles are encoded as **Edges** using a k-Nearest Neighbors (k-NN) dynamic graph construction. This preserves the precise geometric relationships crucial for identifying displaced vertices.

---

## ⚙️ Technical Architecture (The "Detective")
[Image of Graph Autoencoder architecture for particle physics]

The core model is a **Graph Autoencoder** trained in a purely **Unsupervised** manner.

1. **Input:** A graph of a "Jet" (a spray of particles).
2. **The Encoder (Compression):** A Graph Neural Network (GNN) compresses the complex, high-dimensional graph into a low-dimensional **Latent Vector**. This forces the model to learn the fundamental physical laws (conservation of momentum, typical radiation patterns) of the Standard Model.
3. **The Decoder (Reconstruction):** The network attempts to reconstruct the original graph from the latent vector.

### The Anomaly Logic
* **Standard Model Event:** The model recognizes the pattern and reconstructs it with **Low Error**.
* **New Physics Event:** The model fails to map the "weird" geometry to its learned latent space, resulting in **High Reconstruction Error**.
* **Result:** Events with the highest reconstruction error are flagged as discovery candidates.

[Image of ROC curve for anomaly detection]

---

## 🛠️ Implementation Details

### Dataset
This project relies on the **LHC Olympics 2020 Anomaly Detection Dataset**, the gold standard for model-independent search benchmarks.
* [Official Zenodo Dataset Link](https://zenodo.org/records/4536624)
* [R&D Zenodo Dataset Link](https://zenodo.org/records/4536377)

### Preprocessing & Loss
* **Uproot & Awkward Arrays:** For efficient handling of ragged, non-uniform particle lists (events with varying numbers of particles).
* **Loss Function:** **Chamfer Distance**. Unlike Mean Squared Error (MSE), Chamfer Distance is permutation-invariant, allowing the model to compare two clouds of points regardless of the order in which particles are listed.
* **Robustness:** The model is trained on background data containing typical experimental noise (pile-up), ensuring that the anomaly detector learns to ignore common noise and focuses only on topological anomalies.

---

## 🚀 Key Innovation & Impact
1. **Model Independence:** Unlike supervised classifiers, this system does not require simulated training data for the "New Physics" signal. It can find discoveries that theorists haven't even predicted yet.
2. **Geometric Fidelity:** By using GNNs instead of CNNs, the model handles the sparsity of detector data efficiently, avoiding the computational waste of processing "