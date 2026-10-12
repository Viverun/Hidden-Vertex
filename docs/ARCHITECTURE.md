# Architecture Guide - The Hidden Vertex

> **Note (2026-10):** Parts of this document describe the *planned* design (k-NN graphs, a 10D
> event-level latent, global pooling). The implemented Part 1 baseline uses ΔR < 0.4 radius graphs and a
> per-particle 1D latent (`configs/part1_notebook_parity.yaml`). For verified Part 1 numbers see
> [`PART1_VERIFICATION.md`](PART1_VERIFICATION.md).

> **Technical deep dive into the Graph Autoencoder architecture for unsupervised particle physics anomaly detection**

## Table of Contents
- [Overview](#overview)
- [Why Graph Neural Networks?](#why-graph-neural-networks)
- [Model Architecture](#model-architecture)
- [Mathematical Formulation](#mathematical-formulation)
- [Design Decisions](#design-decisions)
- [Loss Functions](#loss-functions)
- [Theoretical Foundations](#theoretical-foundations)

---

## Overview

The Hidden Vertex uses a **Graph Autoencoder (GAE)** to learn the manifold of Standard Model physics in an unsupervised manner. By compressing particle collision events through a narrow 10-dimensional bottleneck, the model is forced to learn the fundamental physical laws governing particle interactions.

### High-Level Architecture

```
Input: Particle Graph (100-300 nodes, 3 features each)
                    ↓
         ┌──────────────────────┐
         │   ENCODER (GNN)      │
         │  3 → 64 → 32 → 10    │  ← Compression forces learning
         └──────────────────────┘
                    ↓
         [10D Latent Vector]      ← Physics manifold
                    ↓
         ┌──────────────────────┐
         │   DECODER (MLP)      │
         │  10 → 64 → 3         │  ← Reconstruction
         └──────────────────────┘
                    ↓
  Reconstructed Graph (pT, η, φ)
                    ↓
         [Reconstruction Error]
                    ↓
  Low error = Standard Model ✅
  High error = Anomaly! 🚨
```

---

## Why Graph Neural Networks?

### The Problem with CNNs

Traditional approaches in particle physics use **Convolutional Neural Networks (CNNs)**, treating detector data as images:

```
Particle Data (Sparse)  →  Image Grid (Dense)  →  CNN
   [10-300 particles]        [64×64 pixels]
```

**Problems:**
1. **Sparsity waste:** Most pixels are empty (0 occupancy)
2. **Information loss:** Precise (η, φ) coordinates → grid quantization
3. **Fixed structure:** CNNs expect regular grids, particles form irregular graphs
4. **Computational waste:** 99% of operations on zeros

### The Graph Approach

**Particles are naturally graphs!**

```
Particles = Nodes (features: pT, η, φ)
   ↓
Build edges via k-NN in (η, φ) space
   ↓
Result: Graph G = (V, E, X)
```

**Advantages:**
1. **Sparsity-aware:** Only process actual particles
2. **Permutation invariant:** Order doesn't matter (like real physics!)
3. **Geometric:** Preserves precise spatial relationships
4. **Efficient:** ~100 nodes vs ~4096 pixels

---

## Model Architecture

### Full Architecture Diagram

```
Input Graph G = (V, E, X)
  V: Nodes (particles)
  E: Edges (k-NN relationships)  
  X: Node features [N, 3]
        ↓
┌─────────────────────────────────┐
│        ENCODER (GNN)            │
│                                 │
│  GCNConv Layer 1: 3 → 64        │ ← Message passing
│    • Aggregates neighbor info  │
│    • ReLU activation            │
│    • Dropout (0.1)              │
│                                 │
│  GCNConv Layer 2: 64 → 32       │ ← Higher-level features
│    • Global jet structure       │
│    • ReLU activation            │
│    • Dropout (0.1)              │
│                                 │
│  Global Mean Pooling            │ ← Graph-level embedding
│    [N, 32] → [1, 32]            │
│                                 │
│  MLP Encoder: 32 → 10           │ ← Bottleneck compression
│    • Linear + ReLU + Dropout    │
│    • Linear → latent vector     │
└─────────────────────────────────┘
        ↓
    [z ∈ ℝ¹⁰]  ← Latent space (physics manifold)
        ↓
┌─────────────────────────────────┐
│        DECODER (MLP)            │
│                                 │
│  MLP Layer 1: 10 → 32           │ ← Expand latent
│    • Linear + ReLU + Dropout    │
│                                 │
│  MLP Layer 2: 32 → 64           │ ← Intermediate
│    • Linear + ReLU              │
│                                 │
│  Broadcasting                   │ ← Replicate to all nodes
│    [1, 64] → [N, 64]            │
│                                 │
│  Node Reconstruction: 64 → 3    │ ← Predict (pT, η, φ)
│    • Linear (no activation)     │
└─────────────────────────────────┘
        ↓
    X̂ ∈ ℝᴺˣ³  ← Reconstructed features
        ↓
    Loss = MSE(X, X̂)
```

### Encoder: Graph → Latent

#### 1. Graph Convolution Layers

**GCNConv (Graph Convolutional Network):**

```python
class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(3, 64)
        self.conv2 = GCNConv(64, 32)
        self.fc_encode = nn.Sequential(
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 10)
        )
    
    def forward(self, x, edge_index, batch):
        # Layer 1: Learn local jet structure
        h = F.relu(self.conv1(x, edge_index))
        h = F.dropout(h, p=0.1, training=self.training)
        
        # Layer 2: Learn global momentum conservation
        h = F.relu(self.conv2(h, edge_index))
        h = F.dropout(h, p=0.1, training=self.training)
        
        # Global pooling: graph → vector
        h_graph = global_mean_pool(h, batch)  # [batch_size, 32]
        
        # Compress to latent space
        z = self.fc_encode(h_graph)  # [batch_size, 10]
        
        return z
```

**What GCNConv Does:**

For each node i:
```
hᵢ⁽ˡ⁺¹⁾ = σ(Σⱼ∈N(i) (1/√(dᵢdⱼ)) · W⁽ˡ⁾ · hⱼ⁽ˡ⁾)
```

Where:
- N(i): Neighbors of node i (via k-NN edges)
- dᵢ, dⱼ: Node degrees (normalization)
- W⁽ˡ⁾: Learnable weight matrix
- σ: Activation function (ReLU)

**Physical Interpretation:**
- Layer 1: Learns relationships between nearby particles (jet substructure)
- Layer 2: Learns global event topology (momentum balance)

#### 2. Global Pooling

Converts variable-size graphs → fixed-size vectors:

```python
h_graph = global_mean_pool(h, batch)
```

**Mean Pooling:**
```
h_graph = (1/N) Σᵢ₌₁ᴺ hᵢ
```

**Alternatives:**
- `global_max_pool`: Takes maximum activation
- `global_add_pool`: Sums activations
- Attention pooling: Weighted sum

**Why Mean?** Provides robust summary, invariant to particle ordering.

#### 3. Latent Bottleneck

**The Critical Compression:**

```python
z = self.fc_encode(h_graph)  # 32 → 10 dimensions
```

**Why 10 Dimensions?**

This is the **information bottleneck** that forces learning:

- **Too large (e.g., 100D):** Model memorizes, no generalization
- **Too small (e.g., 2D):** Cannot capture physics complexity
- **Just right (10D):** Forces model to learn:
  - Conservation laws (energy, momentum)
  - Jet topology patterns
  - Radiation structure
  - Fundamental symmetries

**Latent Space = Physics Manifold**

Background events cluster tightly in 10D space:
```
Standard Model → Smooth manifold in ℝ¹⁰
New Physics → Off-manifold → Cannot reconstruct
```

### Decoder: Latent → Graph

#### 1. Expansion

```python
class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc_decode = nn.Sequential(
            nn.Linear(10, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 64),
            nn.ReLU()
        )
        self.node_reconstructor = nn.Linear(64, 3)
    
    def forward(self, z, batch):
        # Expand latent code
        h = self.fc_decode(z)  # [batch_size, 64]
        
        # Broadcast to all nodes in each graph
        h_broadcast = h[batch]  # [total_nodes, 64]
        
        # Reconstruct node features
        x_recon = self.node_reconstructor(h_broadcast)  # [total_nodes, 3]
        
        return x_recon
```

#### 2. Broadcasting

**Key Challenge:** Latent vector is per-graph, but we need per-node predictions.

**Solution:** Replicate graph embedding to all nodes:

```python
# z: [batch_size, 10]
# batch: [total_nodes] - indicates which graph each node belongs to
# h[batch]: [total_nodes, 64] - broadcasts graph embedding to nodes
```

**Example:**
```
Graph 0: nodes [0, 1, 2]     → all get h[0]
Graph 1: nodes [3, 4, 5, 6]  → all get h[1]
```

#### 3. Node Feature Reconstruction

```python
x_recon = self.node_reconstructor(h_broadcast)
```

Predicts (pT, η, φ) for each particle:
- **No activation** on output (regression task)
- **Normalized features** (log(pT), η, φ)

---

## Mathematical Formulation

### Complete Forward Pass

Given input graph **G = (V, E, X)**:

**Encoding:**
```
H⁽¹⁾ = ReLU(GCN(X, E))           ∈ ℝᴺˣ⁶⁴
H⁽²⁾ = ReLU(GCN(H⁽¹⁾, E))        ∈ ℝᴺˣ³²
h_graph = (1/N) Σᵢ H⁽²⁾ᵢ          ∈ ℝ³²
z = MLP(h_graph)                  ∈ ℝ¹⁰
```

**Decoding:**
```
h_decoded = MLP(z)                ∈ ℝ⁶⁴
H_broadcast = h_decoded[batch]    ∈ ℝᴺˣ⁶⁴
X̂ = Linear(H_broadcast)           ∈ ℝᴺˣ³
```

**Loss:**
```
ℒ = MSE(X, X̂) = (1/N) Σᵢ ||Xᵢ - X̂ᵢ||²
```

### Anomaly Score

For a test graph G:
```
z = Encoder(G)
X̂ = Decoder(z)
score = ||X - X̂||² / N

if score > threshold:
    → Anomaly detected! 🚨
else:
    → Standard Model ✅
```

---

## Design Decisions

### 1. Why k-NN Graph Construction?

**Approach:** Build edges between k nearest neighbors in (η, φ) space.

```python
edge_index = knn_graph(x[:, 1:3], k=6, loop=False)
```

**Rationale:**
- **Physics-informed:** Particles close in detector space are causally related
- **k=6:** Captures immediate neighborhood without over-connecting
- **No self-loops:** Avoids trivial message passing

**Alternatives Considered:**
- **Radius graph:** Variable connectivity (unstable)
- **Fully connected:** O(N²) edges (expensive, noisy)
- **Distance-weighted:** Adds complexity without clear benefit

### 2. Why Global Mean Pooling?

**Options:**
```python
# Mean (chosen)
h = global_mean_pool(h_node, batch)

# Max
h = global_max_pool(h_node, batch)

# Sum
h = global_add_pool(h_node, batch)
```

**Why Mean?**
- **Size invariant:** Works for 10 or 300 particles
- **Robust:** Less sensitive to outliers than max
- **Physics:** Represents "average" event characteristics

### 3. Why 10D Latent Space?

**Empirical Testing:**

| Dimensions | Reconstruction | Separation | Training |
|------------|---------------|------------|----------|
| 2D | Poor | Good | Fast |
| 5D | Fair | Good | Fast |
| **10D** | **Good** | **Best** | **Fast** |
| 20D | Good | Moderate | Medium |
| 50D | Excellent | Poor | Slow |

**10D is the sweet spot:**
- Enough capacity to encode physics
- Small enough to force generalization
- Fast training convergence

### 4. Architecture Depth

**Why 2 GCN Layers?**

Tested 1-4 layers:
```
1 layer:  Local features only, poor performance
2 layers: ✅ Captures local + global structure
3 layers: Over-smoothing, diminishing returns
4 layers: Severe over-smoothing
```

**Over-smoothing:** Node features become too similar after many layers of aggregation.

### 5. Dropout Rate

**p = 0.1** chosen empirically:
- **p = 0:** Overfitting on training data
- **p = 0.1:** ✅ Good generalization
- **p = 0.3:** Underfitting, poor reconstruction

---

## Loss Functions

### Mean Squared Error (MSE)

**Current implementation:**

```python
loss = F.mse_loss(x_reconstructed, x_original)
```

**Formula:**
```
ℒ_MSE = (1/N) Σᵢ₌₁ᴺ Σⱼ₌₁³ (xᵢⱼ - x̂ᵢⱼ)²
```

**Pros:**
- Simple, differentiable
- Works well with our graph structure
- Fast to compute

**Cons:**
- Assumes fixed particle ordering
- Sensitive to outliers

### Chamfer Distance (Advanced)

**For permutation invariance:**

```python
def chamfer_distance(x, x_hat):
    """
    Permutation-invariant point cloud distance
    """
    # Distance matrix: [N, N]
    dist_matrix = torch.cdist(x, x_hat)
    
    # For each original particle, find nearest reconstruction
    min_orig = dist_matrix.min(dim=1)[0].mean()
    
    # For each reconstructed particle, find nearest original
    min_recon = dist_matrix.min(dim=0)[0].mean()
    
    # Symmetric Chamfer distance
    return min_orig + min_recon
```

**Formula:**
```
ℒ_CD = (1/N) Σᵢ min_ⱼ ||xᵢ - x̂ⱼ||² + (1/N) Σⱼ min_ᵢ ||xᵢ - x̂ⱼ||²
```

**When to use:**
- Particles are truly unordered
- Need strict permutation invariance
- Have computational budget (O(N²))

**Trade-off:** MSE works well in practice, Chamfer is more principled.

---

## Theoretical Foundations

### Manifold Learning Perspective

**Hypothesis:** Standard Model events lie on a low-dimensional manifold M ⊂ ℝᴺˣ³.

**Autoencoder Goal:** Learn mapping that:
1. Projects onto manifold: **Encoder(x) ∈ M**
2. Reconstructs from manifold: **Decoder(z) ≈ x** if **x ∈ M**

**Anomaly Detection:**
```
If x ∈ M:     Reconstruction succeeds (low error)
If x ∉ M:     Projection fails (high error)
```

### Information Bottleneck Theory

The 10D bottleneck enforces:
```
I(Z; X) ≤ 10 bits  (limited information flow)
```

**Result:** Model must compress X to its most informative features:
- Conservation laws
- Symmetries  
- Typical topologies

**Irrelevant details discarded:**
- Noise
- Pile-up
- Random fluctuations

### Comparison to Other Architectures

**vs. Variational Autoencoder (VAE):**
```
VAE: p(z|x) is Gaussian, samples from distribution
GAE: z = f(x) is deterministic, learns manifold directly
```
**Why GAE?** Simpler, no KL divergence tuning, works well for anomaly detection.

**vs. CNN Autoencoder:**
```
CNN: X → Image → CNN → Latent → CNN → Image → X
GAE: X → Graph → GNN → Latent → MLP → Graph → X
```
**Why GAE?** Sparsity-aware, preserves geometry, permutation invariant.

**vs. Transformer:**
```
Transformer: Attention over all particles (O(N²))
GNN: Message passing over edges (O(E))
```
**Why GNN?** More efficient, physics-informed structure.

---

## Implementation Details

### Input Normalization

**Critical for training stability:**

```python
# Raw features (WRONG - causes loss = 1500+)
x = [pT, η, φ]  # pT: 0.1-850, η: -5-5, φ: -π-π

# Normalized features (CORRECT)
x_norm = [
    torch.log(pT + 1e-8),  # log(pT): -2 to 7
    η,                      # η: -5 to 5 (already good)
    φ                       # φ: -π to π (already good)
]
```

**Why log(pT)?**
- pT has exponential distribution
- Log transform → roughly Gaussian
- All features on similar scale

### Edge Construction

```python
# Build k-NN graph in (η, φ) space
coords = x[:, 1:3]  # [η, φ]
edge_index = knn_graph(coords, k=6, loop=False)
```

**Result:**
- ~6 edges per node
- Sparse connectivity
- Physics-informed structure

### Batching Strategy

PyTorch Geometric batches by **concatenation**:

```python
# Batch of 3 graphs
Graph 0: 100 nodes → indices 0-99
Graph 1: 150 nodes → indices 100-249  
Graph 2: 80 nodes  → indices 250-329

# Batch tensor tracks assignment
batch = [0,0,...,0, 1,1,...,1, 2,2,...,2]
        └─ 100x ──┘ └─ 150x ──┘ └─ 80x ─┘
```

**Advantages:**
- Efficient GPU operations
- No padding needed
- Handles variable sizes naturally

---

## Extensions and Future Work

### 1. Attention Mechanisms

Add attention to capture importance:

```python
class AttentionPooling(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attention = nn.Linear(hidden_dim, 1)
    
    def forward(self, h, batch):
        # Compute attention weights
        weights = F.softmax(self.attention(h), dim=0)
        
        # Weighted sum per graph
        h_graph = scatter_add(weights * h, batch, dim=0)
        return h_graph
```

### 2. Hierarchical Encoding

Multi-scale latent space:

```python
# Different resolutions
z_coarse = encoder_coarse(x)   # 5D - global topology
z_fine = encoder_fine(x)       # 20D - detailed features

# Combine for reconstruction
x_recon = decoder([z_coarse, z_fine])
```

### 3. Contrastive Learning

Learn representations via similarity:

```python
# Pull similar events together
# Push dissimilar events apart
loss = contrastive_loss(z_anchor, z_positive, z_negative)
```

### 4. Graph Generation

Fully generative model:

```python
# Sample from latent space
z ~ N(0, I)

# Decode to graph
x, edges = decoder(z)

# Generate new physics events!
```

---

## Hyperparameter Summary

**Model Architecture:**
```yaml
encoder:
  conv1: 3 → 64
  conv2: 64 → 32
  latent: 32 → 10

decoder:
  expand: 10 → 32 → 64
  reconstruct: 64 → 3

pooling: global_mean
dropout: 0.1
```

**Training:**
```yaml
optimizer: AdamW
learning_rate: 1e-3
weight_decay: 1e-5
batch_size: 32
max_epochs: 50
early_stopping: 10
gradient_clip: 1.0
```

**Data:**
```yaml
graph_construction: knn
k_neighbors: 6
feature_norm: log(pT)
train_fraction: 0.7
val_fraction: 0.15
test_fraction: 0.15
```

---

## References

**Graph Neural Networks:**
- Kipf & Welling (2017): "Semi-Supervised Classification with Graph Convolutional Networks"
- Fey & Lenssen (2019): "Fast Graph Representation Learning with PyTorch Geometric"

**Autoencoders for Anomaly Detection:**
- Goodfellow et al. (2016): "Deep Learning" (Chapter 14)
- Chalapathy & Chawla (2019): "Deep Learning for Anomaly Detection"

**Particle Physics Applications:**
- Komiske et al. (2019): "Energy Flow Networks"
- Qu & Gouskos (2020): "ParticleNet"
- Kasieczka et al. (2021): "The LHC Olympics 2020"

---

## Code Reference

See implementation in:
- `src/model/autoencoder.py` - Main model
- `src/model/layers.py` - Custom GNN layers
- `src/data/preprocessing.py` - Graph construction
- `src/training/losses.py` - Loss functions

---

**Questions?** Open an issue or see [TRAINING.md](TRAINING.md) for practical usage.