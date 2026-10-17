# Architecture: The Hidden Vertex

This document describes the two models and the anomaly scores. Part 1 uses a node-level graph autoencoder.
Part 2 uses an event-level graph autoencoder.

## 1. Why a graph

A collision event is a set of particles. The number of particles changes from event to event, and the sequence
of the particles has no physical meaning. A graph is a good representation of this data:

- Each particle is a node.
- Edges connect particles that are near each other in (η, φ).
- A graph neural network gives the same result for any sequence of the nodes.

An image representation (for a CNN) puts the particles on a grid. Most pixels of the grid are empty.

## 2. Part 1: node-level autoencoder (`PhysicsAE`)

Code: `src/model/autoencoder.py`. Configuration: `configs/part1_notebook_parity.yaml`.

### 2.1 Structure

```
Input graph: nodes (log pT, η, φ), edges ΔR ≤ 0.4
        ↓
GCNConv 3 → 16, ReLU, dropout 0.1
        ↓
GCNConv 16 → 8, ReLU
        ↓
Linear 8 → 1          (latent value for each particle)
        ↓
Linear 1 → 8, ReLU, dropout 0.1
Linear 8 → 16, ReLU
Linear 16 → 3         (reconstructed log pT, η, φ for each particle)
```

The model has 420 trainable parameters. The notebook model also had an attention gate with 9 parameters, but
it did not use the output of the gate. The repository model does not have this gate.

### 2.2 Loss

`reconstruction_loss()` in `src/training/engine.py` has two modes:

| Mode | Definition |
|---|---|
| `pt_only` (Part 1 default) | Mean squared error of the standardized log pT only |
| `weighted` | Mean squared error of all three features with the weights [10, 1, 1] |

### 2.3 Anomaly score

`compute_anomaly_scores()` in `src/utils/metrics.py` calculates the reconstruction error of each particle and
combines the errors of an event:

| Reduction | Score | AUC (seed 0) |
|---|---|---|
| `max` (Part 1 default) | Largest particle error | 0.478 |
| `mean` | Mean particle error | 0.781 |

### 2.4 What the model learns

The two ablations in `PART1_VERIFICATION.md` show:

- With the `pt_only` loss, the 1D latent does not limit the model. A per-particle MLP with the same 1D bottleneck
  reconstructs pT almost perfectly (MSE 0.0001).
- All the reconstruction error comes from the GCN layers. Each GCN layer averages a particle with its
  neighbours. Thus, the model cannot reconstruct the pT of a particle that is very different from its neighbours.
- The maximum error follows the pT of the leading particle (Spearman 0.70).

## 3. Part 2: event-level autoencoder (`EventAE`)

Code: `src/model/event_ae.py`. Configuration: `configs/part2_main.yaml`.

### 3.1 Input

The input is a batch of padded events (`EventBatch` in `src/data/event_features.py`):

| Tensor | Shape | Contents |
|---|---|---|
| `feats` | [B, P, 7] | Standardized particle features (see `DATA.md`, Section 5.3) |
| `mask` | [B, P] | True for real particles |
| `adj` | [B, P, P] | True if ΔR ≤ 0.4 (periodic Δφ) |
| `pt_frac` | [B, P] | pT / HT of each particle |
| `target` | [B, 48, 4] | Decoder targets for the 48 particles with the highest pT |

P is 350.

### 3.2 Structure

```
Particle features [B, 350, 7]
        ↓
Linear 7 → 64 (embedding)
        ↓
3 × NeighbourLayer (width 64):
    h ← LayerNorm(h + GELU(W · [h, mean of the neighbours of h]))
        ↓
Pooling over the particles: mean, max and pT-weighted sum → 3 × 64 values
        ↓
Linear 192 → 128, GELU, Linear 128 → 16    (latent z, 16 values for each event)
        ↓
Linear 16 → 128, GELU, Linear 128 → 128, GELU, Linear 128 → 48 × 4
        ↓
48 reconstructed particles: (log pT and η scaled with fixed constants, cos φ, sin φ)
```

The model has 95,888 parameters.

### 3.3 Loss

The loss is the symmetric Chamfer distance (`chamfer_per_event()` in `src/training/set_loss.py`). It compares
the 48 predicted particles with the real target particles. The loss does not depend on the sequence of the
particles.

The Chamfer distance has two terms:

1. For each predicted particle: the squared distance to the nearest real target particle.
2. For each real target particle: the squared distance to the nearest predicted particle.

If an event has fewer than 48 particles, the mask removes the empty target slots. The option
`training.loss: chamfer_ptw` gives each target particle a weight equal to its pT in the second term.

### 3.4 Anomaly scores

The model gives three scores for each event. A higher score means a more anomalous event.

| Score | Definition | Code |
|---|---|---|
| Chamfer | The Chamfer reconstruction error of the event | `chamfer_per_event()` |
| kNN | Mean distance from z to the 20 nearest training latents | `KNNScorer` in `src/utils/latent_scores.py` |
| GMM | Negative log-likelihood of z under a Gaussian mixture model with 8 components | `GMMScorer` in `src/utils/latent_scores.py` |

The kNN and GMM scorers use the latents of the training events only. They standardize each latent dimension
with the mean and the standard deviation of the training latents. The Gaussian mixture uses full covariance
matrices, `reg_covar = 1e-4` and `random_state = seed`.

### 3.5 Results

All three scores have an AUC below 0.5 (Chamfer 0.428, kNN 0.401, GMM 0.436, mean of 3 seeds). The kNN and GMM
scores probably follow the particle multiplicity. For the full analysis, see `PART2_RESULTS.md`.

## 4. Hyperparameters

| Parameter | Part 1 | Part 2 |
|---|---|---|
| Input features | 3 (log pT, η, φ) | 7 |
| Graph | Radius graph, ΔR ≤ 0.4, Δφ not periodic | Dense adjacency, ΔR ≤ 0.4, Δφ periodic |
| Layers | 2 GCNConv (16, 8) | 3 NeighbourLayer (64) |
| Latent | 1 for each particle | 16 for each event |
| Decoder output | 3 for each particle | 48 particles × 4 |
| Loss | MSE of log pT | Symmetric Chamfer distance |
| Parameters | 420 | 95,888 |

## 5. Code reference

| Component | File |
|---|---|
| Part 1 model | `src/model/autoencoder.py` |
| Part 1 training and loss | `src/training/engine.py` |
| Part 1 scores and AUC | `src/utils/metrics.py` |
| Part 2 model | `src/model/event_ae.py` |
| Part 2 features and batches | `src/data/event_features.py`, `src/data/dense_graph.py` |
| Part 2 loss | `src/training/set_loss.py` |
| Part 2 training | `src/training/event_trainer.py` |
| Part 2 latent scorers | `src/utils/latent_scores.py` |
| Part 2 pipeline | `src/pipeline/part2.py` |
| Baselines and report | `src/benchmarks/baselines.py`, `src/benchmarks/report.py` |

The trained Part 2 checkpoints and a standalone copy of the model code are on
[Hugging Face](https://huggingface.co/Viverun/lhco-rnd-event-gae).
