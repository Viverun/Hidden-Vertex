# Data Documentation - The Hidden Vertex

> **Complete guide to dataset structure, preprocessing pipeline, and graph construction**

## Table of Contents
- [Dataset Overview](#dataset-overview)
- [Data Format](#data-format)
- [Preprocessing Pipeline](#preprocessing-pipeline)
- [Graph Construction](#graph-construction)
- [Data Splits](#data-splits)
- [Storage & Access](#storage--access)

---

## Dataset Overview

### LHC Olympics 2020 Anomaly Detection Dataset

**Source:** [Zenodo - LHC Olympics 2020](https://zenodo.org/records/4536624)

**Purpose:** Benchmark dataset for model-independent anomaly detection in particle physics.

**Composition:**
```
Total Events: 1,100,000
├── Background: 1,000,000 (90.9%)  ← Standard Model QCD jets
└── Anomalies:    100,000 (9.1%)   ← Simulated BSM signals
```

**File:** `events_anomalydetection.h5` (~2.3 GB)

### Physics Context

**Background Events (y = 0.0):**
- Quantum Chromodynamics (QCD) dijet production
- Standard Model processes (W/Z bosons, top quarks)
- Typical LHC collision signatures
- "Boring" physics we understand well

**Anomaly Events (y = 1.0):**
- Simulated Long-Lived Particles (LLPs)
- Exotic decay topologies
- Displaced vertices (particles traveling before decay)
- Beyond Standard Model (BSM) signatures

**Why This Dataset?**
- Gold standard for unsupervised learning benchmarks
- Realistic detector-level simulation
- Controlled signal injection (known ground truth)
- Challenging separation (signal hidden in background)

---

## Data Format

### HDF5 File Structure

**Format:** Hierarchical Data Format version 5 (HDF5)

**Hierarchy:**
```
events_anomalydetection.h5
└── /df/
    └── block0_values  [1,100,000 × 2,101]
        ├── Rows: Events (1.1M)
        └── Columns: Features (2,101)
```

**Column Structure:**
```
Columns 0-2099:  Particle features (700 particles × 3 features)
Column 2100:     Label (0.0 = background, 1.0 = anomaly)
```

### Particle Features

Each event contains **up to 700 particles**, stored sequentially:

```
Event Structure:
[p0_pT, p0_eta, p0_phi, p1_pT, p1_eta, p1_phi, ..., p699_pT, p699_eta, p699_phi, label]
 └─────────────────┘  └──────────────────┘       └────────────────────┘
    Particle 0           Particle 1                  Particle 699
```

**Feature Definitions:**

| Feature | Symbol | Description | Range | Units |
|---------|--------|-------------|-------|-------|
| **Transverse Momentum** | pT | Energy perpendicular to beam | 0.1 - 850 | GeV |
| **Pseudorapidity** | η (eta) | Angle from beam axis | -5 to +5 | dimensionless |
| **Azimuthal Angle** | φ (phi) | Rotation around beam | -π to +π | radians |

**Physics Meaning:**
- **pT:** How much energy/momentum the particle has
- **η:** Where in the detector it hits (forward/backward)
- **φ:** Angular position around the beam pipe

### Zero-Padding

Most events have fewer than 700 particles. Unused slots are **zero-padded**:

```python
# Example event with 150 real particles:
particles[0:150]   = Real particle data (pT > 0)
particles[150:700] = Zero padding (0.0, 0.0, 0.0)
```

**Padding Pattern:**
```
Real particles: pT > 0 (typically 0.1 - 850 GeV)
Padded entries: pT = 0, η = 0, φ = 0  (exactly zero)
```

**Detection:**
```python
# Filter real particles
real_particles = particles[particles[:, 0] > 0]  # Keep only pT > 0
```

### Data Statistics

**Event-level Statistics:**

| Statistic | Background | Anomaly |
|-----------|-----------|---------|
| Mean particles/event | 175 ± 59 | 138 ± 37 |
| Min particles/event | 43 | 68 |
| Max particles/event | 409 | 304 |
| Padding fraction | ~75% | ~80% |

**Feature Distributions:**

**Background Events:**
```
pT:  Mean = 4.2 GeV,  Std = 12.8,  Range = [0.17, 847]
η:   Mean = 0.0,      Std = 2.1,   Range = [-4.97, 4.99]
φ:   Mean = 0.0,      Std = 1.8,   Range = [-3.14, 3.14]
```

**Anomaly Events:**
```
pT:  Mean = 3.8 GeV,  Std = 10.2,  Range = [0.39, 457]
η:   Mean = 0.0,      Std = 1.9,   Range = [-4.95, 4.98]
φ:   Mean = 0.0,      Std = 1.8,   Range = [-3.14, 3.14]
```

---

## Preprocessing Pipeline

### Step 1: Load HDF5 Data

**Using PyTables:**

```python
import tables
import numpy as np

# Open HDF5 file
h5_file = tables.open_file('events_anomalydetection.h5', mode='r')
dataset = h5_file.get_node('/df/block0_values')

# Dataset shape
print(dataset.shape)  # (1100000, 2101)

# Load one event
event_data = dataset[0]  # [2101]
particles = event_data[:-1].reshape(700, 3)  # [700, 3]
label = event_data[-1]  # scalar
```

**Chunked Loading** (memory-efficient):

```python
# Process in chunks of 10,000 events
chunk_size = 10000

for start_idx in range(0, 1_100_000, chunk_size):
    end_idx = min(start_idx + chunk_size, 1_100_000)
    
    # Load chunk
    chunk = dataset[start_idx:end_idx]
    
    # Process chunk
    for event in chunk:
        # ... process event ...
```

### Step 2: Remove Zero-Padding

**Extract Real Particles:**

```python
def extract_particles(event_data):
    """
    Extract real particles from zero-padded event.
    
    Args:
        event_data: [2101] array from HDF5
        
    Returns:
        particles: [N, 3] array of real particles
        label: scalar (0.0 or 1.0)
    """
    # Split features and label
    features = event_data[:-1]
    label = event_data[-1]
    
    # Reshape to particles
    particles = features.reshape(700, 3)
    
    # Filter real particles (pT > 0)
    real_particles = particles[particles[:, 0] > 0]
    
    return real_particles, label
```

**Example:**
```python
event_data = dataset[0]
particles, label = extract_particles(event_data)

print(f"Original: 700 slots")
print(f"Real particles: {len(particles)}")
print(f"Label: {label}")

# Output:
# Original: 700 slots
# Real particles: 173
# Label: 0.0
```

### Step 3: Feature Normalization

**Critical for Training Stability!**

**Problem:** Features have vastly different scales:
```
pT:  0.1 → 850   (factor of 8,500!)
η:   -5 → +5     (range of 10)
φ:   -π → +π     (range of 6.28)
```

**Solution:** Log-scale pT

```python
def normalize_features(particles):
    """
    Normalize particle features.
    
    Args:
        particles: [N, 3] array (pT, η, φ)
        
    Returns:
        normalized: [N, 3] array with log(pT), η, φ
    """
    normalized = particles.copy()
    
    # Log-transform pT (compresses exponential distribution)
    normalized[:, 0] = np.log(particles[:, 0] + 1e-8)
    
    # η and φ already in good ranges
    # (leave unchanged)
    
    return normalized
```

**Effect:**
```
Before: pT ∈ [0.1, 850]     → Range of 850
After:  log(pT) ∈ [-2, 7]   → Range of 9

All features now on comparable scales!
```

**Why This Matters:**
- Without normalization: Loss = 1500+, training fails ❌
- With normalization: Loss = 0.1-1.0, model learns ✅

### Step 4: Build Graph Structure

**k-Nearest Neighbors in (η, φ) Space:**

```python
from torch_cluster import knn_graph
import torch

def build_graph(particles):
    """
    Construct k-NN graph from particle coordinates.
    
    Args:
        particles: [N, 3] tensor (log(pT), η, φ)
        
    Returns:
        edge_index: [2, E] tensor of edges
    """
    # Extract spatial coordinates
    coords = particles[:, 1:3]  # [N, 2] (η, φ)
    
    # Build k-NN graph (k=6 neighbors)
    edge_index = knn_graph(coords, k=6, loop=False)
    
    return edge_index
```

**Example:**
```python
particles = torch.tensor([
    [1.2, 0.5, 1.0],  # Particle 0
    [1.5, 0.6, 1.1],  # Particle 1 (near 0)
    [2.0, 2.0, 0.0],  # Particle 2 (far)
    [1.3, 0.4, 0.9],  # Particle 3 (near 0)
])

edge_index = build_graph(particles)
print(edge_index)

# Output (example):
# tensor([[0, 0, 1, 1, 2, 2],  ← source nodes
#         [1, 3, 0, 3, 0, 1]]) ← target nodes
# 
# Edges: 0→1, 0→3, 1→0, 1→3, 2→0, 2→1
```

**Why k=6?**
- Too small (k=2): Disconnected graph, poor message passing
- Just right (k=6): Captures local neighborhood ✅
- Too large (k=20): Over-connected, noise in messages

### Step 5: Create PyTorch Geometric Data Object

```python
from torch_geometric.data import Data

def create_graph_data(particles, label, edge_index):
    """
    Create PyTorch Geometric Data object.
    
    Args:
        particles: [N, 3] tensor (features)
        label: scalar tensor
        edge_index: [2, E] tensor (edges)
        
    Returns:
        data: PyG Data object
    """
    data = Data(
        x=particles,              # Node features [N, 3]
        edge_index=edge_index,    # Edge connectivity [2, E]
        y=torch.tensor([label])   # Graph label [1]
    )
    
    return data
```

**Complete Preprocessing Function:**

```python
def preprocess_event(event_data):
    """
    Full preprocessing pipeline: HDF5 → PyG Data.
    """
    # Step 1: Extract particles
    particles, label = extract_particles(event_data)
    
    if len(particles) == 0:
        return None  # Skip empty events
    
    # Step 2: Normalize features
    particles_norm = normalize_features(particles)
    particles_tensor = torch.tensor(particles_norm, dtype=torch.float32)
    
    # Step 3: Build graph
    edge_index = build_graph(particles_tensor)
    
    # Step 4: Create Data object
    data = create_graph_data(particles_tensor, label, edge_index)
    
    return data
```

---

## Graph Construction

### Why Graph Neural Networks?

**Particles form natural point clouds:**
- Variable number of particles (43-409 per event)
- Sparse structure (~100 particles in 4π detector)
- Geometric relationships matter (nearby particles correlated)

**Graph representation captures this naturally:**
```
Nodes (V): Particles
Edges (E): Physical relationships
Features (X): Particle properties (pT, η, φ)
```

### k-NN Graph Algorithm

**Algorithm:**
```
For each particle i:
  1. Compute distances to all other particles in (η, φ) space
  2. Select k=6 nearest neighbors
  3. Create edges: i → neighbor_j for j in top-k
  4. Optionally make bidirectional: i ↔ j
```

**Distance Metric:**
```
Distance in (η, φ) space:
d(i, j) = √[(ηᵢ - ηⱼ)² + (φᵢ - φⱼ)²]
```

**Edge Construction:**
```python
# Compute pairwise distances
from torch_geometric.nn import knn_graph

coords = particles[:, 1:3]  # [N, 2] - (η, φ) coordinates
edge_index = knn_graph(
    coords, 
    k=6,           # Number of neighbors
    loop=False,    # No self-loops
    flow='source_to_target'
)
```

### Graph Properties

**Typical Event:**
```
Nodes (particles): 175
Edges:            ~1,050  (6 per node)
Edge density:     ~0.07%  (sparse!)
Average degree:   6
Max degree:       6
```

**Graph Structure:**
```
Dense region (jet core):
  • Many particles close together
  • High connectivity
  • Rich message passing

Sparse region (periphery):
  • Few particles
  • Lower connectivity
  • Limited messages
```

### Why Not Fully Connected?

**Fully connected graph:**
```
Edges = N × (N-1) / 2
For N=175: E = 15,225 edges  (14x more than k-NN!)
```

**Problems:**
- Computational cost: O(N²) vs O(N·k)
- Noise: Distant particles shouldn't interact
- Physics: Causality limits particle correlations

**k-NN graph advantages:**
- Efficient: O(N log N) construction
- Physics-informed: Connects nearby particles
- Scalable: Works for N=10 or N=300

---

## Data Splits

### Split Strategy

**Objective:** Train on background, test on both.

```
Total Dataset: 1,100,000 events
  ├── Background: 1,000,000 (90.9%)
  │   ├── Training:   700,000 (70%)  ← Learn Standard Model
  │   ├── Validation: 150,000 (15%)  ← Tune hyperparameters
  │   └── Test:       150,000 (15%)  ← Evaluate on background
  │
  └── Anomalies: 100,000 (9.1%)
      └── Test: 100,000 (100%)       ← Evaluate on anomalies
```

**Final Splits:**
```
Training:   700,000 background only
Validation: 150,000 background only  
Test:       250,000 mixed (150k BG + 100k anomaly)
```

### Implementation

```python
import numpy as np
from torch.utils.data import Subset, ConcatDataset

# Set random seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# 1. Filter background and anomaly indices
background_indices = [i for i, label in enumerate(labels) if label == 0.0]
anomaly_indices = [i for i, label in enumerate(labels) if label == 1.0]

# 2. Shuffle background indices
np.random.shuffle(background_indices)

# 3. Split background
num_bg = len(background_indices)
train_end = int(0.70 * num_bg)
val_end = int(0.85 * num_bg)

train_idx = background_indices[:train_end]
val_idx = background_indices[train_end:val_end]
test_bg_idx = background_indices[val_end:]

# 4. Create subsets
train_dataset = Subset(full_dataset, train_idx)
val_dataset = Subset(full_dataset, val_idx)
test_bg_dataset = Subset(full_dataset, test_bg_idx)
anomaly_dataset = Subset(full_dataset, anomaly_indices)

# 5. Combine test sets
test_dataset = ConcatDataset([test_bg_dataset, anomaly_dataset])
```

### Split Verification

**Security Check:**
```python
def verify_split_purity(dataset, name, expected_label=0.0):
    """Verify dataset contains only expected labels."""
    sample_size = min(100, len(dataset))
    samples = np.random.choice(len(dataset), sample_size, replace=False)
    
    labels = [dataset[i].y.item() for i in samples]
    
    contamination = sum(1 for l in labels if l != expected_label)
    
    print(f"{name}: {len(dataset):,} events")
    print(f"  Sampled: {sample_size}")
    print(f"  Contamination: {contamination}")
    
    if contamination > 0:
        raise ValueError(f"❌ CONTAMINATION in {name}!")
    print(f"  ✅ CLEAN")

# Verify
verify_split_purity(train_dataset, "Training", expected_label=0.0)
verify_split_purity(val_dataset, "Validation", expected_label=0.0)

# Test set should have both
print(f"Test: {len(test_dataset):,} events (mixed)")
```

**Expected Output:**
```
Training: 700,000 events
  Sampled: 100
  Contamination: 0
  ✅ CLEAN

Validation: 150,000 events
  Sampled: 100
  Contamination: 0
  ✅ CLEAN

Test: 250,000 events (mixed)
```

---

## Storage & Access

### Preprocessed Data Structure

**Organization:**
```
dataset/
├── raw/
│   └── events_anomalydetection.h5  (2.3 GB)
│
└── processed/
    ├── data_chunk_0.pt   (110 chunks)
    ├── data_chunk_1.pt
    ├── ...
    └── data_chunk_109.pt
    
    Total: ~200 MB (10,000 graphs per chunk)
```

**Chunk Files:**
```python
# Each chunk contains a list of PyG Data objects
chunk = torch.load('processed/data_chunk_0.pt')

print(type(chunk))        # list
print(len(chunk))         # 10,000
print(type(chunk[0]))     # torch_geometric.data.Data
print(chunk[0])
# Data(x=[N, 3], edge_index=[2, E], y=[1])
```

### Dataset Class

**Custom PyTorch Geometric Dataset:**

```python
from torch_geometric.data import Dataset
import os

class LHCAnomalyDataset(Dataset):
    def __init__(self, root, h5_file_path):
        self.h5_file_path = h5_file_path
        self.chunk_size = 10000
        
        # Get total events
        with tables.open_file(h5_file_path, 'r') as f:
            self.total_events = f.get_node('/df/block0_values').shape[0]
        
        super().__init__(root)
    
    @property
    def processed_file_names(self):
        num_chunks = (self.total_events + self.chunk_size - 1) // self.chunk_size
        return [f'data_chunk_{i}.pt' for i in range(num_chunks)]
    
    def process(self):
        """Preprocess HDF5 → PyG graphs (one-time operation)."""
        with tables.open_file(self.h5_file_path, 'r') as f:
            dataset = f.get_node('/df/block0_values')
            
            num_chunks = len(self.processed_file_names)
            
            for chunk_idx in range(num_chunks):
                start = chunk_idx * self.chunk_size
                end = min(start + self.chunk_size, self.total_events)
                
                chunk_graphs = []
                
                for event_data in dataset[start:end]:
                    graph = preprocess_event(event_data)
                    if graph is not None:
                        chunk_graphs.append(graph)
                
                # Save chunk
                save_path = os.path.join(self.processed_dir, 
                                        f'data_chunk_{chunk_idx}.pt')
                torch.save(chunk_graphs, save_path)
    
    def len(self):
        return self.total_events
    
    def get(self, idx):
        """Load single graph."""
        chunk_idx = idx // self.chunk_size
        idx_in_chunk = idx % self.chunk_size
        
        chunk_path = os.path.join(self.processed_dir, 
                                  f'data_chunk_{chunk_idx}.pt')
        chunk = torch.load(chunk_path)
        
        return chunk[idx_in_chunk]
```

### DataLoader

**PyTorch Geometric DataLoader:**

```python
from torch_geometric.loader import DataLoader

# Create loaders
train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=0,  # Important for Colab!
    pin_memory=True
)
```

**Batching Mechanism:**

PyG batches graphs by **concatenation**:
```python
# Batch of 3 graphs:
# Graph 0: 100 nodes
# Graph 1: 150 nodes  
# Graph 2: 80 nodes

batch.x:          [330, 3]      # All node features concatenated
batch.edge_index: [2, ~2000]    # All edges (with offset indices)
batch.batch:      [330]         # Assigns nodes to graphs
                                # [0,0,...,0,1,1,...,1,2,2,...,2]
batch.y:          [3]           # Graph labels
```

**Efficient Access:**
```python
for batch in train_loader:
    # batch.num_graphs = 32
    # batch.num_nodes = ~5,600 (175 * 32)
    # batch.num_edges = ~33,600 (1050 * 32)
    
    # Process batch on GPU
    batch = batch.to('cuda')
    predictions = model(batch)
```

---

## Data Quality Checks

### Preprocessing Validation

**Run these checks after preprocessing:**

```python
# 1. Check all chunks exist
import os
processed_dir = 'dataset/processed'
expected_chunks = 110

actual_chunks = len([f for f in os.listdir(processed_dir) 
                     if f.startswith('data_chunk_')])

assert actual_chunks == expected_chunks, f"Missing chunks! {actual_chunks}/110"
print(f"✅ All {expected_chunks} chunks found")

# 2. Check chunk sizes
for i in range(expected_chunks):
    chunk = torch.load(f'{processed_dir}/data_chunk_{i}.pt')
    expected_size = 10000 if i < 109 else (1_100_000 % 10000)
    
    assert len(chunk) == expected_size, f"Chunk {i} wrong size!"

print("✅ All chunks have correct size")

# 3. Check label distribution
all_labels = []
for i in range(110):
    chunk = torch.load(f'{processed_dir}/data_chunk_{i}.pt')
    all_labels.extend([g.y.item() for g in chunk])

bg_count = sum(1 for l in all_labels if l == 0.0)
anom_count = sum(1 for l in all_labels if l == 1.0)

print(f"✅ Labels: {bg_count:,} BG, {anom_count:,} Anomaly")
assert bg_count + anom_count == 1_100_000

# 4. Check graph structure
sample_graph = torch.load(f'{processed_dir}/data_chunk_0.pt')[0]

assert hasattr(sample_graph, 'x'), "Missing node features!"
assert hasattr(sample_graph, 'edge_index'), "Missing edges!"
assert hasattr(sample_graph, 'y'), "Missing label!"

print(f"✅ Graph structure valid")
print(f"   Sample: {sample_graph}")
```

---

## Troubleshooting

### Common Issues

**1. "HDF5 file not found"**
```
Solution: Download dataset from Zenodo and place in dataset/raw/
```

**2. "Out of memory during preprocessing"**
```
Solution: Reduce chunk_size from 10,000 to 5,000
```

**3. "Graphs have no edges"**
```
Cause: k-NN couldn't find neighbors (very rare)
Solution: Skip events with 0 edges or use k=1
```

**4. "Inconsistent normalization"**
```
Cause: Forgot to log-scale pT
Solution: Always apply normalize_features() before creating graph
```

**5. "DataLoader hangs"**
```
Cause: num_workers > 0 on Colab
Solution: Set num_workers=0
```

---

## Best Practices

### Data Handling

1. **Always set random seeds** for reproducible splits
2. **Verify split purity** (no anomalies in training)
3. **Normalize features** before graph construction
4. **Check for empty graphs** and skip them
5. **Use chunked loading** for large datasets

### Performance

1. **Preprocess once, train many times** (save preprocessed graphs)
2. **Use DataLoader** with appropriate batch_size
3. **Pin memory** for GPU transfer speedup
4. **Monitor I/O bottlenecks** (copy data to local disk if slow)

### Reproducibility

```python
# Always set seeds at the start
import random
import numpy as np
import torch

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
```

---

## References

**Dataset:**
- Kasieczka et al. (2021): "The LHC Olympics 2020: A Community Challenge for Anomaly Detection in High Energy Physics"
- Zenodo DOI: 10.5281/zenodo.4536624

**File Formats:**
- HDF5: https://www.hdfgroup.org/solutions/hdf5/
- PyTorch Geometric: https://pytorch-geometric.readthedocs.io/

**Related Documentation:**
- [ARCHITECTURE.md](ARCHITECTURE.md) - Graph construction details
- [TRAINING.md](TRAINING.md) - Using the preprocessed data
- [PHYSICS.md](PHYSICS.md) - Physics context

---

## Quick Reference

### Key Files
```
dataset/raw/events_anomalydetection.h5     # Original data (2.3 GB)
dataset/processed/data_chunk_*.pt          # Preprocessed (200 MB)
src/data/preprocessing.py                  # Preprocessing code
scripts/preprocess_data.py                 # CLI preprocessing
```

### Key Functions
```python
extract_particles(event_data)              # Remove padding
normalize_features(particles)              # Log-scale pT
build_graph(particles)                     # k-NN construction
preprocess_event(event_data)               # Full pipeline
```

### Key Parameters
```python
k = 6                                      # k-NN neighbors
chunk_size = 10000                         # Events per chunk
batch_size = 32                            # Graphs per batch
num_workers = 0                            # DataLoader workers (Colab)
```

---

**Questions?** See [TRAINING.md](TRAINING.md) for usage examples or open an issue on GitHub.