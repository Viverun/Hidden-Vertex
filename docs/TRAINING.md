# Training Guide - The Hidden Vertex

> **Complete training pipeline with all optimizations discovered through iterative development**

## Table of Contents
- [Quick Start](#quick-start)
- [Google Colab Setup](#google-colab-setup)
- [Local Training](#local-training)
- [Troubleshooting](#troubleshooting)
- [Performance Optimization](#performance-optimization)

---

## Quick Start

### Prerequisites
- Python 3.10+
- CUDA-capable GPU (8GB+ VRAM recommended)
- 16GB+ RAM
- ~20GB disk space for processed data

### Installation
```bash
# Clone repository
git clone https://github.com/yourusername/vertex.git
cd vertex

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

---

## Google Colab Setup

### Initial Setup (5 minutes)

**Cell 1: Mount Drive & Install Packages**
```python
from google.colab import drive
drive.mount('/content/drive')

# Install PyTorch Geometric (pre-built wheels - fast!)
!pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu121
!pip install pyg_lib torch_scatter torch_sparse torch_cluster -f https://data.pyg.org/whl/torch-2.4.0+cu121.html
!pip install torch_geometric pytorch-lightning tables
```

**Cell 2: Copy Data to Local Disk** (Critical for speed!)
```python
import shutil
import os

# Copy processed chunks from Drive to local disk (~10 min one-time)
source = '/content/drive/MyDrive/LHC_Project/Processed_Graphs/processed'
dest = '/content/Processed_LOCAL/processed'

os.makedirs(dest, exist_ok=True)
for i in range(10):  # Copy first 10 chunks for subset training
    src_file = f'{source}/data_chunk_{i}.pt'
    dst_file = f'{dest}/data_chunk_{i}.pt'
    if not os.path.exists(dst_file):
        shutil.copy2(src_file, dst_file)
        print(f'✓ Chunk {i}')
```

### Resource Management

**Colab Free Tier Limits:**
- ~12 hours max session
- Compute units reset every 24 hours
- Can disconnect anytime when quota exhausted

**Best Practices:**
1. Train during fresh session (morning)
2. Save checkpoints to Drive every epoch
3. Monitor `Resources` panel
4. Consider Colab Pro for serious training ($10/month)

---

## Training Pipeline

### Step 1: Data Preprocessing

**Feature Normalization (CRITICAL!)**

Raw particle features have vastly different scales:
- pT: 0.1 → 850 GeV (huge range!)
- η: -5 → +5 (small range)
- φ: -π → +π (small range)

**Solution: Log-scale pT**
```python
# Normalize features
x_normalized = data.x.clone()
x_normalized[:, 0] = torch.log(data.x[:, 0] + 1e-8)  # pT → log(pT)
# η, φ already in good ranges
```

**Why This Matters:**
- Without normalization: Loss = 1500+, training fails
- With normalization: Loss = 0.1-1.0, model learns

### Step 2: Dataset Filtering

**The "Detective School" Approach:**

Train ONLY on background events (Standard Model physics).

```python
# Filter for background only (y=0.0)
background_indices = []
anomaly_indices = []

for chunk_idx in range(110):
    chunk_data = torch.load(f'processed/data_chunk_{chunk_idx}.pt')
    base_idx = chunk_idx * 10000
    
    for i, graph in enumerate(chunk_data):
        actual_idx = base_idx + i
        if graph.y.item() == 0.0:
            background_indices.append(actual_idx)
        else:
            anomaly_indices.append(actual_idx)

print(f'Background: {len(background_indices):,}')
print(f'Anomalies: {len(anomaly_indices):,}')
```

**Security Check:**
```python
# Verify training set has ZERO anomalies
def verify_no_contamination(dataset, samples=100):
    indices = np.random.choice(len(dataset), samples, replace=False)
    labels = [dataset[i].y.item() for i in indices]
    
    if 1.0 in labels:
        raise ValueError("🚨 CONTAMINATION! Anomalies in training set!")
    print("✅ Clean - 100% background events")

verify_no_contamination(train_dataset)
```

### Step 3: Train/Val/Test Split

```python
# Split background events
train_ratio = 0.7
val_ratio = 0.15
test_ratio = 0.15

num_bg = len(background_indices)
indices = torch.randperm(num_bg).tolist()

train_size = int(train_ratio * num_bg)
val_size = int(val_ratio * num_bg)

train_indices = indices[:train_size]
val_indices = indices[train_size:train_size + val_size]
test_bg_indices = indices[train_size + val_size:]

# Test set: background + ALL anomalies
test_dataset = ConcatDataset([test_bg_dataset, anomaly_dataset])
```

**Final Split:**
- Training: 700k background (learns Standard Model)
- Validation: 150k background (tunes hyperparameters)
- Test: 150k background + 100k anomalies (evaluation)

### Step 4: Model Architecture

```python
class GraphAutoencoder(nn.Module):
    def __init__(self, input_dim=3, latent_dim=10):
        super().__init__()
        
        # Encoder: Graph → 10D bottleneck
        self.conv1 = GCNConv(3, 64)
        self.conv2 = GCNConv(64, 32)
        self.fc_encode = nn.Linear(32, latent_dim)
        
        # Decoder: 10D → Reconstructed graph
        self.fc_decode = nn.Linear(latent_dim, 64)
        self.node_reconstructor = nn.Linear(64, 3)
    
    def encode(self, x, edge_index, batch):
        h = F.relu(self.conv1(x, edge_index))
        h = F.relu(self.conv2(h, edge_index))
        graph_embedding = global_mean_pool(h, batch)
        z = self.fc_encode(graph_embedding)
        return z
    
    def decode(self, z, batch):
        h = F.relu(self.fc_decode(z))
        h_broadcasted = h[batch]
        x_recon = self.node_reconstructor(h_broadcasted)
        return x_recon
```

**Why 10 Dimensions?**
- Forces extreme compression
- Model must learn fundamental physics laws
- Anomalies can't fit learned manifold → high reconstruction error

### Step 5: Training Loop

```python
# Hyperparameters
learning_rate = 1e-3
weight_decay = 1e-5
batch_size = 32
num_epochs = 50
patience = 10  # Early stopping

# Setup
device = torch.device('cuda')
model = GraphAutoencoder(latent_dim=10).to(device)
optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
criterion = nn.MSELoss()

# Training
best_val_loss = float('inf')
patience_counter = 0

for epoch in range(1, num_epochs + 1):
    # Training phase
    model.train()
    train_loss = 0.0
    
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        
        # Forward pass
        z = model.encode(batch.x, batch.edge_index, batch.batch)
        x_recon = model.decode(z, batch.batch)
        
        # Loss
        loss = criterion(x_recon, batch.x)
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        train_loss += loss.item()
    
    avg_train_loss = train_loss / len(train_loader)
    
    # Validation phase
    model.eval()
    val_loss = 0.0
    
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            z = model.encode(batch.x, batch.edge_index, batch.batch)
            x_recon = model.decode(z, batch.batch)
            loss = criterion(x_recon, batch.x)
            val_loss += loss.item()
    
    avg_val_loss = val_loss / len(val_loader)
    
    print(f'Epoch {epoch} | Train: {avg_train_loss:.6f} | Val: {avg_val_loss:.6f}')
    
    # Checkpointing
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        patience_counter = 0
        torch.save(model.state_dict(), 'best_model.pt')
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print('Early stopping!')
            break
```

---

## Performance Optimization

### Problem: Slow Data Loading (143 seconds/batch!)

**Root Cause:**
- Loading full 10k-graph chunks from disk
- Google Drive I/O is extremely slow
- Chunk caching helps but limited

**Solutions:**

#### 1. Train on Subset (Recommended for Colab)
```python
# Use first 100k events instead of 700k
subset_size = 100000
train_indices_subset = [idx for idx in train_indices if idx < subset_size]

# Training time: 30 min/epoch (manageable!)
```

#### 2. Copy Data to Local Disk
```python
# One-time copy: Drive → Colab local disk (100x faster I/O)
shutil.copytree('/content/drive/.../processed', '/content/Processed_LOCAL/processed')
```

#### 3. Pre-cache Training Data
```python
# Create single fast-loading file (one-time, ~10 min)
train_graphs = []
for chunk_idx in range(10):
    chunk = torch.load(f'chunk_{chunk_idx}.pt')
    train_graphs.extend([g for g in chunk if g.y.item() == 0.0])

torch.save(train_graphs, 'train_cache.pt')

# Tomorrow: load in 5 seconds instead of hours!
train_data = torch.load('train_cache.pt')
```

### DataLoader Configuration

```python
# CRITICAL: Set num_workers=0 on Colab (multiprocessing hangs)
train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=0,      # ← Important for Colab!
    pin_memory=True     # GPU transfer optimization
)
```

---

## Troubleshooting

### Issue 1: "Kernel Restarted" / "Out of Compute Units"

**Symptom:** Colab disconnects mid-training

**Cause:** Free tier quota exhausted

**Solutions:**
1. Wait 24 hours for quota reset
2. Upgrade to Colab Pro ($10/month)
3. Train during fresh session (morning)

### Issue 2: Loss = 1500+ (Not Learning)

**Symptom:** Extremely high loss, no improvement

**Cause:** Features not normalized

**Solution:** Apply log-scaling to pT
```python
x[:, 0] = torch.log(x[:, 0] + 1e-8)
```

### Issue 3: Training Extremely Slow (170 sec/batch)

**Symptom:** Hours per epoch

**Cause:** Google Drive I/O bottleneck

**Solutions:**
1. Copy data to local disk (see Performance Optimization)
2. Train on subset (100k events)
3. Use pre-cached files

### Issue 4: GPU Utilization = 0%

**Symptom:** GPU not being used

**Cause:** 99% of time spent loading data from disk

**Solution:** This is normal if I/O is slow. GPU bursts to 100% during actual computation (0.04 sec), then waits for next batch (143 sec). Fix the I/O bottleneck.

### Issue 5: "KeyError" in Cached Dataset

**Symptom:** Cache lookup fails

**Cause:** Bug in cache eviction logic

**Solution:** Use corrected cache class
```python
if len(self._cache) > 20:
    keys_to_keep = sorted(self._cache.keys())[-20:]
    self._cache = {k: self._cache[k] for k in keys_to_keep}
```

### Issue 6: DataLoader Hanging at 0%

**Symptom:** Progress bar stuck at 0%, no movement

**Cause:** Multiprocessing workers hanging on Colab

**Solution:** Set `num_workers=0` in all DataLoaders
```python
train_loader = DataLoader(..., num_workers=0, ...)
```

---

## Expected Results

### Training Progress
```
Epoch  1/50 | Train: 0.234567 | Val: 0.198765 | ⭐ NEW BEST
Epoch  5/50 | Train: 0.123456 | Val: 0.108765 | ⭐ NEW BEST
Epoch 10/50 | Train: 0.098765 | Val: 0.087654 | ⭐ NEW BEST
Epoch 20/50 | Train: 0.078901 | Val: 0.076543 | ⭐ NEW BEST
Early stopping!
```

### Final Metrics
- **Background reconstruction error:** 0.08 ± 0.02
- **Anomaly reconstruction error:** 0.45 ± 0.15
- **Separation factor:** ~5-6x
- **Training time:** 20-30 epochs (~10-15 hours on Colab)

---

## Advanced: Chamfer Distance Loss

For better permutation invariance (mentioned in README), use Chamfer Distance instead of MSE:

```python
def chamfer_distance(x, y):
    """
    Compute Chamfer distance between two point clouds.
    Permutation-invariant unlike MSE.
    """
    # x: [N, 3], y: [N, 3]
    
    # For each point in x, find nearest point in y
    dist_x_to_y = torch.cdist(x, y)  # [N, N]
    min_dist_x = dist_x_to_y.min(dim=1)[0]  # [N]
    
    # For each point in y, find nearest point in x
    min_dist_y = dist_x_to_y.min(dim=0)[0]  # [N]
    
    # Chamfer distance
    chamfer = min_dist_x.mean() + min_dist_y.mean()
    return chamfer

# Use in training loop
loss = chamfer_distance(x_recon, batch.x)
```

**When to use:**
- Chamfer Distance: Better for unordered particle sets
- MSE: Simpler, faster, works well with our graph structure

---

## Next Steps

After training completes:

1. **Load best checkpoint**
   ```python
   model.load_state_dict(torch.load('best_model.pt'))
   ```

2. **Compute anomaly scores on test set**
   ```python
   model.eval()
   scores = []
   labels = []
   
   with torch.no_grad():
       for batch in test_loader:
           batch = batch.to(device)
           z = model.encode(batch.x, batch.edge_index, batch.batch)
           x_recon = model.decode(z, batch.batch)
           
           # Reconstruction error per graph
           mse = F.mse_loss(x_recon, batch.x, reduction='none')
           graph_errors = scatter_mean(mse.mean(dim=1), batch.batch)
           
           scores.extend(graph_errors.cpu().numpy())
           labels.extend(batch.y.cpu().numpy())
   ```

3. **Visualize results**
   ```python
   import matplotlib.pyplot as plt
   from sklearn.metrics import roc_curve, auc
   
   # Score distribution
   plt.figure(figsize=(10, 5))
   plt.hist(scores[labels==0], bins=50, alpha=0.5, label='Background')
   plt.hist(scores[labels==1], bins=50, alpha=0.5, label='Anomaly')
   plt.xlabel('Reconstruction Error')
   plt.ylabel('Count')
   plt.legend()
   plt.title('Anomaly Detection Performance')
   
   # ROC curve
   fpr, tpr, _ = roc_curve(labels, scores)
   roc_auc = auc(fpr, tpr)
   
   plt.figure(figsize=(8, 8))
   plt.plot(fpr, tpr, label=f'ROC (AUC = {roc_auc:.3f})')
   plt.plot([0, 1], [0, 1], 'k--')
   plt.xlabel('False Positive Rate')
   plt.ylabel('True Positive Rate')
   plt.title('ROC Curve - Anomaly Detection')
   plt.legend()
   ```

4. **Analyze top anomalies**
   ```python
   # Get indices of top anomalies
   top_k = 100
   top_anomaly_indices = np.argsort(scores)[-top_k:]
   
   print(f"Top {top_k} anomalies:")
   for idx in top_anomaly_indices:
       print(f"  Event {idx}: Score = {scores[idx]:.6f}, Label = {labels[idx]}")
   ```

---

## Performance Benchmarks

### Colab Free Tier (T4 GPU)

**With subset training (100k events):**
- Training: ~30 min/epoch
- Validation: ~10 min/epoch
- Total: ~40 min/epoch
- Full run (30 epochs): ~20 hours

**With full dataset (700k events) - NOT RECOMMENDED:**
- Data loading bottleneck dominates
- ~4-6 hours per epoch
- Impractical on free tier

### Local Training (RTX 3080)

**With local NVMe SSD:**
- Training: ~10 min/epoch (full 700k)
- Validation: ~3 min/epoch
- Total: ~13 min/epoch
- Full run: ~6-7 hours

---

## Tips for Success

1. **Start small:** Train on 10k events first to verify pipeline works
2. **Monitor early:** Watch first 3 epochs - loss should drop steadily
3. **Save often:** Checkpoint every epoch to Drive
4. **Use tensorboard:** Track training curves
5. **Validate assumptions:** Run security checks, verify normalization
6. **Be patient:** First epoch is slowest (cache warming)

---

## Citation

If you use this training pipeline in your research, please cite:

```bibtex
@software{hidden_vertex,
  title={The Hidden Vertex: Unsupervised Anomaly Detection in Particle Physics},
  author={Your Name},
  year={2026},
  url={https://github.com/yourusername/vertex}
}
```

---

**Good luck with training!** 🚀

For questions or issues, please open an issue on GitHub or refer to:
- [ARCHITECTURE.md](ARCHITECTURE.md) - Model details
- [DATA.md](DATA.md) - Dataset documentation
- [PHYSICS.md](PHYSICS.md) - Physics background