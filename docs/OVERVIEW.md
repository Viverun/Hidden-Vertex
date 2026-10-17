# Overview - The Hidden Vertex

> **Note (2026-10):** Parts of this document describe the *planned* design (k-NN graphs, a 10D
> event-level latent, global pooling). The implemented Part 1 baseline uses ΔR < 0.4 radius graphs and a
> per-particle 1D latent (`configs/part1_notebook_parity.yaml`). For verified Part 1 numbers see
> [`PART1_VERIFICATION.md`](PART1_VERIFICATION.md).

> **A comprehensive introduction to unsupervised particle physics anomaly detection using Graph Neural Networks**

---

## 🎯 Executive Summary

**The Hidden Vertex** is a deep learning system designed to discover unknown particles at the Large Hadron Collider (LHC) without being told what to look for. By learning what "normal" Standard Model physics looks like, it can automatically flag unusual events that might indicate New Physics - such as Dark Matter, supersymmetry, or other beyond-Standard-Model phenomena.

**Key Innovation:** Instead of searching for specific predicted particles (supervised learning), our system learns the entire landscape of known physics and identifies anything that doesn't fit (unsupervised learning).

---

## 🔬 The Problem: The Lamppost Effect

### What We're Solving

Imagine searching for your lost keys under a lamppost because that's where the light is brightest - even though you lost them elsewhere. This is exactly what happens in particle physics today.

**Current Approach (Supervised Learning):**
```
Theorist predicts: "Dark Matter might look like X"
    ↓
Build detector looking for X
    ↓  
Search data for X
    ↓
Find X? ✅  or  Don't find X? ❌ (but miss Y and Z!)
```

**The Problem:**
- LHC produces **40 million collisions per second**
- Detectors can only save ~1,000 per second (0.0025%)
- **99.998% of data is immediately deleted** by pre-programmed filters
- If new physics doesn't match our predictions → **deleted forever**

**Example:** Long-Lived Particles (LLPs)
- Exotic particles that travel a few millimeters before decaying
- Standard triggers reject them as "noise" or "detector glitches"
- Could be Dark Matter candidates
- We've been throwing them away for decades! 🤦‍♂️

### The Solution: Learn Normal, Detect Anomalies

**Our Approach (Unsupervised Learning):**
```
Train AI on Standard Model physics ONLY
    ↓
AI learns what "normal" looks like
    ↓
Run AI on ALL collision data
    ↓
High reconstruction error = Something weird = Discovery candidate! 🎉
```

**Advantage:** Finds **any** deviation from known physics, not just predicted ones.

---

## 🔁 Notebook-to-Repository Mapping (Part 1)

The notebook is the **primary user workflow** for Part 1:

- Start with `notebooks/lhc-anamoly-detector-part(1).ipynb`
- Run cells sequentially for the full analysis narrative
- Use repository scripts when you need automation/reproducibility outside notebooks

The notebook `notebooks/lhc-anamoly-detector-part(1).ipynb` now maps directly to repository workflows:

- **Act 1 (Data pipeline):** `src/data/dataset.py` (`create_radius_graph`, `LHCH5Dataset`)
- **Act 2 (Strict bottleneck model):** `src/model/autoencoder.py` (`PhysicsAE` with configurable hidden layers/latent size)
- **Act 3 (Oracle purification):** `src/data/oracle_split.py` + `python main.py prepare-part1 --config configs/part1_notebook_parity.yaml`
- **Act 4 (ROC evaluation):** `python scripts/run_part1_eval.py --config configs/part1_notebook_parity.yaml`

Primary outputs:

- `results/models/part1_purified_baseline.pt`
- `results/scores/part1_metrics.json`
- `results/figures/part1_roc.png`

---

## 🧠 How It Works: The "Detective School" Analogy

### Training Phase: Teaching the Detective

**Goal:** Train AI to be an expert on "normal" particle collisions.

```
Input: 1,000,000 Standard Model collision events
    ↓
AI learns patterns:
  • Energy/momentum conservation
  • Typical jet structures
  • Radiation patterns
  • Detector signatures
    ↓
AI becomes expert on "boring" physics
```

**Key Constraint:** The AI must compress each event through a narrow **10-dimensional bottleneck**. This forces it to learn fundamental physics laws, not just memorize events.

```
Input: ~300 particles × 3 features = 900 numbers
    ↓
[Compression]
    ↓
Latent Code: 10 numbers  ← Must capture essence of physics!
    ↓
[Reconstruction]
    ↓
Output: ~300 particles × 3 features
```

If the AI can successfully reconstruct events through this narrow bottleneck, it has learned the underlying physics rules.

### Detection Phase: Spotting the Unusual

**Test on Unknown Data:**

```
Standard Model Event:
  Input → [Bottleneck] → Reconstruction
  Error: LOW ✅  (AI recognizes the pattern)
  
New Physics Event:
  Input → [Bottleneck] → Reconstruction
  Error: HIGH 🚨  (AI can't fit it through learned rules)
  
Conclusion: Flag high-error events for physicist review!
```

---

## 🏗️ Technical Approach

### Why Graphs?

Particle collisions are naturally **point clouds** - collections of particles with positions and momenta. Traditional approaches force this sparse data into dense image grids (CNNs), wasting computation on empty space.

**Our Approach: Graph Neural Networks**

```
Each Particle = Node
  Features: (pT, η, φ)
      ↓
Connect nearby particles = Edges
  (k-Nearest Neighbors in detector space)
      ↓
Result: Graph G = (V, E, X)
```

**Advantages:**
- ✅ **Sparse:** Only processes actual particles (~100-300 per event)
- ✅ **Geometric:** Preserves precise spatial relationships
- ✅ **Permutation invariant:** Order doesn't matter (like real physics!)
- ✅ **Efficient:** ~100 nodes vs ~64×64 = 4,096 pixels

### Architecture Overview

```
Particle Graph (100-300 nodes)
         ↓
┌────────────────────────┐
│  Graph Neural Network  │  ← Learns particle relationships
│  (2 GCN layers)        │
└────────────────────────┘
         ↓
┌────────────────────────┐
│  Global Pooling        │  ← Summarize entire event
└────────────────────────┘
         ↓
┌────────────────────────┐
│  10D Bottleneck        │  ← CRITICAL: Forces compression
└────────────────────────┘
         ↓
┌────────────────────────┐
│  Decoder Network       │  ← Reconstructs particles
└────────────────────────┘
         ↓
Reconstructed Graph
         ↓
Reconstruction Error = Anomaly Score
```

**The 10D Bottleneck is Key:**
- Too wide (100D): AI memorizes, doesn't generalize
- Too narrow (2D): Can't capture physics complexity  
- Just right (10D): Forces learning of fundamental laws ✅

---

## 📊 Dataset

### LHC Olympics 2020 Dataset

**Source:** Simulated LHC collision events from the LHC Olympics 2020 challenge

**Composition:**
- **1,100,000 total collision events**
- **1,000,000 background** (Standard Model physics)
- **100,000 anomalies** (Simulated New Physics signals)

**Event Structure:**
```
Each event:
  • Up to 700 particles
  • Each particle: (pT, η, φ)
    - pT: Transverse momentum (energy)
    - η: Pseudorapidity (angle from beam)
    - φ: Azimuthal angle (rotation around beam)
```

**Real-world Equivalent:**
- Background = QCD jets, W/Z bosons, top quarks (known physics)
- Anomalies = Long-Lived Particles, exotic decays (BSM physics)

### Data Processing Pipeline

```
Raw HDF5 File (events_anomalydetection.h5)
         ↓
[Step 1: Load & Parse]
  • Extract particle features
  • Remove zero-padding (pT > 0)
         ↓
[Step 2: Normalize]
  • Log-scale pT: log(pT) ∈ [-2, 7]
  • Keep η, φ as-is: [-5, 5], [-π, π]
         ↓
[Step 3: Build Graphs]
  • Construct k-NN edges (k=6)
  • Create PyTorch Geometric Data objects
         ↓
[Step 4: Split]
  • Training: 700k background (learns normal)
  • Validation: 150k background (tunes model)
  • Test: 150k background + 100k anomalies (evaluates)
         ↓
Preprocessed Graphs (110 chunk files, ~200MB total)
```

---

## 🎓 Training Strategy

### The Unsupervised Approach

**Critical Decision:** Train on background events ONLY.

```
Training Set (700k events):
  ✅ Standard Model physics
  ❌ NO anomalies
  
Why? Force the AI to learn "normal" so well that
anything abnormal sticks out!
```

**Security Check:**
```python
# Verify ZERO contamination
for event in random_sample(training_set, 100):
    assert event.label == 0.0  # background only
    
if any anomalies found:
    🚨 ABORT TRAINING! 🚨
```

### Training Process

**Objective:** Minimize reconstruction error on background events.

```python
for epoch in range(50):
    for batch in train_loader:
        # Forward pass
        latent_code = encoder(batch)      # Graph → 10D
        reconstructed = decoder(latent_code)  # 10D → Graph
        
        # Compute loss
        loss = MSE(reconstructed, original)
        
        # Update weights
        loss.backward()
        optimizer.step()
    
    # Check validation performance
    val_loss = evaluate(val_loader)
    
    # Save if improved
    if val_loss < best_loss:
        save_checkpoint(model)
```

**Expected Progress:**
```
Epoch  1: Loss = 0.234 (learning basic patterns)
Epoch  5: Loss = 0.123 (learning jet structure)
Epoch 10: Loss = 0.098 (learning conservation laws)
Epoch 20: Loss = 0.078 (convergence) ✅
```

**Early Stopping:** Halt if no improvement for 10 epochs.

---

## 🎯 Results & Performance

### Anomaly Separation

**Key Metric:** Reconstruction error behavior on an oracle-purified test set.

```
Background Events (Standard Model):
  Reconstructs similarly to signal under node-level constraints
  
Anomaly Events (New Physics):
  Also reconstructs similarly at event level with this architecture
  
Separation: weak (near-random discrimination)
```

**Interpretation:**
- Node-level compression captures local kinematics but misses event-level structure.
- Heavy-resonance signals are not reliably separated by this Part 1 baseline.
- This is a meaningful negative baseline that motivates Part 2.

### ROC Curve Performance

```
Purified baseline ROC-AUC: ~0.46
```

**Translation:** Performance is near random, supporting the Part 1 conclusion that node-level-only reconstruction is insufficient for event-level anomaly discovery.

### Computational Performance

**Training Time:**
- Dataset: 700k events
- Hardware: NVIDIA T4 GPU (Google Colab)
- Time: ~20 epochs × 40 min/epoch = **~13 hours**

**Inference Speed:**
- **~100 events/second** on GPU
- Can process entire test set (250k events) in ~40 minutes

**Scalability:**
- Handles variable-size graphs (10-700 particles)
- Memory efficient (batch processing)
- Baseline-ready for Part 2 architecture iteration

---

## 💡 Key Innovations

### 1. Geometric Deep Learning for Particle Physics

**First application of GNNs to unsupervised anomaly detection at the LHC.**

Traditional: CNNs on image grids  
Our approach: GNNs on sparse graphs ✨

### 2. Physics-Informed Architecture

**10D bottleneck forces learning of conservation laws:**
- Energy-momentum conservation
- Charge conservation  
- Symmetries (rotational, translational)

Not just pattern matching - actual physics understanding!

### 3. Model-Agnostic Discovery

**Finds ANY deviation from Standard Model:**
- Long-Lived Particles ✅
- Exotic decays ✅
- Unknown signatures ✅
- Detector anomalies ✅

No theoretical bias - pure data-driven discovery.

### 4. Production-Ready Pipeline

**End-to-end system:**
- Efficient preprocessing (chunked HDF5 → PyG graphs)
- Robust training (early stopping, checkpointing)
- Fast inference (100 events/sec)
- Interpretable results (reconstruction errors)

---

## 🛠️ Technology Stack

**Deep Learning:**
- PyTorch 2.4.0
- PyTorch Geometric (GNN library)
- PyTorch Lightning (training framework)

**Data Processing:**
- HDF5 / PyTables (large-scale storage)
- NumPy / Pandas (numerical computing)

**Visualization:**
- Matplotlib / Seaborn (plotting)
- TensorBoard (training monitoring)

**Development:**
- Python 3.10+
- Google Colab (cloud GPU training)
- Git / GitHub (version control)

---

## 📈 Use Cases

### 1. LHC Physics Discovery

**Primary Application:** Discover new particles/phenomena at the LHC.

**Workflow:**
```
Real LHC Data Stream
    ↓
Apply Hidden Vertex Model
    ↓
Flag high-error events (top 0.1%)
    ↓
Physicist review candidates
    ↓
Potential discovery! 🏆
```

### 2. Trigger System Enhancement

**Application:** Improve real-time event selection at detectors.

**Current:** Hard-coded rules (discard 99.998% of data)  
**Enhanced:** AI-guided selection (preserve anomalous events)

### 3. Data Quality Monitoring

**Application:** Detect detector malfunctions or calibration issues.

**Benefit:** AI flags unusual detector responses automatically.

### 4. Beyond Particle Physics

**Transferable to:**
- Astrophysics (gravitational wave anomalies)
- Medical imaging (tumor detection)
- Cybersecurity (network intrusion detection)
- Financial fraud detection

Any domain with:
- High-dimensional data
- Rare anomalies to detect
- Expensive labels

---

## 📚 Project Structure

```
vertex/
├── README.md                 ← Project overview
├── docs/
│   ├── OVERVIEW.md          ← This file
│   ├── ARCHITECTURE.md      ← Technical deep dive
│   ├── TRAINING.md          ← Training guide
│   ├── DATA.md              ← Dataset documentation
│   └── PHYSICS.md           ← Physics background
│
├── src/                      ← Source code
│   ├── model/               ← Graph Autoencoder
│   ├── data/                ← Dataset & preprocessing
│   ├── training/            ← Training pipeline
│   └── utils/               ← Utilities
│
├── notebooks/                ← Jupyter notebooks
│   ├── 01_data_exploration.ipynb
│   ├── 02_training.ipynb
│   └── 03_evaluation.ipynb
│
├── scripts/                  ← Standalone scripts
│   ├── preprocess_data.py   ← HDF5 → PyG graphs
│   ├── train_model.py       ← CLI training
│   └── evaluate_model.py    ← Compute anomaly scores
│
└── dataset/                  ← Data (gitignored)
    ├── raw/                  ← Original HDF5
    └── processed/            ← Graph chunks
```

---

## 🚀 Getting Started

### Quick Start (5 minutes)

```bash
# 1. Clone repository
git clone https://github.com/yourusername/vertex.git
cd vertex

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download dataset (see dataset/README.md)
# Place events_anomalydetection.h5 in dataset/raw/

# 4. Preprocess data
python scripts/preprocess_data.py

# 5. Train model
python scripts/train_model.py --epochs 50

# 6. Evaluate
python scripts/evaluate_model.py --checkpoint best_model.pt
```

### For Detailed Instructions

See documentation:
- **[TRAINING.md](TRAINING.md)** - Complete training walkthrough
- **[ARCHITECTURE.md](ARCHITECTURE.md)** - Model details
- **[DATA.md](DATA.md)** - Dataset setup

---

## 🎓 Learning Path

**New to the project? Follow this path:**

1. **Start here:** `README.md` (project summary)
2. **Big picture:** `docs/OVERVIEW.md` (this file)
3. **Physics context:** `docs/PHYSICS.md` (why this matters)
4. **Technical details:** `docs/ARCHITECTURE.md` (how it works)
5. **Hands-on:** `docs/TRAINING.md` (build it yourself)
6. **Data deep dive:** `docs/DATA.md` (dataset details)

**For researchers:**
→ Read PHYSICS.md + ARCHITECTURE.md

**For ML engineers:**
→ Read ARCHITECTURE.md + TRAINING.md

**For data scientists:**
→ Read DATA.md + notebooks/

---

## 🤝 Contributing

We welcome contributions! Areas of interest:

**Research:**
- Test on additional physics datasets
- Explore alternative architectures (Transformers, etc.)
- Investigate interpretability methods

**Engineering:**
- Optimize inference speed
- Add more preprocessing options
- Improve visualization tools

**Documentation:**
- Tutorial notebooks
- Video walkthroughs
- Blog posts

See `CONTRIBUTING.md` for guidelines.

---

## 📊 Roadmap

### Phase 1: Node-Level Baseline ✅ (Complete)

- [x] Radius-graph data pipeline
- [x] Oracle-purified baseline training/evaluation
- [x] Reproducible Part 1 repository workflow
- [x] Part 1 conclusion documented

### Phase 2: Event-Level Learning 🔄 (Current)

- [ ] Global event pooling and event-level latent bottleneck
- [ ] Invariant-mass-sensitive feature learning
- [ ] Improved reconstruction objective for whole-event structure
- [ ] Ablation studies against Part 1 baseline

### Phase 3: Scale-Up Validation

- [ ] Larger-scale training/evaluation sweeps
- [ ] Robustness checks across anomaly regimes
- [ ] Physics-driven validation and interpretability

---

## 📜 Citation

If you use this work in your research, please cite:

```bibtex
@software{hidden_vertex_2026,
  title={The Hidden Vertex: Unsupervised Discovery of Long-Lived Particles using Graph Autoencoders},
  author={Jamil Khan},
  year={2026},
  url={https://github.com/Viverun/Hidden-Vertex},
  note={Graph Neural Network-based anomaly detection for particle physics}
}
```

---

## 📞 Contact & Support

**Questions?**
- Open an issue on GitHub
- Email: your.email@example.com
- Twitter: @yourhandle

**Resources:**
- [Documentation](docs/)
- [Tutorials](notebooks/)
- [Discussions](https://github.com/yourusername/vertex/discussions)

---

## 🙏 Acknowledgments

**Data:**
- LHC Olympics 2020 organizers
- CERN Open Data Portal

**Software:**
- PyTorch Geometric team
- PyTorch developers
- Google Colab (free GPU access)

**Inspiration:**
- Graph Neural Network pioneers (Kipf, Welling, et al.)
- Particle physics ML community

---

## 📄 License

MIT License - see [LICENSE](../LICENSE) for details.

---

## 🌟 Impact

**Why This Matters:**

Discovery in particle physics has stagnated. The Higgs boson (2012) was the last major finding. The LHC has been running for over a decade since, and we've found... nothing new.

**But what if new physics is there, and we're just not looking in the right way?**

This project represents a paradigm shift:
- From "searching for what we predict" 
- To "discovering what we didn't expect"

**If even one new particle is discovered using this approach, it will revolutionize experimental physics.**

That's the goal. That's the dream. 🚀

---

**Ready to start?** Jump to [TRAINING.md](TRAINING.md) and build it yourself! 🎯