"""
Part 1 diagnostics (see docs/PART1_VERIFICATION.md).

Needs a split saved by `main.py prepare-part1` and per-event scores saved by
`scripts/run_part1_eval.py` (evaluation.scores_path). Computes trivial event-level
baselines, score correlations, and the bottleneck ablations.

    python scripts/part1_diagnostics.py --split results/scores/part1_oracle_split.pt \
        --scores results/scores/part1_scores.npz \
        --features dataset/raw/events_anomalydetection_v2.features.h5
"""
import argparse, json, os, sys, time
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data.transforms import load_norm_stats
from src.model.autoencoder import PhysicsAE
from src.training.engine import train_one_epoch
from src.utils.metrics import compute_anomaly_scores

parser = argparse.ArgumentParser(description="Part 1 diagnostics")
parser.add_argument("--split", default="results/scores/part1_oracle_split.pt")
parser.add_argument("--scores", default="results/scores/part1_scores.npz")
parser.add_argument("--features", default=None, help="events_anomalydetection_v2.features.h5 (optional)")
parser.add_argument("--out", default="results/scores/part1_diagnostics.json")
parser.add_argument("--epochs", type=int, default=5)
args = parser.parse_args()

OUT = {}
def log(k, v):
    OUT[k] = v; print(f"[{k}] {v}", flush=True)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(OUT, open(args.out, "w"), indent=2, default=str)

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
split = torch.load(args.split, weights_only=False)
tr, te = split["train_graphs"], split["test_graphs"]
y = np.asarray(split["test_labels"]); te_idx = np.asarray(split["test_indices"])
log("split_counts", {"train": len(tr), "test_bg": int((y == 0).sum()), "test_sig": int((y == 1).sum()),
                     "sig_rows_min_max": [int(te_idx[y == 1].min()), int(te_idx[y == 1].max())]})

# ── Event-level baselines straight from the raw graphs ────────────────────────
def feats(g):
    pt, eta, phi = g.x[:, 0].numpy().astype(np.float64), g.x[:, 1].numpy(), g.x[:, 2].numpy()
    px, py, pz, E = pt * np.cos(phi), pt * np.sin(phi), pt * np.sinh(eta), pt * np.cosh(eta)
    m2 = E.sum() ** 2 - px.sum() ** 2 - py.sum() ** 2 - pz.sum() ** 2
    return dict(multiplicity=g.num_nodes, max_pt=pt.max(), HT=pt.sum(), m_all=np.sqrt(max(m2, 0)), n_edges=g.num_edges)
fe = pd.DataFrame([feats(g) for g in te])
log("baseline_auc", {c: round(roc_auc_score(y, fe[c]), 4) for c in fe.columns})
log("mean_bg_vs_sig", {c: [round(float(fe[c][y == 0].mean()), 2), round(float(fe[c][y == 1].mean()), 2)] for c in fe.columns})

if args.features:
    j = pd.read_hdf(args.features).iloc[te_idx]
    log("features_label_agree", float((j["label"].values.astype(int) == y).mean()))
    E1 = np.sqrt(j.mj1**2 + j.pxj1**2 + j.pyj1**2 + j.pzj1**2); E2 = np.sqrt(j.mj2**2 + j.pxj2**2 + j.pyj2**2 + j.pzj2**2)
    mjj = np.sqrt(np.maximum((E1 + E2)**2 - (j.pxj1 + j.pxj2)**2 - (j.pyj1 + j.pyj2)**2 - (j.pzj1 + j.pzj2)**2, 0))
    t21 = (j.tau2j1 / j.tau1j1.replace(0, np.nan)).fillna(1)
    log("baseline_auc_jet", {"mjj": round(roc_auc_score(y, mjj), 4), "-|mjj-3500|": round(roc_auc_score(y, -np.abs(mjj - 3500)), 4),
                             "mj1": round(roc_auc_score(y, j.mj1), 4), "mj2": round(roc_auc_score(y, j.mj2), 4),
                             "-tau21_j1": round(roc_auc_score(y, -t21), 4)})

# ── Score correlations (seed 0 of the repo parity run) ────────────────────────
sc = np.load(args.scores)["scores"]
log("seed0_auc_check", round(roc_auc_score(y, sc), 4))
for c in ["multiplicity", "max_pt", "HT"]:
    log(f"spearman(score,{c})", {"all": round(float(spearmanr(sc, fe[c]).correlation), 3),
                                 "bg": round(float(spearmanr(sc[y == 0], fe[c][y == 0]).correlation), 3),
                                 "sig": round(float(spearmanr(sc[y == 1], fe[c][y == 1]).correlation), 3)})
for c in ["multiplicity", "max_pt"]:
    q = pd.qcut(fe[c], 10, labels=False, duplicates="drop").values
    log(f"auc_within_{c}_deciles", [round(roc_auc_score(y[q == k], sc[q == k]), 3) for k in np.unique(q) if len(np.unique(y[q == k])) == 2])

# ── Ablations: is the "1D bottleneck" actually the constraint? ───────────────
mean, std = load_norm_stats(None, dev)
class MLPAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(3, 16), nn.ReLU(), nn.Linear(16, 8), nn.ReLU(), nn.Linear(8, 1))
        self.dec = nn.Sequential(nn.Linear(1, 8), nn.ReLU(), nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 3))
    def forward(self, d): return self.dec(self.enc(d.x))

def run(fn, trg, teg, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    m = fn().to(dev); opt = torch.optim.Adam(m.parameters(), lr=0.005)
    tl = DataLoader(trg, batch_size=256, shuffle=True)
    for _ in range(args.epochs):
        loss = train_one_epoch(m, tl, opt, dev, mean, std, loss_mode="pt_only")
    s = compute_anomaly_scores(m, DataLoader(teg, batch_size=256), dev, mean, std, reduction="max", score_features=[0])
    return round(loss, 5), round(roc_auc_score(y, s), 4)

noedge = lambda gs: [Data(x=g.x, edge_index=torch.empty((2, 0), dtype=torch.long)) for g in gs]
tr_ne, te_ne = noedge(tr), noedge(te)
for name, fn, trg, teg in [("mlp_no_graph", MLPAE, tr, te),
                           ("gcn_no_edges", lambda: PhysicsAE(latent_dim=1, hidden_layers=(16, 8)), tr_ne, te_ne)]:
    t = time.time(); r = [run(fn, trg, teg, s) for s in range(3)]
    log(f"ablation_{name}", {"final_loss": [a for a, _ in r], "auc_max": [b for _, b in r], "seconds": round(time.time() - t)})
print("DIAG DONE")
