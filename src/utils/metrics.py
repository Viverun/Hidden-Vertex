import numpy as np
import torch
from sklearn.metrics import roc_auc_score, roc_curve
from torch_geometric.utils import scatter

from src.data.transforms import standardize_batch


def _extract_reconstruction(model_output):
    if isinstance(model_output, tuple):
        return model_output[0]
    return model_output


def compute_anomaly_scores(model, loader, device, gpu_mean, gpu_std, reduction='max', score_features=None):
    """
    Calculates reconstruction error per graph.

    score_features: feature columns to score (None = all three; [0] = pT only, as in
    the Part 1 notebook). reduction: how node errors become an event score ('max' finds
    the single worst-reconstructed particle; 'mean' averages over particles).
    """
    model.eval()
    all_scores = []

    with torch.no_grad():
        for batch in loader:
            # 1. Preprocess exactly as in training (inputs and targets)
            batch = standardize_batch(batch.to(device), gpu_mean, gpu_std)
            target = batch.x

            # 2. Get Reconstruction
            out = _extract_reconstruction(model(batch))
            if score_features is not None:
                out = out[:, score_features]
                target = target[:, score_features]

            # 3. Node-level MSE
            node_mse = torch.mean((out - target) ** 2, dim=1)

            # 4. Graph-level score
            graph_scores = scatter(node_mse, batch.batch, dim=0, dim_size=batch.num_graphs, reduce=reduction)
            all_scores.append(graph_scores.cpu())

    if not all_scores:
        return np.array([], dtype=np.float32)
    return torch.cat(all_scores).numpy()

def get_performance_stats(bg_scores, sig_scores):
    """
    Calculates the ROC-AUC and separation factor.
    """
    y_true = np.concatenate([np.zeros(len(bg_scores)), np.ones(len(sig_scores))])
    y_scores = np.concatenate([bg_scores, sig_scores])

    auc = roc_auc_score(y_true, y_scores)

    # Separation: How much larger is the signal error than the background error?
    separation = np.mean(sig_scores) / np.mean(bg_scores)

    return auc, separation


def get_roc_curve(y_true, y_scores):
    return roc_curve(y_true, y_scores)


def bootstrap_auc_ci(y_true, y_scores, n_boot=1000, level=0.95, seed=0):
    """Percentile bootstrap confidence interval for ROC-AUC."""
    y_true = np.asarray(y_true)
    y_scores = np.asarray(y_scores)
    rng = np.random.default_rng(seed)
    aucs = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y_true), len(y_true))
        if y_true[idx].min() == y_true[idx].max():
            continue
        aucs.append(roc_auc_score(y_true[idx], y_scores[idx]))
    if not aucs:
        return float('nan'), float('nan')
    alpha = (1 - level) / 2
    return float(np.quantile(aucs, alpha)), float(np.quantile(aucs, 1 - alpha))
