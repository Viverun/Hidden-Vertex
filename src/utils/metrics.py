import numpy as np
import torch
from sklearn.metrics import roc_auc_score, roc_curve
from torch_geometric.utils import scatter


def _extract_reconstruction(model_output):
    if isinstance(model_output, tuple):
        return model_output[0]
    return model_output


def compute_anomaly_scores(model, loader, device, gpu_mean, gpu_std, reduction='max'):
    """
    Calculates reconstruction error per graph. 
    Baseline AUC target: ~0.4065 (documented from initial experiments).
    """
    model.eval()
    all_scores = []
    
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            
            # 1. Preprocess as in training
            target = batch.x.clone()
            target[:, 0] = torch.log(target[:, 0].clamp(min=1e-8))
            target = (target - gpu_mean) / gpu_std
            
            # 2. Get Reconstruction
            out = _extract_reconstruction(model(batch))
            
            # 3. Node-level MSE
            node_mse = torch.mean((out - target)**2, dim=1)
            
            # 4. Graph-level score (The Max-Node "Sound the Alarm" strategy)
            # Find the single worst-reconstructed particle in each event
            graph_scores = scatter(node_mse, batch.batch, dim=0, reduce=reduction)
            
            all_scores.extend(graph_scores.cpu().numpy())
            
    return np.array(all_scores)

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