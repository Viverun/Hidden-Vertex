import argparse
import json
import os
import random
import sys

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from sklearn.metrics import roc_auc_score, roc_curve
from torch_geometric.loader import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.oracle_split import build_oracle_purified_split
from src.data.transforms import load_norm_stats
from src.model.autoencoder import PhysicsAE
from src.training.engine import train_one_epoch
from src.utils.metrics import bootstrap_auc_ci, compute_anomaly_scores


def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as handle:
        return yaml.safe_load(handle)


def ensure_parent_dir(file_path):
    parent = os.path.dirname(file_path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_or_build_split(config):
    """Reuses the split saved by `main.py prepare-part1` when present, else builds it."""
    split_path = config['data'].get('split_path')
    if split_path and os.path.exists(split_path):
        print(f'Loading cached oracle split: {split_path}')
        split = torch.load(split_path, map_location='cpu', weights_only=False)
        return split['train_graphs'], split['test_graphs'], split['test_labels']

    print('Building oracle-purified split...')
    return build_oracle_purified_split(
        h5_path=config['data']['h5_path'],
        train_target=config['oracle']['train_target'],
        test_bg_target=config['oracle']['test_bg_target'],
        test_sig_target=config['oracle']['test_sig_target'],
        scan_limit=config['data']['scan_limit'],
        radius=config['data']['radius'],
        wrap_phi=config['data'].get('wrap_phi', False),
    )


def save_outputs(model, config, auc, auc_ci, fpr, tpr, y_scores, test_labels, extra=None):
    model_path = config['evaluation']['model_path']
    metrics_path = config['evaluation']['metrics_path']
    roc_path = config['evaluation']['roc_path']

    ensure_parent_dir(model_path)
    ensure_parent_dir(metrics_path)
    ensure_parent_dir(roc_path)

    torch.save({'model_state_dict': model.state_dict()}, model_path)

    num_signal = int(np.sum(test_labels))
    metrics = {
        'auc': float(auc),
        'auc_ci95': list(auc_ci),
        'seed': config['training'].get('seed'),
        'score_reduction': config['evaluation']['score_reduction'],
        'num_test_events': int(len(test_labels)),
        'num_signal': num_signal,
        'num_background': int(len(test_labels) - num_signal),
        'mean_score': float(np.mean(y_scores)),
        'std_score': float(np.std(y_scores)),
    }
    metrics.update(extra or {})
    with open(metrics_path, 'w', encoding='utf-8') as handle:
        json.dump(metrics, handle, indent=2)

    scores_path = config['evaluation'].get('scores_path')
    if scores_path:
        ensure_parent_dir(scores_path)
        np.savez(scores_path, scores=np.asarray(y_scores), labels=np.asarray(test_labels))
        print(f'✓ Saved per-event scores: {scores_path}')

    plt.figure(figsize=(10, 6))
    plt.plot(fpr, tpr, color='indigo', lw=2.5, label=f'Part 1 Baseline (AUC = {auc:.4f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--', alpha=0.6, label='Random (AUC = 0.50)')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Part 1 Purified ROC-AUC')
    plt.legend(loc='lower right')
    plt.grid(alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig(roc_path, dpi=200)
    plt.close()

    print(f'✓ Saved model: {model_path}')
    print(f'✓ Saved metrics: {metrics_path}')
    print(f'✓ Saved ROC: {roc_path}')


def main():
    parser = argparse.ArgumentParser(description='Run Part 1 purified baseline evaluation.')
    parser.add_argument('--config', default='configs/part1_notebook_parity.yaml')
    parser.add_argument('--seed', type=int, default=None, help='Overrides training.seed')
    args = parser.parse_args()

    config = load_config(args.config)
    if args.seed is not None:
        config['training']['seed'] = args.seed
    seed = config['training'].get('seed')
    if seed is not None:
        set_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_graphs, test_graphs, test_labels = load_or_build_split(config)
    if len(train_graphs) == 0 or len(test_graphs) == 0:
        raise RuntimeError('Oracle split produced empty datasets. Check h5 path and scan limits.')
    if len(set(test_labels)) < 2:
        raise RuntimeError(
            f'Test set has only label(s) {sorted(set(test_labels))}; ROC-AUC needs signal and background. '
            'For the sorted events_anomalydetection_v2.h5 set data.scan_limit to null.'
        )

    train_loader = DataLoader(train_graphs, batch_size=config['data']['batch_size'], shuffle=True)
    test_loader = DataLoader(test_graphs, batch_size=config['data']['batch_size'], shuffle=False)

    mean_vec, std_vec = load_norm_stats(config['data'].get('stats_path'), device)

    model = PhysicsAE(
        input_dim=config['model']['input_dim'],
        latent_dim=config['model']['latent_dim'],
        hidden_layers=tuple(config['model']['hidden_layers']),
        dropout=config['model']['dropout'],
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=float(config['training'].get('weight_decay', 0.0)),
    )
    loss_mode = config['training'].get('loss_mode', 'pt_only')
    loss_weights = torch.tensor(config['training']['loss_weights'], device=device)

    avg_loss = float('nan')
    for epoch in range(1, config['training']['epochs'] + 1):
        avg_loss = train_one_epoch(
            model, train_loader, optimizer, device, mean_vec, std_vec,
            loss_weights, loss_mode=loss_mode, desc=f'Train Epoch {epoch}',
        )
        print(f'Epoch {epoch:02d} | Purified MSE: {avg_loss:.4f}')

    # Score on the same feature(s) the model was trained to reconstruct
    score_features = [0] if loss_mode == 'pt_only' else None
    y_true = np.asarray(test_labels)
    reduction = config['evaluation']['score_reduction']
    scores_by_reduction = {
        r: compute_anomaly_scores(
            model, test_loader, device, mean_vec, std_vec, reduction=r, score_features=score_features,
        )
        for r in dict.fromkeys([reduction, 'max', 'mean'])
    }
    y_scores = scores_by_reduction[reduction]

    auc = roc_auc_score(y_true, y_scores)
    auc_ci = bootstrap_auc_ci(y_true, y_scores)
    fpr, tpr, _ = roc_curve(y_true, y_scores)

    print(f'Final Purified ROC-AUC: {auc:.4f}  (95% CI {auc_ci[0]:.4f}–{auc_ci[1]:.4f})')
    extra = {
        'auc_by_reduction': {r: float(roc_auc_score(y_true, s)) for r, s in scores_by_reduction.items()},
        'final_train_loss': float(avg_loss),
        'loss_mode': loss_mode,
        'latent_dim': config['model']['latent_dim'],
        'hidden_layers': list(config['model']['hidden_layers']),
        'num_parameters': sum(p.numel() for p in model.parameters()),
    }
    save_outputs(model, config, auc, auc_ci, fpr, tpr, y_scores, test_labels, extra)


if __name__ == '__main__':
    main()
