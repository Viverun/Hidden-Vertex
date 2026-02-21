import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from sklearn.metrics import roc_auc_score, roc_curve
from torch_geometric.loader import DataLoader
from torch_geometric.utils import scatter
from tqdm import tqdm

from src.data.oracle_split import build_oracle_purified_split
from src.model.autoencoder import PhysicsAE


def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as handle:
        return yaml.safe_load(handle)


def ensure_parent_dir(file_path):
    parent = os.path.dirname(file_path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def standardize_graph(batch, mean_vec, std_vec):
    x = batch.x.clone()
    x[:, 0] = torch.log(x[:, 0].clamp(min=1e-8))
    batch.x = (x - mean_vec) / std_vec
    return batch


def train_part1(model, train_loader, device, mean_vec, std_vec, lr, epochs):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0

        for batch in tqdm(train_loader, desc=f'Train Epoch {epoch}', leave=False):
            batch = standardize_graph(batch.to(device), mean_vec, std_vec)
            optimizer.zero_grad()
            out = model(batch)
            loss = F.mse_loss(out[:, 0], batch.x[:, 0])
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / max(len(train_loader), 1)
        print(f'Epoch {epoch:02d} | Purified MSE: {avg_loss:.4f}')


def evaluate_part1(model, test_loader, test_labels, device, mean_vec, std_vec, reduction='max'):
    model.eval()
    y_scores = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc='Evaluating', leave=False):
            batch = standardize_graph(batch.to(device), mean_vec, std_vec)
            out = model(batch)
            node_errors = (out[:, 0] - batch.x[:, 0]) ** 2
            graph_scores = scatter(node_errors, batch.batch, dim=0, reduce=reduction)
            y_scores.extend(graph_scores.detach().cpu().numpy())

    y_scores = np.array(y_scores)
    y_true = np.array(test_labels)

    auc = roc_auc_score(y_true, y_scores)
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    return auc, fpr, tpr, y_scores


def save_outputs(model, config, auc, fpr, tpr, y_scores, test_labels):
    model_path = config['evaluation']['model_path']
    metrics_path = config['evaluation']['metrics_path']
    roc_path = config['evaluation']['roc_path']

    ensure_parent_dir(model_path)
    ensure_parent_dir(metrics_path)
    ensure_parent_dir(roc_path)

    torch.save({'model_state_dict': model.state_dict()}, model_path)

    metrics = {
        'auc': float(auc),
        'score_reduction': config['evaluation']['score_reduction'],
        'num_test_events': int(len(test_labels)),
        'num_signal': int(sum(test_labels)),
        'num_background': int(len(test_labels) - sum(test_labels)),
        'mean_score': float(np.mean(y_scores)),
        'std_score': float(np.std(y_scores)),
    }
    with open(metrics_path, 'w', encoding='utf-8') as handle:
        json.dump(metrics, handle, indent=2)

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
    args = parser.parse_args()

    config = load_config(args.config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('Building oracle-purified split...')
    train_graphs, test_graphs, test_labels = build_oracle_purified_split(
        h5_path=config['data']['h5_path'],
        train_target=config['oracle']['train_target'],
        test_bg_target=config['oracle']['test_bg_target'],
        test_sig_target=config['oracle']['test_sig_target'],
        scan_limit=config['data']['scan_limit'],
        radius=config['data']['radius'],
    )

    if len(train_graphs) == 0 or len(test_graphs) == 0:
        raise RuntimeError('Oracle split produced empty datasets. Check h5 path and scan limits.')

    train_loader = DataLoader(train_graphs, batch_size=config['data']['batch_size'], shuffle=True)
    test_loader = DataLoader(test_graphs, batch_size=config['data']['batch_size'], shuffle=False)

    stats = torch.load(config['data']['stats_path'], map_location=device)
    mean_vec = stats['mean'].to(device)
    std_vec = stats['std'].to(device)

    model = PhysicsAE(
        input_dim=config['model']['input_dim'],
        latent_dim=config['model']['latent_dim'],
        hidden_layers=tuple(config['model']['hidden_layers']),
        dropout=config['model']['dropout'],
    ).to(device)

    train_part1(
        model,
        train_loader,
        device,
        mean_vec,
        std_vec,
        lr=config['training']['learning_rate'],
        epochs=config['training']['epochs'],
    )

    auc, fpr, tpr, y_scores = evaluate_part1(
        model,
        test_loader,
        test_labels,
        device,
        mean_vec,
        std_vec,
        reduction=config['evaluation']['score_reduction'],
    )

    print(f'Final Purified ROC-AUC: {auc:.4f}')
    save_outputs(model, config, auc, fpr, tpr, y_scores, test_labels)


if __name__ == '__main__':
    main()
