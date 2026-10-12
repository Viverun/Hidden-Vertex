import os
import argparse
import random
import numpy as np
import yaml
import torch
from src.model.autoencoder import PhysicsAE
from src.data.loader import get_discovery_loaders
from src.data.oracle_split import build_oracle_purified_split
from src.data.transforms import load_norm_stats
from src.training.engine import train_one_epoch, validate

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def load_background_indices(source_dir):
    """
    Background events are selected by truth label: the original R&D file is shuffled
    (~9% signal), and v2 is sorted, so index ranges are not a safe proxy. `scripts/process_h5.py` writes these labels to `labels.npy`.
    """
    labels_path = os.path.join(source_dir, 'labels.npy')
    if not os.path.exists(labels_path):
        raise FileNotFoundError(
            f"{labels_path} not found. Re-run scripts/process_h5.py to regenerate the chunks "
            "and labels; training on unfiltered indices would mix signal into the background."
        )
    labels = np.load(labels_path)
    return np.flatnonzero(labels == 0).tolist()

def setup_results_dirs():
    """Ensure the results folder isn't empty anymore once we run!"""
    for folder in ['results/models', 'results/figures', 'results/logs', 'results/scores']:
        os.makedirs(folder, exist_ok=True)

def run_default_training(config_path):
    setup_results_dirs()
    config = load_config(config_path)
    if config['training'].get('seed') is not None:
        set_seed(config['training']['seed'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Load Physics Normalization Stats
    # These are vital for the Log-pT transform in your engine
    gpu_mean, gpu_std = load_norm_stats(config['data']['stats_path'], device)
    
    # 2. Setup Data
    # Background-only indices of the LHCO 2020 R&D set (selected by truth label)
    bg_idx = load_background_indices(config['data']['source_dir'])
    train_ld, val_ld, _ = get_discovery_loaders(
        config['data']['source_dir'], 
        bg_idx, 
        batch_size=config['data']['batch_size'],
        num_workers=config['data'].get('num_workers', 0),
        chunk_size=config['data'].get('chunk_size', 10000),
    )
    
    # 3. Initialize Model & Training Tools
    model = PhysicsAE(
        input_dim=config['model']['input_dim'],
        latent_dim=config['model']['latent_dim'],
        hidden_layers=tuple(config['model'].get('hidden_layers', [32, 16])),
        dropout=config['model'].get('dropout', 0.1),
    ).to(device)
    
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=config['training']['learning_rate'],
        weight_decay=float(config['training'].get('weight_decay', 0.0)),
    )
    
    loss_weights = torch.tensor(config['training']['loss_weights']).to(device)
    
    # 4. Discovery Training Loop
    print(f"🚀 Starting Discovery on {device}...")
    loss_mode = config['training'].get('loss_mode', 'weighted')
    for epoch in range(1, config['training']['epochs'] + 1):
        train_loss = train_one_epoch(
            model,
            train_ld,
            optimizer,
            device,
            gpu_mean,
            gpu_std,
            loss_weights,
            loss_mode=loss_mode,
        )
        val_loss = validate(
            model,
            val_ld,
            device,
            gpu_mean,
            gpu_std,
            loss_weights,
            loss_mode=loss_mode,
        )
        
        print(f"Epoch {epoch:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
        # Save the model after training is complete
        if epoch == config['training']['epochs']:
            save_path = f"results/models/strict_ae_v1_final.pt"
            torch.save(model.state_dict(), save_path)
            print(f"💾 Model weights secured at {save_path}")


def prepare_part1(config_path):
    setup_results_dirs()
    config = load_config(config_path)

    train_graphs, test_graphs, test_labels, train_indices, test_indices = build_oracle_purified_split(
        h5_path=config['data']['h5_path'],
        train_target=config['oracle']['train_target'],
        test_bg_target=config['oracle']['test_bg_target'],
        test_sig_target=config['oracle']['test_sig_target'],
        scan_limit=config['data']['scan_limit'],
        radius=config['data']['radius'],
        wrap_phi=config['data'].get('wrap_phi', False),
        return_indices=True,
    )

    output_path = config['data'].get('split_path', 'results/scores/part1_oracle_split.pt')
    torch.save(
        {
            'train_graphs': train_graphs,
            'test_graphs': test_graphs,
            'test_labels': test_labels,
            'train_indices': train_indices,
            'test_indices': test_indices,
        },
        output_path,
    )

    print('✅ Part 1 oracle split prepared')
    print(f"   Train graphs: {len(train_graphs):,}")
    print(f"   Test graphs:  {len(test_graphs):,}")
    print(f"   Saved split:  {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Hidden-Vertex training entrypoint')
    parser.add_argument(
        'command',
        nargs='?',
        default='train-default',
        choices=['train-default', 'prepare-part1'],
    )
    parser.add_argument('--config', default=None)
    args = parser.parse_args()

    if args.command == 'prepare-part1':
        config_path = args.config or 'configs/part1_notebook_parity.yaml'
        prepare_part1(config_path)
    else:
        config_path = args.config or 'configs/default_search.yaml'
        run_default_training(config_path)

if __name__ == "__main__":
    main()