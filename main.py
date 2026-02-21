import os
import argparse
import yaml
import torch
from src.model.autoencoder import PhysicsAE
from src.data.loader import get_discovery_loaders
from src.data.oracle_split import build_oracle_purified_split
from src.training.engine import train_one_epoch, validate
from src.utils.metrics import compute_anomaly_scores, get_performance_stats

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def setup_results_dirs():
    """Ensure the results folder isn't empty anymore once we run!"""
    for folder in ['results/models', 'results/figures', 'results/logs', 'results/scores']:
        os.makedirs(folder, exist_ok=True)

def run_default_training(config_path):
    setup_results_dirs()
    config = load_config(config_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Load Physics Normalization Stats
    # These are vital for the Log-pT transform in your engine
    stats = torch.load(config['data']['stats_path'], map_location=device)
    gpu_mean, gpu_std = stats['mean'], stats['std']
    
    # 2. Setup Data
    # Using the LHCO 2020 background indices (1 million events)
    bg_idx = list(range(1000000))
    train_ld, val_ld, _ = get_discovery_loaders(
        config['data']['source_dir'], 
        bg_idx, 
        batch_size=config['data']['batch_size']
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
        lr=config['training']['learning_rate']
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

    train_graphs, test_graphs, test_labels = build_oracle_purified_split(
        h5_path=config['data']['h5_path'],
        train_target=config['oracle']['train_target'],
        test_bg_target=config['oracle']['test_bg_target'],
        test_sig_target=config['oracle']['test_sig_target'],
        scan_limit=config['data']['scan_limit'],
        radius=config['data']['radius'],
    )

    output_path = 'results/scores/part1_oracle_split.pt'
    torch.save(
        {
            'train_graphs': train_graphs,
            'test_graphs': test_graphs,
            'test_labels': test_labels,
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