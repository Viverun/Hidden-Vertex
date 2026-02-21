from torch_geometric.loader import DataLoader
from .dataset import RawChunkDataset

def get_discovery_loaders(source_dir, bg_indices, anom_indices=None, batch_size=64):
    """
    Prepares DataLoaders for the Discovery Pipeline.
    Separates background for training and reserves anomalies for sensitivity testing.
    """
    # 1. Split Background (90% Train, 10% Val)
    split_idx = int(0.9 * len(bg_indices))
    train_idx = bg_indices[:split_idx]
    val_idx = bg_indices[split_idx:]
    
    # 2. Initialize Datasets
    train_set = RawChunkDataset(source_dir, train_idx)
    val_set = RawChunkDataset(source_dir, val_idx)
    
    # 3. Create Loaders
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    
    # 4. Optional: Anomaly Test Loader (The "Truth" Set)
    test_loader = None
    if anom_indices is not None:
        test_set = RawChunkDataset(source_dir, anom_indices)
        test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
        
    return train_loader, val_loader, test_loader