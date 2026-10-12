from torch_geometric.loader import DataLoader
from .dataset import ChunkGroupedSampler, RawChunkDataset

def get_discovery_loaders(source_dir, bg_indices, anom_indices=None, batch_size=64, num_workers=0, chunk_size=10000):
    """
    Prepares DataLoaders for the Discovery Pipeline.
    Separates background for training and reserves anomalies for sensitivity testing.
    """
    # 1. Split Background (90% Train, 10% Val)
    split_idx = int(0.9 * len(bg_indices))
    train_idx = bg_indices[:split_idx]
    val_idx = bg_indices[split_idx:]

    # 2. Initialize Datasets
    train_set = RawChunkDataset(source_dir, train_idx, chunk_size=chunk_size)
    val_set = RawChunkDataset(source_dir, val_idx, chunk_size=chunk_size)

    # 3. Create Loaders (chunk-grouped shuffling keeps chunk reloads to one per epoch)
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        sampler=ChunkGroupedSampler(train_idx, chunk_size=chunk_size),
        num_workers=num_workers,
    )
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # 4. Optional: Anomaly Test Loader (The "Truth" Set)
    test_loader = None
    if anom_indices is not None:
        test_set = RawChunkDataset(source_dir, anom_indices, chunk_size=chunk_size)
        test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader
