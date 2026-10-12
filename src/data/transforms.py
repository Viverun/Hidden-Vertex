import os

import torch

# Pre-calculated SM background stats hard-coded in the Part 1 notebook (Act 1, cell 3).
# Order: [log pT, eta, phi]
NOTEBOOK_FEAT_MEAN = (1.1831, 0.0012, -0.0034)
NOTEBOOK_FEAT_STD = (1.5210, 1.4121, 1.8150)

_MEAN_KEYS = ('mean', 'feat_mean')
_STD_KEYS = ('std', 'feat_std')


def _pick(stats, keys, path):
    for key in keys:
        if key in stats:
            return torch.as_tensor(stats[key], dtype=torch.float)
    raise KeyError(f"{path} has none of {keys}; found {list(stats)}")


def load_norm_stats(stats_path=None, device='cpu'):
    """
    Returns (mean, std) tensors for [log pT, eta, phi].

    Accepts both key conventions ('mean'/'std' and the Kaggle 'feat_mean'/'feat_std').
    Falls back to the notebook constants when no stats file is given or found.
    """
    if stats_path and os.path.exists(stats_path):
        stats = torch.load(stats_path, map_location='cpu', weights_only=False)
        mean = _pick(stats, _MEAN_KEYS, stats_path)
        std = _pick(stats, _STD_KEYS, stats_path)
    else:
        if stats_path:
            print(f"⚠️  Stats file '{stats_path}' not found; using notebook constants.")
        mean = torch.tensor(NOTEBOOK_FEAT_MEAN)
        std = torch.tensor(NOTEBOOK_FEAT_STD)
    return mean.to(device), std.to(device)


def standardize_features(x, mean, std):
    """Log-scale pT (column 0) and z-score all features. Does not modify `x` in place."""
    x = x.clone()
    x[:, 0] = torch.log(x[:, 0].clamp(min=1e-8))
    return (x - mean) / std


def standardize_batch(batch, mean, std):
    """Standardizes `batch.x` in place on the batch object and returns the batch."""
    batch.x = standardize_features(batch.x, mean, std)
    return batch
