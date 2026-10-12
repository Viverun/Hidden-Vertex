import torch
import torch.nn as nn
from tqdm import tqdm

from src.data.transforms import standardize_batch


def _extract_reconstruction(model_output):
    if isinstance(model_output, tuple):
        return model_output[0]
    return model_output


def reconstruction_loss(out, target, loss_weights=None, loss_mode="weighted"):
    """
    pt_only:  MSE on standardized log pT only (Part 1 notebook objective).
    weighted: per-feature weighted MSE; standard weights are [10.0, 1.0, 1.0] to
              prioritize pT (index 0) over angular noise.
    """
    if loss_mode == "pt_only":
        return nn.functional.mse_loss(out[:, 0], target[:, 0])
    if loss_mode == "weighted":
        diff = (out - target) ** 2
        return (diff * loss_weights).mean()
    raise ValueError(f"Unknown loss_mode '{loss_mode}' (expected 'weighted' or 'pt_only')")


def train_one_epoch(
    model,
    loader,
    optimizer,
    device,
    gpu_mean,
    gpu_std,
    loss_weights=None,
    loss_mode="weighted",
    desc="Training",
):
    """
    Standard training heartbeat. Inputs and targets are both the log-pT, z-scored
    features, so the model sees the same space it is asked to reconstruct.
    """
    model.train()
    total_loss = 0.0
    num_batches = 0

    for batch in tqdm(loader, desc=desc, leave=False):
        batch = standardize_batch(batch.to(device), gpu_mean, gpu_std)

        optimizer.zero_grad()
        out = _extract_reconstruction(model(batch))
        loss = reconstruction_loss(out, batch.x, loss_weights, loss_mode)

        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        num_batches += 1

    return total_loss / max(num_batches, 1)


def validate(model, loader, device, gpu_mean, gpu_std, loss_weights=None, loss_mode="weighted"):
    """
    Evaluates reconstruction performance on background events.
    Used to monitor for 'Latent Collapse' or overfitting.
    """
    model.eval()
    val_loss = 0.0
    num_batches = 0
    with torch.no_grad():
        for batch in loader:
            batch = standardize_batch(batch.to(device), gpu_mean, gpu_std)
            out = _extract_reconstruction(model(batch))
            val_loss += reconstruction_loss(out, batch.x, loss_weights, loss_mode).item()
            num_batches += 1

    return val_loss / max(num_batches, 1)
