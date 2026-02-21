import torch
import torch.nn as nn
from tqdm import tqdm


def _extract_reconstruction(model_output):
    if isinstance(model_output, tuple):
        return model_output[0]
    return model_output


def train_one_epoch(
    model,
    loader,
    optimizer,
    device,
    gpu_mean,
    gpu_std,
    loss_weights,
    loss_mode="weighted",
):
    """
    Standard training heartbeat. It applies physics-informed weighting 
    to the MSE loss to focus the model on momentum signatures.
    """
    model.train()
    total_loss = 0
    
    for batch in tqdm(loader, desc="Training", leave=False):
        batch = batch.to(device)
        
        # 1. Target Preprocessing: Log-scale pT and standardize
        target = batch.x.clone()
        target[:, 0] = torch.log(target[:, 0].clamp(min=1e-8))
        target = (target - gpu_mean) / gpu_std
        
        # 2. Forward pass
        optimizer.zero_grad()
        out = _extract_reconstruction(model(batch))
        
        # 3. Weighted MSE Loss: Prioritize pT (index 0) over angular noise
        # Standard weights are [10.0, 1.0, 1.0] for LHCO 2020 R&D
        if loss_mode == "pt_only":
            loss = nn.functional.mse_loss(out[:, 0], target[:, 0])
        else:
            diff = (out - target) ** 2
            loss = (diff * loss_weights).mean()
        
        # 4. Backprop
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
    return total_loss / len(loader)

def validate(model, loader, device, gpu_mean, gpu_std, loss_weights, loss_mode="weighted"):
    """
    Evaluates reconstruction performance on background events.
    Used to monitor for 'Latent Collapse' or overfitting.
    """
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            target = batch.x.clone()
            target[:, 0] = torch.log(target[:, 0].clamp(min=1e-8))
            target = (target - gpu_mean) / gpu_std
            
            out = _extract_reconstruction(model(batch))
            if loss_mode == "pt_only":
                loss = nn.functional.mse_loss(out[:, 0], target[:, 0])
            else:
                diff = (out - target) ** 2
                loss = (diff * loss_weights).mean()
            val_loss += loss.item()
            
    return val_loss / len(loader)