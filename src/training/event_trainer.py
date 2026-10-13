import copy
import time
from dataclasses import dataclass

import numpy as np
import torch

from src.data.event_features import FeatureStats, make_event_batch
from src.training.set_loss import chamfer_per_event


class TrainingError(RuntimeError):
    """Raised when training must be aborted (non-finite loss, collapsed latent)."""


@dataclass
class BatchContext:
    stats: FeatureStats
    k: int
    radius: float
    wrap_phi: bool
    loss: str = 'chamfer'          # 'chamfer' | 'chamfer_ptw'


def per_event_loss(model, particles, ctx):
    batch = make_event_batch(particles, ctx.stats, ctx.k, ctx.radius, ctx.wrap_phi)
    reconstruction, z = model(batch)
    weights = batch.target_pt if ctx.loss == 'chamfer_ptw' else None
    return chamfer_per_event(reconstruction, batch.target, batch.target_mask, weights), z


@torch.no_grad()
def infer(model, particles, ctx, batch_size=1024):
    """Per-event loss and latent for every event (eval mode). `particles` may live on the CPU."""
    model.eval()
    device = next(model.parameters()).device
    losses, latents = [], []
    for start in range(0, len(particles), batch_size):
        loss, z = per_event_loss(model, particles[start:start + batch_size].to(device), ctx)
        losses.append(loss.cpu())
        latents.append(z.cpu())
    return torch.cat(losses).numpy(), torch.cat(latents).numpy()


def train_event_ae(model, train_particles, val_particles, ctx, *, epochs, batch_size, lr, weight_decay,
                   patience, grad_clip, collapse_std, seed, log=print):
    device = next(model.parameters()).device
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    generator = torch.Generator().manual_seed(seed)
    n = len(train_particles)
    best_val, best_state, bad_epochs, history = float('inf'), None, 0, []

    for epoch in range(1, epochs + 1):
        started = time.time()
        model.train()
        order = torch.randperm(n, generator=generator)
        total = 0.0
        for start in range(0, n, batch_size):
            idx = order[start:start + batch_size].to(train_particles.device)
            loss = per_event_loss(model, train_particles[idx].to(device), ctx)[0].mean()
            if not torch.isfinite(loss):
                raise TrainingError(f'non-finite loss at epoch {epoch}')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            total += loss.item() * len(idx)

        val_losses, z_val = infer(model, val_particles, ctx)
        val_loss = float(val_losses.mean())
        z_std_max = float(np.std(z_val, axis=0).max())
        if not np.isfinite(val_loss):
            raise TrainingError(f'non-finite validation loss at epoch {epoch}')
        if z_std_max < collapse_std:
            raise TrainingError(f'latent collapsed at epoch {epoch}: max per-dimension std {z_std_max:.2e}')
        history.append({'epoch': epoch, 'train_loss': total / n, 'val_loss': val_loss,
                        'z_std_max': z_std_max, 'seconds': round(time.time() - started, 1)})
        log(f"epoch {epoch:02d} | train {total / n:.4f} | val {val_loss:.4f} | z_std_max {z_std_max:.3f} | "
            f"{history[-1]['seconds']}s")

        if val_loss < best_val - 1e-6:
            best_val, bad_epochs = val_loss, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return history
