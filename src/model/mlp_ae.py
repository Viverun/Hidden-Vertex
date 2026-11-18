"""Small fully connected autoencoder (d-32-32-3-32-32-d, GELU) and its plain MSE training loop."""
import copy
import time

import numpy as np
import torch
from torch import nn


class MLPAE(nn.Module):
    """Encoder d-32-32-3, decoder 3-32-32-d. GELU after every hidden layer; linear latent and output."""

    def __init__(self, n_inputs, hidden=(32, 32), latent=3):
        super().__init__()

        def stack(sizes):
            layers = []
            for i, (a, b) in enumerate(zip(sizes[:-1], sizes[1:])):
                layers.append(nn.Linear(a, b))
                if i < len(sizes) - 2:
                    layers.append(nn.GELU())
            return nn.Sequential(*layers)

        self.encoder = stack([n_inputs, *hidden, latent])
        self.decoder = stack([latent, *reversed(hidden), n_inputs])

    def forward(self, x):
        return self.decoder(self.encoder(x))

    def energy(self, x):
        """Per-example reconstruction error: mean over features of the squared error."""
        return ((x - self(x)) ** 2).mean(dim=1)


@torch.no_grad()
def recon_error(model, x, batch_size=8192):
    model.eval()
    x = torch.as_tensor(x, dtype=torch.float32)
    return np.concatenate([model.energy(x[i:i + batch_size]).numpy() for i in range(0, len(x), batch_size)])


def train_ae(model, x_fit, x_val, epochs=30, batch_size=256, lr=1e-3, weight_decay=1e-5, patience=5, seed=0):
    """AdamW on the mean reconstruction error; early stop on the validation loss; the best state is restored.

    Raises FloatingPointError on a non-finite loss."""
    x_fit = torch.as_tensor(x_fit, dtype=torch.float32)
    x_val = torch.as_tensor(x_val, dtype=torch.float32)
    gen = torch.Generator().manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    best, best_state, wait, history = float('inf'), None, 0, []
    for epoch in range(1, epochs + 1):
        started = time.time()
        model.train()
        perm = torch.randperm(len(x_fit), generator=gen)
        total = 0.0
        for i in range(0, len(x_fit), batch_size):
            xb = x_fit[perm[i:i + batch_size]]
            loss = model.energy(xb).mean()
            if not torch.isfinite(loss):
                raise FloatingPointError(f'non-finite training loss at epoch {epoch}')
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss) * len(xb)
        val = float(recon_error(model, x_val).mean())
        history.append({'epoch': epoch, 'train_loss': total / len(x_fit), 'val_loss': val,
                        'seconds': time.time() - started})
        if val < best:
            best, best_state, wait = val, copy.deepcopy(model.state_dict()), 0
        else:
            wait += 1
            if wait >= patience:
                break
    model.load_state_dict(best_state)
    best_epoch = min(history, key=lambda h: h['val_loss'])['epoch']
    return history, best_epoch
