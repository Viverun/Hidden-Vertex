import torch
import torch.nn as nn
import torch.nn.functional as F

from src.data.event_features import EventBatch


class NeighbourLayer(nn.Module):
    """h <- LayerNorm(h + GELU(W [h, mean of neighbours' h])), neighbours = ΔR < radius particles."""

    def __init__(self, width):
        super().__init__()
        self.lin = nn.Linear(2 * width, width)
        self.norm = nn.LayerNorm(width)

    def forward(self, h, adj, mask):
        adj_f = adj.to(h.dtype)
        agg = torch.bmm(adj_f, h) / adj_f.sum(-1, keepdim=True).clamp(min=1)
        h = self.norm(h + F.gelu(self.lin(torch.cat([h, agg], dim=-1))))
        return h * mask.unsqueeze(-1)


class EventAE(nn.Module):
    """Event-level graph autoencoder on dense padded tensors.

    Encoder: embed -> `layers` neighbour-averaging layers -> pool (mean / max / pT-weighted sum) -> MLP -> z.
    Decoder: MLP z -> k_recon x out_dim (the hardest particles: log pT, eta, cos phi, sin phi).
    """

    def __init__(self, in_dim=7, width=64, layers=3, hidden=128, latent_dim=16, k_recon=48, out_dim=4,
                 pooling=('mean', 'max', 'ptsum')):
        super().__init__()
        unknown = set(pooling) - {'mean', 'max', 'ptsum'}
        if unknown or not pooling:
            raise ValueError(f'bad pooling {pooling}')
        self.pooling = tuple(pooling)
        self.k_recon, self.out_dim = k_recon, out_dim
        self.embed = nn.Linear(in_dim, width)
        self.layers = nn.ModuleList(NeighbourLayer(width) for _ in range(layers))
        self.to_latent = nn.Sequential(nn.Linear(width * len(self.pooling), hidden), nn.GELU(),
                                       nn.Linear(hidden, latent_dim))
        self.decoder = nn.Sequential(nn.Linear(latent_dim, hidden), nn.GELU(), nn.Linear(hidden, hidden),
                                     nn.GELU(), nn.Linear(hidden, k_recon * out_dim))

    def encode(self, batch: EventBatch):
        mask = batch.mask
        h = self.embed(batch.feats) * mask.unsqueeze(-1)
        for layer in self.layers:
            h = layer(h, batch.adj, mask)
        n = mask.sum(1, keepdim=True)
        parts = []
        if 'mean' in self.pooling:
            parts.append((h * mask.unsqueeze(-1)).sum(1) / n.clamp(min=1))
        if 'max' in self.pooling:
            hmax = h.masked_fill(~mask.unsqueeze(-1), -1e9).max(dim=1).values
            parts.append(torch.where(n > 0, hmax, torch.zeros_like(hmax)))
        if 'ptsum' in self.pooling:
            parts.append((h * batch.pt_frac.unsqueeze(-1)).sum(1))
        return self.to_latent(torch.cat(parts, dim=-1))

    def forward(self, batch: EventBatch):
        z = self.encode(batch)
        return self.decoder(z).view(-1, self.k_recon, self.out_dim), z
