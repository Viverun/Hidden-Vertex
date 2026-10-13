"""Event-level particle features, decoder targets and batches for the Part 2 model.

Particles arrive as dense tensors [B, P, 3] = (pT, eta, phi), sorted by pT (slot 0 is the
leading particle) and zero-padded; a slot is a real particle iff pT > 0.
"""
import math
from typing import NamedTuple

import numpy as np
import torch

from .dense_graph import radius_adjacency

SUMMARY_KEYS = ('ht', 'multiplicity', 'mean_pt', 'max_pt', 'm_all')
FEATURE_NAMES = ['log_pt', 'eta', 'phi', 'log_e', 'pt_frac', 'd_eta_lead', 'd_phi_lead']
PT_FRAC_INDEX = FEATURE_NAMES.index('pt_frac')

# Part 1 notebook constants, reused to put decoder targets on a unit scale
LOGPT_MEAN, LOGPT_STD = 1.1831, 1.5210
ETA_MEAN, ETA_STD = 0.0012, 1.4121


def particle_summary(particles):
    """Label-free event quantities from [N, P, 3] arrays (pT == 0 is padding). Use full, untruncated
    events so multiplicity and HT are exact."""
    particles = np.asarray(particles, dtype=np.float64)
    pt, eta, phi = particles[..., 0], particles[..., 1], particles[..., 2]
    n = (pt > 0).sum(axis=1)
    ht = pt.sum(axis=1)
    e = (pt * np.cosh(eta)).sum(axis=1)
    px = (pt * np.cos(phi)).sum(axis=1)
    py = (pt * np.sin(phi)).sum(axis=1)
    pz = (pt * np.sinh(eta)).sum(axis=1)
    return {
        'ht': ht,
        'multiplicity': n.astype(np.float64),
        'mean_pt': ht / np.maximum(n, 1),
        'max_pt': pt.max(axis=1),
        'm_all': np.sqrt(np.maximum(e**2 - px**2 - py**2 - pz**2, 0.0)),
    }


def wrap_delta_phi(d):
    return torch.remainder(d + math.pi, 2 * math.pi) - math.pi


def build_event_features(particles, mask):
    """Raw (unstandardized) per-particle features [B, P, 7]; zero at padded slots."""
    pt, eta, phi = particles.unbind(-1)
    safe_pt = torch.where(mask, pt, torch.ones_like(pt))
    log_pt = torch.log(safe_pt)
    log_e = torch.log(safe_pt * torch.cosh(eta))
    ht = (pt * mask).sum(1, keepdim=True).clamp(min=1e-8)
    pt_frac = pt * mask / ht
    d_eta = eta - eta[:, :1]
    d_phi = wrap_delta_phi(phi - phi[:, :1])
    feats = torch.stack([log_pt, eta, phi, log_e, pt_frac, d_eta, d_phi], dim=-1)
    return feats * mask.unsqueeze(-1)


class FeatureStats:
    """Per-feature mean/std over real particles of the training set."""

    def __init__(self, mean, std):
        self.mean, self.std = mean, std

    def to(self, device):
        return FeatureStats(self.mean.to(device), self.std.to(device))


@torch.no_grad()
def fit_feature_stats(particles, chunk=20000):
    n_feat = len(FEATURE_NAMES)
    total = torch.zeros(n_feat, dtype=torch.float64)
    total_sq = torch.zeros(n_feat, dtype=torch.float64)
    count = 0
    for start in range(0, len(particles), chunk):
        p = particles[start:start + chunk]
        mask = p[..., 0] > 0
        f = build_event_features(p, mask)[mask].double().cpu()
        total += f.sum(0)
        total_sq += (f * f).sum(0)
        count += f.shape[0]
    mean = total / count
    std = (total_sq / count - mean * mean).clamp(min=1e-12).sqrt()
    return FeatureStats(mean.float(), std.float().clamp(min=1e-6))


def standardize(feats, mask, stats):
    return ((feats - stats.mean) / stats.std) * mask.unsqueeze(-1)


def build_targets(particles, mask, k):
    """Decoder targets for the k hardest particles: (std log pT, std eta, cos phi, sin phi)."""
    if particles.shape[1] < k:
        raise ValueError(f'need at least k={k} particle slots, got {particles.shape[1]}')
    pt, eta, phi = particles[:, :k].unbind(-1)
    m = mask[:, :k]
    safe_pt = torch.where(m, pt, torch.ones_like(pt))
    target = torch.stack([
        (torch.log(safe_pt) - LOGPT_MEAN) / LOGPT_STD,
        (eta - ETA_MEAN) / ETA_STD,
        torch.cos(phi),
        torch.sin(phi),
    ], dim=-1)
    return target * m.unsqueeze(-1), m


class EventBatch(NamedTuple):
    feats: torch.Tensor        # [B, P, 7] standardized, zero at padding
    mask: torch.Tensor         # [B, P] bool
    adj: torch.Tensor          # [B, P, P] bool
    pt_frac: torch.Tensor      # [B, P] pT / HT, zero at padding
    target: torch.Tensor       # [B, k, 4]
    target_mask: torch.Tensor  # [B, k] bool
    target_pt: torch.Tensor    # [B, k] raw pT of the targets (weights for the pT-weighted loss)


def make_event_batch(particles, stats, k, radius=0.4, wrap_phi=True):
    mask = particles[..., 0] > 0
    raw = build_event_features(particles, mask)
    target, target_mask = build_targets(particles, mask, k)
    return EventBatch(
        feats=standardize(raw, mask, stats),
        mask=mask,
        adj=radius_adjacency(particles[..., 1], particles[..., 2], mask, radius, wrap_phi),
        pt_frac=raw[..., PT_FRAC_INDEX],
        target=target,
        target_mask=target_mask,
        target_pt=particles[:, :k, 0] * target_mask,
    )
