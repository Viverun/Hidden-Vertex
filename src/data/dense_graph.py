import math

import torch


def radius_adjacency(eta, phi, mask, radius=0.4, wrap_phi=True):
    """Dense ΔR <= radius adjacency on padded tensors.

    eta, phi: [B, P]; mask: [B, P] bool (False = padding). Returns [B, P, P] bool without self
    loops or padded slots. The arithmetic mirrors `create_radius_graph` in src/data/dataset.py
    (squared distance, same wrap expression) so both give identical edges.
    """
    d_eta = eta[:, :, None] - eta[:, None, :]
    d_phi = phi[:, :, None] - phi[:, None, :]
    if wrap_phi:
        d_phi = torch.remainder(d_phi + math.pi, 2 * math.pi) - math.pi
    adj = d_eta.square() + d_phi.square() <= radius * radius
    adj = adj & mask[:, :, None] & mask[:, None, :]
    eye = torch.eye(eta.shape[1], dtype=torch.bool, device=eta.device)
    return adj & ~eye
