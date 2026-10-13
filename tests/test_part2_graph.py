import numpy as np
import torch

from src.data.dataset import create_radius_graph
from src.data.dense_graph import radius_adjacency


def _event_row(rng, n=40, spread=0.6):
    row = np.zeros(2101)
    pt = rng.uniform(1, 100, n)
    eta = rng.normal(0, spread, n)
    phi = rng.uniform(-np.pi, np.pi, n)
    phi[:4] = [3.1, -3.1, 3.05, -3.14]          # force pairs that only connect when phi wraps
    row[:3 * n] = np.stack([pt, eta, phi], 1).ravel()
    return row


def _edges(adj):
    return set(map(tuple, adj.nonzero().tolist()))


def test_dense_adjacency_matches_pyg_radius_graph():
    rng = np.random.default_rng(0)
    for wrap in (True, False):
        for _ in range(5):
            row = _event_row(rng)
            graph = create_radius_graph(row, radius=0.4, wrap_phi=wrap)
            x = torch.from_numpy(row[:2100].astype(np.float32).reshape(-1, 3))
            x = x[x[:, 0] > 0][None]
            mask = torch.ones(1, x.shape[1], dtype=torch.bool)
            adj = radius_adjacency(x[..., 1], x[..., 2], mask, radius=0.4, wrap_phi=wrap)[0]
            assert _edges(adj) == set(map(tuple, graph.edge_index.t().tolist()))


def test_wrap_adds_edges_across_pi():
    eta = torch.zeros(1, 2); phi = torch.tensor([[3.1, -3.1]])
    mask = torch.ones(1, 2, dtype=torch.bool)
    assert radius_adjacency(eta, phi, mask, wrap_phi=False).sum() == 0
    assert radius_adjacency(eta, phi, mask, wrap_phi=True).sum() == 2


def test_padding_and_diagonal_excluded():
    eta = torch.zeros(1, 3); phi = torch.zeros(1, 3)
    mask = torch.tensor([[True, True, False]])
    adj = radius_adjacency(eta, phi, mask)[0]
    assert adj.tolist() == [[False, True, False], [True, False, False], [False, False, False]]
