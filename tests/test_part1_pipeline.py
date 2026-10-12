import math

import h5py
import numpy as np
import pytest
import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.utils import scatter

from src.data.dataset import (
    ChunkGroupedSampler,
    LHCH5Dataset,
    RawChunkDataset,
    create_radius_graph,
)
from src.data.loader import get_discovery_loaders
from src.data.oracle_split import build_oracle_purified_split
from src.data.transforms import (
    NOTEBOOK_FEAT_MEAN,
    NOTEBOOK_FEAT_STD,
    load_norm_stats,
    standardize_features,
)
from src.model.autoencoder import PhysicsAE
from src.training.engine import train_one_epoch, validate
from src.utils.metrics import compute_anomaly_scores


# ── Reference implementations copied from the Part 1 notebook ────────────────
def notebook_create_radius_graph(event_row, r=0.4):
    particles = event_row[:2100].reshape(-1, 3)
    particles = particles[particles[:, 0] > 0]
    if len(particles) < 2:
        return None
    x = torch.tensor(particles, dtype=torch.float)
    eta_phi = x[:, 1:3]
    dist_matrix = torch.cdist(eta_phi, eta_phi, p=2.0)
    edge_index = torch.nonzero(dist_matrix <= r, as_tuple=False).t().contiguous()
    edge_index = edge_index[:, edge_index[0] != edge_index[1]]
    return Data(x=x, edge_index=edge_index)


def notebook_oracle(rows, train_target, test_bg_target, test_sig_target):
    train_graphs, test_graphs, test_labels = [], [], []
    for row in rows:
        label = int(row[2100])
        graph = notebook_create_radius_graph(row)
        if graph is None:
            continue
        if label == 0:
            if len(train_graphs) < train_target:
                train_graphs.append(graph)
            elif test_labels.count(0) < test_bg_target:
                test_graphs.append(graph)
                test_labels.append(0)
        elif label == 1 and test_labels.count(1) < test_sig_target:
            test_graphs.append(graph)
            test_labels.append(1)
        if (len(train_graphs) >= train_target and test_labels.count(1) >= test_sig_target
                and test_labels.count(0) >= test_bg_target):
            break
    return train_graphs, test_graphs, test_labels


def make_events(n_events, seed=0, signal_frac=0.1):
    rng = np.random.default_rng(seed)
    rows = np.zeros((n_events, 2101), dtype=np.float64)
    for i in range(n_events):
        n = rng.integers(0, 60)  # includes 0/1-particle events that must be skipped
        pt = rng.exponential(5.0, n) + 0.1
        eta = rng.normal(0, 1.5, n)
        phi = rng.uniform(-math.pi, math.pi, n)
        rows[i, :3 * n] = np.stack([pt, eta, phi], axis=1).ravel()
        rows[i, 2100] = float(rng.random() < signal_frac)
    return rows


@pytest.fixture
def h5_file(tmp_path):
    path = tmp_path / 'events.h5'
    rows = make_events(400)
    with h5py.File(path, 'w') as handle:
        handle.create_dataset('df/block0_values', data=rows)
    return str(path), rows


def same_graph(a, b):
    assert torch.equal(a.x, b.x)
    ea = set(map(tuple, a.edge_index.t().tolist()))
    eb = set(map(tuple, b.edge_index.t().tolist()))
    assert ea == eb


# ── Graph construction ────────────────────────────────────────────────────────
def test_radius_graph_matches_notebook():
    for row in make_events(50, seed=1):
        new, ref = create_radius_graph(row), notebook_create_radius_graph(row)
        assert (new is None) == (ref is None)
        if new is not None:
            same_graph(new, ref)
            assert new.x.dtype == torch.float32


def test_wrap_phi_connects_across_boundary():
    row = np.zeros(2101)
    row[:6] = [10.0, 0.0, math.pi - 0.05, 10.0, 0.0, -math.pi + 0.05]
    assert create_radius_graph(row).num_edges == 0
    assert create_radius_graph(row, wrap_phi=True).num_edges == 2


# ── Oracle split ──────────────────────────────────────────────────────────────
@pytest.mark.parametrize('read_block', [7, 64, 10000])
def test_oracle_split_matches_notebook(h5_file, read_block):
    path, rows = h5_file
    targets = dict(train_target=150, test_bg_target=40, test_sig_target=20)
    tr, te, tl = build_oracle_purified_split(path, scan_limit=10**6, read_block=read_block, **targets)
    rtr, rte, rtl = notebook_oracle(rows, **targets)
    assert tl == rtl
    assert len(tr) == len(rtr) == 150
    assert tl.count(0) == 40 and tl.count(1) == 20
    for a, b in zip(tr + te, rtr + rte):
        same_graph(a, b)


def test_oracle_split_respects_scan_limit(h5_file):
    path, _ = h5_file
    with pytest.warns(UserWarning):
        tr, te, tl = build_oracle_purified_split(path, train_target=10**6, scan_limit=50)
    assert len(tr) + len(te) <= 50


def test_h5_dataset_streams_rows(h5_file):
    path, rows = h5_file
    ds = LHCH5Dataset(path, start_idx=3, num_events=5)
    for i in range(5):
        g = ds[i]
        assert g.y.item() == rows[3 + i, 2100]


# ── Transforms ────────────────────────────────────────────────────────────────
def test_standardize_matches_notebook():
    x = torch.tensor([[5.0, 0.3, -1.0], [120.0, -2.0, 3.0]])
    mean, std = torch.tensor(NOTEBOOK_FEAT_MEAN), torch.tensor(NOTEBOOK_FEAT_STD)
    expected = x.clone()
    expected[:, 0] = torch.log(expected[:, 0])
    expected = (expected - mean) / std
    out = standardize_features(x, mean, std)
    assert torch.allclose(out, expected)
    assert x[0, 0] == 5.0  # input untouched


@pytest.mark.parametrize('keys', [('mean', 'std'), ('feat_mean', 'feat_std')])
def test_load_norm_stats_key_styles(tmp_path, keys):
    path = tmp_path / 'stats.pt'
    torch.save({keys[0]: torch.tensor([1.0, 2.0, 3.0]), keys[1]: torch.tensor([4.0, 5.0, 6.0])}, path)
    mean, std = load_norm_stats(str(path))
    assert mean.tolist() == [1.0, 2.0, 3.0] and std.tolist() == [4.0, 5.0, 6.0]


def test_load_norm_stats_fallback():
    mean, std = load_norm_stats(None)
    assert torch.allclose(mean, torch.tensor(NOTEBOOK_FEAT_MEAN))
    assert torch.allclose(std, torch.tensor(NOTEBOOK_FEAT_STD))


# ── Model / training / scoring ────────────────────────────────────────────────
def test_part1_model_parameter_count():
    # Notebook reports 429; 9 of those belong to an attention gate whose output is never used.
    model = PhysicsAE(input_dim=3, latent_dim=1, hidden_layers=(16, 8), dropout=0.1)
    assert sum(p.numel() for p in model.parameters()) == 420


class InputSpy(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(3, 3)
        self.seen = []

    def forward(self, data):
        self.seen.append(data.x.detach().clone())
        return self.lin(data.x)


def _graphs():
    return [g for g in (create_radius_graph(r) for r in make_events(40, seed=3)) if g is not None]


def test_training_feeds_standardized_inputs():
    graphs = _graphs()
    mean, std = load_norm_stats(None)
    model = InputSpy()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loader = DataLoader(graphs, batch_size=len(graphs), shuffle=False)
    loss = train_one_epoch(model, loader, opt, 'cpu', mean, std, loss_mode='pt_only')
    raw = torch.cat([g.x for g in graphs])
    assert torch.allclose(model.seen[0], standardize_features(raw, mean, std))
    assert math.isfinite(loss)
    assert math.isfinite(validate(model, loader, 'cpu', mean, std, torch.ones(3), loss_mode='weighted'))


def test_unknown_loss_mode_raises():
    graphs = _graphs()
    mean, std = load_norm_stats(None)
    model = InputSpy()
    loader = DataLoader(graphs, batch_size=8)
    with pytest.raises(ValueError):
        train_one_epoch(model, loader, torch.optim.Adam(model.parameters()), 'cpu', mean, std, loss_mode='typo')


def test_scores_match_notebook_eval():
    torch.manual_seed(0)
    graphs = _graphs()
    mean, std = load_norm_stats(None)
    model = PhysicsAE(latent_dim=1, hidden_layers=(16, 8)).eval()
    loader = DataLoader(graphs, batch_size=16, shuffle=False)

    expected = []
    with torch.no_grad():
        for batch in DataLoader(graphs, batch_size=16, shuffle=False):
            batch.x = standardize_features(batch.x, mean, std)
            out = model(batch)
            node_errors = (out[:, 0] - batch.x[:, 0]) ** 2
            expected.extend(scatter(node_errors, batch.batch, dim=0, reduce='max').numpy())

    scores = compute_anomaly_scores(model, loader, 'cpu', mean, std, reduction='max', score_features=[0])
    assert np.allclose(scores, expected)


# ── Chunked dataset / sampler ─────────────────────────────────────────────────
def test_chunk_grouped_sampler_is_chunk_contiguous_permutation():
    indices = list(range(0, 95, 1))
    sampler = ChunkGroupedSampler(indices, chunk_size=10, generator=torch.Generator().manual_seed(0))
    order = list(sampler)
    assert sorted(order) == list(range(len(indices)))
    chunks = [indices[p] // 10 for p in order]
    switches = sum(1 for a, b in zip(chunks, chunks[1:]) if a != b)
    assert switches == len(set(chunks)) - 1


def test_raw_chunk_dataset_lru(tmp_path):
    for c in range(3):
        torch.save([Data(x=torch.full((2, 3), float(c * 10 + i))) for i in range(10)], tmp_path / f'data_chunk_{c}.pt')
    ds = RawChunkDataset(str(tmp_path), list(range(30)), max_cache=2, chunk_size=10)
    assert ds[0].x[0, 0] == 0 and ds[15].x[0, 0] == 15
    ds[1]           # touch chunk 0 -> chunk 1 becomes least recently used
    ds[25]          # evicts chunk 1
    assert list(ds.cache) == [0, 2]

    train_ld, val_ld, _ = get_discovery_loaders(str(tmp_path), list(range(30)), batch_size=4, chunk_size=10)
    assert sum(b.num_graphs for b in train_ld) == 27
    assert sum(b.num_graphs for b in val_ld) == 3


# ── Sorted files (events_anomalydetection_v2.h5 layout: background first, then signal) ──
@pytest.fixture
def sorted_h5_file(tmp_path):
    rows = make_events(600, seed=7, signal_frac=0.0)
    rows[500:, 2100] = 1.0
    path = tmp_path / 'sorted.h5'
    with h5py.File(path, 'w') as handle:
        handle.create_dataset('df/block0_values', data=rows)
    return str(path), rows


def test_oracle_split_full_scan_finds_signal_in_sorted_file(sorted_h5_file):
    path, rows = sorted_h5_file
    targets = dict(train_target=100, test_bg_target=30, test_sig_target=20)
    tr, te, tl, tr_idx, te_idx = build_oracle_purified_split(
        path, scan_limit=None, read_block=64, return_indices=True, **targets)
    rtr, rte, rtl = notebook_oracle(rows, **targets)
    assert tl == rtl and tl.count(1) == 20
    assert (te_idx[np.array(tl) == 1] >= 500).all()
    for a, b in zip(tr + te, rtr + rte):
        same_graph(a, b)
    for g, i in zip(tr + te, list(tr_idx) + list(te_idx)):
        same_graph(g, create_radius_graph(rows[i]))


def test_oracle_split_warns_when_quotas_unmet(sorted_h5_file):
    path, _ = sorted_h5_file
    with pytest.warns(UserWarning, match='test_sig'):
        _, _, tl = build_oracle_purified_split(path, train_target=100, test_bg_target=30,
                                               test_sig_target=20, scan_limit=400)
    assert 1 not in tl


def test_bootstrap_ci_single_class_is_nan():
    from src.utils.metrics import bootstrap_auc_ci
    lo, hi = bootstrap_auc_ci([0, 0, 0], [0.1, 0.2, 0.3], n_boot=10)
    assert math.isnan(lo) and math.isnan(hi)
