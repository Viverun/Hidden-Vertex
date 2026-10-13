import math

import numpy as np
import pytest
import torch

from src.data.event_features import (
    FEATURE_NAMES,
    EventBatch,
    build_event_features,
    build_targets,
    fit_feature_stats,
    make_event_batch,
    particle_summary,
    wrap_delta_phi,
)


def _particles():
    # event 0: three particles (pT-sorted), event 1: one particle, event 2: empty
    p = torch.zeros(3, 4, 3)
    p[0, :3] = torch.tensor([[30.0, 0.5, 3.1], [20.0, -0.5, -3.1], [10.0, 1.0, 0.0]])
    p[1, 0] = torch.tensor([50.0, 0.0, 1.0])
    return p


def test_particle_summary_values():
    particles = np.zeros((1, 5, 3))
    particles[0, :2] = [[10.0, 0.0, 0.0], [30.0, 1.0, 2.0]]
    s = particle_summary(particles)
    assert s['ht'][0] == 40.0 and s['multiplicity'][0] == 2 and s['mean_pt'][0] == 20.0
    assert s['max_pt'][0] == 30.0
    pt = np.array([10.0, 30.0]); eta = np.array([0.0, 1.0]); phi = np.array([0.0, 2.0])
    e = (pt * np.cosh(eta)).sum()
    px, py, pz = (pt * np.cos(phi)).sum(), (pt * np.sin(phi)).sum(), (pt * np.sinh(eta)).sum()
    assert s['m_all'][0] == pytest.approx(np.sqrt(e**2 - px**2 - py**2 - pz**2))


def test_particle_summary_empty_event_is_finite():
    s = particle_summary(np.zeros((1, 5, 3)))
    assert all(np.isfinite(v).all() for v in s.values())
    assert s['multiplicity'][0] == 0 and s['mean_pt'][0] == 0


def test_wrap_delta_phi():
    d = torch.tensor([(math.pi - 0.1) - (-math.pi + 0.1), 0.3, -0.3])
    out = wrap_delta_phi(d)
    assert torch.allclose(out, torch.tensor([-0.2, 0.3, -0.3]), atol=1e-6)


def test_build_event_features_masks_padding_and_wraps_phi():
    p = _particles()
    mask = p[..., 0] > 0
    f = build_event_features(p, mask)
    assert f.shape == (3, 4, len(FEATURE_NAMES))
    assert torch.all(f[~mask] == 0)
    i = {n: k for k, n in enumerate(FEATURE_NAMES)}
    assert torch.allclose(f[0, :3, i['pt_frac']].sum(), torch.tensor(1.0))
    assert f[0, 0, i['d_eta_lead']] == 0 and f[0, 0, i['d_phi_lead']] == 0
    # particle 1 is at phi=-3.1, leader at 3.1: wrapped delta is +0.0832, not -6.2
    assert abs(f[0, 1, i['d_phi_lead']].item() - (2 * math.pi - 6.2)) < 1e-5
    assert abs(f[0, 0, i['log_pt']].item() - math.log(30.0)) < 1e-6
    assert torch.isfinite(f).all()


def test_build_targets_masks_events_with_fewer_than_k_particles():
    p = _particles()
    mask = p[..., 0] > 0
    target, tmask = build_targets(p, mask, k=4)
    assert target.shape == (3, 4, 4) and tmask.sum(1).tolist() == [3, 1, 0]
    assert torch.all(target[~tmask] == 0)
    norm = target[..., 2] ** 2 + target[..., 3] ** 2
    assert torch.allclose(norm[tmask], torch.ones(4))


def test_fit_feature_stats_matches_direct_computation():
    torch.manual_seed(0)
    p = torch.zeros(50, 10, 3)
    n = torch.randint(2, 10, (50,))
    for i in range(50):
        k = int(n[i])
        pt = torch.sort(torch.rand(k) * 100 + 1, descending=True).values
        p[i, :k] = torch.stack([pt, torch.randn(k), torch.rand(k) * 6 - 3], -1)
    mask = p[..., 0] > 0
    feats = build_event_features(p, mask)[mask]
    stats = fit_feature_stats(p, chunk=7)
    assert torch.allclose(stats.mean, feats.mean(0), atol=1e-4)
    assert torch.allclose(stats.std, feats.std(0, unbiased=False), atol=1e-4)


def test_make_event_batch_shapes():
    p = _particles()
    stats = fit_feature_stats(p)
    batch = make_event_batch(p, stats, k=3, radius=0.4, wrap_phi=True)
    assert isinstance(batch, EventBatch)
    assert batch.feats.shape == (3, 4, len(FEATURE_NAMES))
    assert batch.adj.shape == (3, 4, 4) and batch.adj.dtype == torch.bool
    assert batch.target.shape == (3, 3, 4) and batch.target_pt.shape == (3, 3)
    assert torch.all(batch.feats[~batch.mask] == 0)
