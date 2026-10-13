import torch

from src.data.event_features import fit_feature_stats, make_event_batch
from src.model.event_ae import EventAE


def _particles(n_events=6, p=20, seed=0):
    g = torch.Generator().manual_seed(seed)
    out = torch.zeros(n_events, p, 3)
    for i in range(n_events):
        n = int(torch.randint(3, p, (1,), generator=g))
        pt = torch.sort(torch.rand(n, generator=g) * 100 + 1, descending=True).values
        out[i, :n] = torch.stack([pt, torch.randn(n, generator=g), torch.rand(n, generator=g) * 6 - 3], -1)
    return out


def _model(**kw):
    torch.manual_seed(0)
    return EventAE(width=16, layers=2, hidden=32, latent_dim=4, k_recon=8, **kw).eval()


def test_output_shapes_and_gradients():
    p = _particles()
    batch = make_event_batch(p, fit_feature_stats(p), k=8)
    model = _model().train()
    recon, z = model(batch)
    assert recon.shape == (6, 8, 4) and z.shape == (6, 4)
    (recon.square().mean() + z.square().mean()).backward()
    assert all(q.grad is not None and torch.isfinite(q.grad).all() for q in model.parameters())


def test_permutation_invariance():
    p = _particles()
    stats = fit_feature_stats(p)
    batch = make_event_batch(p, stats, k=8)
    model = _model()
    perm = torch.randperm(p.shape[1])
    permuted = batch._replace(
        feats=batch.feats[:, perm], mask=batch.mask[:, perm], pt_frac=batch.pt_frac[:, perm],
        adj=batch.adj[:, perm][:, :, perm],
    )
    with torch.no_grad():
        _, z = model(batch)
        _, z_perm = model(permuted)
    assert torch.allclose(z, z_perm, atol=1e-5)


def test_padding_contributes_nothing():
    p = _particles(p=20)
    stats = fit_feature_stats(p)
    padded = torch.cat([p, torch.zeros(p.shape[0], 15, 3)], dim=1)
    model = _model()
    with torch.no_grad():
        _, z = model(make_event_batch(p, stats, k=8))
        _, z_padded = model(make_event_batch(padded, stats, k=8))
    assert torch.allclose(z, z_padded, atol=1e-5)


def test_pooling_options_and_empty_event():
    p = _particles()
    p[0] = 0                                           # an event with no particles
    batch = make_event_batch(p, fit_feature_stats(_particles()), k=8)
    for pooling in (('mean', 'max'), ('mean', 'max', 'ptsum'), ('ptsum',)):
        _, z = _model(pooling=pooling)(batch)
        assert torch.isfinite(z).all()
