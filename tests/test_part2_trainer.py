import pytest
import torch

from src.data.event_features import fit_feature_stats
from src.model.event_ae import EventAE
from src.training.event_trainer import BatchContext, TrainingError, infer, per_event_loss, train_event_ae


def _events(n=256, p=24, seed=0):
    g = torch.Generator().manual_seed(seed)
    out = torch.zeros(n, p, 3)
    for i in range(n):
        k = int(torch.randint(4, p, (1,), generator=g))
        pt = torch.sort(torch.rand(k, generator=g) * 50 + 1, descending=True).values
        out[i, :k] = torch.stack([pt, torch.randn(k, generator=g), torch.rand(k, generator=g) * 6 - 3], -1)
    return out


def _ctx(p, loss='chamfer'):
    return BatchContext(stats=fit_feature_stats(p), k=8, radius=0.4, wrap_phi=True, loss=loss)


def _model():
    torch.manual_seed(0)
    return EventAE(width=16, layers=2, hidden=32, latent_dim=4, k_recon=8)


KW = dict(epochs=4, batch_size=64, lr=3e-3, weight_decay=0.0, patience=10, grad_clip=5.0,
          collapse_std=1e-6, seed=0, log=lambda *_: None)


def test_training_reduces_validation_loss_and_restores_best_weights():
    p = _events()
    train, val = p[:200], p[200:]
    ctx, model = _ctx(train), _model()
    before, _ = infer(model, val, ctx)
    history = train_event_ae(model, train, val, ctx, **KW)
    after, z = infer(model, val, ctx)
    assert len(history) >= 1 and {'epoch', 'train_loss', 'val_loss', 'z_std_max', 'seconds'} <= set(history[0])
    assert after.mean() < before.mean()
    assert abs(after.mean() - min(h['val_loss'] for h in history)) < 1e-4      # best weights restored
    assert z.shape == (56, 4)


def test_early_stopping_stops_before_max_epochs():
    p = _events()
    kw = dict(KW, epochs=50, patience=1, lr=0.0)                                # nothing learns, so epoch 2 cannot improve
    history = train_event_ae(_model(), p[:200], p[200:], _ctx(p[:200]), **kw)
    assert len(history) == 2


def test_ptweighted_loss_runs():
    p = _events(n=32)
    loss, z = per_event_loss(_model(), p, _ctx(p, 'chamfer_ptw'))
    assert loss.shape == (32,) and torch.isfinite(loss).all()


class _NaNModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.w = torch.nn.Parameter(torch.zeros(1))

    def forward(self, batch):
        b = batch.feats.shape[0]
        return self.w * float('nan') + torch.zeros(b, 8, 4), torch.zeros(b, 4) + self.w


class _CollapsedModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.w = torch.nn.Parameter(torch.zeros(1))

    def forward(self, batch):
        b = batch.feats.shape[0]
        return self.w + torch.zeros(b, 8, 4), torch.zeros(b, 4)


def test_nonfinite_loss_aborts():
    p = _events(n=64)
    with pytest.raises(TrainingError, match='non-finite'):
        train_event_ae(_NaNModel(), p[:48], p[48:], _ctx(p), **KW)


def test_collapsed_latent_aborts():
    p = _events(n=64)
    with pytest.raises(TrainingError, match='collapsed'):
        train_event_ae(_CollapsedModel(), p[:48], p[48:], _ctx(p), **dict(KW, collapse_std=1e-3))
