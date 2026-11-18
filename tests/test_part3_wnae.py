# tests/test_part3_wnae.py
import numpy as np
import pytest
import torch

from src.model.mlp_ae import MLPAE, recon_error, train_ae
from src.model.wnae import SampleBuffer, WNAETrainer, euclidean_cost, langevin, wasserstein

MCMC = {'n_steps': 5, 'step_size': 0.0242, 'noise_std': 0.22, 'temperature': 0.063, 'restart_fraction': 0.05,
        'bound': (-3.0, 3.0)}


def _data(n, d=5, seed=0):
    rng = np.random.default_rng(seed)
    z = rng.normal(size=(n, 2))
    return (z @ rng.normal(size=(2, d)) + 0.1 * rng.normal(size=(n, d))).astype(np.float32)


def test_mlp_ae_shapes_and_energy():
    model = MLPAE(5)
    x = torch.randn(7, 5)
    assert model(x).shape == (7, 5)
    assert model.encoder(x).shape == (7, 3)
    assert torch.allclose(model.energy(x), ((x - model(x)) ** 2).mean(1))
    assert sum(isinstance(m, torch.nn.Linear) for m in model.modules()) == 6


def test_train_ae_learns_and_restores_best():
    torch.manual_seed(0)
    x = _data(2000)
    model = MLPAE(5)
    before = recon_error(model, x).mean()
    history, best_epoch = train_ae(model, x[:1600], x[1600:], epochs=5, batch_size=128, seed=0)
    assert recon_error(model, x[1600:]).mean() < before
    assert recon_error(model, x[1600:]).mean() == pytest.approx(min(h['val_loss'] for h in history), rel=1e-5)
    assert history[best_epoch - 1]['val_loss'] == min(h['val_loss'] for h in history)


def test_buffer_fifo_and_restart_fraction():
    buf = SampleBuffer(max_samples=100)
    x0 = buf.initial(50, 3, 0.05)
    assert x0.shape == (50, 3)                        # empty buffer: all from N(0,1)
    buf.push(torch.full((80, 3), 7.0))
    buf.push(torch.full((40, 3), 9.0))
    assert len(buf) == 100 and float(buf.data[-1, 0]) == 9.0 and int((buf.data == 7.0).all(1).sum()) == 60
    x0 = buf.initial(20000, 3, 0.05, torch.Generator().manual_seed(0))
    fresh = ~((x0 == 7.0) | (x0 == 9.0)).all(1)
    assert 0.04 < float(fresh.float().mean()) < 0.06


def test_langevin_lowers_energy_and_respects_bound():
    energy = lambda x: (x ** 2).sum(1)  # noqa: E731
    x0 = torch.full((100, 2), 2.5)
    x = langevin(energy, x0, 10, 0.01, 0.0, 1.0, bound=(-3, 3))
    assert float(energy(x).mean()) < float(energy(x0).mean())
    x = langevin(lambda x: -(x ** 2).sum(1), x0, 50, 0.1, 0.0, 1.0, bound=(-3, 3))
    assert float(x.abs().max()) <= 3.0


def test_wasserstein_zero_on_identical_sets_and_euclidean_cost():
    a = torch.randn(64, 3)
    assert float(wasserstein(a, a.clone())[0]) == pytest.approx(0.0, abs=1e-4)
    assert torch.allclose(euclidean_cost(a, a + 1.0), torch.cdist(a, a + 1.0), atol=1e-4)
    shifted = float(wasserstein(a, a + torch.tensor([3.0, 4.0, 0.0]))[0])
    assert shifted == pytest.approx(5.0, rel=1e-3)        # pure translation: W1 = |shift|


def test_gradient_flows_through_negative_samples():
    torch.manual_seed(0)
    model = MLPAE(5)
    x = torch.randn(128, 5)
    neg = langevin(model.energy, torch.randn(128, 5), 3, 0.0242, 0.22, 0.063, (-3, 3), create_graph=True)
    loss, _ = wasserstein(x, neg)
    loss.backward()
    grads = [p.grad for p in model.parameters()]
    assert all(g is not None for g in grads) and sum(float(g.abs().sum()) for g in grads) > 0


def test_wnae_trainer_runs_and_selects_lowest_val_w():
    torch.manual_seed(0)
    x = _data(1200)
    model = MLPAE(5)
    trainer = WNAETrainer(model, 5, MCMC, batch_size=256, lr=1e-3, seed=0, buffer_size=1000)
    history, best_epoch, diverged = trainer.fit(x[:1000], x[1000:], max_epochs=3, log=lambda m: None)
    assert diverged is None and len(history) == 3
    assert all(np.isfinite(h['train_w']) and np.isfinite(h['val_w']) for h in history)
    assert history[best_epoch - 1]['val_w'] == min(h['val_w'] for h in history)
    assert len(history[0]['neg_mean']) == 5 and len(trainer.buffer) == 1000
