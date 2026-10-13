# tests/test_part2_loss.py
import numpy as np
import torch

from src.training.set_loss import chamfer_loss, chamfer_per_event


def _brute(pred, target, mask, weights=None):
    out = []
    for b in range(pred.shape[0]):
        valid = np.flatnonzero(mask[b])
        if len(valid) == 0:
            out.append(0.0)
            continue
        d = ((pred[b][:, None, :] - target[b][None, valid, :]) ** 2).sum(-1)   # [K, n_valid]
        pred_term = d.min(1).mean()
        w = np.ones(len(valid)) if weights is None else weights[b][valid]
        w = w / w.sum()
        out.append(pred_term + (d.min(0) * w).sum())
    return np.array(out)


def test_identical_sets_have_zero_loss_and_permutation_invariance():
    torch.manual_seed(0)
    target = torch.randn(2, 6, 4)
    mask = torch.ones(2, 6, dtype=torch.bool)
    assert torch.allclose(chamfer_per_event(target, target, mask), torch.zeros(2), atol=1e-7)
    perm = torch.randperm(6)
    assert torch.allclose(chamfer_per_event(target[:, perm], target, mask), torch.zeros(2), atol=1e-7)


def test_matches_bruteforce_with_masks_and_weights():
    torch.manual_seed(1)
    pred, target = torch.randn(5, 7, 4), torch.randn(5, 7, 4)
    mask = torch.rand(5, 7) > 0.3
    target = target * mask.unsqueeze(-1)
    w = torch.rand(5, 7) + 0.1
    for weights in (None, w):
        got = chamfer_per_event(pred, target, mask, weights).numpy()
        want = _brute(pred.numpy(), target.numpy(), mask.numpy(), None if weights is None else weights.numpy())
        np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-6)


def test_invalid_target_slots_do_not_matter():
    torch.manual_seed(2)
    pred = torch.randn(1, 5, 4)
    target = torch.randn(1, 5, 4)
    mask = torch.tensor([[True, True, True, False, False]])
    a = chamfer_per_event(pred, target * mask.unsqueeze(-1), mask)
    garbage = target.clone(); garbage[:, 3:] = 1e3
    assert torch.allclose(a, chamfer_per_event(pred, garbage, mask))


def test_fewer_particles_than_k_can_be_matched_exactly():
    real = torch.randn(1, 3, 4)
    target = torch.zeros(1, 5, 4); target[:, :3] = real
    mask = torch.tensor([[True, True, True, False, False]])
    pred = torch.cat([real, real[:, :1], real[:, :1]], dim=1)     # extra slots sit on real particles
    assert chamfer_per_event(pred, target, mask).item() < 1e-10


def test_empty_event_is_zero_with_finite_gradients():
    pred = torch.randn(2, 4, 4, requires_grad=True)
    target = torch.zeros(2, 4, 4)
    mask = torch.zeros(2, 4, dtype=torch.bool); mask[1, :2] = True
    loss = chamfer_loss(pred, target, mask)
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(pred.grad).all()
    assert chamfer_per_event(pred, target, mask)[0].item() == 0.0
    assert torch.all(pred.grad[0] == 0)
