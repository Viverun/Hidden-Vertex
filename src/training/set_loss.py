import torch

_BIG = 1e9


def chamfer_per_event(pred, target, target_mask, target_weights=None):
    """Symmetric Chamfer distance per event, masked for events with fewer than K real particles.

    pred [B, K, D]: K predicted particles. target [B, K, D] with target_mask [B, K] marking real ones.
    Term 1: each prediction to its nearest *real* target (extra predictions in events with n < K are
    pulled onto real particles). Term 2: each real target to its nearest prediction, optionally weighted
    (weights normalised over the real targets). Events without real targets give 0.
    """
    diff = pred[:, :, None, :] - target[:, None, :, :]
    d = diff.square().sum(-1)                                            # [B, K_pred, K_tgt]
    has = target_mask.any(dim=1)
    pred_to_target = d.masked_fill(~target_mask[:, None, :], _BIG).min(dim=2).values
    term_pred = torch.where(has[:, None], pred_to_target, torch.zeros_like(pred_to_target)).mean(dim=1)
    target_to_pred = d.min(dim=1).values                                 # [B, K_tgt]
    w = target_mask.to(d.dtype) if target_weights is None else target_weights * target_mask
    w = w / w.sum(dim=1, keepdim=True).clamp(min=1e-8)
    return term_pred + (target_to_pred * w).sum(dim=1)


def chamfer_loss(pred, target, target_mask, target_weights=None):
    return chamfer_per_event(pred, target, target_mask, target_weights).mean()
