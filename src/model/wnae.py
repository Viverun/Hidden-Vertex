"""Wasserstein normalized autoencoder (WNAE; CMS, arXiv:2510.02168), clean reimplementation.

Follows the authors' reference code (github.com/fleble/wnae, read via a public mirror):
- energy E(x) = per-example MSE reconstruction error (mean over features);
- negative samples by Langevin MCMC in input space, x <- clip(x - (lambda/T) grad_x E(x) + sigma * eps),
  with create_graph=True so the negatives keep their dependence on the weights;
- persistent contrastive divergence: a FIFO buffer of past negatives (max 10,000); each chain starts from the
  buffer with probability 1 - restart_fraction, otherwise from N(0, 1) (drawn with replacement, as reference);
- loss = exact optimal-transport distance between the positive batch and the negatives (`ot.emd2`).
Differences from the reference are listed in configs/part3_jet.yaml.
"""
import copy
import time

import numpy as np
import ot
import torch


class SampleBuffer:
    def __init__(self, max_samples=10000):
        self.max_samples = max_samples
        self.data = None

    def __len__(self):
        return 0 if self.data is None else len(self.data)

    def push(self, samples):
        samples = samples.detach().cpu()
        self.data = samples if self.data is None else torch.cat([self.data, samples])
        if len(self.data) > self.max_samples:
            self.data = self.data[-self.max_samples:]

    def initial(self, n, dim, restart_fraction, generator=None):
        """n chain starts: buffer draws with prob 1 - restart_fraction (if the buffer is not empty), else N(0,1)."""
        x0 = torch.randn(n, dim, generator=generator)
        if len(self):
            from_buffer = torch.rand(n, generator=generator) >= restart_fraction
            k = int(from_buffer.sum())
            idx = torch.randint(0, len(self), (k,), generator=generator)
            x0[from_buffer] = self.data[idx]
        return x0


def langevin(energy, x0, n_steps, step_size, noise_std, temperature, bound=None, create_graph=True,
             generator=None):
    """Langevin chain on `energy` (per-example tensor function). Returns the final samples."""
    x = x0.detach().clone().requires_grad_(True)
    for _ in range(n_steps):
        grad = torch.autograd.grad(energy(x).sum(), x, create_graph=create_graph)[0]
        noise = torch.randn(x.shape, generator=generator) * noise_std
        x = x - (step_size / temperature) * grad + noise
        if bound is not None:
            x = torch.clamp(x, bound[0], bound[1])
        if not create_graph:
            x = x.detach().requires_grad_(True)
    return x


def euclidean_cost(a, b):
    sq = (a * a).sum(1, keepdim=True) + (b * b).sum(1)[None, :] - 2.0 * a @ b.T
    return torch.sqrt(sq.clamp_min(1e-12))


def wasserstein(positive, negative, num_iter_max=10_000_000):
    """Exact OT cost with a Euclidean ground cost (W1), uniform weights. Returns (tensor, emd log)."""
    cost = euclidean_cost(positive, negative)
    w_pos = torch.full((len(positive),), 1.0 / len(positive), dtype=cost.dtype)
    w_neg = torch.full((len(negative),), 1.0 / len(negative), dtype=cost.dtype)
    value, log = ot.emd2(w_pos, w_neg, cost, numItermax=num_iter_max, log=True)
    return value, log


class WNAETrainer:
    """Trains an autoencoder with `energy(x)` (per example) by the WNAE objective."""

    def __init__(self, model, n_inputs, mcmc, batch_size=4096, lr=2e-4, scheduler=None, seed=0,
                 buffer_size=10000, ot_num_iter_max=10_000_000):
        self.model = model
        self.n_inputs = n_inputs
        self.mcmc = dict(mcmc)
        self.batch_size = batch_size
        self.seed = seed
        self.ot_num_iter_max = ot_num_iter_max
        self.buffer = SampleBuffer(buffer_size)
        self.opt = torch.optim.Adam(model.parameters(), lr=lr)
        sched = scheduler or {}
        self.sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.opt, mode='min', factor=sched.get('factor', 0.8), patience=sched.get('patience', 20),
            cooldown=sched.get('cooldown', 10), threshold=sched.get('threshold', 1e-3),
            threshold_mode=sched.get('threshold_mode', 'rel'))
        self.gen = torch.Generator().manual_seed(seed)

    def _negatives(self, n, create_graph, generator):
        m = self.mcmc
        x0 = self.buffer.initial(n, self.n_inputs, m['restart_fraction'], generator)
        return langevin(self.model.energy, x0, m['n_steps'], m['step_size'], m['noise_std'], m['temperature'],
                        m.get('bound'), create_graph=create_graph, generator=generator)

    def train_epoch(self, x_fit):
        self.model.train()
        x_fit = torch.as_tensor(x_fit, dtype=torch.float32)
        perm = torch.randperm(len(x_fit), generator=self.gen)
        n_batches = len(x_fit) // self.batch_size        # drop the last partial batch
        out = {'w': 0.0, 'e_pos': 0.0, 'e_neg': 0.0, 'ot_warnings': 0}
        for b in range(n_batches):
            x = x_fit[perm[b * self.batch_size:(b + 1) * self.batch_size]]
            neg = self._negatives(len(x), create_graph=True, generator=self.gen)
            loss, log = wasserstein(x, neg, self.ot_num_iter_max)
            if not torch.isfinite(loss):
                raise FloatingPointError(f'non-finite WNAE loss (batch {b})')
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
            self.buffer.push(neg)
            with torch.no_grad():
                out['e_pos'] += float(self.model.energy(x).mean())
                out['e_neg'] += float(self.model.energy(neg.detach()).mean())
            out['w'] += float(loss)
            out['ot_warnings'] += int(bool(log.get('warning')))
        for k in ('w', 'e_pos', 'e_neg'):
            out[k] /= n_batches
        return out

    def validate(self, x_val, epoch_seed):
        """Mean W over validation batches. The MCMC noise uses a fixed generator so epochs are comparable;
        the buffer is read, not written (as in the reference). Also returns negative/positive feature moments."""
        self.model.eval()
        x_val = torch.as_tensor(x_val, dtype=torch.float32)
        gen = torch.Generator().manual_seed(epoch_seed)
        ws, sizes, e_pos, e_neg, negs = [], [], [], [], []
        for i in range(0, len(x_val), self.batch_size):
            x = x_val[i:i + self.batch_size]
            neg = self._negatives(len(x), create_graph=False, generator=gen).detach()
            with torch.no_grad():
                w, _ = wasserstein(x, neg, self.ot_num_iter_max)
                e_pos.append(float(self.model.energy(x).mean()))
                e_neg.append(float(self.model.energy(neg).mean()))
            ws.append(float(w))
            sizes.append(len(x))
            negs.append(neg.numpy())
        negs = np.concatenate(negs)
        wts = np.asarray(sizes, dtype=float) / sum(sizes)
        return {'w': float(np.dot(ws, wts)), 'e_pos': float(np.dot(e_pos, wts)), 'e_neg': float(np.dot(e_neg, wts)),
                'neg_mean': negs.mean(0).tolist(), 'neg_std': negs.std(0).tolist(),
                'pos_mean': x_val.numpy().mean(0).tolist(), 'pos_std': x_val.numpy().std(0).tolist()}

    def fit(self, x_fit, x_val, max_epochs, log=print, es_patience=None, es_threshold=1e-3, on_epoch=None):
        """Trains up to max_epochs; keeps the state with the lowest validation W. Stops on a non-finite loss.

        With es_patience, also stops when the validation W has not improved by more than es_threshold
        (relative) on the best-so-far value for es_patience epochs. `self.early_stopped` records it."""
        history, best, best_state, diverged = [], float('inf'), None, None
        es_best, es_wait, self.early_stopped = float('inf'), 0, None
        for epoch in range(1, max_epochs + 1):
            started = time.time()
            try:
                tr = self.train_epoch(x_fit)
                va = self.validate(x_val, epoch_seed=self.seed * 100_003 + 7)
            except FloatingPointError as err:
                diverged = {'epoch': epoch, 'reason': str(err)}
                log(f'epoch {epoch}: diverged ({err})')
                break
            if not np.isfinite(va['w']):
                diverged = {'epoch': epoch, 'reason': 'non-finite validation W'}
                break
            self.sched.step(va['w'])
            rec = {'epoch': epoch, 'train_w': tr['w'], 'train_e_pos': tr['e_pos'], 'train_e_neg': tr['e_neg'],
                   'ot_warnings': tr['ot_warnings'], 'val_w': va['w'], 'val_e_pos': va['e_pos'],
                   'val_e_neg': va['e_neg'], 'lr': self.opt.param_groups[0]['lr'],
                   'neg_mean': va['neg_mean'], 'neg_std': va['neg_std'], 'pos_mean': va['pos_mean'],
                   'pos_std': va['pos_std'], 'seconds': time.time() - started}
            history.append(rec)
            log(f"epoch {epoch}: train W {tr['w']:.4f} val W {va['w']:.4f} E+ {va['e_pos']:.4f} "
                f"E- {va['e_neg']:.4f} lr {rec['lr']:.2e} ({rec['seconds']:.0f} s)")
            if va['w'] < best:
                best, best_state = va['w'], copy.deepcopy(self.model.state_dict())
            if on_epoch is not None:
                on_epoch(rec, best_state)
            if es_patience is not None:
                if va['w'] < es_best * (1.0 - es_threshold):
                    es_best, es_wait = va['w'], 0
                else:
                    es_wait += 1
                    if es_wait >= es_patience:
                        self.early_stopped = epoch
                        log(f'epoch {epoch}: early stop (no rel. {es_threshold:g} improvement for {es_patience} epochs)')
                        break
        if best_state is not None:
            self.model.load_state_dict(best_state)
        best_epoch = min(history, key=lambda h: h['val_w'])['epoch'] if history else None
        return history, best_epoch, diverged
