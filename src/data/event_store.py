# src/data/event_store.py
"""Row planning and the compact particle cache for Part 2.

Events are stored as pT-sorted, zero-padded arrays [N, max_particles, 3] (not graphs). Event-level
baseline quantities are computed from the full 700-slot rows before any truncation.
"""
import json
import os
from dataclasses import dataclass

import numpy as np
from tqdm import tqdm

from .dataset import LABEL_COLUMN, NUM_PARTICLE_COLUMNS, open_events
from .event_features import SUMMARY_KEYS, particle_summary

ROLES = ('train_bg', 'test_bg', 'test_sig', 'contam_sig')
BUILD_ORDER = ('test_bg', 'test_sig', 'contam_sig', 'train_bg')   # small roles first: a failing guard fails fast
PERCENTILES = (50, 90, 95, 99, 100)
ROLE_LABEL = {'train_bg': 0, 'test_bg': 0, 'test_sig': 1, 'contam_sig': 1}


@dataclass
class EventSet:
    particles: np.ndarray   # [N, max_particles, 3] float32
    rows: np.ndarray        # [N] HDF5 row of each event
    labels: np.ndarray      # [N] int8
    summary: dict           # SUMMARY_KEYS -> [N] float64, from the full events


def plan_rows(labels, skip_bg=40000, n_test_bg=5000, n_test_sig=5000, n_train=200000, n_contam_sig=0):
    """Chooses the HDF5 rows of each role from the label column (works for sorted or shuffled files).

    test_bg  = background events skip_bg .. skip_bg+n_test_bg (Part 1 trained on the first 40,000)
    test_sig = first n_test_sig signal events
    train_bg = the next n_train background events
    contam_sig = the n_contam_sig signal events after the test ones
    """
    labels = np.asarray(labels)
    bg = np.flatnonzero(labels == 0)
    sig = np.flatnonzero(labels == 1)
    need_bg = skip_bg + n_test_bg + n_train
    need_sig = n_test_sig + n_contam_sig
    if len(bg) < need_bg:
        raise ValueError(f'need {need_bg} background events, file has {len(bg)}')
    if len(sig) < need_sig:
        raise ValueError(f'need {need_sig} signal events, file has {len(sig)}')
    plan = {
        'test_bg': bg[skip_bg:skip_bg + n_test_bg],
        'test_sig': sig[:n_test_sig],
        'train_bg': bg[skip_bg + n_test_bg:need_bg],
        'contam_sig': sig[n_test_sig:need_sig],
    }
    allrows = np.concatenate(list(plan.values()))
    if len(np.unique(allrows)) != len(allrows):
        raise AssertionError('row overlap between roles')
    return {k: v.astype(np.int64) for k, v in plan.items()}


def read_labels(h5_path, cache_path=None, block=50000):
    if cache_path and os.path.exists(cache_path):
        return np.load(cache_path)
    h5_file, dset = open_events(h5_path)
    with h5_file:
        n = dset.shape[0]
        labels = np.empty(n, dtype=np.int8)
        for start in tqdm(range(0, n, block), desc='Reading labels'):
            labels[start:start + block] = dset[start:start + block, LABEL_COLUMN]
    if cache_path:
        os.makedirs(os.path.dirname(cache_path) or '.', exist_ok=True)
        np.save(cache_path, labels)
    return labels


def iter_row_blocks(dset, rows, block=10000):
    """Yields (positions, chunk) with chunk[i] == dset[rows[positions[i]]]; each HDF5 block is read once."""
    rows = np.asarray(rows)
    order = np.argsort(rows, kind='stable')
    sorted_rows = rows[order]
    i = 0
    while i < len(sorted_rows):
        start = (int(sorted_rows[i]) // block) * block
        j = int(np.searchsorted(sorted_rows, start + block, side='left'))
        data = dset[start:start + block]
        yield order[i:j], data[sorted_rows[i:j] - start]
        i = j


def build_role(h5_path, rows, max_particles, expected_label=None, block=10000):
    rows = np.asarray(rows, dtype=np.int64)
    n = len(rows)
    particles = np.zeros((n, max_particles, 3), dtype=np.float32)
    labels = np.zeros(n, dtype=np.int8)
    summary = {k: np.zeros(n) for k in SUMMARY_KEYS}
    if n == 0:
        return EventSet(particles, rows, labels, summary)
    h5_file, dset = open_events(h5_path)
    with h5_file:
        for positions, chunk in tqdm(iter_row_blocks(dset, rows, block), desc='Reading events'):
            chunk = np.asarray(chunk, dtype=np.float64)
            chunk_labels = chunk[:, LABEL_COLUMN].astype(np.int8)
            if expected_label is not None and (chunk_labels != expected_label).any():
                raise ValueError(f'unexpected label in rows (expected {expected_label})')
            raw = chunk[:, :NUM_PARTICLE_COLUMNS].reshape(len(chunk), -1, 3)
            for key, value in particle_summary(raw).items():
                summary[key][positions] = value
            order = np.argsort(-raw[..., 0], axis=1, kind='stable')
            ranked = np.take_along_axis(raw, order[..., None], axis=1)[:, :max_particles]
            particles[positions] = ranked
            labels[positions] = chunk_labels
    return EventSet(particles, rows, labels, summary)


def save_role(path, es):
    """Atomic: writes `<role>.tmp.npz` next to the target (np.savez appends .npz to other names), then renames."""
    base = path[:-len('.npz')] if path.endswith('.npz') else path
    tmp = f'{base}.tmp.npz'
    try:
        np.savez(tmp, particles=es.particles, rows=es.rows, labels=es.labels,
                 **{f'summary_{k}': v for k, v in es.summary.items()})
        os.replace(tmp, f'{base}.npz')
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def multiplicity_percentiles(es):
    """{50: .., 90: .., ...} of the true (pre-truncation) particle multiplicity of a role."""
    if len(es.rows) == 0:
        return {q: 0.0 for q in PERCENTILES}
    values = np.percentile(es.summary['multiplicity'], PERCENTILES)
    return {q: float(v) for q, v in zip(PERCENTILES, values)}


def _format_percentiles(pct):
    return ' '.join(f'p{q}={v:.0f}' for q, v in pct.items())


def load_role(cache_dir, role):
    with np.load(os.path.join(cache_dir, f'{role}.npz')) as data:
        return EventSet(
            particles=data['particles'], rows=data['rows'], labels=data['labels'],
            summary={k: data[f'summary_{k}'] for k in SUMMARY_KEYS},
        )


def build_cache(h5_path, cache_dir, split, max_particles, max_truncated_fraction=0.02):
    """Builds `<cache_dir>/<role>.npz` for every role (skipping existing ones) and returns the plan.

    Roles are built in BUILD_ORDER (small ones first). cache_meta.json is written only just before the
    first role file is saved, so a guard failure on a fresh directory leaves nothing behind.
    """
    os.makedirs(cache_dir, exist_ok=True)
    meta = {'split': dict(split), 'max_particles': max_particles}
    meta_path = os.path.join(cache_dir, 'cache_meta.json')
    if os.path.exists(meta_path):
        with open(meta_path, encoding='utf-8') as f:
            stored = json.load(f)
        if stored != meta:
            raise RuntimeError(f'{cache_dir} was built with different split/max_particles settings; '
                               f'use another cache_dir or delete {cache_dir} and re-run')
    labels = read_labels(h5_path, os.path.join(cache_dir, 'labels.npy'))
    plan = plan_rows(labels, **split)
    for role in BUILD_ORDER:
        path = os.path.join(cache_dir, f'{role}.npz')
        if os.path.exists(path):
            continue
        es = build_role(h5_path, plan[role], max_particles, ROLE_LABEL[role])
        frac = float((es.summary['multiplicity'] > max_particles).mean()) if len(es.rows) else 0.0
        pct = _format_percentiles(multiplicity_percentiles(es))
        print(f'{role}: {len(es.rows)} events, {frac:.2%} truncated at {max_particles} particles; '
              f'multiplicity {pct}', flush=True)
        if frac > max_truncated_fraction:
            raise RuntimeError(f'{role}: {frac:.2%} of events are truncated at max_particles={max_particles} '
                               f'(limit {max_truncated_fraction:.0%}); multiplicity percentiles {pct}; '
                               f'raise data.max_particles, or delete {cache_dir} before re-running '
                               'with a different max_particles')
        if not os.path.exists(meta_path):
            with open(meta_path, 'w', encoding='utf-8') as f:
                json.dump(meta, f)
        save_role(path, es)
    return plan
