# tests/test_part2_store.py
import numpy as np
import pytest

from src.data.event_store import (
    build_cache,
    build_role,
    iter_row_blocks,
    load_role,
    plan_rows,
    read_labels,
)


def test_plan_rows_layout():
    labels = np.r_[np.zeros(100, int), np.ones(30, int)]
    plan = plan_rows(labels, skip_bg=10, n_test_bg=5, n_test_sig=5, n_train=20, n_contam_sig=4)
    assert plan['test_bg'].tolist() == list(range(10, 15))
    assert plan['train_bg'].tolist() == list(range(15, 35))
    assert plan['test_sig'].tolist() == list(range(100, 105))
    assert plan['contam_sig'].tolist() == list(range(105, 109))


def test_plan_rows_is_disjoint_and_label_correct_on_shuffled_labels():
    rng = np.random.default_rng(0)
    labels = rng.permutation(np.r_[np.zeros(300, int), np.ones(80, int)])
    plan = plan_rows(labels, skip_bg=20, n_test_bg=30, n_test_sig=30, n_train=100, n_contam_sig=20)
    allrows = np.concatenate(list(plan.values()))
    assert len(np.unique(allrows)) == len(allrows)
    assert (labels[plan['train_bg']] == 0).all() and (labels[plan['test_bg']] == 0).all()
    assert (labels[plan['test_sig']] == 1).all() and (labels[plan['contam_sig']] == 1).all()


def test_plan_rows_raises_when_quota_unmet():
    with pytest.raises(ValueError, match='signal'):
        plan_rows(np.zeros(500, int), skip_bg=10, n_test_bg=5, n_test_sig=5, n_train=20)


def test_iter_row_blocks_returns_requested_rows_in_any_order(events_h5):
    import h5py
    rows = np.array([205, 3, 150, 4, 3])
    with h5py.File(events_h5) as f:
        dset = f['df']['block0_values']
        out = np.zeros((len(rows), 2101))
        for positions, chunk in iter_row_blocks(dset, rows, block=64):
            out[positions] = chunk
        assert np.array_equal(out, dset[:][rows])


def test_read_labels_uses_and_writes_cache(events_h5, tmp_path):
    cache = tmp_path / 'labels.npy'
    labels = read_labels(events_h5, str(cache), block=50)
    assert labels.shape == (260,) and labels[:200].sum() == 0 and labels[200:].sum() == 60
    assert cache.exists()
    assert np.array_equal(read_labels(events_h5, str(cache)), labels)


def test_build_role_sorts_pads_and_keeps_exact_summary(events_h5):
    rows = np.array([3, 0, 150])
    es = build_role(events_h5, rows, max_particles=10, expected_label=0)
    assert es.particles.shape == (3, 10, 3) and es.particles.dtype == np.float32
    pt = es.particles[..., 0]
    assert (np.diff(pt, axis=1) <= 0).all()                      # descending, zeros at the end
    assert (es.summary['multiplicity'] > 10).any()               # truncated events keep true multiplicity
    assert (es.summary['ht'] > pt.sum(1) - 1e-3).all()
    assert np.array_equal(es.rows, rows)


def test_build_role_checks_expected_label(events_h5):
    with pytest.raises(ValueError, match='label'):
        build_role(events_h5, np.array([210]), max_particles=10, expected_label=0)


def test_build_cache_roundtrip_and_guards(events_h5, tmp_path):
    split = dict(skip_bg=10, n_test_bg=30, n_test_sig=30, n_train=100, n_contam_sig=10)
    cache = str(tmp_path / 'cache')
    plan = build_cache(events_h5, cache, split, max_particles=40)
    sig = load_role(cache, 'test_sig')
    assert sig.rows[0] == 200 and (sig.labels == 1).all() and sig.particles.shape == (30, 40, 3)
    assert len(load_role(cache, 'train_bg').rows) == 100 and len(plan['contam_sig']) == 10
    with pytest.raises(RuntimeError, match='truncated'):
        build_cache(events_h5, str(tmp_path / 'c2'), split, max_particles=5)
    with pytest.raises(RuntimeError, match='different'):
        build_cache(events_h5, cache, split, max_particles=41)
