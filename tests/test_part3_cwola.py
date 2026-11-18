# tests/test_part3_cwola.py  --  Part 3d CWoLa bump hunt
import json
import os

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from src.pipeline import part3_cwola as cw

REAL_CFG = 'configs/part3_cwola.yaml'
INF = np.inf


def _small_cfg(tmp_path=None, min_events=50, ensemble=2):
    cfg = cw.load_config(REAL_CFG)
    cfg['windows']['min_events_per_class'] = min_events
    cfg['classifier'].update({'max_iter': 60, 'ensemble': ensemble})
    if tmp_path is not None:
        d = str(tmp_path / 'cwola')
        cfg['output'] = {'dir': d, 'report': str(tmp_path / 'cwola.md'), 'json': str(tmp_path / 'cwola.json')}
    return cfg


# ----------------------------------------------------------------------------------------------- windows
def test_grid_a_windows_open_ended_and_interior():
    cfg = cw.load_config(REAL_CFG)
    w = cfg['windows']
    win = cw.grid_windows(w['grid_a_edges'], w['width'], w['sideband'])
    assert len(win) == 8
    assert win[0]['sr'] == (-INF, 2.4) and win[0]['sb'] == [(2.4, 2.8)]
    assert win[1]['sr'] == (2.4, 2.8) and np.allclose(win[1]['sb'], [(2.2, 2.4), (2.8, 3.0)])
    assert win[-1]['sr'] == (4.8, INF) and np.allclose(win[-1]['sb'], [(4.4, 4.8)])


def test_grid_b_windows():
    cfg = cw.load_config(REAL_CFG)
    w = cfg['windows']
    win = cw.grid_windows(w['grid_b_edges'], w['width'], w['sideband'])
    assert len(win) == 8
    assert win[0]['sr'] == (-INF, 2.6) and np.allclose(win[0]['sb'], [(2.6, 3.0)])
    assert np.allclose(win[3]['sr'], (3.4, 3.8)) and np.allclose(win[3]['sb'], [(3.2, 3.4), (3.8, 4.0)])
    assert win[-1]['sr'] == (5.0, INF) and np.allclose(win[-1]['sb'], [(4.6, 5.0)])


@pytest.mark.parametrize('grid', ['grid_a_edges', 'grid_b_edges'])
def test_assign_window_covers_axis_once(grid):
    cfg = cw.load_config(REAL_CFG)
    w = cfg['windows']
    win = cw.grid_windows(w[grid], w['width'], w['sideband'])
    m = np.r_[0.5, 1.9, w[grid], 3.5, 9.0, np.linspace(1, 8, 500)]
    idx = cw.assign_window(m, win)
    assert idx[0] == 0 and idx[1] == 0 and idx[-1] == len(win) - 1
    for i, wi in enumerate(win):                      # SR and SB of a window never overlap
        assert not (cw.sr_mask(m, wi) & cw.sb_mask(m, wi)).any()
        assert np.all(cw.sr_mask(m, wi) == (idx == i))


def test_edge_value_goes_to_upper_window():
    win = cw.grid_windows([2.0, 2.4, 2.8, 3.2], 0.4, 0.2)
    assert list(cw.assign_window(np.array([2.4, 2.8, 2.79999]), win)) == [1, 2, 1]


def test_merge_top_window_with_inner_neighbour():
    win = cw.grid_windows([2.0, 2.4, 2.8, 3.2, 3.6, 4.0], 0.4, 0.2)      # 5 windows; top SR [3.6, inf)
    rng = np.random.default_rng(0)
    m = np.r_[rng.uniform(1.5, 3.6, 5000), rng.uniform(3.6, 5.0, 10)]    # top SR has 10 events
    out, log = cw.merge_windows(win, m, 100, 0.2)
    assert len(out) == 4 and len(log) == 1
    assert out[-1]['sr'] == (3.2, INF) and np.allclose(out[-1]['sb'], [(2.8, 3.2)])
    assert out[-2]['sr'] == (2.8, 3.2)                                    # untouched inner window
    assert all(min(cw.window_counts(m, w)) >= 100 for w in out)
    cw.assign_window(m, out)


def test_merge_bottom_window_and_interior_sidebands_recomputed():
    win = cw.grid_windows([2.0, 2.4, 2.8, 3.2, 3.6, 4.0, 4.4], 0.4, 0.2)  # 6 windows
    rng = np.random.default_rng(1)
    m = np.r_[rng.uniform(2.4, 6.0, 20000), rng.uniform(1.0, 2.4, 5)]   # bottom SR (-inf, 2.4) nearly empty
    out, log = cw.merge_windows(win, m, 100, 0.2)
    assert out[0]['sr'] == (-INF, 2.8) and np.allclose(out[0]['sb'], [(2.8, 3.2)])
    assert log[0]['into'] == [-INF, 2.8]
    # an interior merge gets 0.2 sidebands on both sides of the merged SR
    m2 = np.r_[rng.uniform(1.0, 6.0, 30000)]
    m2 = m2[~((m2 >= 2.8) & (m2 < 3.2))]
    m2 = np.r_[m2, rng.uniform(2.8, 3.2, 3)]                             # window [2.8, 3.2) starved
    out2, _ = cw.merge_windows(win, m2, 100, 0.2)
    srs = [w['sr'] for w in out2]
    assert (2.8, 3.6) in srs
    merged = out2[srs.index((2.8, 3.6))]
    assert np.allclose(merged['sb'], [(2.6, 2.8), (3.6, 3.8)])


# ----------------------------------------------------------------------------------------------- scoring
def test_quantile_scoring():
    ref = np.array([0.1, 0.2, 0.3, 0.4])
    q = cw.quantile_scores(ref, np.array([0.05, 0.1, 0.25, 0.4, 0.9]))
    assert np.allclose(q, [0.0, 0.0, 0.5, 0.75, 1.0])
    rng = np.random.default_rng(0)
    r = rng.normal(size=20000)
    assert abs(cw.quantile_scores(r, rng.normal(size=20000)).mean() - 0.5) < 0.01   # same law -> uniform
    with pytest.raises(ValueError):
        cw.quantile_scores([], [0.1])


# ----------------------------------------------------------------------------------------------- rows
def test_no_test_row_leakage_and_injection_count():
    cfg = cw.load_config(REAL_CFG)
    t = cw.test_rows(cfg)
    assert len(t) == 10000 and t[0] == 40000 and t[4999] == 44999 and t[5000] == 1_000_000
    n_bg = 1_000_000 - 45_000
    prev = np.array([], int)
    for f in cfg['injection']['fractions']:
        for s in cfg['injection']['seeds']:
            inj = cw.injected_rows(cfg, f, s)
            assert len(inj) == int(round(f * n_bg)) and len(np.unique(inj)) == len(inj)
            assert inj.size == 0 or (inj.min() >= 1_005_000 and inj.max() < 1_100_000)
            pool = cw.pool_rows(cfg, f, s)
            assert not np.intersect1d(pool, t).size
            train = cw.split_pool(len(pool), cfg['split']['train_fraction'], s)
            assert not np.intersect1d(pool[train], t).size and not np.intersect1d(pool[~train], t).size
        if f > 0:                                      # same seed: injected sets are nested across fractions
            assert np.isin(prev, cw.injected_rows(cfg, f, 0)).all()
        prev = cw.injected_rows(cfg, f, 0)
    assert len(cw.injected_rows(cfg, 0.001, 0)) == 955


def test_pool_overlapping_test_rows_is_refused():
    cfg = cw.load_config(REAL_CFG)
    cfg['data']['pool_bg_rows'] = [44000, 1_000_000]
    with pytest.raises(AssertionError):
        cw.pool_rows(cfg, 0.0, 0)


def test_injection_and_split_are_deterministic_by_seed():
    cfg = cw.load_config(REAL_CFG)
    assert np.array_equal(cw.injected_rows(cfg, 0.002, 1), cw.injected_rows(cfg, 0.002, 1))
    assert not np.array_equal(cw.injected_rows(cfg, 0.002, 1), cw.injected_rows(cfg, 0.002, 2))
    a, b = cw.split_pool(1000, 0.7, 3), cw.split_pool(1000, 0.7, 3)
    assert np.array_equal(a, b) and a.sum() == 700
    assert not np.array_equal(a, cw.split_pool(1000, 0.7, 4))


# ----------------------------------------------------------------------------------------------- synthetic
def _synthetic(n_bg, n_sig, seed):
    """mjj (TeV) falling for background, a bump at 3.5 TeV for signal; signal has distinct jet features."""
    rng = np.random.default_rng(seed)
    m_bg = 1.5 + rng.exponential(0.9, n_bg)
    x_bg = rng.normal(0, 1, (n_bg, 4))
    x_bg[:, 0] += 0.2 * m_bg                                # mild feature-mjj correlation, as in real QCD
    m_sig = rng.normal(3.5, 0.1, n_sig)
    x_sig = rng.normal(0, 1, (n_sig, 4))
    x_sig[:, 1] += 3.0
    x_sig[:, 2] -= 3.0
    x_sig[:, 0] += 0.2 * m_sig
    return np.r_[x_bg, x_sig], np.r_[m_bg, m_sig], np.r_[np.zeros(n_bg), np.ones(n_sig)].astype(int)


def _end_to_end(n_inject, seed=0):
    cfg = _small_cfg()
    x_pool, m_pool, _ = _synthetic(40000, n_inject, seed)
    x_test, m_test, y_test = _synthetic(1000, 1000, seed + 100)
    train = cw.split_pool(len(m_pool), 0.7, seed)
    scores, info = cw.cwola_scores(x_pool[train], m_pool[train], x_pool[~train], m_pool[~train], x_test, m_test,
                                   cfg, seed, log=lambda s: None)
    assert scores['score'].min() >= 0 and scores['score'].max() <= 1
    return roc_auc_score(y_test, scores['score']), roc_auc_score(y_test, scores['oracle_score']), info, scores


def test_planted_bump_is_found():
    auc, oracle_auc, _, _ = _end_to_end(800)
    assert auc > 0.75 and oracle_auc > 0.85


def test_no_signal_gives_auc_near_half():
    auc, _, info, scores = _end_to_end(0)
    assert abs(auc - 0.5) < 0.08
    assert set(info['grids']) == {'a', 'b'} and 'oracle' in info


def test_end_to_end_is_deterministic():
    a = _end_to_end(400, seed=3)[3]
    b = _end_to_end(400, seed=3)[3]
    assert all(np.array_equal(a[k], b[k]) for k in a)


def test_run_one_writes_scores_without_test_leakage(tmp_path):
    n_bg, n_sig = 30000, 3000
    x, m, labels = _synthetic(n_bg, n_sig, 7)
    cfg = _small_cfg(tmp_path)
    cfg['data'].update({'test_bg_rows': [0, 500], 'pool_bg_rows': [500, n_bg],
                        'test_sig_rows': [n_bg, n_bg + 500], 'inject_sig_rows': [n_bg + 500, n_bg + n_sig]})
    cw.check_labels(labels, cfg)
    info = cw.run_one(cfg, 0.01, 0, arrays=(x, m, labels), log=lambda s: None)
    assert info['n_injected'] == round(0.01 * (n_bg - 500))
    with np.load(os.path.join(cfg['output']['dir'], 'scores_f0.01_s0.npz')) as z:
        assert np.array_equal(z['rows'], np.r_[np.arange(500), np.arange(n_bg, n_bg + 500)])
        assert np.array_equal(z['labels'], labels[z['rows']])
        assert {'labels', 'rows', 'score', 'oracle_score'} <= set(z.files)
    with open(os.path.join(cfg['output']['dir'], 'metrics_f0.01_s0.json'), encoding='utf-8') as h:
        meta = json.load(h)
    assert meta['n_train'] + meta['n_ref'] == meta['n_pool'] == n_bg - 500 + info['n_injected']


def test_predictions_verdicts():
    means = {0.0: 0.50, 0.0005: 0.52, 0.001: 0.60, 0.002: 0.595, 0.005: 0.92, 0.01: 0.95}
    p = cw.predictions_verdicts(means, list(means), {'passes': False, 'ci_lo': 0.58, 'within_bin_min': 0.5})
    assert [x['passed'] for x in p] == [True, True, False, True]
    means[0.002] = 0.58
    p = cw.predictions_verdicts(means, list(means), {'passes': True, 'ci_lo': 0.9, 'within_bin_min': 0.7})
    assert [x['passed'] for x in p] == [True, False, True, True]
