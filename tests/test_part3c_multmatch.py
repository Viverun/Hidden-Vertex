"""Part 3C: split.match_multiplicity_to_signal (training-side multiplicity matching)."""
import json
import os
import sys

import numpy as np
import pytest

from src.data.event_store import load_role
from src.pipeline.part2 import apply_overrides, load_config, multiplicity_match_resample, prepare_cache, run_training
from tests.test_part2_pipeline import ROOT, TINY

sys.path.insert(0, os.path.join(ROOT, 'scripts'))
from part2_sweep import build_jobs  # noqa: E402


def _cfg(events_h5, base, extra=()):
    cfg = load_config(os.path.join(ROOT, 'configs/part2_main.yaml'))
    return apply_overrides(cfg, [f'data.h5_path={events_h5}', f'data.cache_dir={base}/cache',
                                 f'output_dir={base}/out', *TINY, *extra])


def _scores(cfg, tag, seed=0):
    with np.load(os.path.join(cfg['output_dir'], 'runs', tag, f'scores_s{seed}.npz')) as data:
        return {k: data[k] for k in data.files}


def test_resample_moves_multiplicity_to_signal_reference():
    rng = np.random.default_rng(1)
    bg, sig = rng.poisson(182, 50000).astype(float), rng.poisson(140, 5000).astype(float)
    idx, diag = multiplicity_match_resample(bg, sig, len(bg), seed=0)
    assert idx.shape == (len(bg),) and idx.min() >= 0 and idx.max() < len(bg)
    drawn = bg[idx]
    edges = np.quantile(bg, np.linspace(0, 1, 11))

    def hist(x):
        return np.histogram(np.clip(x, edges[0], edges[-1]), edges)[0] / len(x)
    assert np.abs(hist(drawn) - hist(sig)).sum() < 0.5 * np.abs(hist(bg) - hist(sig)).sum()
    assert abs(drawn.mean() - sig.mean()) < 0.5 * abs(bg.mean() - sig.mean())
    assert diag['mean_multiplicity_fit_before'] == pytest.approx(bg.mean())
    assert diag['mean_multiplicity_fit_after'] == pytest.approx(drawn.mean())
    assert diag['mean_multiplicity_signal_reference'] == pytest.approx(sig.mean())
    assert 0 < diag['effective_sample_size'] < len(bg)
    assert diag['n_unique_drawn'] == len(np.unique(idx)) < len(bg)
    assert diag['n_bins'] <= 30


def test_resample_is_seeded():
    rng = np.random.default_rng(2)
    bg, sig = rng.poisson(30, 1000).astype(float), rng.poisson(20, 300).astype(float)
    a, _ = multiplicity_match_resample(bg, sig, 1000, seed=3)
    b, _ = multiplicity_match_resample(bg, sig, 1000, seed=3)
    c, _ = multiplicity_match_resample(bg, sig, 1000, seed=4)
    assert np.array_equal(a, b) and not np.array_equal(a, c)


def test_default_off_is_identical_to_explicit_false(events_h5, tmp_path):
    absent = _cfg(events_h5, tmp_path / 'a')
    assert 'match_multiplicity_to_signal' not in absent['split']
    explicit = _cfg(events_h5, tmp_path / 'b', ['split.match_multiplicity_to_signal=false'])
    prepare_cache(absent)
    prepare_cache(explicit)
    m_a = run_training(absent, seed=0, tag='main')
    m_b = run_training(explicit, seed=0, tag='main')
    s_a, s_b = _scores(absent, 'main'), _scores(explicit, 'main')
    assert set(s_a) == set(s_b)
    for k in s_a:
        assert np.array_equal(s_a[k], s_b[k]), k
    assert m_a['auc'] == m_b['auc'] and m_a['history'][-1]['train_loss'] == m_b['history'][-1]['train_loss']
    assert 'multiplicity_resample' not in m_a and 'match_multiplicity_to_signal' not in m_a


def test_multmatch_run_resamples_fit_set_and_leaves_test_set_untouched(events_h5, tmp_path):
    base = _cfg(events_h5, tmp_path)
    prepare_cache(base)
    run_training(base, seed=0, tag='main')
    cfg = apply_overrides(base, ['split.match_multiplicity_to_signal=true'])
    metrics = run_training(cfg, seed=0, tag='multmatch')
    with open(os.path.join(cfg['output_dir'], 'runs', 'multmatch', 'metrics_s0.json'), encoding='utf-8') as h:
        assert json.load(h)['multiplicity_resample'] == metrics['multiplicity_resample']
    r = metrics['multiplicity_resample']
    assert metrics['match_multiplicity_to_signal'] is True and metrics['n_fit'] == 90 and r['n_drawn'] == 90
    cache = cfg['data']['cache_dir']
    fit_mult = load_role(cache, 'train_bg').summary['multiplicity'][:90]
    sig_mult = load_role(cache, 'contam_sig').summary['multiplicity']
    assert r['mean_multiplicity_fit_before'] == pytest.approx(fit_mult.mean())
    assert r['mean_multiplicity_signal_reference'] == pytest.approx(sig_mult.mean())
    # synthetic signal has fewer particles than every background event: the draw goes to the lowest bins
    assert r['mean_multiplicity_fit_after'] < r['mean_multiplicity_fit_before'] - 3
    assert r['n_unique_drawn'] < 90 and 0 < r['effective_sample_size'] < 90
    # the frozen test set (events, order, labels) is the same as in the default run
    main, mm = _scores(base, 'main'), _scores(cfg, 'multmatch')
    assert np.array_equal(main['rows'], mm['rows']) and np.array_equal(main['labels'], mm['labels'])
    assert mm['knn'].shape == (60,) and np.isfinite(mm['gmm']).all()
    assert not np.array_equal(main['knn'], mm['knn'])


def test_multmatch_with_contamination_raises(events_h5, tmp_path):
    cfg = _cfg(events_h5, tmp_path, ['split.match_multiplicity_to_signal=true', 'split.contamination=0.1'])
    prepare_cache(cfg)
    with pytest.raises(ValueError, match='contamination 0'):
        run_training(cfg, seed=0, tag='multmatch')


def test_multmatch_sweep_stage():
    assert build_jobs('multmatch') == [('multmatch', s, ['split.match_multiplicity_to_signal=true'])
                                       for s in (0, 1, 2)]
