import json
import os

import numpy as np
import pytest

from src.pipeline.part2 import (
    apply_overrides,
    build_bank,
    load_config,
    make_report,
    parse_external,
    prepare_cache,
    run_training,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TINY = [
    'data.features_path=null', 'data.max_particles=40',
    'split.skip_bg=10', 'split.n_test_bg=30', 'split.n_test_sig=30', 'split.n_train=100', 'split.n_contam_sig=10',
    'model.width=16', 'model.layers=2', 'model.latent_dim=4', 'model.k_recon=8', 'model.hidden=32',
    'training.epochs=2', 'training.batch_size=32', 'scores.knn_k=5', 'scores.gmm_components=2',
    'benchmark.n_boot=30',
]


@pytest.fixture
def tiny_cfg(events_h5, tmp_path):
    cfg = load_config(os.path.join(ROOT, 'configs/part2_main.yaml'))
    return apply_overrides(cfg, [f'data.h5_path={events_h5}', f'data.cache_dir={tmp_path}/cache',
                                 f'output_dir={tmp_path}/out', *TINY])


def test_apply_overrides_parses_yaml_values():
    cfg = apply_overrides({'a': {'b': 1}}, ['a.b=2', 'a.c=null', 'x.y=[mean,max]', 'z=0.5'])
    assert cfg == {'a': {'b': 2, 'c': None}, 'x': {'y': ['mean', 'max']}, 'z': 0.5}


def test_parse_external():
    assert parse_external('Part 1 AE (max)=/a/b.npz:scores_max') == ('Part 1 AE (max)', '/a/b.npz', 'scores_max')


def test_bank_and_report_without_models(tiny_cfg, tmp_path):
    prepare_cache(tiny_cfg)
    scores = tmp_path / 'ext.npz'
    labels = np.r_[np.zeros(30), np.ones(30)]
    np.savez(scores, scores=np.random.default_rng(0).normal(size=60), labels=labels)
    bank = build_bank(tiny_cfg, externals=[f'Ext={scores}:scores'])
    out = make_report(tiny_cfg, bank, name='stage0')
    names = {r['name'] for r in out['rows']}
    assert {'HT (sum pT)', 'mean pT per particle', '-multiplicity', 'Ext'} <= names
    assert out['gate'] is None
    ht = next(r for r in out['rows'] if r['name'] == 'HT (sum pT)')
    assert 0.0 <= ht['auc'] <= 1.0
    assert os.path.exists(os.path.join(tiny_cfg['output_dir'], 'stage0.md'))
    with open(os.path.join(tiny_cfg['output_dir'], 'stage0.json'), encoding='utf-8') as handle:
        assert json.load(handle)['gate'] is None


def test_bank_rejects_external_with_different_labels(tiny_cfg, tmp_path):
    prepare_cache(tiny_cfg)
    bad = tmp_path / 'bad.npz'
    np.savez(bad, scores=np.zeros(60), labels=np.zeros(60))
    with pytest.raises(ValueError, match='labels'):
        build_bank(tiny_cfg, externals=[f'Bad={bad}:scores'])


def test_run_training_writes_scores_and_skips_existing(tiny_cfg):
    prepare_cache(tiny_cfg)
    metrics = run_training(tiny_cfg, seed=0, tag='main')
    run_dir = os.path.join(tiny_cfg['output_dir'], 'runs', 'main')
    scores = np.load(os.path.join(run_dir, 'scores_s0.npz'))
    assert set(scores.files) == {'chamfer', 'knn', 'gmm', 'labels'}
    assert all(scores[k].shape == (60,) and np.isfinite(scores[k]).all() for k in ('chamfer', 'knn', 'gmm'))
    assert scores['labels'].tolist() == [0] * 30 + [1] * 30
    assert set(metrics['auc']) == {'chamfer', 'knn', 'gmm'} and metrics['n_contaminating_signal'] == 0
    assert metrics['n_fit'] == 90 and metrics['num_parameters'] > 0
    mtime = os.path.getmtime(os.path.join(run_dir, 'scores_s0.npz'))
    assert run_training(tiny_cfg, seed=0, tag='main')['auc'] == metrics['auc']
    assert os.path.getmtime(os.path.join(run_dir, 'scores_s0.npz')) == mtime


def test_contamination_run_replaces_training_events_with_signal(tiny_cfg):
    prepare_cache(tiny_cfg)
    metrics = run_training(tiny_cfg, seed=1, contamination=0.1, tag='contam')
    assert metrics['n_contaminating_signal'] == 9 and metrics['n_fit'] == 90


def test_contamination_larger_than_cache_raises(tiny_cfg):
    prepare_cache(tiny_cfg)
    with pytest.raises(ValueError, match='contamination'):
        run_training(tiny_cfg, seed=0, contamination=0.5, tag='too_much')


def test_unknown_loss_raises(tiny_cfg):
    prepare_cache(tiny_cfg)
    tiny_cfg['training']['loss'] = 'chamfer_typo'
    with pytest.raises(ValueError, match='loss'):
        run_training(tiny_cfg, seed=0, tag='bad_loss')
    assert not os.path.exists(os.path.join(tiny_cfg['output_dir'], 'runs', 'bad_loss', 'scores_s0.npz'))


def test_full_report_with_model_rows_and_gate(tiny_cfg):
    prepare_cache(tiny_cfg)
    for seed in (0, 1):
        run_training(tiny_cfg, seed=seed, tag='main')
    bank = build_bank(tiny_cfg)
    out = make_report(tiny_cfg, bank, run_tags=['main'], name='final')
    names = {r['name'] for r in out['rows']}
    assert {'EventAE chamfer', 'EventAE knn', 'EventAE gmm'} <= names
    assert out['gate'] is not None and out['gate']['bar_name'] in {
        'HT (sum pT)', 'mean pT per particle', 'leading pT', '-multiplicity', 'm_all'}
    assert next(r for r in out['rows'] if r['name'] == 'EventAE knn')['notes'] == '2 seed(s)'


def test_scores_file_is_written_last_as_the_run_finished_marker(tiny_cfg, monkeypatch):
    prepare_cache(tiny_cfg)

    def boom(*args, **kwargs):
        raise OSError('disk full')
    monkeypatch.setattr('src.pipeline.part2.np.savez', boom)
    with pytest.raises(OSError):
        run_training(tiny_cfg, seed=0, tag='main')
    run_dir = os.path.join(tiny_cfg['output_dir'], 'runs', 'main')
    assert os.path.exists(os.path.join(run_dir, 'metrics_s0.json'))
    assert os.path.exists(os.path.join(run_dir, 'model_s0.pt'))
    assert not os.path.exists(os.path.join(run_dir, 'scores_s0.npz'))


def test_metrics_report_last_epoch_and_best_val_loss(tiny_cfg):
    prepare_cache(tiny_cfg)
    metrics = run_training(tiny_cfg, seed=0, tag='main')
    assert 'final_val_loss' not in metrics
    assert metrics['last_epoch_val_loss'] == metrics['history'][-1]['val_loss']
    assert metrics['best_val_loss'] == min(h['val_loss'] for h in metrics['history'])


def test_cache_not_matching_n_train_raises(tiny_cfg):
    prepare_cache(tiny_cfg)
    tiny_cfg['split']['n_train'] = 90
    with pytest.raises(ValueError, match='cache'):
        run_training(tiny_cfg, seed=0, tag='main')
    assert not os.path.exists(os.path.join(tiny_cfg['output_dir'], 'runs', 'main', 'scores_s0.npz'))


def test_stale_failed_file_is_removed_after_success(tiny_cfg):
    prepare_cache(tiny_cfg)
    run_dir = os.path.join(tiny_cfg['output_dir'], 'runs', 'main')
    os.makedirs(run_dir)
    failed = os.path.join(run_dir, 'failed_s0.json')
    with open(failed, 'w', encoding='utf-8') as handle:
        json.dump({'seed': 0, 'tag': 'main', 'reason': 'old failure'}, handle)
    run_training(tiny_cfg, seed=0, tag='main')
    assert not os.path.exists(failed)
    assert os.path.exists(os.path.join(run_dir, 'scores_s0.npz'))


def test_density_scorer_fit_guard_is_an_explicit_raise(tiny_cfg, monkeypatch):
    import src.pipeline.part2 as part2
    prepare_cache(tiny_cfg)
    real = part2.fit_scorers

    class Wrong:
        n_fit = 1

    monkeypatch.setattr(part2, 'fit_scorers', lambda *a, **k: {**real(*a, **k), 'knn': Wrong()})
    with pytest.raises(RuntimeError, match='training latents'):
        run_training(tiny_cfg, seed=0, tag='main')


def test_partial_scores_write_leaves_no_temp_or_scores_file(tiny_cfg, monkeypatch):
    prepare_cache(tiny_cfg)

    def partial(path, *args, **kwargs):
        with open(path, 'wb') as handle:
            handle.write(b'partial')
        raise OSError('disk full')
    monkeypatch.setattr('src.pipeline.part2.np.savez', partial)
    with pytest.raises(OSError):
        run_training(tiny_cfg, seed=0, tag='main')
    run_dir = os.path.join(tiny_cfg['output_dir'], 'runs', 'main')
    assert not [f for f in os.listdir(run_dir) if f.endswith('.npz')]
