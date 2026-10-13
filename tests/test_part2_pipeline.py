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
