# tests/test_part3_pipeline.py
import json
import os

import numpy as np
import pytest

from src.benchmarks.baselines import jet_baselines
from src.pipeline import part3
from tests.test_part3_features import make_jet_frame


def _cfg(tmp_path, max_epochs=1):
    return {
        'data': {'features_path': str(tmp_path / 'features.h5'), 'bank_path': str(tmp_path / 'bank.npz')},
        'split': {'skip_bg': 20, 'n_test_bg': 60, 'n_test_sig': 60, 'n_train': 400, 'val_fraction': 0.1},
        'features': {'headline': 'jet5', 'variants': ['jet4'],
                     'quantile': {'output_distribution': 'normal', 'n_quantiles': 100, 'random_state': 0}},
        'density': {'seeds': [0, 1], 'gmm_components': 2, 'knn_k': 5,
                    'ae': {'hidden': [32, 32], 'latent': 3, 'epochs': 2, 'batch_size': 64, 'learning_rate': 0.001,
                           'weight_decay': 0.00001, 'patience': 5}},
        'wnae': {'feature_set': 'jet5', 'seeds': [0], 'batch_size': 120, 'learning_rate': 0.0002,
                 'scheduler': {'factor': 0.8, 'patience': 20, 'cooldown': 10, 'threshold': 0.001,
                               'threshold_mode': 'rel'},
                 'mcmc': {'n_steps': 3, 'step_size': 0.0242, 'noise_std': 0.22, 'temperature': 0.063,
                          'restart_fraction': 0.05, 'bound': [-3.0, 3.0]},
                 'buffer_size': 500, 'ot_num_iter_max': 1000000, 'max_epochs': max_epochs, 'torch_threads': 1},
        'benchmark': {'n_boot': 20, 'baseline_ci_level': 0.95, 'model_ci_level': 0.983, 'wnae_ci_level': 0.95,
                      'n_bins': 2, 'min_within_bin': 0.6},
        'output_dir': str(tmp_path / 'out'),
    }


def _write_inputs(tmp_path, corrupt_bank=False):
    df = make_jet_frame(500, 80, seed=1)
    df.to_hdf(tmp_path / 'features.h5', key='df')
    test = np.r_[np.arange(20, 80), np.arange(500, 560)]
    base = {b.name: b.scores for b in jet_baselines(df.iloc[test].reset_index(drop=True))}
    rng = np.random.default_rng(0)
    y = np.r_[np.zeros(60), np.ones(60)]
    mult = rng.integers(50, 200, 120).astype(float)
    ht = rng.normal(2500, 200, 120)
    mjj = base['mjj'] + (1.0 if corrupt_bank else 0.0)
    np.savez(tmp_path / 'bank.npz', y=y, cov_ht=ht, cov_multiplicity=mult, cov_max_pt=rng.normal(600, 50, 120),
             s0=ht / mult, s1=mjj, s2=base['-tau21 (leading jet)'])
    with open(tmp_path / 'bank.json', 'w') as handle:
        json.dump([{'name': 'mean pT per particle', 'group': 'particle', 'notes': ''},
                   {'name': 'mjj', 'group': 'jet', 'notes': ''},
                   {'name': '-tau21 (leading jet)', 'group': 'jet', 'notes': ''}], handle)


def test_bank_alignment_detects_misaligned_rows(tmp_path):
    _write_inputs(tmp_path, corrupt_bank=True)
    cfg = _cfg(tmp_path)
    df, rows = part3.load_data(cfg)
    with pytest.raises(AssertionError, match='misaligned'):
        part3.verify_bank_alignment(df, rows, cfg['data']['bank_path'])


def test_wnae_refuses_unfixed_epoch_count(tmp_path):
    _write_inputs(tmp_path)
    with pytest.raises(ValueError, match='max_epochs'):
        part3.run_wnae(_cfg(tmp_path, max_epochs='MAX_EPOCHS'), seed=0)


def test_density_wnae_and_reports_end_to_end(tmp_path):
    _write_inputs(tmp_path)
    cfg = _cfg(tmp_path)
    log = lambda m: None  # noqa: E731
    part3.run_density(cfg, log=log)
    out = cfg['output_dir']
    with np.load(os.path.join(out, 'density', 'scores_s0.npz')) as d:
        assert set(d.files) == {'labels', 'rows', 'gmm_jet5', 'knn_jet5', 'ae_jet5', 'gmm_jet4', 'knn_jet4', 'ae_jet4'}
        assert d['ae_jet5'].shape == (120,) and list(d['rows'][:2]) == [20, 21] and d['rows'][60] == 500
        knn0 = d['knn_jet5']
    with np.load(os.path.join(out, 'density', 'scores_s1.npz')) as d:
        assert np.allclose(d['knn_jet5'], knn0)               # kNN is deterministic
    metrics = part3.run_wnae(cfg, seed=0, log=log)
    assert metrics['best_epoch'] == 1 and metrics['diverged'] is None
    assert 'auc' not in json.dumps(metrics).lower()          # no test AUC is logged during training
    result = part3.make_reports(cfg, log=log)
    for name in ('jet_density.md', 'jet_density.json', 'wnae.md', 'wnae.json'):
        assert os.path.exists(os.path.join(out, name))
    gate = result['jet_density']['gate']
    assert {m['name'] for m in gate['models']} == {'GMM NLL [jet5]', 'kNN dist [jet5]', 'AE recon [jet5]'}
    assert gate['bar_name'] == 'mean pT per particle'
    rows = {r['name']: r for r in result['wnae']['rows']}
    assert rows['WNAE recon [jet5]']['group'] == 'variant' and rows['WNAE recon [jet5]']['level'] == 0.95
    assert rows['kNN dist [jet4]']['group'] == 'variant' and rows['GMM NLL [jet5]']['level'] == 0.983
    extra = result['wnae']['extras'][-1]
    assert set(extra['spearman_seed0']) == {'multiplicity', 'ht', 'mean_pt'}
    assert set(extra['spearman_seed0']['ht']) == {'all', 'bg', 'sig'}
    md = open(os.path.join(out, 'wnae.md')).read()
    assert 'Negative samples vs positives' in md and 'Spearman' in md
