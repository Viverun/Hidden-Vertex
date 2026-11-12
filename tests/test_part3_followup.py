# tests/test_part3_followup.py
import json
import os

import numpy as np
import pytest
import torch

from src.model.mlp_ae import MLPAE
from src.model.wnae import WNAETrainer
from src.pipeline import part3
from tests.test_part3_pipeline import _cfg, _write_inputs
from tests.test_part3_wnae import MCMC, _data


def test_sic_metrics_known_values():
    y = np.r_[np.zeros(1000), np.ones(1000)]
    perfect = part3.sic_metrics(y, y.copy())
    assert perfect['tpr@0.01'] == pytest.approx(1.0) and perfect['sic@0.01'] == pytest.approx(10.0)
    assert perfect['max_sic'] == pytest.approx(1.0 / np.sqrt(1e-3)) and perfect['max_sic_fpr'] == 1e-3
    rng = np.random.default_rng(0)
    rand = part3.sic_metrics(y, rng.normal(size=2000))
    assert rand['tpr@0.1'] == pytest.approx(0.1, abs=0.04)
    assert rand['sic@0.1'] == pytest.approx(rand['tpr@0.1'] / np.sqrt(0.1))
    assert rand['max_sic_fpr'] >= 1e-3
    summ = part3.sic_summary(y, [rng.normal(size=2000), rng.normal(size=2000)])
    assert summ['mean']['max_sic'] == pytest.approx(np.mean([p['max_sic'] for p in summ['per_seed']]))


def test_early_stop_triggers_on_flat_validation(monkeypatch):
    torch.manual_seed(0)
    x = _data(600)
    trainer = WNAETrainer(MLPAE(5), 5, MCMC, batch_size=200, lr=0.0, seed=0, buffer_size=400)
    monkeypatch.setattr(trainer, 'validate', lambda xv, epoch_seed: {
        'w': 1.0, 'e_pos': 0.0, 'e_neg': 0.0, 'neg_mean': [0] * 5, 'neg_std': [1] * 5,
        'pos_mean': [0] * 5, 'pos_std': [1] * 5})
    history, best_epoch, diverged = trainer.fit(x[:400], x[400:], max_epochs=10, log=lambda m: None,
                                                es_patience=3, es_threshold=1e-3)
    assert trainer.early_stopped == 4 and len(history) == 4 and best_epoch == 1 and diverged is None


def test_reverse_rows_and_long_run_settings(tmp_path):
    _write_inputs(tmp_path)
    cfg = _cfg(tmp_path)
    cfg['reverse'] = {'skip_sig': 60, 'n_fit': 15, 'n_val': 5}
    labels = np.r_[np.zeros(500), np.ones(80)].astype(int)
    fit, val = part3.reverse_rows(labels, cfg)
    assert list(fit) == list(range(560, 575)) and list(val) == list(range(575, 580))
    cfg['wnae_long'] = {'batch_size': 64, 'max_epochs': 2, 'es_patience': 30}
    w = part3.wnae_settings(cfg, 'wnae_long')
    assert w['batch_size'] == 64 and w['mcmc'] == cfg['wnae']['mcmc'] and part3.wnae_settings(cfg)['batch_size'] == 120
    with pytest.raises(ValueError):
        part3.wnae_settings(cfg, 'wnae_other')


def test_reverse_long_run_and_sic_report_end_to_end(tmp_path):
    _write_inputs(tmp_path)
    cfg = _cfg(tmp_path)
    cfg['reverse'] = {'skip_sig': 60, 'n_fit': 15, 'n_val': 5}
    cfg['density']['gmm_components'] = 1
    cfg['wnae_long'] = {'batch_size': 60, 'max_epochs': 2, 'es_patience': 30}
    log = lambda m: None  # noqa: E731
    part3.run_density(cfg, log=log)
    m = part3.run_wnae(cfg, seed=0, log=log, run='wnae_long')
    assert m['run'] == 'wnae_long' and m['batch_size'] == 60 and m['early_stopped_at'] is None
    assert os.path.exists(os.path.join(cfg['output_dir'], 'wnae_long', 'progress_s0.json'))
    info = part3.run_reverse(cfg, log=log)
    assert info['positive_class'].startswith('background') and len(info['rows']) == 6
    assert 'DIAGNOSTIC' in open(os.path.join(cfg['output_dir'], 'reverse.md')).read()
    res = part3.make_reports(cfg, log=log)
    names = {r['name'] for r in res['wnae']['rows']}
    assert 'WNAE-long recon [jet5]' in names and 'WNAE recon [jet5]' not in names
    assert all('sic' in r for r in res['wnae']['rows'])
    payload = json.load(open(os.path.join(cfg['output_dir'], 'jet_density.json')))
    assert 'max_sic' in payload['rows'][0]['sic']['mean']
    assert 'Significance improvement' in open(os.path.join(cfg['output_dir'], 'wnae.md')).read()
