"""Part 3C: scripts/part3c_all_scores_matched.py on synthetic inputs and on the saved results."""
import json
import os
import sys

import numpy as np
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, 'scripts'))
import part3c_all_scores_matched as m  # noqa: E402

BANK_NAMES = ['HT (sum pT)', 'mean pT per particle', '-multiplicity', 'mjj', '-tau21 (leading jet)',
              '-|mjj-3500| (oracle)', 'Part 1 AE (mean)']


def _write_synthetic(tmp_path, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    y = np.r_[np.zeros(n), np.ones(n)].astype(int)
    mult = np.round(np.r_[rng.normal(180, 40, n), rng.normal(140, 40, n)]).clip(20)
    ht = mult * np.r_[rng.normal(10, 1, n), rng.normal(12, 1, n)]
    p2, p3 = tmp_path / 'part2', tmp_path / 'part3'
    bank = {'y': y, 'cov_multiplicity': mult, 'cov_ht': ht}
    for i, _ in enumerate(BANK_NAMES):
        bank[f's{i}'] = rng.normal(size=2 * n) + y * (i == 0)  # HT column is informative
    os.makedirs(p2 / 'runs' / 'main')
    np.savez(p2 / 'bank.npz', **bank)
    (p2 / 'bank.json').write_text(json.dumps([{'name': k} for k in BANK_NAMES]))
    keys = {'density': ['knn_jet4', 'gmm_jet4', 'knn_jet5', 'gmm_jet5', 'ae_jet4', 'ae_jet5'],
            'wnae': ['wnae_jet5'], 'wnae_long': ['wnae_jet5']}
    for s in range(2):
        np.savez(p2 / 'runs' / 'main' / f'scores_s{s}.npz', labels=y,
                 **{k: rng.normal(size=2 * n) for k in ('chamfer', 'knn', 'gmm')})
        for sub, ks in keys.items():
            os.makedirs(p3 / sub, exist_ok=True)
            np.savez(p3 / sub / f'scores_s{s}.npz', labels=y, **{k: rng.normal(size=2 * n) for k in ks})
    return p2, p3


def test_synthetic_rows_and_columns(tmp_path):
    p2, p3 = _write_synthetic(tmp_path)
    out = m.run(str(p2), str(p3))
    assert len(out['results']) == 18
    assert out['results']['kNN, jet4']['n_seeds'] == 2
    assert out['results']['HT (sum pT)']['plain'] > 0.6
    for r in out['results'].values():
        assert set(m.VARIANTS) <= set(r)
    assert 0 < out['diagnostics']['mult_ht']['bg_effective_sample_size'] <= 1000
    assert 'mult-matched' in m.render(out)


def test_label_mismatch_raises(tmp_path):
    p2, p3 = _write_synthetic(tmp_path)
    f = p3 / 'wnae' / 'scores_s0.npz'
    d = dict(np.load(f))
    d['labels'] = d['labels'][::-1]
    np.savez(f, **d)
    with pytest.raises(ValueError):
        m.run(str(p2), str(p3))


EXPECTED = {'-|mjj-3500 GeV| (oracle)': 0.817, '-tau21 (leading jet)': 0.805, 'kNN, jet4': 0.729,
            'GMM, jet4': 0.718, 'WNAE pilot, jet5': 0.680, 'kNN, jet5': 0.651, 'GMM, jet5': 0.630,
            'WNAE long, jet5': 0.571, 'Node-level GAE (Part 1 AE, mean)': 0.570, 'mjj': 0.540,
            'HT (sum pT)': 0.510, 'mean pT per particle': 0.509}


@pytest.mark.skipif(not os.path.exists(os.path.join(REPO, 'results', 'part3', 'wnae_long', 'scores_s0.npz')),
                    reason='saved score files not present')
def test_reproduces_paper_table():
    out = m.run(os.path.join(REPO, 'results', 'part2'), os.path.join(REPO, 'results', 'part3'))
    for name, auc in EXPECTED.items():
        assert round(out['results'][name]['mult_ht'], 3) == pytest.approx(auc, abs=1e-9), name
    assert round(out['diagnostics']['mult_ht']['bg_effective_sample_size']) == 831
