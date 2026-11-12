"""Part 3C / C1: scripts/part3c_matched_eval.py on synthetic inputs."""
import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'scripts'))
from part3c_matched_eval import BASELINES, cell_weights, quantile_bins, render, run  # noqa: E402


def _synthetic(n=2000, seed=0):
    rng = np.random.default_rng(seed)
    y = np.r_[np.zeros(n), np.ones(n)].astype(int)
    mult = np.round(np.r_[rng.normal(180, 40, n), rng.normal(140, 40, n)]).clip(20)  # broad overlap, like the data
    ht = mult * np.r_[rng.normal(10, 1, n), rng.normal(12, 1, n)]
    return y, mult, ht


def test_weighted_background_histogram_matches_signal():
    y, mult, _ = _synthetic()
    cell, n = quantile_bins(mult, 20)
    w, _ = cell_weights(y, cell, n, clip_percentile=100)
    h_sig = np.bincount(cell[y == 1], minlength=n) / (y == 1).sum()
    h_bg = np.bincount(cell[y == 0], weights=w[y == 0], minlength=n) / w[y == 0].sum()
    assert np.allclose(h_sig, h_bg)
    assert np.all(w[y == 1] == 1)


def test_clip_caps_background_weights():
    y, mult, _ = _synthetic()
    cell, n = quantile_bins(mult, 20)
    w, d = cell_weights(y, cell, n, clip_percentile=99)
    assert w[y == 0].max() == pytest.approx(d['clip_value'])


def _write_inputs(tmp_path, y, mult, ht, model_scores):
    out = tmp_path / 'p2'
    run_dir = out / 'runs' / 'main'
    run_dir.mkdir(parents=True)
    for s, sc in enumerate(model_scores):
        np.savez(run_dir / f'scores_s{s}.npz', labels=y, chamfer=sc, knn=sc, gmm=sc)
    names = ['HT (sum pT)', 'mean pT per particle', 'leading pT', '-multiplicity']
    np.savez(out / 'bank.npz', y=y, cov_ht=ht, cov_multiplicity=mult, cov_max_pt=ht / 10,
             s0=ht, s1=ht / mult, s2=ht / 10, s3=-mult)
    with open(out / 'bank.json', 'w', encoding='utf-8') as handle:
        json.dump([{'name': n, 'group': 'particle', 'notes': ''} for n in names], handle)
    return str(out)


def test_score_that_only_follows_multiplicity_is_near_half_after_matching(tmp_path):
    y, mult, ht = _synthetic()
    rng = np.random.default_rng(1)
    model = [mult + rng.normal(0, 3, len(y)) for _ in range(2)]   # complexity only: inverted before matching
    out_dir = _write_inputs(tmp_path, y, mult, ht, model)
    out = run(out_dir, os.path.join(out_dir, 'bank.npz'), 'main', n_boot=40, n_jobs=1)
    knn = out['results']['EventAE knn']
    assert knn['n_seeds'] == 2
    assert knn['plain']['auc'] < 0.35
    assert abs(knn['mult']['auc'] - 0.5) < 0.06 and abs(knn['mult_ht']['auc'] - 0.5) < 0.08
    for v in ('plain', 'mult', 'mult_ht'):
        assert knn[v]['ci_lo'] <= knn[v]['auc'] <= knn[v]['ci_hi']
    assert set(BASELINES) <= set(out['results'])
    assert out['results']['-multiplicity']['plain']['auc'] > 0.7
    d = out['diagnostics']['mult']
    assert abs(d['bg_mean_multiplicity_weighted'] - mult[y == 1].mean()) < 2
    md = render(out)
    assert 'EventAE gmm' in md and 'no `rows` key' in md


def test_label_mismatch_raises(tmp_path):
    y, mult, ht = _synthetic(200)
    out_dir = _write_inputs(tmp_path, y, mult, ht, [mult])
    np.savez(os.path.join(out_dir, 'runs', 'main', 'scores_s1.npz'), labels=1 - y, chamfer=mult, knn=mult, gmm=mult)
    with pytest.raises(ValueError, match='labels'):
        run(out_dir, os.path.join(out_dir, 'bank.npz'), 'main', n_boot=5, n_jobs=1)
