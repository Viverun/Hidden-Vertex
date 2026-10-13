# tests/test_part2_report.py
import json

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from src.benchmarks.report import (
    auc_ci,
    benchmark_row,
    binned_auc,
    check_reference,
    evaluate_gate,
    mean_auc,
    render_markdown,
    within_bin_min,
    write_report,
)


def _data(n=4000, seed=0, shift=1.0):
    rng = np.random.default_rng(seed)
    y = np.r_[np.zeros(n // 2), np.ones(n // 2)]
    return y, rng.normal(size=n) + shift * y


def _row(name, group, auc, lo, within):
    return {
        'name': name, 'group': group, 'auc': auc, 'ci_lo': lo, 'ci_hi': auc + 0.01,
        'level': 0.95, 'notes': '',
        'within_bin': {'ht': [{'lo': 0.0, 'hi': 1.0, 'n': 100, 'auc': within}]},
    }


def test_mean_auc_matches_sklearn():
    y, s = _data()
    assert mean_auc(y, s) == pytest.approx(roc_auc_score(y, s))


def test_perfect_and_inverted_scores():
    y = np.r_[np.zeros(50), np.ones(50)]
    s = y.copy()
    assert mean_auc(y, s) == 1.0
    assert mean_auc(y, -s) == 0.0
    assert auc_ci(y, s, n_boot=50) == (1.0, 1.0)


def test_ci_contains_point_estimate_and_widens_with_level():
    y, s = _data()
    lo95, hi95 = auc_ci(y, s, level=0.95, n_boot=300)
    lo99, hi99 = auc_ci(y, s, level=0.99, n_boot=300)
    assert lo95 < mean_auc(y, s) < hi95
    assert lo99 <= lo95 and hi99 >= hi95


def test_seed_list_is_averaged():
    y, s1 = _data(seed=1)
    _, s2 = _data(seed=2)
    expected = (roc_auc_score(y, s1) + roc_auc_score(y, s2)) / 2
    assert mean_auc(y, [s1, s2]) == pytest.approx(expected)
    lo, hi = auc_ci(y, [s1, s2], n_boot=200)
    assert lo < expected < hi


def test_binned_auc_removes_covariate_information():
    rng = np.random.default_rng(0)
    n = 20000
    cov = rng.normal(size=n)
    y = (cov + rng.normal(size=n) > 0).astype(int)
    assert roc_auc_score(y, cov) > 0.75
    bins = binned_auc(y, cov, cov, n_bins=5)
    assert len(bins) == 5
    assert all(b['auc'] < 0.7 for b in bins)


def test_binned_auc_skips_bins_without_both_classes():
    y = np.r_[np.zeros(100), np.ones(100)]
    cov = np.r_[np.arange(100), np.arange(100, 200)]
    s = np.random.default_rng(0).normal(size=200)
    assert binned_auc(y, s, cov, n_bins=4, min_per_class=20) == []


def test_benchmark_row_and_within_bin_min():
    y, s = _data(n=6000)
    cov = {'ht': np.random.default_rng(1).normal(size=6000)}
    row = benchmark_row('toy', y, s, cov, 'particle', level=0.95, n_boot=50, notes='x')
    assert row['name'] == 'toy' and row['group'] == 'particle' and row['notes'] == 'x'
    assert row['ci_lo'] <= row['auc'] <= row['ci_hi']
    assert within_bin_min(row) == min(b['auc'] for b in row['within_bin']['ht'])


def test_gate_uses_best_particle_baseline_and_ignores_jet_rows():
    rows = [_row('HT', 'particle', 0.79, 0.78, 0.7),
            _row('mjj', 'jet', 0.85, 0.84, 0.7),
            _row('m', 'model', 0.80, 0.795, 0.65)]
    gate = evaluate_gate(rows)
    assert gate['bar_name'] == 'HT'
    assert gate['passed'] is True

    rows[2] = _row('m', 'model', 0.80, 0.785, 0.65)   # CI overlaps the bar
    assert evaluate_gate(rows)['passed'] is False

    rows[2] = _row('m', 'model', 0.82, 0.81, 0.55)    # beats the bar, but within-bin AUC too low
    gate = evaluate_gate(rows)
    assert gate['models'][0]['beats_bar'] is True
    assert gate['passed'] is False


def test_gate_requires_both_row_kinds():
    with pytest.raises(ValueError):
        evaluate_gate([_row('HT', 'particle', 0.79, 0.78, 0.7)])


def test_check_reference():
    rows = [_row('HT', 'particle', 0.79, 0.78, 0.7)]
    rows[0]['ci_hi'] = 0.80
    ref = check_reference(rows, {'HT': 0.792, 'missing': 0.5})
    assert ref['HT']['ok'] is True
    assert ref['missing']['ok'] is False and ref['missing']['missing'] is True
    assert check_reference(rows, {'HT': 0.9})['HT']['ok'] is False


def test_render_and_write(tmp_path):
    rows = [_row('HT', 'particle', 0.79, 0.78, 0.7), _row('m', 'model', 0.80, 0.795, 0.65)]
    gate = evaluate_gate(rows)
    md = render_markdown(rows, gate, check_reference(rows, {'HT': 0.79}), title='T')
    assert '| HT |' in md and 'PASSED' in md
    write_report(rows, gate, None, tmp_path / 'r.json', tmp_path / 'r.md', 'T', meta={'a': 1})
    assert json.loads((tmp_path / 'r.json').read_text())['meta'] == {'a': 1}
    assert (tmp_path / 'r.md').read_text().startswith('# T')
