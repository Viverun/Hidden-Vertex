# tests/test_part3_features.py
import numpy as np
import pandas as pd
import pytest

from src.benchmarks.baselines import jet_baselines
from src.data.jet_features import FEATURE_SETS, feature_matrix, fit_quantile, mjj, prepare_sets, split_rows, tau21


def make_jet_frame(n_bg, n_sig, seed=0):
    """Sorted synthetic jet-feature frame (bg first, then signal) in the LHCO features layout."""
    rng = np.random.default_rng(seed)
    n = n_bg + n_sig
    sig = np.r_[np.zeros(n_bg), np.ones(n_sig)].astype(int)
    df = {}
    for j in (1, 2):
        pt = rng.normal(1300, 150, n)
        phi = rng.uniform(-np.pi, np.pi, n) + (np.pi if j == 2 else 0.0)
        df[f'pxj{j}'] = pt * np.cos(phi)
        df[f'pyj{j}'] = pt * np.sin(phi)
        df[f'pzj{j}'] = rng.normal(0, 800, n)
        df[f'mj{j}'] = np.where(sig == 1, rng.normal(500 if j == 1 else 100, 20, n), rng.exponential(80, n) + 5)
        tau1 = rng.uniform(0.1, 1.0, n)
        tau1[:3] = 0.0                                  # exercises the tau1 == 0 rule
        df[f'tau1j{j}'] = tau1
        df[f'tau2j{j}'] = tau1 * np.where(sig == 1, rng.uniform(0.1, 0.5, n), rng.uniform(0.4, 0.9, n))
        df[f'tau3j{j}'] = df[f'tau2j{j}'] * 0.8
    df['label'] = sig.astype(float)
    return pd.DataFrame(df)


def test_tau21_and_mjj_match_jet_baselines():
    df = make_jet_frame(50, 10)
    base = {b.name: b.scores for b in jet_baselines(df)}
    assert np.allclose(mjj(df), base['mjj'])
    assert np.allclose(-tau21(df, 1), base['-tau21 (leading jet)'])
    assert np.all(tau21(df, 1)[:3] == 1.0)


def test_feature_matrix_columns():
    df = make_jet_frame(30, 5)
    x5, x4 = feature_matrix(df, 'jet5'), feature_matrix(df, 'jet4')
    assert x5.shape == (35, 5) and x4.shape == (35, 4)
    assert np.allclose(x5[:, :4], x4)
    assert np.allclose(x5[:, 4], mjj(df))
    assert np.allclose(x5[:, 0], df['mj1']) and np.allclose(x5[:, 3], tau21(df, 2))
    assert FEATURE_SETS['jet5'][-1] == 'mjj'
    with pytest.raises(ValueError):
        feature_matrix(df, 'jet9')


def test_split_rows_matches_part2_layout():
    labels = np.r_[np.zeros(100), np.ones(30)].astype(int)
    split = {'skip_bg': 10, 'n_test_bg': 5, 'n_test_sig': 5, 'n_train': 50, 'val_fraction': 0.1}
    rows = split_rows(labels, split)
    assert list(rows['test']) == list(range(10, 15)) + list(range(100, 105))
    assert list(rows['test_labels']) == [0] * 5 + [1] * 5
    assert list(rows['fit']) == list(range(15, 60))
    assert list(rows['val']) == list(range(60, 65))


def test_quantile_is_fit_on_fit_set_only():
    df = make_jet_frame(400, 50)
    rows = {'fit': np.arange(0, 300), 'val': np.arange(300, 400), 'test': np.arange(380, 450)}
    x, qt = prepare_sets(df, rows, 'jet5', n_quantiles=100)
    ref = fit_quantile(feature_matrix(df.iloc[:300].reset_index(drop=True), 'jet5'), 100)
    assert np.allclose(qt.quantiles_, ref.quantiles_)
    assert x['fit'].dtype == np.float32 and x['test'].shape == (70, 5)
    assert abs(float(np.median(x['fit']))) < 0.1
