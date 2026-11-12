"""Jet-level feature sets for Part 3 (dijet summary: jet masses, tau21, mjj).

All functions take the pandas frame of `events_anomalydetection_v2.features.h5` (columns pxj1, pyj1, pzj1,
mj1, tau1j1, tau2j1, tau3j1, the same for j2, and label). The definitions of tau21 and mjj are the same as in
`src.benchmarks.baselines.jet_baselines`.
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import QuantileTransformer

from src.data.event_store import plan_rows

FEATURE_SETS = {
    'jet5': ('mj1', 'mj2', 'tau21_j1', 'tau21_j2', 'mjj'),
    'jet4': ('mj1', 'mj2', 'tau21_j1', 'tau21_j2'),
}


def tau21(df, jet):
    """tau2/tau1 of jet 1 or 2; 1.0 where tau1 == 0 (as in jet_baselines)."""
    tau1 = df[f'tau1j{jet}'].to_numpy(dtype=float)
    tau2 = df[f'tau2j{jet}'].to_numpy(dtype=float)
    return np.where(tau1 > 0, tau2 / np.where(tau1 > 0, tau1, 1.0), 1.0)


def mjj(df):
    def energy(i):
        return np.sqrt(df[f'mj{i}'] ** 2 + df[f'pxj{i}'] ** 2 + df[f'pyj{i}'] ** 2 + df[f'pzj{i}'] ** 2)

    px = df['pxj1'] + df['pxj2']
    py = df['pyj1'] + df['pyj2']
    pz = df['pzj1'] + df['pzj2']
    return np.sqrt(np.maximum((energy(1) + energy(2)) ** 2 - px**2 - py**2 - pz**2, 0.0)).to_numpy()


def feature_matrix(df, feature_set):
    """(n, d) float64 matrix of the named feature set, in the column order of FEATURE_SETS."""
    if feature_set not in FEATURE_SETS:
        raise ValueError(f'unknown feature set {feature_set!r}; choose from {sorted(FEATURE_SETS)}')
    columns = {
        'mj1': lambda: df['mj1'].to_numpy(dtype=float),
        'mj2': lambda: df['mj2'].to_numpy(dtype=float),
        'tau21_j1': lambda: tau21(df, 1),
        'tau21_j2': lambda: tau21(df, 2),
        'mjj': lambda: mjj(df),
    }
    return np.stack([columns[name]() for name in FEATURE_SETS[feature_set]], axis=1).astype(np.float64)


def load_features(path):
    return pd.read_hdf(path)


def split_rows(labels, split):
    """Row ids of the Part 2 split: test (bg then signal), fit (first 90% of train_bg), val (last 10%)."""
    plan = plan_rows(labels, skip_bg=split['skip_bg'], n_test_bg=split['n_test_bg'],
                     n_test_sig=split['n_test_sig'], n_train=split['n_train'], n_contam_sig=0)
    n_val = int(round(split['val_fraction'] * split['n_train']))
    n_fit = split['n_train'] - n_val
    train = plan['train_bg']
    return {
        'test': np.concatenate([plan['test_bg'], plan['test_sig']]),
        'test_labels': np.r_[np.zeros(len(plan['test_bg'])), np.ones(len(plan['test_sig']))].astype(int),
        'fit': train[:n_fit],
        'val': train[n_fit:],
    }


def fit_quantile(x_fit, n_quantiles=1000, random_state=0):
    """QuantileTransformer to a standard normal, fit on the fit events only."""
    return QuantileTransformer(output_distribution='normal', n_quantiles=n_quantiles,
                               random_state=random_state).fit(x_fit)


def prepare_sets(df, rows, feature_set, n_quantiles=1000, random_state=0):
    """Transformed float32 arrays {'fit', 'val', 'test'} for one feature set, plus the fitted transformer."""
    x = {k: feature_matrix(df.iloc[rows[k]].reset_index(drop=True), feature_set) for k in ('fit', 'val', 'test')}
    qt = fit_quantile(x['fit'], n_quantiles, random_state)
    return {k: qt.transform(v).astype(np.float32) for k, v in x.items()}, qt
