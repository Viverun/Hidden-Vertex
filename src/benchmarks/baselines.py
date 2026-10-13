# src/benchmarks/baselines.py
"""Label-free baseline scores. Orientation is fixed here: higher = more anomalous.

Signs follow the physics of a heavy resonance decaying to four quarks (more energy, fewer and
harder particles, jet mass and two-prong substructure). The signs of -multiplicity and -tau21 were
first seen on labelled data in Part 1, so they are reference points, not blind predictions.
`-|mjj - m| (oracle)` uses the signal mass and is not model independent.
"""
from typing import NamedTuple

import numpy as np
import pandas as pd


class Baseline(NamedTuple):
    name: str
    scores: np.ndarray
    group: str      # 'particle' | 'jet' | 'oracle'
    notes: str


def particle_baselines(summary):
    return [
        Baseline('HT (sum pT)', np.asarray(summary['ht']), 'particle', 'more energy = anomalous'),
        Baseline('mean pT per particle', np.asarray(summary['mean_pt']), 'particle',
                 '= HT / multiplicity'),
        Baseline('leading pT', np.asarray(summary['max_pt']), 'particle', ''),
        Baseline('-multiplicity', -np.asarray(summary['multiplicity']), 'particle',
                 'fewer particles = anomalous (sign seen on labelled data in Part 1)'),
        Baseline('m_all', np.asarray(summary['m_all']), 'particle', 'invariant mass of all particles'),
    ]


def load_jet_features(path, rows, labels):
    """Jet features for the given HDF5 rows; raises if their label column disagrees with `labels`."""
    df = pd.read_hdf(path).iloc[np.asarray(rows)].reset_index(drop=True)
    if not np.array_equal(df['label'].to_numpy().astype(int), np.asarray(labels).astype(int)):
        raise ValueError('jet-feature file labels do not match the event labels (row misalignment)')
    return df


def jet_baselines(df, oracle_mass=3500.0):
    def energy(i):
        return np.sqrt(df[f'mj{i}'] ** 2 + df[f'pxj{i}'] ** 2 + df[f'pyj{i}'] ** 2 + df[f'pzj{i}'] ** 2)

    px = df['pxj1'] + df['pxj2']
    py = df['pyj1'] + df['pyj2']
    pz = df['pzj1'] + df['pzj2']
    mjj = np.sqrt(np.maximum((energy(1) + energy(2)) ** 2 - px**2 - py**2 - pz**2, 0.0)).to_numpy()
    tau1 = df['tau1j1'].to_numpy(dtype=float)
    tau2 = df['tau2j1'].to_numpy(dtype=float)
    tau21 = np.where(tau1 > 0, tau2 / np.where(tau1 > 0, tau1, 1.0), 1.0)
    mass_label = f'{oracle_mass:g}'
    return [
        Baseline('mjj', mjj, 'jet', 'dijet invariant mass'),
        Baseline('mj1', df['mj1'].to_numpy(dtype=float), 'jet', 'leading jet mass'),
        Baseline('mj2', df['mj2'].to_numpy(dtype=float), 'jet', 'second jet mass'),
        Baseline('-tau21 (leading jet)', -tau21, 'jet', 'two-prong substructure'),
        Baseline(f'-|mjj-{mass_label}| (oracle)', -np.abs(mjj - oracle_mass), 'oracle',
                 'needs the signal mass; not model independent'),
    ]
