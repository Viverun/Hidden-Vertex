# tests/test_part2_baselines.py
import numpy as np
import pandas as pd
import pytest

from src.benchmarks.baselines import jet_baselines, load_jet_features, particle_baselines


def test_particle_baselines_orientation():
    summary = {'ht': np.array([10.0, 20]), 'multiplicity': np.array([4.0, 2]), 'mean_pt': np.array([2.5, 10]),
               'max_pt': np.array([5.0, 15]), 'm_all': np.array([1.0, 3])}
    got = {b.name: b for b in particle_baselines(summary)}
    assert set(got) == {'HT (sum pT)', 'mean pT per particle', 'leading pT', '-multiplicity', 'm_all'}
    assert all(b.group == 'particle' for b in got.values())
    assert got['-multiplicity'].scores.tolist() == [-4.0, -2.0]
    assert got['HT (sum pT)'].scores.tolist() == [10.0, 20.0]


def _jets():
    return pd.DataFrame({
        'mj1': [0.0, 50.0], 'pxj1': [100.0, 100.0], 'pyj1': [0.0, 0.0], 'pzj1': [0.0, 0.0],
        'mj2': [0.0, 60.0], 'pxj2': [-100.0, -100.0], 'pyj2': [0.0, 0.0], 'pzj2': [0.0, 0.0],
        'tau1j1': [0.5, 0.0], 'tau2j1': [0.25, 0.3], 'label': [0, 1],
    })


def test_jet_baselines_values():
    got = {b.name: b for b in jet_baselines(_jets(), oracle_mass=200.0)}
    assert got['mjj'].scores[0] == pytest.approx(200.0)
    assert got['-|mjj-200| (oracle)'].scores[0] == pytest.approx(0.0)
    assert got['-|mjj-200| (oracle)'].group == 'oracle'
    assert got['-tau21 (leading jet)'].scores.tolist() == [-0.5, -1.0]   # tau1 == 0 -> tau21 := 1
    assert got['mj1'].group == 'jet'


def test_load_jet_features_checks_label_alignment(tmp_path):
    df = _jets()
    path = tmp_path / 'f.h5'
    df.to_hdf(path, key='df')
    out = load_jet_features(str(path), np.array([1, 0]), np.array([1, 0]))
    assert out['mj1'].tolist() == [50.0, 0.0]
    with pytest.raises(ValueError, match='labels'):
        load_jet_features(str(path), np.array([0, 1]), np.array([1, 0]))
