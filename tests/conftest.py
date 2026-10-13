# tests/conftest.py
import h5py
import numpy as np
import pytest


def make_events(n_bg, n_sig, seed=0):
    """Sorted synthetic LHCO-like rows (bg first, then signal); signal = fewer, harder particles."""
    rng = np.random.default_rng(seed)
    rows = np.zeros((n_bg + n_sig, 2101))
    for i in range(n_bg + n_sig):
        sig = i >= n_bg
        n = int(rng.integers(8, 15)) if sig else int(rng.integers(15, 40))
        pt = rng.exponential(30.0 if sig else 10.0, n) + 1.0
        eta = rng.normal(0.0, 1.0, n)
        phi = rng.uniform(-np.pi, np.pi, n)
        rows[i, :3 * n] = np.stack([pt, eta, phi], 1).ravel()
        rows[i, 2100] = float(sig)
    return rows


@pytest.fixture
def events_h5(tmp_path):
    """HDF5 file with 200 background rows followed by 60 signal rows, in the LHCO layout."""
    path = tmp_path / 'events.h5'
    with h5py.File(path, 'w') as f:
        f.create_group('df').create_dataset('block0_values', data=make_events(200, 60))
    return str(path)
