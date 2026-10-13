import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from src.utils.latent_scores import GMMScorer, KNNScorer, fit_scorers


def _blobs(seed=0):
    rng = np.random.default_rng(seed)
    train = rng.normal(0, 1, (2000, 6))
    inliers = rng.normal(0, 1, (300, 6))
    outliers = rng.normal(0, 1, (300, 6)) + 5.0
    return train, np.vstack([inliers, outliers]), np.r_[np.zeros(300), np.ones(300)]


@pytest.mark.parametrize('scorer', [KNNScorer(k=10), GMMScorer(n_components=3, seed=0)])
def test_outliers_score_higher(scorer):
    train, test, y = _blobs()
    scorer.fit(train)
    assert scorer.n_fit == 2000
    assert roc_auc_score(y, scorer.score(test)) > 0.99


def test_score_before_fit_raises():
    with pytest.raises(RuntimeError):
        KNNScorer().score(np.zeros((2, 3)))
    with pytest.raises(RuntimeError):
        GMMScorer().score(np.zeros((2, 3)))


def test_fit_scorers_and_constant_dimension_is_safe():
    train, test, y = _blobs()
    train[:, 0] = 1.0; test[:, 0] = 1.0                    # a dead latent dimension must not break scoring
    scorers = fit_scorers(train, knn_k=10, gmm_components=2, seed=0)
    assert set(scorers) == {'knn', 'gmm'}
    for s in scorers.values():
        assert np.isfinite(s.score(test)).all()
        assert roc_auc_score(y, s.score(test)) > 0.99
