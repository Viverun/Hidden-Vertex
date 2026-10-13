"""Density-style anomaly scores in latent space. Always fit on training latents only."""
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors


class _Standardizer:
    def fit(self, z):
        z = np.asarray(z, dtype=np.float64)
        self.mean = z.mean(0)
        self.std = np.where(z.std(0) > 1e-8, z.std(0), 1.0)      # dead dimensions pass through unscaled
        return self

    def __call__(self, z):
        return (np.asarray(z, dtype=np.float64) - self.mean) / self.std


class KNNScorer:
    """Mean distance to the k nearest training latents."""

    def __init__(self, k=20):
        self.k, self.n_fit, self._nn = k, 0, None

    def fit(self, z_train):
        self._scale = _Standardizer().fit(z_train)
        self._nn = NearestNeighbors(n_neighbors=self.k).fit(self._scale(z_train))
        self.n_fit = len(z_train)
        return self

    def score(self, z):
        if self._nn is None:
            raise RuntimeError('KNNScorer.score called before fit')
        distances, _ = self._nn.kneighbors(self._scale(z))
        return distances.mean(axis=1)


class GMMScorer:
    """Negative log-likelihood of z under a Gaussian mixture fit to the training latents."""

    def __init__(self, n_components=8, seed=0):
        self.n_components, self.seed, self.n_fit, self._gmm = n_components, seed, 0, None

    def fit(self, z_train):
        self._scale = _Standardizer().fit(z_train)
        self._gmm = GaussianMixture(self.n_components, covariance_type='full', reg_covar=1e-4,
                                    max_iter=200, random_state=self.seed).fit(self._scale(z_train))
        self.n_fit = len(z_train)
        return self

    def score(self, z):
        if self._gmm is None:
            raise RuntimeError('GMMScorer.score called before fit')
        return -self._gmm.score_samples(self._scale(z))


def fit_scorers(z_train, knn_k=20, gmm_components=8, seed=0):
    return {'knn': KNNScorer(knn_k).fit(z_train), 'gmm': GMMScorer(gmm_components, seed).fit(z_train)}
