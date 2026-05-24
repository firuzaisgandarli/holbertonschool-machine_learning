#!/usr/bin/env python3
import numpy as np
from sklearn.mixture import GaussianMixture


def gmm(X, k):
    """
    Performs Gaussian Mixture Model clustering on a dataset.

    Parameters:
    - X: numpy.ndarray of shape (n, d), dataset
    - k: number of clusters

    Returns:
    - pi: cluster priors (k,)
    - m: cluster means (k, d)
    - S: covariance matrices (k, d, d)
    - clss: cluster assignments (n,)
    - bic: Bayesian Information Criterion score
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None, None, None
    if not isinstance(k, int) or k <= 0:
        return None, None, None, None, None

    # Fit Gaussian Mixture Model
    gmm_model = GaussianMixture(n_components=k, covariance_type='full')
    gmm_model.fit(X)

    # Priors (weights)
    pi = gmm_model.weights_

    # Means
    m = gmm_model.means_

    # Covariances
    S = gmm_model.covariances_

    # Cluster assignments
    clss = gmm_model.predict(X)

    # BIC score
    bic = gmm_model.bic(X)

    return pi, m, S, clss, bic
