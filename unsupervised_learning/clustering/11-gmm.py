#!/usr/bin/env python3
import numpy as np
import sklearn.mixture


def gmm(X, k):
    """
    Performs Gaussian Mixture Model clustering on a dataset.
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None, None, None
    if not isinstance(k, int) or k <= 0:
        return None, None, None, None, None

    # Initialize and fit model
    model = sklearn.mixture.GaussianMixture(
        n_components=k,
        covariance_type='full'
    )
    model.fit(X)

    # Extract parameters
    pi = model.weights_
    m = model.means_
    S = model.covariances_
    clss = model.predict(X)
    bic = model.bic(X)

    return pi, m, S, clss, bic
