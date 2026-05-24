#!/usr/bin/env python3
import numpy as np
kmeans = __import__('1-kmeans').kmeans


def initialize(X, k):
    """
    Initializes variables for a Gaussian Mixture Model.
    """

    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None
    if not isinstance(k, int) or k <= 0:
        return None, None, None

    n, d = X.shape

    # uniform priors
    pi = np.full(k, 1 / k)

    # means from K-means
    _, m = kmeans(X, k)

    # identity covariance matrices
    S = np.ones((k, d, d))
    S = np.eye(d)[None, :, :].repeat(k, axis=0)

    return pi, m, S
