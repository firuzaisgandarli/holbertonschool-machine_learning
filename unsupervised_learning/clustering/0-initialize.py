#!/usr/bin/env python3
import numpy as np


def initialize(X, k):
    """
    Initializes cluster centroids for K-means

    X: numpy.ndarray of shape (n, d)
    k: number of clusters

    Returns:
        numpy.ndarray of shape (k, d) or None on failure
    """

    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None
    if not isinstance(k, int) or k <= 0:
        return None

    n, d = X.shape

    # min and max along each feature dimension
    X_min = X.min(axis=0)
    X_max = X.max(axis=0)

    # single uniform sampling (IMPORTANT requirement)
    centroids = np.random.uniform(X_min, X_max, (k, d))

    return centroids
