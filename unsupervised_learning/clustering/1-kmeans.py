#!/usr/bin/env python3
import numpy as np


def initialize(X, k):
    """Initialize centroids (from task 0)"""
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None
    if not isinstance(k, int) or k <= 0:
        return None

    X_min = X.min(axis=0)
    X_max = X.max(axis=0)

    return np.random.uniform(X_min, X_max, (k, X.shape[1]))


def kmeans(X, k, iterations=1000):
    """
    Performs K-means clustering

    Returns:
        C: centroids (k, d)
        clss: cluster assignment (n,)
    """

    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None
    if not isinstance(k, int) or k <= 0:
        return None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None

    n, d = X.shape

    # initialize centroids
    C = initialize(X, k)
    if C is None:
        return None, None

    clss = np.zeros(n, dtype=int)

    for _ in range(iterations):

        # distance computation (vectorized)
        distances = np.linalg.norm(X[:, np.newaxis] - C, axis=2)
        new_clss = np.argmin(distances, axis=1)

        # stop if no change
        if np.array_equal(new_clss, clss):
            break

        clss = new_clss

        # update centroids
        new_C = np.zeros((k, d))

        for i in range(k):  # allowed second loop
            points = X[clss == i]

            if len(points) == 0:
                # reinitialize empty cluster
                new_C[i] = np.random.uniform(X.min(axis=0), X.max(axis=0))
            else:
                new_C[i] = points.mean(axis=0)

        C = new_C

    return C, clss
