#!/usr/bin/env python3
import numpy as np


def variance(X, C):
    """
    Calculates total intra-cluster variance

    X: (n, d) data points
    C: (k, d) centroids

    Returns:
        float variance or None on failure
    """

    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None
    if not isinstance(C, np.ndarray) or len(C.shape) != 2:
        return None

    # compute distances from every point to every centroid
    distances = np.linalg.norm(X[:, np.newaxis] - C, axis=2)

    # assign each point to closest centroid
    min_dist = np.min(distances, axis=1)

    # total variance (sum of squared distances)
    var = np.sum(min_dist ** 2)

    return var
