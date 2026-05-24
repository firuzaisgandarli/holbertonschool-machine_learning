#!/usr/bin/env python3
import numpy as np
import sklearn.cluster


def kmeans(X, k):
    """
    Performs K-means clustering using sklearn.

    Parameters:
    - X: numpy.ndarray of shape (n, d)
    - k: number of clusters

    Returns:
    - C: centroids (k, d)
    - clss: cluster labels (n,)
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None
    if not isinstance(k, int) or k <= 0:
        return None, None

    model = sklearn.cluster.KMeans(n_clusters=k, n_init=10, random_state=0)
    model.fit(X)

    C = model.cluster_centers_
    clss = model.labels_

    return C, clss
