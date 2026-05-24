#!/usr/bin/env python3
import numpy as np

def maximization(X, g):
    """
    Performs the maximization step in the EM algorithm for a GMM.
    """

    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None
    if not isinstance(g, np.ndarray) or len(g.shape) != 2:
        return None, None, None

    n, d = X.shape
    k = g.shape[0]

    if g.shape[1] != n:
        return None, None, None

    # Effective number of points per cluster
    nk = np.sum(g, axis=1)  # (k,)

    # Avoid division by zero
    nk_safe = np.clip(nk, 1e-300, None)

    # Priors
    pi = nk / n

    # Means (k, d)
    m = np.dot(g, X) / nk_safe[:, np.newaxis]

    # Covariances (k, d, d)
    S = np.zeros((k, d, d))

    for i in range(k):   # only loop allowed
        diff = X - m[i]  # (n, d)
        weighted = diff.T * g[i]  # (d, n)
        S[i] = np.dot(weighted, diff) / nk_safe[i]

    return pi, m, S
