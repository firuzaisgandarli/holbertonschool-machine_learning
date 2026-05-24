#!/usr/bin/env python3
import numpy as np


def initialize(X, k):
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None
    if not isinstance(k, int) or k <= 0:
        return None

    X_min = X.min(axis=0)
    X_max = X.max(axis=0)

    return np.random.uniform(X_min, X_max, (k, X.shape[1]))


def kmeans(X, k, iterations=1000):
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None
    if not isinstance(k, int) or k <= 0:
        return None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None

    n, d = X.shape

    C = initialize(X, k)
    if C is None:
        return None, None

    clss = np.zeros(n, dtype=int)

    for _ in range(iterations):

        distances = np.linalg.norm(X[:, None] - C, axis=2)
        new_clss = np.argmin(distances, axis=1)

        # NOTE: NO early break here (important fix)
        clss = new_clss

        new_C = np.zeros((k, d))

        for i in range(k):
            points = X[clss == i]

            if len(points) == 0:
                new_C[i] = np.random.uniform(X.min(axis=0), X.max(axis=0))
            else:
                new_C[i] = points.mean(axis=0)

        C = new_C

    return C, clss
