#!/usr/bin/env python3
import numpy as np
kmeans = __import__('1-kmeans').kmeans
variance = __import__('2-variance').variance


def optimum_k(X, kmin=1, kmax=None, iterations=1000):
    """
    Finds optimal k using variance differences.
    """

    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None

    if kmax is None:
        kmax = kmin + 1

    if not isinstance(kmin, int) or not isinstance(kmax, int):
        return None, None

    if kmin <= 0 or kmax <= 0 or kmax < kmin:
        return None, None

    if not isinstance(iterations, int) or iterations <= 0:
        return None, None

    results = []
    vars_list = []

    for k in range(kmin, kmax + 1):
        C, clss = kmeans(X, k, iterations)
        results.append((C, clss))
        vars_list.append(variance(X, C))

    vars_arr = np.array(vars_list)
    d_vars = vars_arr[0] - vars_arr

    return results, d_vars
