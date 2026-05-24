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

    if kmin <= 0 or kmax <= 0 or kmax < kmin:
        return None, None

    results = []
    vars_ = []

    for k in range(kmin, kmax + 1):
        C, clss = kmeans(X, k, iterations)
        results.append((C, clss))
        vars_.append(variance(X, C))

    vars_ = np.array(vars_)
    d_vars = vars_[0] - vars_

    return results, d_vars
