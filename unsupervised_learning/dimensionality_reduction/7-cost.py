#!/usr/bin/env python3
"""t-SNE cost function (KL divergence)."""

import numpy as np


def cost(P, Q):
    """
    Calculates the cost of the t-SNE transformation.

    Args:
        P (numpy.ndarray): P affinities (n, n)
        Q (numpy.ndarray): Q affinities (n, n)

    Returns:
        float: Cost
    """
    # Avoid log(0) and division by 0
    P = np.maximum(P, 1e-12)
    Q = np.maximum(Q, 1e-12)

    C = np.sum(P * np.log(P / Q))

    return C
