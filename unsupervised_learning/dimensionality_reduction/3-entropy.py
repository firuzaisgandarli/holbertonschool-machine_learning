#!/usr/bin/env python3
"""Entropy and P affinities for t-SNE."""

import numpy as np


def HP(Di, beta):
    """
    Calculates Shannon entropy and P affinities.

    Args:
        Di (numpy.ndarray): shape (n - 1,) distances
        beta (numpy.ndarray): shape (1,) precision

    Returns:
        Hi (float): Shannon entropy
        Pi (numpy.ndarray): P affinities
    """
    # Compute P affinities
    Pi = np.exp(-Di * beta)

    # Avoid division by zero
    sum_Pi = np.sum(Pi)
    if sum_Pi == 0:
        Pi = np.zeros_like(Pi)
        Hi = 0
        return Hi, Pi

    Pi = Pi / sum_Pi

    # Compute entropy
    Hi = -np.sum(Pi * np.log2(Pi + 1e-10))

    return Hi, Pi
