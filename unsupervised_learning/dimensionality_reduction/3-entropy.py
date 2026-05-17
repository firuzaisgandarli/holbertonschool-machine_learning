#!/usr/bin/env python3
"""Entropy and P affinities for t-SNE."""

import numpy as np


def HP(Di, beta):
    """
    Calculates Shannon entropy and P affinities.

    Args:
        Di (numpy.ndarray): shape (n - 1,) distances
        beta (numpy.ndarray): shape (1,) beta value

    Returns:
        tuple: Shannon entropy Hi and P affinities Pi
    """
    Pi = np.exp(-Di * beta)
    sum_Pi = np.sum(Pi)

    Pi = Pi / sum_Pi

    Hi = -np.sum(Pi * np.log2(Pi))

    return Hi, Pi
