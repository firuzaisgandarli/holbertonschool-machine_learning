#!/usr/bin/env python3
"""
Module for calculating the definiteness of a square matrix.
"""

import numpy as np


def definiteness(matrix):
    """
    Calculates the definiteness of a square matrix.

    Args:
        matrix (numpy.ndarray): The matrix whose definiteness is to be calculated.

    Returns:
        str: One of 'Positive definite', 'Positive semi-definite',
             'Negative semi-definite', 'Negative definite', 'Indefinite',
             or None if the matrix is invalid or does not fit categories.

    Raises:
        TypeError: If matrix is not a numpy.ndarray.
    """
    if not isinstance(matrix, np.ndarray):
        raise TypeError("matrix must be a numpy.ndarray")

    # Kare matris ve boş olmama kontrolü
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or matrix.size == 0:
        return None

    # Simetri kontrolü
    if not np.allclose(matrix, matrix.T):
        return None

    eigvals = np.linalg.eigvals(matrix)
    eigvals = np.real_if_close(eigvals)

    if np.all(eigvals > 0):
        return "Positive definite"
    if np.all(eigvals >= 0):
        return "Positive semi-definite"
    if np.all(eigvals < 0):
        return "Negative definite"
    if np.all(eigvals <= 0):
        return "Negative semi-definite"
    if np.any(eigvals > 0) and np.any(eigvals < 0):
        return "Indefinite"

    return None

