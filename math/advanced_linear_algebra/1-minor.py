#!/usr/bin/env python3
"""
Module for calculating the minor matrix of a square matrix.
"""


def minor(matrix):
    """
    Calculates the minor matrix of a square matrix.

    Args:
        matrix (list of lists): The matrix whose minor matrix is to be calculated.

    Returns:
        list of lists: The minor matrix.

    Raises:
        TypeError: If matrix is not a list of lists.
        ValueError: If matrix is not square or is empty.
    """
    if not isinstance(matrix, list) or \
       any(not isinstance(row, list) for row in matrix):
        raise TypeError("matrix must be a list of lists")

    n = len(matrix)
    if n == 0 or any(len(row) != n for row in matrix):
        raise ValueError("matrix must be a non-empty square matrix")

    from 0-determinant import determinant

    if n == 1:
        return [[1]]

    minors = []
    for i in range(n):
        row_minors = []
        for j in range(n):
            submatrix = [r[:j] + r[j+1:] for k, r in enumerate(matrix) if k != i]
            row_minors.append(determinant(submatrix))
        minors.append(row_minors)

    return minors

