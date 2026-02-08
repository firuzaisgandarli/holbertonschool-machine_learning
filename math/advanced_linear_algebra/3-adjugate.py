#!/usr/bin/env python3
"""
Module for calculating the adjugate matrix of a square matrix.
"""


def adjugate(matrix):
    """
    Calculates the adjugate matrix of a square matrix.

    Args:
        matrix (list of lists): The matrix whose adjugate matrix is to be calculated.

    Returns:
        list of lists: The adjugate matrix.

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

    def determinant(m):
        """Helper function to calculate determinant."""
        if m == [[]]:
            return 1
        size = len(m)
        if size == 1:
            return m[0][0]
        if size == 2:
            return m[0][0] * m[1][1] - m[0][1] * m[1][0]
        det = 0
        for col in range(size):
            sub = [row[:col] + row[col+1:] for row in m[1:]]
            det += ((-1) ** col) * m[0][col] * determinant(sub)
        return det

    def minor(m):
        """Helper function to calculate minor matrix."""
        size = len(m)
        if size == 1:
            return [[1]]
        minors = []
        for i in range(size):
            row_minors = []
            for j in range(size):
                sub = [row[:j] + row[j+1:] for k, row in enumerate(m)
                       if k != i]
                row_minors.append(determinant(sub))
            minors.append(row_minors)
        return minors

    def cofactor(m):
        """Helper function to calculate cofactor matrix."""
        size = len(m)
        if size == 1:
            return [[1]]
        minors = minor(m)
        cofactors = []
        for i in range(size):
            row_cofactors = []
            for j in range(size):
                row_cofactors.append(((-1) ** (i + j)) * minors[i][j])
            cofactors.append(row_cofactors)
        return cofactors

    if n == 1:
        return [[1]]

    cofactors = cofactor(matrix)
    adj = []
    for i in range(n):
        row = []
        for j in range(n):
            row.append(cofactors[j][i])  # transpose
        adj.append(row)

    return adj

