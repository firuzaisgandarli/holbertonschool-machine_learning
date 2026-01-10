#!/usr/bin/env python3
"""
Module for creating a pandas DataFrame from a NumPy ndarray
with alphabetically labeled, capitalized columns (A–Z).
"""

import pandas as pd


def from_numpy(array):
    """
    Create a pandas DataFrame from a NumPy ndarray with columns labeled A–Z.

    Parameters
    ----------
    array : np.ndarray
        The NumPy array to convert into a DataFrame. It is assumed to be
        2-dimensional. There will not be more than 26 columns.

    Returns
    -------
    pd.DataFrame
        A DataFrame constructed from the input array, with columns labeled
        in alphabetical order and capitalized (A, B, C, ...).
    """
    n_cols = array.shape[1]
    if n_cols > 26:
        raise ValueError("There will not be more than 26 columns")

    # Generate column labels: 'A', 'B', ..., up to n_cols
    columns = [chr(ord('A') + i) for i in range(n_cols)]

    # Construct and return the DataFrame
    return pd.DataFrame(array, columns=columns)
