#!/usr/bin/env python3
"""
Module that defines a function to remove rows
where the 'Close' column has NaN values.
"""


def prune(df):
    """
    Remove any entries where 'Close' has NaN values.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing a 'Close' column.

    Returns
    -------
    pd.DataFrame
        Modified DataFrame with rows containing NaN in 'Close' removed.
    """
    return df.dropna(subset=["Close"])
