#!/usr/bin/env python3
"""
Module that defines a function to sort a DataFrame
by the 'High' column in descending order.
"""


def high(df):
    """
    Sort the DataFrame by the 'High' column in descending order.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing a 'High' column.

    Returns
    -------
    pd.DataFrame
        DataFrame sorted by 'High' in descending order.
    """
    return df.sort_values("High", ascending=False)
