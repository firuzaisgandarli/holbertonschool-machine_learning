#!/usr/bin/env python3
"""
Module that defines a function to sort a DataFrame
in reverse chronological order and transpose it.
"""


def flip_switch(df):
    """
    Sort the DataFrame in reverse chronological order
    and transpose the result.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing a 'Timestamp' column.

    Returns
    -------
    pd.DataFrame
        Transposed DataFrame sorted in reverse chronological order.
    """
    # Sort by Timestamp descending
    df_sorted = df.sort_values("Timestamp", ascending=False)

    # Transpose the DataFrame
    return df_sorted.transpose()
