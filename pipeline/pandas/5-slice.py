#!/usr/bin/env python3
"""
Module that defines a function to slice a DataFrame
by selecting specific columns and every 60th row.
"""


def slice(df):
    """
    Extract the columns 'High', 'Low', 'Close', and 'Volume_(BTC)'
    and select every 60th row.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing the required columns.

    Returns
    -------
    pd.DataFrame
        Sliced DataFrame with selected columns and every 60th row.
    """
    return df[["High", "Low", "Close", "Volume_(BTC)"]].iloc[::60]
