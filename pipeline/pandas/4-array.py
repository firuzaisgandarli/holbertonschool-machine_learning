#!/usr/bin/env python3
"""
Module that defines a function to convert
the last 10 rows of specific DataFrame columns
into a NumPy ndarray.
"""


def array(df):
    """
    Select the last 10 rows of 'High' and 'Close' columns
    from the DataFrame and convert them into a NumPy ndarray.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing 'High' and 'Close' columns.

    Returns
    -------
    numpy.ndarray
        A NumPy array with the last 10 rows of 'High' and 'Close'.
    """
    return df[["High", "Close"]].tail(10).to_numpy()
