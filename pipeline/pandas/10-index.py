#!/usr/bin/env python3
"""
Module that defines a function to set the Timestamp column
as the index of a DataFrame.
"""


def index(df):
    """
    Set the 'Timestamp' column as the index of the DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing a 'Timestamp' column.

    Returns
    -------
    pd.DataFrame
        Modified DataFrame with 'Timestamp' set as index.
    """
    return df.set_index("Timestamp")
