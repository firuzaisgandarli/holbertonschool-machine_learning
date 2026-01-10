#!/usr/bin/env python3
"""
Module that defines a function to rename and transform
the Timestamp column in a pandas DataFrame.
"""

import pandas as pd


def rename(df):
    """
    Rename the 'Timestamp' column to 'Datetime',
    convert values to datetime, and keep only
    'Datetime' and 'Close' columns.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing a 'Timestamp' column.

    Returns
    -------
    pd.DataFrame
        Modified DataFrame with 'Datetime' and 'Close' columns.
    """
    # Rename column
    df = df.rename(columns={"Timestamp": "Datetime"})

    # Convert to datetime (timestamps are in seconds)
    df["Datetime"] = pd.to_datetime(df["Datetime"], unit="s")

    # Keep only Datetime and Close
    return df[["Datetime", "Close"]]
