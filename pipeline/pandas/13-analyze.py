#!/usr/bin/env python3
"""
Module that defines a function to compute descriptive statistics
for all columns except the Timestamp column.
"""


def analyze(df):
    """
    Compute descriptive statistics for all columns except 'Timestamp'.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing a 'Timestamp' column.

    Returns
    -------
    pd.DataFrame
        DataFrame containing descriptive statistics for all columns
        except 'Timestamp'.
    """
    # Drop Timestamp column if it exists
    if "Timestamp" in df.columns:
        df = df.drop(columns=["Timestamp"])

    # Compute descriptive statistics
    return df.describe()
