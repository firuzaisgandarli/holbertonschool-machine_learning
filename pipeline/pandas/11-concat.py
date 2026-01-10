#!/usr/bin/env python3
"""
Module that defines a function to concatenate two DataFrames
with specific indexing and filtering rules.
"""

import pandas as pd
index = __import__('10-index').index


def concat(df1, df2):
    """
    Index both DataFrames on 'Timestamp', include all timestamps from df2
    up to and including 1417411920, then concatenate df2 above df1 with keys.

    Parameters
    ----------
    df1 : pd.DataFrame
        First DataFrame (coinbase).
    df2 : pd.DataFrame
        Second DataFrame (bitstamp).

    Returns
    -------
    pd.DataFrame
        Concatenated DataFrame with keys 'bitstamp' and 'coinbase'.
    """
    # Index both DataFrames on Timestamp
    df1 = index(df1)
    df2 = index(df2)

    # Filter df2 up to and including timestamp 1417411920
    df2 = df2.loc[:1417411920]

    # Concatenate with keys
    return pd.concat([df2, df1], keys=["bitstamp", "coinbase"])
