#!/usr/bin/env python3
"""
Module that defines a function to rearrange MultiIndex
so that Timestamp is the first level, and concatenate
bitstamp and coinbase data in a given range.
"""

import pandas as pd
index = __import__('10-index').index


def hierarchy(df1, df2):
    """
    Rearrange MultiIndex so that Timestamp is the first level.
    Concatenate bitstamp and coinbase tables from timestamps
    1417411980 to 1417417980 inclusive, with keys.

    Parameters
    ----------
    df1 : pd.DataFrame
        First DataFrame (coinbase).
    df2 : pd.DataFrame
        Second DataFrame (bitstamp).

    Returns
    -------
    pd.DataFrame
        Concatenated DataFrame with Timestamp as first level
        and keys 'bitstamp' and 'coinbase'.
    """
    # Index both DataFrames on Timestamp
    df1 = index(df1)
    df2 = index(df2)

    # Filter both DataFrames within the given timestamp range
    df1 = df1.loc[1417411980:1417417980]
    df2 = df2.loc[1417411980:1417417980]

    # Concatenate with keys
    df = pd.concat([df2, df1], keys=["bitstamp", "coinbase"])

    # Swap levels so Timestamp is first
    df = df.swaplevel(0, 1)

    # Sort by Timestamp to ensure chronological order
    return df.sort_index(level=0)
