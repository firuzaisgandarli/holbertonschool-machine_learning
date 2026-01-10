#!/usr/bin/env python3
"""
Module that defines a function to clean a DataFrame
by removing Weighted_Price and filling missing values.
"""


def fill(df):
    """
    Remove the 'Weighted_Price' column, fill missing values in 'Close'
    with the previous row’s value, fill missing values in 'High', 'Low',
    and 'Open' with the corresponding 'Close' value in the same row,
    and set missing values in 'Volume_(BTC)' and 'Volume_(Currency)' to 0.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing the required columns.

    Returns
    -------
    pd.DataFrame
        Modified DataFrame with missing values handled.
    """
    # Remove Weighted_Price column
    if "Weighted_Price" in df.columns:
        df = df.drop(columns=["Weighted_Price"])

    # Fill Close with previous row’s value
    df["Close"] = df["Close"].fillna(method="ffill")

    # Fill High, Low, Open with corresponding Close value
    for col in ["High", "Low", "Open"]:
        df[col] = df[col].fillna(df["Close"])

    # Set missing values in Volume_(BTC) and Volume_(Currency) to 0
    for col in ["Volume_(BTC)", "Volume_(Currency)"]:
        df[col] = df[col].fillna(0)

    return df
