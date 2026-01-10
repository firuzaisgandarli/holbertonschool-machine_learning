#!/usr/bin/env python3
"""
Module that defines a function to load data from a file
into a pandas DataFrame.
"""

import pandas as pd


def from_file(filename, delimiter):
    """
    Load data from a file into a pandas DataFrame.

    Parameters
    ----------
    filename : str
        The path to the file to load.
    delimiter : str
        The column separator used in the file.

    Returns
    -------
    pd.DataFrame
        The loaded DataFrame.
    """
    return pd.read_csv(filename, delimiter=delimiter)
