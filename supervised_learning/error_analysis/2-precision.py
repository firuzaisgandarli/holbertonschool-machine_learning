#!/usr/bin/env python3
"""
Module that calculates precision for each class
"""
import numpy as np


def precision(confusion):
    """
    Calculates the precision for each class in a confusion matrix

    Parameters:
    confusion (numpy.ndarray): matrix of shape (classes, classes)
                               where rows are correct labels and
                               columns are predicted labels

    Returns:
    numpy.ndarray: shape (classes,) containing precision of each class
    """
    true_positives = np.diag(confusion)
    predicted_positives = np.sum(confusion, axis=0)

    return true_positives / predicted_positives
