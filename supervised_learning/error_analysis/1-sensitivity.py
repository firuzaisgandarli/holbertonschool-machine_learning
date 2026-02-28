#!/usr/bin/env python3
"""
Module that calculates sensitivity for each class
"""
import numpy as np


def sensitivity(confusion):
    """
    Calculates the sensitivity for each class in a confusion matrix

    Parameters:
    confusion (numpy.ndarray): matrix of shape (classes, classes)
                               where rows are correct labels and
                               columns are predicted labels

    Returns:
    numpy.ndarray: shape (classes,) containing sensitivity of each class
    """
    true_positives = np.diag(confusion)
    actual_positives = np.sum(confusion, axis=1)

    return true_positives / actual_positives
