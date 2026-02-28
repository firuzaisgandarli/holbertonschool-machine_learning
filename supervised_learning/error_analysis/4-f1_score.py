#!/usr/bin/env python3
"""
Module that calculates the F1 score for each class
"""
import numpy as np

sensitivity = __import__('1-sensitivity').sensitivity
precision = __import__('2-precision').precision


def f1_score(confusion):
    """
    Calculates the F1 score for each class in a confusion matrix

    Parameters:
    confusion (numpy.ndarray): matrix of shape (classes, classes)
                               where rows are correct labels and
                               columns are predicted labels

    Returns:
    numpy.ndarray: shape (classes,) containing F1 score of each class
    """
    sens = sensitivity(confusion)
    prec = precision(confusion)

    return 2 * (prec * sens) / (prec + sens)
