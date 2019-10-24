# coding=utf-8
"""Logistic loss objective function."""

import numpy as np


def logistic_loss(y, y_1):
    """Calculate logistic loss.

    Formula:
        LL = E(yln(1 + e^-y_1) + (1-y)ln(1+e^y_1))

    Args:
        y: observed features.
        y_1: predicted features.

    Returns:
        calculated logistic loss.
    """
    return np.sum(y * np.log(1 + np.exp(-y_1)) + (-y + 1) * (np.log(1 + np.exp(y_1))))
