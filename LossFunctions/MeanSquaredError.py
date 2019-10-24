# coding=utf-8
"""Mean squared error measure function."""

import numpy as np


def mean_squared_error(y, y_1):
    """Calculate mean squared error.

    Formula:
        MSE = E(y - y_1)^2

    Args:
        y: observed features.
        y_1: predicted features.

    Returns:
        calculated mean squared error.
    """
    return np.sum((y - y_1) ** 2)
