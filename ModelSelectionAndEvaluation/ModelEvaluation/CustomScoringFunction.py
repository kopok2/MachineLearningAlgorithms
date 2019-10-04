# coding=utf-8
"""Scikit-Learn custom scoring function."""

import numpy as np


def custom_scoring_function(y_true, y_pred):
    return np.sum((y_true - y_pred) ** 2)
