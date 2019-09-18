# coding=utf-8
"""Removing features with variance not meeting specified threshold."""

import numpy as np
from sklearn import feature_selection


if __name__ == "__main__":
    X = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0], [0, 1, 1], [0, 1, 0], [0, 1, 1]])
    print(X)

    threshold_alpha = 0.8
    selector = feature_selection.VarianceThreshold(threshold=threshold_alpha * (1.0 - threshold_alpha))
    selector.fit(X)
    X_t = selector.transform(X)
    print(X_t)
