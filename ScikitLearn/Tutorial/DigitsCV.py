# coding=utf-8
"""Digits dataset crossvalidation classifier."""

import numpy as np
from sklearn import datasets, svm
from sklearn.model_selection import cross_val_score


if __name__ == "__main__":
    digits = datasets.load_digits()
    X = digits.data
    y = digits.target

    model = svm.SVC(kernel="linear")
    cs = np.logspace(-10, 0, 10)
    scores = []
    for c in cs:
        model.C = c
        scores.append(np.mean(cross_val_score(model, X, y, cv=5, n_jobs=1)))
    print(scores)
