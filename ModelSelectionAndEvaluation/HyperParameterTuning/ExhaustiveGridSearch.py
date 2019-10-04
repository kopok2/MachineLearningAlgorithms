# coding=utf-8
"""Exhaustive Grid Search."""

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.svm import SVC

if __name__ == '__main__':
    print("Loading dataset...")
    X, y = load_digits(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    print("Performing grid search...")
    gs = GridSearchCV(SVC(),
                      [{'kernel': ['rbf'], 'gamma': np.logspace(0.01, 0.00001, 10), 'C': np.logspace(1, 1000, 10)},
                       {'kernel': ['linear'], 'C': np.logspace(1, 1000, 10)},
                       {'kernel': ['poly'], 'C': np.logspace(1, 1000, 10), 'degree': np.linspace(1, 100, 10)}],
                      cv=5, scoring="accuracy")
    gs.fit(X_train, y_train)
    print(gs.cv_results_["mean_test_score"])
