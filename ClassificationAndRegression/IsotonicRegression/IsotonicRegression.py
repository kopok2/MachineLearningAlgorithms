# coding=utf-8
"""Isotonic Regression."""

import matplotlib.pyplot as plt
from sklearn import isotonic, metrics, datasets, model_selection


if __name__ == '__main__':
    print("Generating data...")
    X, y = datasets.make_regression(n_samples=1000, n_features=1, noise=0.7, bias=1.2, tail_strength=0.0)
    X = X[:, 0]
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y)

    print("Fitting model...")
    reg = isotonic.IsotonicRegression()
    reg.fit(X_train, y_train)

    print("Evaluating model...")
    print(metrics.r2_score(y_test, reg.predict(X_test)))

    print("Plotting regression...")
    plt.scatter(X, y)
    plt.scatter(X, reg.predict(X))
    plt.show()
