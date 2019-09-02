# coding=utf-8
"""Passive aggressive algorithms regression.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn import linear_model


if __name__ == "__main__":
    print("Generating data...")
    n_samples, n_features = 200, 1
    X = np.random.randn(n_samples, n_features)
    idx = np.arange(n_features)
    idd = np.arange(n_samples)
    coef = (-1) ** idx * np.exp(-idx / 12)
    coef[10:] = 0
    y = np.dot(X, coef)
    y += 0.01 * np.random.normal(size=n_samples)
    X_train, X_test = X[:n_samples // 2], X[n_samples // 2:]
    y_train, y_test = y[:n_samples // 2], y[n_samples // 2:]

    print("Fitting model...")
    model = linear_model.PassiveAggressiveRegressor()
    model.fit(X_train, y_train)
    print(model)
    print("R2 score: {0}".format(r2_score(y_test, model.predict(X_test))))

    plt.scatter(np.arange(n_samples // 2), y_train, color="purple")
    plt.scatter(np.arange(n_samples // 2) + (n_samples // 2), y_test, color="red")
    plt.plot(np.arange(n_samples // 2), model.predict(X_train), color="purple")
    plt.plot(np.arange(n_samples // 2) + (n_samples // 2), model.predict(X_test), color="red")
    plt.show()
