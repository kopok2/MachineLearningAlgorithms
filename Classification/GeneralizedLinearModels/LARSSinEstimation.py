# coding=utf-8
"""LARS.

Least Angle Regression - high dimensional data regression.
Sinus function regression.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn import linear_model


if __name__ == "__main__":
    print("Generating data...")
    x = np.arange(-20, 20, 0.01)
    y = np.sin(x)
    X = x.reshape(-1, 1)
    n_samples = X.shape[0]
    print(X, y)
    X_train, X_test = X[:n_samples // 2], X[n_samples // 2:]
    y_train, y_test = y[:n_samples // 2], y[n_samples // 2:]

    print("Fitting model...")
    lars = linear_model.Lars()
    lars.fit(X_train, y_train)
    print(lars)
    print("R2 score: {0}".format(r2_score(y_test, lars.predict(X_test))))

    print("Plotting predictions...")
    plt.scatter(x[n_samples // 2:], y_train, color="red")
    plt.scatter(x[:n_samples // 2], y_test, color="fuchsia")
    plt.plot(x, lars.predict(X), color="purple")
    plt.show()
