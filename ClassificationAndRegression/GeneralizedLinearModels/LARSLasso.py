# coding=utf-8
"""LARS Lasso regression.

Least angle regression least absolute shrinkage and selection operator.
"""


import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn import linear_model


if __name__ == "__main__":
    print("Generating data...")
    n_samples, n_features = 200, 30
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
    larslasso = linear_model.LassoLars(alpha=0.1)
    larslasso.fit(X_train, y_train)
    print(larslasso)
    print("R2 score: {0}".format(r2_score(y_test, larslasso.predict(X_test))))

    print("Plotting predictions...")
    plt.scatter(idd[:n_samples // 2], y_train, color="red")
    plt.scatter(idd[n_samples // 2:], y_test, color="fuchsia")
    plt.plot(idd, larslasso.predict(X), color="purple")
    plt.show()
