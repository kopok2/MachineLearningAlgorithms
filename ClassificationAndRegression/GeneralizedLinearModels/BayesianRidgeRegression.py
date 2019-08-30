# coding=utf-8
"""Bayesian Ridge Regression"""

import numpy as np
from sklearn import linear_model
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from scipy import stats

if __name__ == "__main__":
    print("Generating data...")
    n_samples, n_features = 1000, 50
    X = np.random.randn(n_samples, n_features)
    rel_features = 10
    lambda_ = 0.4
    w = np.zeros(n_features)
    rel_f = np.random.randint(0, n_features, rel_features)
    for i in rel_f:
        w[i] = stats.norm.rvs(loc=0, scale=1.0 / np.sqrt(lambda_))
    alpha_ = 0.30
    noise = stats.norm.rvs(loc=0, scale=1.0 / np.sqrt(lambda_), size=n_samples)
    y = np.dot(X, w) + noise
    X_train, X_test = X[n_samples // 2:], X[:n_samples // 2]
    y_train, y_test = y[n_samples // 2:], y[:n_samples // 2]

    print("Fitting model...")
    brr = linear_model.BayesianRidge()
    brr.fit(X_train, y_train)
    print("R2 score: {0}".format(r2_score(y_test, brr.predict(X_test))))

    print("Plotting predictions...")
    plt.scatter(np.arange(n_samples // 2), y_train, color="red")
    plt.scatter(np.arange(n_samples // 2) + n_samples // 2, y_test, color="fuchsia")
    plt.plot(np.arange(n_samples), brr.predict(X), color="purple")
    plt.show()
