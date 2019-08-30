# coding=utf-8
"""Multi-task Elastic Net regression model."""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import ElasticNet, MultiTaskElasticNet
from sklearn.metrics import r2_score

if __name__ == "__main__":
    print("Generating data...")
    rr = np.random.RandomState(17)
    n_samples = 100
    n_features = 40
    n_tasks = 12
    rel_f = 7
    coef = np.zeros((n_tasks, n_features))
    times = np.linspace(0, 2 * np.pi, n_tasks)
    for k in range(rel_f):
        coef[:, k] = np.sin((1.0 + rr.randn(1)) * times + 3 * rr.randn(1))
    X = rr.randn(n_samples, n_features)
    y = np.dot(X, coef.T) + rr.randn(n_samples, n_tasks)
    X_train = X[:-20]
    y_train = y[:-20]
    X_test = X[-20:]
    y_test = y[-20:]

    print("Fitting Elastic Net model...")
    ll = ElasticNet(alpha=0.45)
    ll.fit(X_train, y_train)
    print("R2 score: {0}".format(r2_score(y_test, ll.predict(X_test))))

    print("Fitting Multitask Elastic Net model...")
    ml = MultiTaskElasticNet(alpha=0.45)
    ml.fit(X_train, y_train)
    print("R2 score: {0}".format(r2_score(y_test, ml.predict(X_test))))

    print("Plotting predictions...")
    plt.scatter(X[:, 1], y[:, 1])
    plt.scatter(X[:, 1], ll.predict(X)[:, 1], color="blue")
    plt.scatter(X[:, 1], ml.predict(X)[:, 1], color="red")
    plt.show()
