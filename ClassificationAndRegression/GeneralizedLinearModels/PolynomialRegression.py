# coding=utf-8
"""Polynomial regression.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline


if __name__ == "__main__":
    print("Generating data...")
    nf = 1000
    kernel = np.array([1, 2.12, 23])
    X = np.column_stack((np.arange(nf), np.arange(nf) ** 2, np.arange(nf) ** 3))
    y = np.dot(X, kernel.T)
    X = np.arange(nf).reshape(-1, 1)
    y = y + 0.02 * np.random.randn(len(y))
    y = y.flatten()
    X_train, X_test = X[nf // 2:], X[:nf // 2]
    y_train, y_test = y[nf // 2:], y[:nf // 2]
    print("Fitting model...")
    model = Pipeline([('Poly', PolynomialFeatures(degree=3)), ("LinReg", linear_model.LinearRegression())])
    model.fit(X_train, y_train)
    print("R2 score: {0}".format(r2_score(y_test, model.predict(X_test))))

    plt.scatter(np.arange(nf // 2), y_train, color="purple")
    plt.scatter(np.arange(nf // 2) + (nf // 2), y_test, color="red")
    plt.plot(np.arange(nf // 2), model.predict(X_train), color="purple")
    plt.plot(np.arange(nf // 2) + (nf // 2), model.predict(X_test), color="red")
    plt.show()
