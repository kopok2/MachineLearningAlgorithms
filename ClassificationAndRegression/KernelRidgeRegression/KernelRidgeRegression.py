# coding=utf-8
"""Kernel Ridge Regression.

L2-norm regularization combined with kernel trick.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn import kernel_ridge


if __name__ == "__main__":
    print("Generating data...")
    x = np.arange(-20, 20, 0.01)
    y = np.sin(x)
    X = x.reshape(-1, 1)
    n_samples = X.shape[0]
    print(X, y)
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    print("Fitting model...")
    kr = kernel_ridge.KernelRidge(kernel="rbf")
    kr.fit(X_train, y_train)
    print(kr)
    print("R2 score: {0}".format(r2_score(y_test, kr.predict(X_test))))

    print("Plotting predictions...")
    plt.scatter(x, y, color="gold")
    plt.plot(x, kr.predict(X), color="purple")
    plt.show()
