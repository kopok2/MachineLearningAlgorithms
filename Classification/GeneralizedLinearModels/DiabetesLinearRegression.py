# coding=utf-8
"""Diabetes dataset regression using sklearn linear regression."""

import numpy as np
from sklearn import datasets
from sklearn import linear_model


if __name__ == "__main__":
    diabetes = datasets.load_diabetes()
    X_train = diabetes.data[:-20]
    X_test = diabetes.data[-20:]
    y_train = diabetes.target[:-20]
    y_test = diabetes.target[-20:]
    regr = linear_model.LinearRegression()
    regr.fit(X_train, y_train)
    print(np.mean((regr.predict(X_test) - y_test) ** 2))
    print(regr.score(X_test, y_test))
