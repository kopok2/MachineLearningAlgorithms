# coding=utf-8
"""Theil-Sen robust regression."""

import numpy as np
from sklearn import linear_model, datasets
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


if __name__ == "__main__":
    print("Generating data...")
    n_samples, n_outliers, n_features = 2000, 500, 1
    X, y, coef = datasets.make_regression(n_samples=n_samples, n_features=n_features,
                                          n_informative=n_features, noise=10, coef=True)
    X[:n_outliers] = 3 + 0.7 * np.random.normal(size=(n_outliers, n_features))
    y[:n_outliers] = -12 + 10 * np.random.normal(size=n_outliers)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

    print("Fitting models...")
    lr_model = linear_model.LinearRegression()
    lr_model.fit(X_train, y_train)
    robust_model = linear_model.TheilSenRegressor()
    robust_model.fit(X, y)

    print("Evaluating models...")
    print("Linear regression R2 score: {0}".format(r2_score(y_test, lr_model.predict(X_test))))
    print("Theil-Sen regression R2 score: {0}".format(r2_score(y_test, robust_model.predict(X_test))))
    print("Coefficients: Original - {0} LR - {1} T-SR - {2}".format(coef, lr_model.coef_, robust_model.coef_))

    print("Plotting regression...")
    line_X = X
    line_y = lr_model.predict(X)
    line_y_rnc = robust_model.predict(X)

    plt.scatter(X, y, color="green", marker=".", label="Data")
    plt.plot(line_X, line_y, color="blue", linewidth=2, label="Linear Regression")
    plt.plot(line_X, line_y_rnc, color="fuchsia", linewidth=2, label="Theil-Sen Regression")
    plt.legend(loc="upper left")
    plt.xlabel("In")
    plt.ylabel("Y")
    plt.show()
