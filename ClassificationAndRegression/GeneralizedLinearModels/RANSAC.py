# coding=utf-8
"""RANSAC.

Random Sample Consensus regression.
"""

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
    rnc_model = linear_model.RANSACRegressor()
    rnc_model.fit(X, y)

    print("Evaluating models...")
    print("Linear regression R2 score: {0}".format(r2_score(y_test, lr_model.predict(X_test))))
    print("RANSAC regression R2 score: {0}".format(r2_score(y_test, rnc_model.predict(X_test))))
    print("Coefficients: Original - {0} LR - {1} RANSAC - {2}".format(coef, lr_model.coef_, rnc_model.estimator_.coef_))

    print("Plotting regression...")
    inlier_mask = rnc_model.inlier_mask_
    outlier_mask = np.logical_not(inlier_mask)
    line_X = X
    line_y = lr_model.predict(X)
    line_y_rnc = rnc_model.predict(X)

    plt.scatter(X[inlier_mask], y[inlier_mask], color="green", marker=".", label="Inliers")
    plt.scatter(X[outlier_mask], y[outlier_mask], color="gold", marker=".", label="Outliers")
    plt.plot(line_X, line_y, color="blue", linewidth=2, label="Linear Regression")
    plt.plot(line_X, line_y_rnc, color="fuchsia", linewidth=2, label="RANSAC Regression")
    plt.legend(loc="upper left")
    plt.xlabel("In")
    plt.ylabel("Y")
    plt.show()
