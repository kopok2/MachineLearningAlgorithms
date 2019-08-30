# coding=utf-8
"""Ridge regression.

Linear regression penalizing size of coefficients.

Model prediction:
    y^(w, x) = w0 + w1x1 + w2x2 + ... + wpxp

Model fitting:
    minw||Xw - y||22 + alpha||w||22
"""

import numpy as np
from sklearn import linear_model, datasets
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt


if __name__ == "__main__":
    print("Ridge regression model.")

    print("Loading data...")
    data = datasets.load_diabetes()

    print("Spliting data...")
    X = data.data[:, np.newaxis, 2]
    X_train = X[:-20]
    X_test = X[-20:]
    y = data.target
    y_train = y[:-20]
    y_test = y[-20:]

    print("Fitting model...")
    regression = linear_model.Ridge(alpha=0.5)
    regression.fit(X_train, y_train)

    print("Model coefficients:")
    print(regression.coef_)
    print("Model intercept:")
    print(regression.intercept_)

    print("Predicting with model...")
    pred = regression.predict(X_test)

    print("Evaluating model...")
    print("Mean squared error:")
    print(mean_squared_error(y_test, pred))
    print("Variance score:")
    print(r2_score(y_test, pred))

    print("Plotting predictions...")
    plt.scatter(X_test, y_test, color="purple")
    plt.plot(X_test, pred, color="blue", linewidth=2)
    plt.xticks(())
    plt.yticks(())
    plt.show()
