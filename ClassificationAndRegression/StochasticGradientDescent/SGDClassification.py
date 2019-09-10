# coding=utf-8
"""Stochastic Gradient Classification using Scikit-Learn."""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_blobs
from sklearn.metrics import classification_report, confusion_matrix


if __name__ == "__main__":
    print("Generating data...")
    X, y = make_blobs(n_samples=50, centers=2, cluster_std=0.6)
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    print("Fitting model...")
    sgd = SGDClassifier(alpha=0.0057, max_iter=200)
    sgd.fit(X_train, y_train)

    print("Evaluating model...")
    print(classification_report(y_test, sgd.predict(X_test)))
    print(confusion_matrix(y_test, sgd.predict(X_test)))

    print("Plotting decision boundary...")
    xx = np.linspace(-1, 5, 10)
    yy = np.linspace(-1, 5, 10)
    x1, x2 = np.meshgrid(xx, yy)
    Z = np.empty(x1.shape)
    for (i, j), val in np.ndenumerate(x1):
        X1 = val
        X2 = x2[i, j]
        p = sgd.decision_function([[X1, X2]])
        Z[i, j] = p[0]
    levels = [-1.0, 0.0, 1.0]
    line_styles = ['dashed', 'solid', 'dashed']
    colors = "k"
    plt.contour(x1, x2, Z, levels, colors=colors, linestyles=line_styles)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired, edgecolors="black", s=20)
    plt.axis("tight")
    plt.show()
