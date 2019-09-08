# coding=utf-8
"""Support Vector Machines decision margin visualization."""

import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets


if __name__ == "__main__":
    print("Generating data...")
    X, y = datasets.make_blobs(n_samples=1222, centers=2)

    print("Fitting model...")
    model = svm.SVC(kernel="linear", C=200)
    model.fit(X, y)

    print("Plotting model...")
    plt.scatter(X[:, 0], X[:, 1], c=y, s=30, cmap=plt.cm.Paired)
    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    xx = np.linspace(*xlim, 30)
    yy = np.linspace(*ylim, 30)
    YY, XX = np.meshgrid(yy, xx)
    xy = np.vstack([XX.ravel(), YY.ravel()]).T
    Z = model.decision_function(xy).reshape(XX.shape)
    ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5, linestyles=["--", "-", "--"])
    ax.scatter(model.support_vectors_[:, 0], model.support_vectors_[:, 1], s=100, linewidth=1,
               facecolors="none", edgecolors="k")
    plt.show()
