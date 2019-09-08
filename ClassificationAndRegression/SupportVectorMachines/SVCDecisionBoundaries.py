# coding=utf-8
"""SVC decision boundaries visualized."""


import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets


def msh_grid(x, y, step=0.01, padding=2):
    mnx, mxx = x.min() - padding, x.max() + padding
    mny, mxy = y.min() - padding, y.max() + padding
    return np.meshgrid(np.arange(mnx, mxx, step), np.arange(mny, mxy, step))


def plt_contours(ax, clf, xx, yy, **params):
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out


if __name__ == "__main__":
    print("Loading data...")
    data = datasets.load_iris()
    X = data.data[:, :2]
    y = data.target

    print("Fitting models...")
    C = 1.01
    models = (svm.SVC(kernel="linear", C=C),
              svm.LinearSVC(C=C, max_iter=20000),
              svm.SVC(kernel="rbf", gamma=0.4, C=C),
              svm.SVC(kernel="poly", degree=4, gamma="auto", C=C))
    models = (model.fit(X, y) for model in models)
    names = ("Linear", "LinearSVC", "RBF", "Polynomial")

    print("Plotting decisions...")
    fig, sub = plt.subplots(2, 2)
    plt.subplots_adjust(wspace=0.3, hspace=0.3)
    x0, x1 = X[:, 0], X[:, 1]
    xx, yy = msh_grid(x0, x1)
    for model, name, ax in zip(models, names, sub.flatten()):
        plt_contours(ax, model, xx, yy, cmap=plt.cm.coolwarm, alpha=0.9)
        ax.scatter(x0, x1, c=y, cmap=plt.cm.coolwarm, s=20, edgecolors="k")
        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        ax.set_xlabel("Sepal height")
        ax.set_ylabel("Sepal width")
        ax.set_xticks(())
        ax.set_yticks(())
        ax.set_title(name)
    plt.show()
