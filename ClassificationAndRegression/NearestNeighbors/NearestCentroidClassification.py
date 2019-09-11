# coding=utf-8
"""Nearest Centroid Classification."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import datasets, neighbors

neighbors_cnt = 15

if __name__ == "__main__":
    print("Loading data...")
    data = datasets.load_iris()
    X, y = data.data[:, :2], data.target
    step = 0.01
    cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
    cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

    for shrinkage in [None, 0.2]:
        model = neighbors.NearestCentroid(shrink_threshold=shrinkage)
        model.fit(X, y)
        y_prediction = model.predict(X)
        print(shrinkage, np.mean(y_prediction == y))
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, step),
                             np.arange(y_min, y_max, step))
        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        plt.figure()
        plt.pcolormesh(xx, yy, Z, cmap=cmap_light)
        plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold,
                    edgecolor='k', s=20)
        plt.title("3-Class classification (shrink_threshold=%r)"
                  % shrinkage)
        plt.axis('tight')
    plt.show()
