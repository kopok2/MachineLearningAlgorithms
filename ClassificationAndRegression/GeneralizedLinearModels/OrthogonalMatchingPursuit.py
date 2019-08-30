# coding=utf-8
"""OMP.

Orthogonal Matching Pursuit.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import r2_score
from sklearn.datasets import make_sparse_coded_signal


if __name__ == "__main__":
    print("Generating data...")
    ncomp, nf, nncoef = 256, 10000, 32
    y, X, w = make_sparse_coded_signal(n_samples=1, n_components=ncomp, n_features=nf, n_nonzero_coefs=nncoef)
    idx, = w.nonzero()
    y = y + 0.02 * np.random.randn(len(y))
    y = y.flatten()
    X_train, X_test = X[nf // 2:], X[:nf // 2]
    y_train, y_test = y[nf // 2:], y[:nf // 2]
    print(X, y)
    print("Fitting model...")
    omp = linear_model.OrthogonalMatchingPursuit(n_nonzero_coefs=nncoef)
    omp.fit(X_train, y_train)
    print("R2 score: {0}".format(r2_score(y_test, omp.predict(X_test))))

    plt.scatter(np.arange(nf // 2), y_train, color="purple")
    plt.scatter(np.arange(nf // 2) + (nf // 2), y_test, color="red")
    plt.plot(np.arange(nf // 2), omp.predict(X_train), color="purple")
    plt.plot(np.arange(nf // 2) + (nf // 2), omp.predict(X_test), color="red")
    plt.show()
